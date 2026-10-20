# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared duplex websocket connection with per-task demultiplexing.

``WsConnection`` owns one physical socket plus a background reader
thread that routes JSON events to tasks by ``header.task_id``. A
``WsTask`` exposes the same interface as ``WsSession``, so the audio
resources (asr / tts / translation / tingwu) can run several tasks
over one connection -- concurrently for JSON-framed traffic, or
sequentially with no re-handshake between tasks.

Binary frames carry no task id, so each direction allows a single
owner at a time: the first task to send binary claims the uplink, and
the downlink goes to the task that claimed it
(``claim_binary_downlink``) or, with exactly one active task, to that
task implicitly. Ownership is released when the owning task finishes
or closes. Concurrent binary output on one connection is impossible
by protocol; use one connection per binary-downlink task in that
case.

The realtime family is not multiplexable: its protocol has no task id
(one socket is one session, closed when the session ends), so it keeps
opening a dedicated connection per session.
"""

# WsConnection and WsTask deliberately share the owning client's
# internals (send/claim/detach) — they are one transport layer.
# pylint: disable=redefined-builtin,protected-access

from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from dashscope.v2._config import (
    WS_INFERENCE_PATH,
    settings,
    workspace_ws_url,
)
from dashscope.v2.errors import APIConnectionError
from dashscope.v2.transport.ws import (
    EVENT_FAILED,
    EVENT_FINISHED,
    EVENT_STARTED,
    Connector,
    WsEventError,
    _assemble_headers,
    _close_websocket,
    _connect,
    _EventProcessor,
    _recv_frame,
)

__all__ = ["WsConnection", "WsTask"]

_FRAME = "frame"
_ERROR = "error"

_QueueItem = Tuple[str, Any]


class WsTask(_EventProcessor):
    """One task on a shared connection; same interface as ``WsSession``.

    ``close()`` only detaches the task from the connection -- the
    shared socket stays open for the remaining tasks.
    """

    def __init__(
        self,
        connection: "WsConnection",
        task_id: str,
        timeout: Optional[float],
    ) -> None:
        self._conn = connection
        self._task_id = task_id
        self._timeout = timeout
        self._queue: "queue.SimpleQueue[_QueueItem]" = queue.SimpleQueue()
        self._request_id = task_id
        self._lock = threading.Lock()
        self._run_sent = False
        self._started = False
        self._input_finished = False
        self._cancel_sent = False
        self._finished = False
        self._closed = False
        # Reader-side routing uses the connection lock, not the send lock.
        self._routing_active = False
        self._routing_finished = False
        self._error: Optional[BaseException] = None

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def finished(self) -> bool:
        return self._finished

    def _enqueue(self, item: _QueueItem) -> None:
        """Called with the connection lock held; the first error ends input."""
        if not self._closed and self._error is None:
            if item[0] == _ERROR:
                self._error = item[1]
            self._queue.put(item)

    def _frame(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "header": {
                "action": action,
                "task_id": self._task_id,
                "streaming": "duplex",
            },
            "payload": payload,
        }

    def run_task(
        self,
        *,
        task_group: str,
        task: str,
        function: str,
        model: str,
        parameters: Optional[Dict[str, Any]] = None,
        input: Optional[Dict[str, Any]] = None,
        resources: Optional[List[Dict[str, Any]]] = None,
        streaming: str = "duplex",
    ) -> Dict[str, Any]:
        """Send run-task and wait for task-started before allowing input."""
        payload: Dict[str, Any] = {
            "task_group": task_group,
            "task": task,
            "function": function,
            "model": model,
            "input": input if input is not None else {},
        }
        if parameters is not None:
            payload["parameters"] = parameters
        if resources is not None:
            payload["resources"] = resources
        frame = self._frame("run-task", payload)
        frame["header"]["streaming"] = streaming
        with self._lock:
            if self._closed or self._run_sent:
                raise RuntimeError("websocket task already started or closed")
            self._run_sent = True
            self._conn._send(frame)
        try:
            event = self.recv_event()
            if event["header"].get("event") != EVENT_STARTED:
                raise WsEventError(
                    "expected task-started, got "
                    f"{event['header'].get('event')!r}",
                    request_id=self.request_id,
                )
            self._started = True
            return event
        except BaseException:
            self.close()
            raise

    def _check_input(self) -> None:
        if self._closed or not self._started or self._finished:
            raise RuntimeError("websocket task is not active")
        if self._input_finished:
            raise RuntimeError("websocket task input is finished")

    def claim_binary_downlink(self) -> None:
        """Route binary output frames to this task.

        Binary frames carry no task id, so when more than one task is
        active the downlink owner must be claimed explicitly (with a
        single active task it is assigned implicitly).
        """
        self._conn._claim_binary_downlink(self)

    def send_binary(self, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("audio data must be bytes")
        with self._lock:
            self._check_input()
            self._conn._claim_binary_uplink(self)
            self._conn._send(bytes(data), opcode=2)

    def continue_task(self, input: Dict[str, Any]) -> None:
        with self._lock:
            self._check_input()
            self._conn._send(self._frame("continue-task", {"input": input}))

    def finish_task(self, input: Optional[Dict[str, Any]] = None) -> None:
        """End input, or cancel synthesis while output is still arriving."""
        with self._lock:
            if self._closed or not self._started:
                raise RuntimeError("websocket task is not active")
            if self._finished:
                return
            cancel = input is not None and input.get("directive") == "cancel"
            if self._cancel_sent or (self._input_finished and not cancel):
                return
            self._conn._send(
                self._frame("finish-task", {"input": input or {}}),
            )
            self._input_finished = True
            self._cancel_sent = cancel

    def recv_event(self) -> Dict[str, Any]:
        """Read the next control event; use recv for binary output."""
        while True:
            frame = self.recv()
            if frame is None:
                raise RuntimeError("websocket task is finished")
            if isinstance(frame, dict):
                return frame

    def recv(self) -> Union[Dict[str, Any], bytes, None]:
        """Read queued frames before surfacing a persistent close/error."""
        if self._finished:
            return None
        try:
            kind, payload = self._queue.get(timeout=self._timeout)
        except queue.Empty:
            raise TimeoutError(
                "timed out waiting for a websocket event",
            ) from None
        if kind == _ERROR:
            # Requeue the terminal marker so subsequent readers also fail.
            self._queue.put((kind, payload))
            raise payload
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        return self._process_event(payload)

    def close(self) -> None:
        """Detach and wake receivers; the shared socket stays open."""
        self._conn._detach(self)


class WsConnection:
    """One physical duplex websocket shared by multiple tasks.

    Construction performs the handshake once; every ``open_task`` then
    runs an independent task on the same socket. ``close()`` shuts the
    socket down and fails every task still attached.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        region: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        module: Optional[str] = None,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        self._url = (
            url
            or settings.websocket_base_url
            or workspace_ws_url(workspace, WS_INFERENCE_PATH, region)
        )
        self._timeout = timeout
        self._tasks: Dict[str, WsTask] = {}
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._closed = False
        self._binary_uplink_owner: Optional[WsTask] = None
        self._binary_downlink_owner: Optional[WsTask] = None
        self._ws = _connect(
            self._url,
            _assemble_headers(api_key, workspace, headers, module),
            connector,
            timeout,
        )
        # Reads happen only on the reader thread and per-task timeouts
        # are enforced on the demux queues, so the socket itself must
        # block forever -- a socket read timeout would kill the reader
        # during any quiet period longer than the timeout.
        try:
            settimeout = getattr(self._ws, "settimeout", None)
            if callable(settimeout):
                settimeout(None)
            self._reader = threading.Thread(
                target=self._reader_main,
                name="dashscope-ws-reader",
                daemon=True,
            )
            self._reader.start()
        except BaseException:
            self.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def open_task(self, *, timeout: Optional[float] = None) -> WsTask:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._lock:
            if self._closed:
                raise RuntimeError("websocket connection is closed")
            task = WsTask(
                self,
                uuid.uuid4().hex,
                self._timeout if timeout is None else timeout,
            )
            self._tasks[task.task_id] = task
            return task

    def _send(self, data: Any, *, opcode: int = 1) -> None:
        with self._send_lock:
            if self._closed:
                raise RuntimeError("websocket connection is closed")
            try:
                if opcode == 2:
                    self._ws.send(data, opcode=2)
                else:
                    self._ws.send(json.dumps(data))
            except BaseException:
                self.close()
                raise

    def _check_binary_owner(self, task: WsTask) -> None:
        """Validate under the connection lock; pre-start claims are allowed."""
        if self._closed:
            raise RuntimeError("websocket connection is closed")
        if (
            self._tasks.get(task.task_id) is not task
            or task._closed
            or task._finished
            or task._routing_finished
        ):
            raise RuntimeError("websocket task is not active")

    def _claim_binary_uplink(self, task: WsTask) -> None:
        with self._lock:
            self._check_binary_owner(task)
            owner = self._binary_uplink_owner
            if owner is not None and owner is not task:
                raise RuntimeError(
                    "another task owns the binary uplink on this connection",
                )
            self._binary_uplink_owner = task

    def _claim_binary_downlink(self, task: WsTask) -> None:
        with self._lock:
            self._check_binary_owner(task)
            owner = self._binary_downlink_owner
            if owner is not None and owner is not task:
                raise RuntimeError(
                    "another task owns the binary downlink on this connection",
                )
            self._binary_downlink_owner = task

    def _release_binary(self, task: WsTask) -> None:
        """Release both directions with the connection lock already held."""
        if self._binary_uplink_owner is task:
            self._binary_uplink_owner = None
        if self._binary_downlink_owner is task:
            self._binary_downlink_owner = None

    def _detach(self, task: WsTask) -> None:
        with self._lock:
            task._enqueue((_ERROR, RuntimeError("websocket task is closed")))
            task._closed = True
            task._routing_active = False
            if self._tasks.get(task.task_id) is task:
                self._tasks.pop(task.task_id)
            self._release_binary(task)

    def _reader_main(self) -> None:
        try:
            while True:
                raw = _recv_frame(self._ws)
                if raw is None or raw == "":
                    raise APIConnectionError(
                        "websocket connection closed by peer",
                    )
                if isinstance(raw, (bytes, bytearray)):
                    self._route_binary(bytes(raw))
                else:
                    self._route_event(json.loads(raw))
        except BaseException as exc:
            self._fail(exc)

    def _route_event(self, event: Any) -> None:
        header = event.get("header") if isinstance(event, dict) else None
        if not isinstance(header, dict):
            raise WsEventError("expected a websocket event header")
        task_id = header.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise WsEventError("expected a non-empty websocket task_id")
        with self._lock:
            task = self._tasks.get(task_id)
            if self._closed or task is None or task._routing_finished:
                # Late events for detached or terminal tasks are harmless.
                return
            name = header.get("event")
            if name == EVENT_STARTED:
                task._routing_active = True
            elif name in (EVENT_FINISHED, EVENT_FAILED):
                task._routing_active = False
                task._routing_finished = True
                self._release_binary(task)
            task._enqueue((_FRAME, event))

    def _route_binary(self, data: bytes) -> None:
        with self._lock:
            if self._closed:
                return
            owner = self._binary_downlink_owner
            if owner is None:
                active = [
                    task
                    for task in self._tasks.values()
                    if task._routing_active
                ]
                if not active:
                    # Late or garbage binary with no task to receive it.
                    return
                if len(active) != 1:
                    raise WsEventError(
                        "binary frame with multiple active tasks; claim the "
                        "downlink with claim_binary_downlink()",
                    )
                owner = active[0]
                self._binary_downlink_owner = owner
            owner._enqueue((_FRAME, data))

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._closed:
                # close() already broadcast; the reader's socket error is
                # just the shutdown echo.
                return
            self._closed = True
            self._binary_uplink_owner = None
            self._binary_downlink_owner = None
        _close_websocket(self._ws)
        self._broadcast(exc)

    def _broadcast(self, exc: BaseException) -> None:
        with self._lock:
            for task in self._tasks.values():
                task._routing_active = False
                task._enqueue((_ERROR, exc))

    def close(self) -> None:
        self._fail(APIConnectionError("websocket connection closed"))
        # Initialization may fail before the reader is created or started.
        reader = getattr(self, "_reader", None)
        if (
            reader is not None
            and reader.is_alive()
            and threading.current_thread() is not reader
        ):
            reader.join(timeout=2)
