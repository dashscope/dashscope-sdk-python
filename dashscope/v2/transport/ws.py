# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""DashScope duplex websocket framing for realtime audio."""

# pylint: disable=redefined-builtin

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from dashscope.v2._config import (
    WS_INFERENCE_PATH,
    settings,
    workspace_ws_url,
)
from dashscope.v2.errors import (
    APIConnectionError,
    DashScopeError,
    error_from_business_code,
)
from dashscope.v2.transport import default_user_agent
from dashscope.v2.transport._ident import sdk_headers

__all__ = [
    "Connector",
    "EVENT_FINISHED",
    "EVENT_GENERATED",
    "EVENT_STARTED",
    "WsEventError",
    "WsSession",
]

EVENT_STARTED = "task-started"
EVENT_GENERATED = "result-generated"
EVENT_FINISHED = "task-finished"
EVENT_FAILED = "task-failed"

Connector = Callable[[str, Dict[str, str]], Any]


class WsEventError(DashScopeError):
    """The server sent an unexpected websocket protocol event."""


def _default_connector(
    url: str,
    headers: Dict[str, str],
    *,
    timeout: Optional[float] = None,
) -> Any:
    import websocket

    return websocket.create_connection(url, header=headers, timeout=timeout)


def _close_websocket(ws: Any) -> None:
    from websocket import WebSocket

    try:
        try:
            ws.close()
        finally:
            # Peer close sets connected=False without releasing the socket.
            if isinstance(ws, WebSocket):
                ws.shutdown()
    except Exception:
        pass


def _assemble_headers(
    api_key: Optional[str],
    workspace: Optional[str],
    headers: Optional[Mapping[str, str]],
    module: Optional[str],
) -> Dict[str, str]:
    all_headers = dict(headers or {})
    if api_key:
        all_headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        all_headers["X-DashScope-WorkSpace"] = workspace
    all_headers.setdefault("user-agent", default_user_agent())
    provided = {key.lower() for key in all_headers}
    for key, value in sdk_headers(module).items():
        if key.lower() not in provided:
            all_headers[key] = value
    return all_headers


def _connect(
    url: str,
    headers: Dict[str, str],
    connector: Optional[Connector],
    timeout: Optional[float],
) -> Any:
    if connector is None:
        ws = _default_connector(url, headers, timeout=timeout)
    else:
        ws = connector(url, headers)
    try:
        if timeout is not None:
            ws.settimeout(timeout)
    except BaseException:
        _close_websocket(ws)
        raise
    return ws


def _recv_frame(ws: Any) -> Any:
    """Read one frame, normalizing a library close to a clean EOF (``None``).

    A peer abort or a concurrent close() surfaces as
    ``WebSocketConnectionClosedException`` rather than a clean EOF; mapping
    it to ``None`` lets every caller treat it as premature EOF and keep the
    error inside the DashScope hierarchy. Shared by ``WsSession.recv`` and
    ``WsConnection``'s reader thread so the normalization cannot drift.
    """
    from websocket import (  # pylint: disable=import-outside-toplevel
        WebSocketConnectionClosedException,
    )

    try:
        return ws.recv()
    except WebSocketConnectionClosedException:
        return None


class _EventProcessor:
    """Shared event-header bookkeeping for WsSession and WsTask."""

    _request_id: str
    _finished: bool

    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(event, dict) or not isinstance(
            event.get("header"),
            dict,
        ):
            raise WsEventError("expected a websocket event header")
        header = event["header"]
        self._request_id = (
            header.get("request_id")
            or header.get("task_id")
            or self._request_id
        )
        if header.get("event") == EVENT_FAILED:
            self._finished = True
            raise error_from_business_code(
                message=header.get("error_message") or "task failed",
                code=header.get("error_code"),
                body=event,
                headers={"x-request-id": self._request_id},
            )
        if header.get("event") == EVENT_FINISHED:
            self._finished = True
        return event


class WsSession(_EventProcessor):
    """One task with an independent sender and receiver on a duplex socket."""

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
        self._task_id = uuid.uuid4().hex
        self._request_id = self._task_id
        self._lock = threading.Lock()
        self._run_sent = False
        self._started = False
        self._input_finished = False
        self._cancel_sent = False
        self._finished = False
        self._closed = False
        self._ws = _connect(
            self._url,
            _assemble_headers(api_key, workspace, headers, module),
            connector,
            timeout,
        )

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def finished(self) -> bool:
        return self._finished

    def _frame(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "header": {
                "action": action,
                "task_id": self._task_id,
                "streaming": "duplex",
            },
            "payload": payload,
        }

    def _send(self, data: Any, *, opcode: int = 1) -> None:
        try:
            if opcode == 2:
                self._ws.send(data, opcode=2)
            else:
                self._ws.send(json.dumps(data))
        except BaseException:
            self.close()
            raise

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
            self._send(frame)
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

    def send_binary(self, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("audio data must be bytes")
        with self._lock:
            self._check_input()
            self._send(bytes(data), opcode=2)

    def continue_task(self, input: Dict[str, Any]) -> None:
        with self._lock:
            self._check_input()
            self._send(self._frame("continue-task", {"input": input}))

    def finish_task(self, input: Optional[Dict[str, Any]] = None) -> None:
        """End input, or cancel synthesis while output is still arriving.

        Most tasks finish with an empty input; TingWu realtime sends its
        stop directive (appId + directive) here instead.
        """
        with self._lock:
            if self._closed or not self._started:
                raise RuntimeError("websocket task is not active")
            if self._finished:
                return
            cancel = input is not None and input.get("directive") == "cancel"
            if self._cancel_sent or (self._input_finished and not cancel):
                return
            self._send(self._frame("finish-task", {"input": input or {}}))
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
        """Read JSON or audio; premature EOF is not task completion."""
        if self._finished:
            return None
        if self._closed:
            raise RuntimeError("websocket session is closed")
        try:
            raw = _recv_frame(self._ws)
            if raw is None or raw == "":
                raise APIConnectionError(
                    "websocket closed before task-finished",
                    request_id=self.request_id,
                )
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw)
            return self._process_event(json.loads(raw))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _close_websocket(self._ws)
