# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Raw WebSocket transport shared by Live and Omni realtime sessions.

Session payloads, terminal events and error policy belong to the callers.
This layer only assembles connections, exchanges raw frames and releases I/O.
"""

from __future__ import annotations

import base64
import socket
import threading
from typing import Any, Mapping, Optional, TypeVar, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dashscope.v2._config import WS_REALTIME_PATH, workspace_ws_url
from dashscope.v2.transport import default_user_agent
from dashscope.v2.transport._ident import sdk_headers
from dashscope.v2.transport.ws import Connector, _close_websocket, _connect

__all__ = ["WsSessionBase"]

MODULE = "audio"

_UNSET = object()
_CLOSE_TIMEOUT = 1.0

_SelfT = TypeVar("_SelfT", bound="WsSessionBase")


def _as_base64(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    raise TypeError("expected a base64 string or bytes")


class WsSessionBase:
    """One socket per session; subclasses own all protocol state."""

    _LABEL = "realtime"

    def __init__(
        self,
        client,
        *,
        model: str,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> None:
        config = client.config
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        # Keep the derived gateway and workspace header consistent.
        space = workspace or config.workspace
        url = ws_url or workspace_ws_url(
            space,
            WS_REALTIME_PATH,
            config.region,
        )
        parts = urlsplit(url)
        query = parse_qsl(parts.query, keep_blank_values=True)
        if not any(key == "model" for key, _ in query):
            query.append(("model", model))
        url = urlunsplit(parts._replace(query=urlencode(query)))
        all_headers = {
            "user-agent": default_user_agent(),
            **sdk_headers(MODULE),
        }
        key = api_key or config.api_key
        if key:
            all_headers["Authorization"] = f"Bearer {key}"
        if space:
            all_headers["X-DashScope-WorkSpace"] = space
        user_headers = dict(headers or {})
        user_keys = {key.lower() for key in user_headers}
        all_headers = {
            key: value
            for key, value in all_headers.items()
            if key.lower() not in user_keys
        }
        all_headers.update(user_headers)
        self._model = model
        self._wire_model = next(
            value for key, value in query if key == "model"
        )
        self._url = url
        self._headers = all_headers
        self._connector = connector
        self._timeout = config.timeout if timeout is None else timeout
        self._lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._recv_lock = threading.Lock()
        self._generation = 0
        self._ws = None
        self._closing = False
        self._reset_state()

    def _reset_state(self) -> None:
        self._closed = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def url(self) -> str:
        return self._url

    def connect(self: _SelfT) -> _SelfT:
        """Open the connection, either explicitly or on context entry."""
        with self._lock:
            if self._closing:
                raise RuntimeError(f"{self._LABEL} session is closing")
            if self._ws is not None:
                raise RuntimeError(
                    f"{self._LABEL} session is already open",
                )
            self._generation += 1
            try:
                self._ws = _connect(
                    self._url,
                    self._headers,
                    self._connector,
                    self._timeout,
                )
                # pylint: disable=import-outside-toplevel
                from websocket import WebSocket

                if isinstance(self._ws, WebSocket):
                    # Raw writes share the native lock with automatic Close
                    # replies; send_frame() acquires this same lock again.
                    self._ws.lock = threading.RLock()
            except BaseException:
                self.close()
                self._reset_state()
                raise
            self._reset_state()
        return self

    def __enter__(self: _SelfT) -> _SelfT:
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _send_raw(self, ws: Any, raw: Union[str, bytes]) -> None:
        """Write without retries or holding the session state lock."""
        from websocket import (  # pylint: disable=import-outside-toplevel
            WebSocket,
            WebSocketConnectionClosedException,
        )

        with self._lock:
            if ws is not self._ws:
                raise WebSocketConnectionClosedException("socket is closed")
        if isinstance(ws, WebSocket):
            with ws.lock:
                if not ws.connected:
                    raise WebSocketConnectionClosedException(
                        "socket is closed",
                    )
                ws.send(raw)
        else:
            ws.send(raw)

    def _recv_raw(self, ws: Any, generation: Optional[int] = None) -> Any:
        """Preserve active I/O errors, but discard failures from detached sockets."""
        with self._lock:
            if ws is not self._ws or (
                generation is not None and generation != self._generation
            ):
                return None
        try:
            return ws.recv()
        except Exception:
            with self._lock:
                if ws is not self._ws or (
                    generation is not None and generation != self._generation
                ):
                    return None
            raise

    def close(self) -> None:
        """Send a normal WebSocket Close, allowing one second for the handshake."""
        self._close(abort=False)

    def _close(self, *, abort: bool) -> None:
        from websocket import (
            WebSocket,
        )  # pylint: disable=import-outside-toplevel

        with self._lock:
            ws, self._ws = self._ws, None
            if ws is not None:
                self._closed = self._closing = True
        if ws is None:
            return

        deadline = None
        sending = False
        try:
            if isinstance(ws, WebSocket) and ws.sock is not None:
                sock = ws.sock

                def interrupt() -> None:
                    # close() sets connected=False before sending, so abort()
                    # cannot wake its blocked writer or reader at the deadline.
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

                def release() -> None:
                    interrupt()
                    # Retain the descriptor until the native writer wakes;
                    # closing it too early can strand socket.send on macOS.
                    writing = ws.lock.acquire(timeout=0.1)
                    try:
                        ws.shutdown()
                    finally:
                        if writing:
                            ws.lock.release()

                if abort:
                    release()
                    return
                deadline = threading.Timer(_CLOSE_TIMEOUT, interrupt)
                deadline.daemon = True
                try:
                    deadline.start()
                except Exception:
                    release()
                    return
                # Complete an in-flight data frame before sending Close. The
                # reentrant lock also permits cleanup from a failed send.
                # A context manager cannot bound lock-acquisition time.
                # pylint: disable-next=consider-using-with
                sending = self._send_lock.acquire(timeout=_CLOSE_TIMEOUT)
                if not sending:
                    release()
                    return
            _close_websocket(ws)
        finally:
            if deadline is not None:
                deadline.cancel()
            if sending:
                self._send_lock.release()
            with self._lock:
                self._closing = False
