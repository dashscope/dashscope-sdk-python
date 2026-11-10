# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Qwen-Live protocol over a single business WebSocket."""

from __future__ import annotations

import base64
import json
import math
import socket
import uuid
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union, cast

from dashscope.v2.resources.audio._ws_session import (
    _UNSET,
    WsSessionBase,
    _as_base64,
)
from dashscope.v2.errors import (
    APIConnectionError,
    APITimeoutError,
    DashScopeError,
    error_from_business_code,
    error_from_status,
)
from dashscope.v2.transport.ws import Connector, WsEventError

__all__ = [
    "EVENT_ERROR",
    "EVENT_RESPONSE_CREATE",
    "EVENT_RESPONSE_EVENT",
    "EVENT_RESPONSE_ITEM_CREATE",
    "EVENT_SESSION_CLOSE",
    "EVENT_SESSION_CLOSED",
    "EVENT_SESSION_COMMENTARY_APPEND",
    "EVENT_SESSION_COMMENTARY_APPENDED",
    "EVENT_SESSION_CREATED",
    "EVENT_SESSION_DELEGATION_CREATED",
    "EVENT_SESSION_EVENT",
    "EVENT_SESSION_INPUT_AUDIO_APPEND",
    "EVENT_SESSION_INPUT_AUDIO_MUTE",
    "EVENT_SESSION_INPUT_AUDIO_MUTED",
    "EVENT_SESSION_INPUT_AUDIO_UNMUTE",
    "EVENT_SESSION_INPUT_AUDIO_UNMUTED",
    "EVENT_SESSION_INPUT_TRANSCRIPT_DELTA",
    "EVENT_SESSION_INSTRUCTION_APPEND",
    "EVENT_SESSION_INSTRUCTIONS_APPENDED",
    "EVENT_SESSION_ITEM_CREATE",
    "EVENT_SESSION_OUTPUT_AUDIO_DELTA",
    "EVENT_SESSION_OUTPUT_AUDIO_INTERRUPTED",
    "EVENT_SESSION_OUTPUT_TRANSCRIPT_DELTA",
    "EVENT_SESSION_THINKING_APPEND",
    "EVENT_SESSION_THINKING_APPENDED",
    "EVENT_SESSION_UPDATE",
    "EVENT_SESSION_UPDATED",
    "EVENT_SESSION_USAGE_UPDATED",
    "Live",
    "LiveSession",
]

# Wire names of the client events this session sends.
EVENT_SESSION_UPDATE = "session.update"
EVENT_SESSION_INPUT_AUDIO_APPEND = "session.input_audio.append"
EVENT_SESSION_INPUT_AUDIO_MUTE = "session.input_audio.mute"
EVENT_SESSION_INPUT_AUDIO_UNMUTE = "session.input_audio.unmute"
EVENT_SESSION_INSTRUCTION_APPEND = "session.instruction.append"
EVENT_SESSION_THINKING_APPEND = "session.thinking.append"
EVENT_SESSION_COMMENTARY_APPEND = "session.commentary.append"
EVENT_SESSION_ITEM_CREATE = "session.item.create"
EVENT_RESPONSE_ITEM_CREATE = "response.item.create"
EVENT_RESPONSE_CREATE = "response.create"
EVENT_SESSION_CLOSE = "session.close"

# Wire names of the server events this session receives.
EVENT_ERROR = "error"
EVENT_SESSION_CREATED = "session.created"
EVENT_SESSION_UPDATED = "session.updated"
EVENT_SESSION_INPUT_AUDIO_MUTED = "session.input_audio.muted"
EVENT_SESSION_INPUT_AUDIO_UNMUTED = "session.input_audio.unmuted"
EVENT_SESSION_OUTPUT_AUDIO_DELTA = "session.output_audio.delta"
EVENT_SESSION_OUTPUT_AUDIO_INTERRUPTED = "session.output_audio.interrupted"
EVENT_SESSION_INPUT_TRANSCRIPT_DELTA = "session.input_transcript.delta"
EVENT_SESSION_OUTPUT_TRANSCRIPT_DELTA = "session.output_transcript.delta"
EVENT_SESSION_EVENT = "session.event"
EVENT_RESPONSE_EVENT = "response.event"
EVENT_SESSION_USAGE_UPDATED = "session.usage.updated"
EVENT_SESSION_CLOSED = "session.closed"
EVENT_SESSION_DELEGATION_CREATED = "session.delegation.created"
EVENT_SESSION_INSTRUCTIONS_APPENDED = "session.instructions.appended"
EVENT_SESSION_THINKING_APPENDED = "session.thinking.appended"
EVENT_SESSION_COMMENTARY_APPENDED = "session.commentary.appended"

# Typed update_session arguments; the parameters passthrough may not shadow them.
_RESERVED_SESSION_FIELDS = frozenset(
    {
        "model",
        "instructions",
        "voice",
        "tts",
        "input_audio_format",
        "output_audio_format",
        "tools",
        "delegation",
        "transport",
        "modalities",
        "context",
        "turn_detection",
        "input_audio_transcription",
    },
)


def _transport_error(exc: BaseException, operation: str) -> BaseException:
    """Normalize Live failures without changing Omni's native error contract."""
    from websocket import (  # pylint: disable=import-outside-toplevel
        WebSocketBadStatusException,
        WebSocketException,
        WebSocketTimeoutException,
    )

    if isinstance(exc, DashScopeError):
        return exc
    if (
        isinstance(exc, WebSocketBadStatusException)
        and exc.status_code is not None
    ):
        body = exc.resp_body
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            pass
        return error_from_status(
            exc.status_code,
            message=str(exc),
            body=body,
            headers=exc.resp_headers,
        )
    if isinstance(
        exc,
        (TimeoutError, socket.timeout, WebSocketTimeoutException),
    ):
        return APITimeoutError(f"websocket {operation} timed out: {exc}")
    if isinstance(exc, (OSError, WebSocketException)):
        return APIConnectionError(f"websocket {operation} failed: {exc}")
    return exc


def _ms_window(
    start_ms: Optional[float],
    end_ms: Optional[float],
) -> Dict[str, Any]:
    if (start_ms is None) != (end_ms is None):
        raise ValueError("start_ms and end_ms must be given together")
    if start_ms is None or end_ms is None:
        return {}
    for value in (start_ms, end_ms):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("start_ms and end_ms must be finite numbers")
        if value < 0:
            raise ValueError("start_ms and end_ms must be non-negative")
    if end_ms < start_ms:
        raise ValueError("end_ms must be >= start_ms")
    return {"start_ms": start_ms, "end_ms": end_ms}


class LiveSession(WsSessionBase):
    """Live session; send helpers return request IDs, not acknowledgements.

    events() yields recoverable error events; recv_event() raises them while
    leaving the connection open. Fatal errors and unexpected disconnects raise
    SDK exceptions in both forms. close() cancels locally, without draining.
    """

    _FINISH_EVENT = EVENT_SESSION_CLOSE
    _TERMINAL_EVENT = EVENT_SESSION_CLOSED
    _LABEL = "live"

    def _reset_state(self) -> None:
        super()._reset_state()
        self._finish_requested = False
        self._finish_event_id: Optional[str] = None
        self._finished = False
        self._failure: Optional[DashScopeError] = None
        self.session_id: Optional[str] = None
        self.last_message: Optional[Dict[str, Any]] = None
        self.close_reason: Optional[str] = None

    def connect(self) -> LiveSession:
        """Open the shared transport, mapping only Live's connection errors."""
        try:
            return super().connect()
        except BaseException as exc:
            error = _transport_error(exc, "connect")
            if error is exc:
                raise
            raise error from exc

    def send_event(self, event: Dict[str, Any]) -> str:
        """Send a raw event and return its supplied or generated event_id."""
        with self._send_lock:
            with self._lock:
                ws, generation = self._ws, self._generation
                if ws is None:
                    raise RuntimeError("live session is not open")
                if self._finish_requested:
                    if event.get("type") == EVENT_SESSION_CLOSE:
                        assert self._finish_event_id is not None
                        return self._finish_event_id
                    raise RuntimeError("live session is finishing")
                payload = dict(event)
                kind = payload.get("type")
                if not isinstance(kind, str) or not kind.strip():
                    raise ValueError("event type must be a nonempty string")
                payload.setdefault("event_id", "event_" + uuid.uuid4().hex)
                event_id = payload["event_id"]
                if (
                    not isinstance(event_id, str)
                    or not 1 <= len(event_id) <= 128
                ):
                    raise ValueError(
                        "event_id must be a string of 1 to 128 characters",
                    )
                raw = json.dumps(payload)
                if payload.get("type") == EVENT_SESSION_CLOSE:
                    self._finish_requested = True
                    self._finish_event_id = event_id
            try:
                self._send_raw(ws, raw)
            except BaseException as exc:
                error = _transport_error(exc, "send")
                with self._lock:
                    if ws is self._ws and generation == self._generation:
                        if isinstance(error, DashScopeError):
                            self._failure = error
                        self.close()
                if error is exc:
                    raise
                raise error from exc
            return event_id

    def recv_event(self) -> Optional[Dict[str, Any]]:
        """Receive an event; recoverable errors raise but leave the socket open."""
        return self._recv_event(raise_request_errors=True)

    # Keep ownership checks and protocol state in one receive critical section.
    # pylint: disable-next=too-many-branches,too-many-statements
    def _recv_event(  # pylint: disable=too-many-return-statements
        self,
        *,
        raise_request_errors: bool,
        expected_generation: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        # Socket I/O never holds the state lock needed by send or close.
        with self._recv_lock:
            with self._lock:
                if (
                    expected_generation is not None
                    and expected_generation != self._generation
                ):
                    return None
                ws, generation = self._ws, self._generation
                if ws is None:
                    self._check_unfinished()
                    if self._closed:
                        return None
                    raise RuntimeError("live session is not open")
            failure = None
            recoverable = False
            try:
                try:
                    raw = self._recv_raw(ws, generation)
                except Exception as exc:
                    with self._lock:
                        if generation != self._generation:
                            return None
                        if ws is not self._ws:
                            self._check_unfinished()
                            return None
                    error = _transport_error(exc, "receive")
                    if error is exc:
                        raise
                    raise error from exc
                with self._lock:
                    if generation != self._generation:
                        return None
                    if ws is not self._ws:
                        self._check_unfinished()
                        return None
                    if raw is None or raw == "" or raw == b"":
                        raise APIConnectionError(
                            "live websocket disconnected before session.closed",
                        )
                    try:
                        if isinstance(raw, (bytes, bytearray)):
                            raw = raw.decode("utf-8")
                        event = json.loads(raw)
                    except (ValueError, TypeError) as exc:
                        raise WsEventError(
                            "expected a UTF-8 JSON event object",
                            body=raw,
                        ) from exc
                    if not isinstance(event, dict):
                        raise WsEventError(
                            "expected a JSON event object",
                            body=event,
                        )
                    if (
                        not isinstance(event.get("type"), str)
                        or not event["type"]
                    ):
                        raise WsEventError(
                            "expected a non-empty string event type",
                            body=event,
                        )
                    self.last_message = event
                    self._track_event(event)
                    if event["type"] == EVENT_SESSION_CLOSED:
                        self._finished = True
                        self.close()
                        self._check_unfinished()
                    elif event["type"] == EVENT_ERROR:
                        error = event.get("error")
                        if error is None:
                            error = {}
                        if not isinstance(error, dict):
                            raise WsEventError(
                                "expected a JSON error object",
                                body=event,
                            )
                        code = error.get("code")
                        if code is None or code == "":
                            code = event.get("code")
                        if code is not None and not isinstance(code, str):
                            raise WsEventError(
                                "expected a string error code",
                                body=event,
                            )
                        failure = error_from_business_code(
                            message=error.get("message") or "live error",
                            code=code,
                            body=event,
                        )
                        failure.status_code = None
                        recoverable = (
                            error.get("type") == "invalid_request_error"
                        )
                        if recoverable:
                            if self._finish_event_id is not None and (
                                error.get("event_id") == self._finish_event_id
                            ):
                                self._finish_requested = False
                                self._finish_event_id = None
                        else:
                            self._failure = failure
                            self.close()
            except BaseException as exc:
                with self._lock:
                    if ws is self._ws and generation == self._generation:
                        if isinstance(exc, DashScopeError):
                            self._failure = exc
                        self.close()
                raise
            if failure is not None and (
                raise_request_errors or not recoverable
            ):
                raise failure
            return event

    def _check_unfinished(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _track_event(self, event: Dict[str, Any]) -> None:
        if event["type"] in {EVENT_SESSION_CREATED, EVENT_SESSION_UPDATED}:
            session = event.get("session")
            if session is None:
                session = {}
            if not isinstance(session, dict):
                raise WsEventError(
                    "expected a JSON session object",
                    body=event,
                )
            self.session_id = session.get("id", self.session_id)
        elif event["type"] == EVENT_SESSION_CLOSED:
            self.close_reason = event.get("reason")

    def events(self) -> Iterator[Dict[str, Any]]:
        """Yield recoverable errors; fatal failures raise, never becoming EOF.

        Once started, an iterator belongs to that connection. Reconnecting
        requires a new iterator; audio and tasks are never replayed.
        """
        with self._lock:
            generation = self._generation
        while True:
            event = self._recv_event(
                raise_request_errors=False,
                expected_generation=generation,
            )
            if event is None:
                return
            yield event
            if event["type"] == EVENT_SESSION_CLOSED:
                return

    def cancel(self) -> None:
        """Immediately abort the socket without a WebSocket close handshake."""
        self._close(abort=True)

    def _send(self, event: Dict[str, Any], event_id: Optional[str]) -> str:
        if event_id is not None:
            event["event_id"] = event_id
        return self.send_event(event)

    def update_session(
        self,
        *,
        model: str,
        delegation: Dict[str, Any],
        instructions: str = cast(str, _UNSET),
        voice: str = cast(str, _UNSET),
        tts: Dict[str, Any] = cast(Dict[str, Any], _UNSET),
        input_audio_format: str = cast(str, _UNSET),
        output_audio_format: str = cast(str, _UNSET),
        tools: List[Dict[str, Any]] = cast(List[Dict[str, Any]], _UNSET),
        transport: Dict[str, Any] = cast(Dict[str, Any], _UNSET),
        modalities: List[str] = cast(List[str], _UNSET),
        context: Dict[str, Any] = cast(Dict[str, Any], _UNSET),
        turn_detection: Optional[Dict[str, Any]] = cast(
            Optional[Dict[str, Any]],
            _UNSET,
        ),
        input_audio_transcription: Optional[Dict[str, Any]] = cast(
            Optional[Dict[str, Any]],
            _UNSET,
        ),
        parameters: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Configure the front end and delegation; confirmed by session.updated.

        ``parameters`` merges extra keys into the ``session`` node for server
        fields this SDK does not type yet; it may not shadow a typed argument.
        """
        if not isinstance(delegation, dict):
            raise ValueError("delegation must be an object")
        kind = delegation.get("type")
        if kind not in ("none", "client", "responses"):
            raise ValueError(
                "delegation.type must be none, client or responses",
            )
        if kind == "responses":
            responses = delegation.get("responses")
            if not isinstance(responses, dict):
                raise ValueError("delegation.responses must be an object")
            backend_model = responses.get("model")
            if not isinstance(backend_model, str) or not backend_model.strip():
                raise ValueError(
                    "delegation.responses.model must be a nonempty string",
                )
        elif "responses" in delegation:
            raise ValueError("delegation.responses requires responses mode")
        for value in (input_audio_format, output_audio_format):
            if value is not _UNSET and value != "pcm":
                raise ValueError("Live audio formats must be pcm")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ValueError("parameters must be an object")
            shadowed = sorted(set(parameters) & _RESERVED_SESSION_FIELDS)
            if shadowed:
                raise ValueError(
                    "parameters cannot override typed session fields: "
                    + ", ".join(shadowed),
                )
        session = {
            key: value
            for key, value in (
                ("model", model),
                ("instructions", instructions),
                ("voice", voice),
                ("tts", tts),
                ("input_audio_format", input_audio_format),
                ("output_audio_format", output_audio_format),
                ("tools", tools),
                ("delegation", delegation),
                ("transport", transport),
                ("modalities", modalities),
                ("context", context),
                ("turn_detection", turn_detection),
                ("input_audio_transcription", input_audio_transcription),
            )
            if value is not _UNSET
        }
        if parameters:
            session.update(parameters)
        return self._send(
            {"type": EVENT_SESSION_UPDATE, "session": session},
            event_id,
        )

    def append_audio(
        self,
        audio_b64: Union[str, bytes],
        *,
        event_id: Optional[str] = None,
    ) -> str:
        """Send raw or base64 16 kHz mono PCM16LE; AOQ input uses its own SDK."""
        audio = _as_base64(audio_b64)
        pcm = (
            audio_b64
            if isinstance(audio_b64, bytes)
            else base64.b64decode(audio, validate=True)
        )
        if not pcm or len(pcm) % 2:
            raise ValueError(
                "audio must contain nonempty, even-length PCM16LE",
            )
        return self._send(
            {"type": EVENT_SESSION_INPUT_AUDIO_APPEND, "audio": audio},
            event_id,
        )

    def mute_input_audio(self, *, event_id: Optional[str] = None) -> str:
        """Replace subsequent input with silence without stopping playback."""
        return self._send({"type": EVENT_SESSION_INPUT_AUDIO_MUTE}, event_id)

    def unmute_input_audio(self, *, event_id: Optional[str] = None) -> str:
        """Resume normal input without replaying muted audio."""
        return self._send({"type": EVENT_SESSION_INPUT_AUDIO_UNMUTE}, event_id)

    def _append_task_event(
        self,
        kind: str,
        content: str,
        *,
        delegation_id: Optional[str],
        task_id: Optional[str],
        start_ms: Optional[float],
        end_ms: Optional[float],
        event_id: Optional[str],
        error: Optional[Dict[str, Any]] = None,
    ) -> str:
        if kind != "commentary" or delegation_id is not None:
            if not isinstance(delegation_id, str) or not delegation_id:
                raise ValueError("delegation_id must be a nonempty string")
        if delegation_id is None and (
            task_id is not None or error is not None
        ):
            raise ValueError("task_id and error require a delegation_id")
        event: Dict[str, Any] = {"type": f"session.{kind}.append"}
        if content is not _UNSET:
            event["content"] = content
        for key, value in (
            ("delegation_id", delegation_id),
            ("task_id", task_id),
            ("error", error),
        ):
            if value is not None:
                event[key] = value
        event.update(_ms_window(start_ms, end_ms))
        return self._send(event, event_id)

    def append_instruction(
        self,
        content: str,
        *,
        delegation_id: str,
        task_id: Optional[str] = None,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Yield for user input; acknowledged by session.instructions.appended."""
        return self._append_task_event(
            "instruction",
            content,
            delegation_id=delegation_id,
            task_id=task_id,
            start_ms=start_ms,
            end_ms=end_ms,
            event_id=event_id,
        )

    def append_thinking(
        self,
        content: str,
        *,
        delegation_id: str,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Report silent client-task progress without ending the execution."""
        return self._append_task_event(
            "thinking",
            content,
            delegation_id=delegation_id,
            task_id=None,
            start_ms=start_ms,
            end_ms=end_ms,
            event_id=event_id,
        )

    def append_commentary(
        self,
        content: str = cast(str, _UNSET),
        *,
        delegation_id: Optional[str] = None,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        error: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Send user text or a client-task result; content may be omitted."""
        return self._append_task_event(
            "commentary",
            content,
            delegation_id=delegation_id,
            task_id=None,
            start_ms=start_ms,
            end_ms=end_ms,
            error=error,
            event_id=event_id,
        )

    def create_session_item(
        self,
        item: Dict[str, Any] = cast(Dict[str, Any], _UNSET),
        *,
        event_id: Optional[str] = None,
    ) -> str:
        """Send session.item.create with an optional front-end item."""
        event: Dict[str, Any] = {"type": EVENT_SESSION_ITEM_CREATE}
        if item is not _UNSET:
            event["item"] = item
        return self._send(event, event_id)

    def submit_function_output(
        self,
        call_id: str,
        output: Any,
        *,
        event_id: Optional[str] = None,
    ) -> str:
        """Reply to a front-end function from session.event, never a backend tool."""
        return self.create_session_item(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": (
                    output
                    if isinstance(output, str)
                    else json.dumps(output, ensure_ascii=False)
                ),
            },
            event_id=event_id,
        )

    def create_response_item(
        self,
        item: Dict[str, Any],
        *,
        event_id: Optional[str] = None,
    ) -> str:
        """Buffer next-task input in Responses mode, not an active task's result."""
        return self._send(
            {"type": EVENT_RESPONSE_ITEM_CREATE, "item": item},
            event_id,
        )

    def create_response(
        self,
        response: Optional[Dict[str, Any]] = None,
        *,
        event_id: Optional[str] = None,
    ) -> str:
        """Start a new Responses task, or request a reply from a capable front end."""
        event: Dict[str, Any] = {"type": EVENT_RESPONSE_CREATE}
        if response is not None:
            event["response"] = response
        return self._send(event, event_id)

    def end_session(self, *, event_id: Optional[str] = None) -> str:
        """Send session.close; keep receiving until session.closed or disconnect."""
        return self._send({"type": EVENT_SESSION_CLOSE}, event_id)


class Live:
    """client.audio.live: Qwen-Live delegated voice sessions."""

    def __init__(self, client) -> None:
        self._client = client

    def connect(
        self,
        *,
        model: str,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> LiveSession:
        """Create an unopened session; model routes the URL, not the front-end model."""
        return LiveSession(
            self._client,
            model=model,
            ws_url=ws_url,
            connector=connector,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
