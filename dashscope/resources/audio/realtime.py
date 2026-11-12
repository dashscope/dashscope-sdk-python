# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Omni, Qwen-Audio and Livetranslate sessions over realtime WebSockets."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union, cast

from dashscope._logging import get_logger, request_id_from_headers
from dashscope.errors import (
    APIConnectionError,
    APITimeoutError,
    error_from_business_code,
)
from dashscope.resources.audio._ws_session import (
    _UNSET,
    WsSessionBase,
    _as_base64,
)
from dashscope.transport.ws import Connector

__all__ = [
    "Realtime",
    "RealtimeSession",
]


_logger = get_logger("audio.realtime")


def _event_request_id(event: Any) -> Optional[str]:
    """Read real request IDs only; event_id is a protocol correlation ID."""
    if not isinstance(event, dict):
        return None
    error = event.get("error")
    response = event.get("response")
    details = (
        response.get("status_details") if isinstance(response, dict) else None
    )
    response_error = (
        details.get("error") if isinstance(details, dict) else None
    )
    for source in (event, error, response_error):
        if isinstance(source, dict):
            value = source.get("request_id")
            if isinstance(value, str) and value.strip():
                return value
    return None


class RealtimeSession(WsSessionBase):
    """One realtime websocket session (context manager).

    Usage::

        with client.audio.realtime.connect(
            model="qwen3.8-omni-flash-realtime",
        ) as rt:
            rt.send_event({"type": "session.update", "session": {...}})
            for event in rt.events():
                ...

    Unlike the run-task family (asr/tts/tingwu), the
    realtime protocol has no task id: one socket IS one session and the
    server closes it when the session ends, so connection reuse does
    not apply here -- every session opens its own socket (see
    ``WsConnection`` for the multiplexable family). Latency-sensitive
    callers can shave the handshake by connecting ahead of use rather
    than by pooling.
    """

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
        super().__init__(
            client,
            model=model,
            ws_url=ws_url,
            connector=connector,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        wire_model = self._wire_model
        if wire_model.startswith("qwen3.8-omni-"):
            self._protocol = "omni_v38"
        elif wire_model.startswith("qwen-audio-"):
            self._protocol = "qwen_audio"
        elif wire_model.startswith("qwen3.8-livetranslate-"):
            self._protocol = "livetranslate_v38"
        elif wire_model.startswith(
            ("qwen3-livetranslate-", "qwen3.5-livetranslate-"),
        ):
            self._protocol = "livetranslate"
        else:
            self._protocol = "omni"

    def _reset_state(self) -> None:
        super()._reset_state()
        # Logging context is separate from the public latency/response metrics.
        self._request_id: Optional[str] = None
        self._protocol_request_id: Optional[str] = None
        self._finish_requested = False
        self._finished = False
        self._response_active = False
        self._pending_response_ids: List[Any] = []
        self.session_id: Optional[str] = None
        self.last_message: Optional[Dict[str, Any]] = None
        self.last_response_id: Optional[str] = None
        self.last_response_create_time: Optional[float] = None
        self.last_first_text_delay: Optional[float] = None
        self.last_first_audio_delay: Optional[float] = None

    def connect(self) -> "RealtimeSession":
        """Open the connection, either explicitly or on context entry."""
        from websocket import (  # pylint: disable=import-outside-toplevel
            WebSocketTimeoutException,
        )

        try:
            return super().connect()
        except WebSocketTimeoutException as exc:
            raise APITimeoutError(detail=str(exc)) from exc

    def _on_connect_error(self, exc: BaseException) -> None:
        body = getattr(exc, "resp_body", None)
        if isinstance(body, (str, bytes, bytearray)):
            try:
                body = json.loads(body)
            except (ValueError, UnicodeError, RecursionError):
                body = None
        request_id = (
            _event_request_id(body)
            or request_id_from_headers(
                getattr(exc, "resp_headers", None),
            )
            or self._header_request_id
        )
        _logger.error(
            "Realtime handshake failed: %s, request_id=%s",
            type(exc).__name__,
            request_id or "unknown",
        )

    def _request_id_for_log(self) -> str:
        return (
            self._request_id
            or self._header_request_id
            or self._protocol_request_id
            or "unknown"
        )

    def send_event(self, event: Dict[str, Any]) -> None:
        """Send a raw event, assigning an event_id unless one is supplied."""
        with self._send_lock:
            with self._lock:
                ws, generation = self._ws, self._generation
                if ws is None:
                    raise RuntimeError("realtime session is not open")
                if self._finish_requested:
                    if event.get("type") == "session.finish":
                        return
                    raise RuntimeError("realtime session is finishing")
                payload = dict(event)
                payload.setdefault("event_id", "event_" + uuid.uuid4().hex)
                raw = json.dumps(payload)
                if payload.get("type") == "session.finish":
                    self._finish_requested = True
                if payload.get("type") == "response.create":
                    # Register before I/O: a response may arrive before send returns.
                    self._pending_response_ids.append(payload["event_id"])
            try:
                self._send_raw(ws, raw)
            except BaseException as exc:
                closing_ws = None
                with self._lock:
                    if ws is self._ws and generation == self._generation:
                        _logger.error(
                            "Realtime send failed: %s, request_id=%s",
                            type(exc).__name__,
                            self._request_id_for_log(),
                        )
                        if payload.get("type") == "response.create" and (
                            payload["event_id"] in self._pending_response_ids
                        ):
                            self._pending_response_ids.remove(
                                payload["event_id"],
                            )
                        closing_ws = self._detach_socket()
                self._close_socket(closing_ws)
                raise

    def update_session(
        self,
        output_modalities: List[str] = cast(List[str], _UNSET),
        *,
        voice: Optional[str] = cast(Optional[str], _UNSET),
        input_audio_format: str = cast(str, _UNSET),
        output_audio_format: str = cast(str, _UNSET),
        enable_input_audio_transcription: bool = cast(bool, _UNSET),
        input_audio_transcription_model: Optional[str] = cast(
            Optional[str],
            _UNSET,
        ),
        enable_turn_detection: bool = cast(bool, _UNSET),
        turn_detection_type: str = cast(str, _UNSET),
        prefix_padding_ms: int = cast(int, _UNSET),
        turn_detection_threshold: float = cast(float, _UNSET),
        turn_detection_silence_duration_ms: int = cast(int, _UNSET),
        turn_detection_param: Optional[Dict[str, Any]] = cast(
            Optional[Dict[str, Any]],
            _UNSET,
        ),
        translation_params: Optional[Dict[str, Any]] = cast(
            Optional[Dict[str, Any]],
            _UNSET,
        ),
        transcription_params: Optional[Dict[str, Any]] = cast(
            Optional[Dict[str, Any]],
            _UNSET,
        ),
        input_audio_config: Optional[Dict[str, Any]] = cast(
            Optional[Dict[str, Any]],
            _UNSET,
        ),
        output_audio_config: Optional[Dict[str, Any]] = cast(
            Optional[Dict[str, Any]],
            _UNSET,
        ),
        input_video_representation_compact: Optional[str] = cast(
            Optional[str],
            _UNSET,
        ),
        **kwargs: Any,
    ) -> None:
        """Patch modern sessions; pre-3.8 Omni retains its legacy defaults.

        ``**kwargs`` is the protocol-evolution escape hatch: extra keys
        merge into the wire ``session`` object after the named helpers,
        so on key overlap the kwargs value wins and falsey values are
        preserved as sent.
        """
        options = {
            key: value
            for key, value in {
                "output_modalities": output_modalities,
                "voice": voice,
                "input_audio_format": input_audio_format,
                "output_audio_format": output_audio_format,
                "enable_input_audio_transcription": (
                    enable_input_audio_transcription
                ),
                "input_audio_transcription_model": (
                    input_audio_transcription_model
                ),
                "enable_turn_detection": enable_turn_detection,
                "turn_detection_type": turn_detection_type,
                "prefix_padding_ms": prefix_padding_ms,
                "turn_detection_threshold": turn_detection_threshold,
                "turn_detection_silence_duration_ms": (
                    turn_detection_silence_duration_ms
                ),
                "turn_detection_param": turn_detection_param,
                "translation_params": translation_params,
                "transcription_params": transcription_params,
                "input_audio_config": input_audio_config,
                "output_audio_config": output_audio_config,
                "input_video_representation_compact": (
                    input_video_representation_compact
                ),
            }.items()
            if value is not _UNSET
        }
        if self._protocol == "omni":
            session = self._omni_session(**options, **kwargs)
        else:
            session = self._incremental_session(options, kwargs)
        self.send_event({"type": "session.update", "session": session})

    def _omni_session(
        self,
        output_modalities: List[str],
        *,
        voice: Optional[str] = None,
        input_audio_format: str = "pcm16",
        output_audio_format: str = "pcm16",
        enable_input_audio_transcription: bool = True,
        input_audio_transcription_model: Optional[str] = None,
        enable_turn_detection: bool = True,
        turn_detection_type: str = "server_vad",
        prefix_padding_ms: int = 300,
        turn_detection_threshold: float = 0.2,
        turn_detection_silence_duration_ms: int = 800,
        turn_detection_param: Optional[Dict[str, Any]] = None,
        translation_params: Optional[Dict[str, Any]] = None,
        transcription_params: Optional[Dict[str, Any]] = None,
        input_audio_config: Optional[Dict[str, Any]] = None,
        output_audio_config: Optional[Dict[str, Any]] = None,
        input_video_representation_compact: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        session: Dict[str, Any] = {
            "modalities": output_modalities,
            "voice": voice,
        }
        if input_audio_config is None and output_audio_config is None:
            session["input_audio_format"] = input_audio_format
            session["output_audio_format"] = output_audio_format
        else:
            audio = {}
            if input_audio_config is not None:
                audio["input"] = {"format": dict(input_audio_config)}
            if output_audio_config is not None:
                audio["output"] = {"format": dict(output_audio_config)}
            session["audio"] = audio
        session["input_audio_transcription"] = (
            {"model": input_audio_transcription_model}
            if enable_input_audio_transcription
            else None
        )
        session["turn_detection"] = None
        if enable_turn_detection:
            session["turn_detection"] = {
                "type": turn_detection_type,
                "threshold": turn_detection_threshold,
                "prefix_padding_ms": prefix_padding_ms,
                "silence_duration_ms": turn_detection_silence_duration_ms,
            }
            if turn_detection_param is not None:
                session["turn_detection"].update(turn_detection_param)
        if translation_params is not None:
            session["translation"] = {"language": None, **translation_params}
        if transcription_params is not None:
            transcription = {
                key: value
                for key, value in transcription_params.items()
                if key
                not in {
                    "input_audio_format",
                    "sample_rate",
                    "corpus_text",
                }
                and value is not None
            }
            if transcription_params.get("corpus_text") is not None:
                transcription["corpus"] = {
                    "text": transcription_params["corpus_text"],
                }
            session["input_audio_transcription"] = transcription
            if "audio" not in session:
                session["input_audio_format"] = transcription_params.get(
                    "input_audio_format",
                    "pcm",
                )
                session["sample_rate"] = transcription_params.get(
                    "sample_rate",
                    16000,
                )
        if input_video_representation_compact is not None:
            session["video"] = {
                "input": {
                    "representation_compact": (
                        input_video_representation_compact
                    ),
                },
            }
        session.update(kwargs)
        return session

    # Protocol event routing is a deliberate branch table.
    # pylint: disable-next=too-many-branches,too-many-statements
    def _incremental_session(
        self,
        options: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        session = {}
        for option, field in {
            "output_modalities": "modalities",
            "voice": "voice",
            "input_audio_format": "input_audio_format",
            "output_audio_format": "output_audio_format",
            "translation_params": "translation",
        }.items():
            if option in options:
                session[field] = options[option]

        vad = {}
        if options.get("enable_turn_detection"):
            vad["type"] = (
                "speaker_detection"
                if self._protocol == "livetranslate_v38"
                else "server_vad"
            )
        for option, field in {
            "turn_detection_type": "type",
            "prefix_padding_ms": "prefix_padding_ms",
            "turn_detection_threshold": "threshold",
            "turn_detection_silence_duration_ms": "silence_duration_ms",
        }.items():
            if option in options:
                vad[field] = options[option]
        params = options.get("turn_detection_param", _UNSET)
        if params is not _UNSET and params is not None:
            vad.update(params)
        if options.get("enable_turn_detection") is False or params is None:
            session["turn_detection"] = None
        elif vad or params is not _UNSET:
            session["turn_detection"] = vad

        transcription: Any = _UNSET
        enabled = options.get("enable_input_audio_transcription", _UNSET)
        if enabled is True and self._protocol != "livetranslate_v38":
            transcription = (
                {"model": "qwen3-asr-flash-realtime"}
                if self._protocol in {"livetranslate", "omni_v38"}
                else {}
            )
        if "input_audio_transcription_model" in options:
            transcription = {
                "model": options["input_audio_transcription_model"],
            }
        if "transcription_params" in options:
            params = options["transcription_params"]
            if params is None:
                transcription = (
                    {"model": None}
                    if self._protocol == "livetranslate"
                    else None
                )
            else:
                transcription = {
                    **(
                        transcription
                        if isinstance(transcription, dict)
                        else {}
                    ),
                    **{
                        key: value
                        for key, value in params.items()
                        if key
                        not in {
                            "input_audio_format",
                            "sample_rate",
                            "corpus_text",
                        }
                    },
                }
                if "corpus_text" in params:
                    transcription["corpus"] = {"text": params["corpus_text"]}
                for key in ("input_audio_format", "sample_rate"):
                    if key in params:
                        session[key] = params[key]
        if enabled is False:
            transcription = (
                {"model": None} if self._protocol == "livetranslate" else None
            )
        if transcription is not _UNSET:
            session["input_audio_transcription"] = transcription

        audio = {}
        for direction in ("input", "output"):
            key = f"{direction}_audio_config"
            if key in options:
                audio[direction] = {"format": options[key]}
        if audio:
            session["audio"] = audio
            if self._protocol not in {"livetranslate_v38", "omni_v38"}:
                for key in (
                    "input_audio_format",
                    "output_audio_format",
                    "sample_rate",
                ):
                    session.pop(key, None)
        if "input_video_representation_compact" in options:
            session["video"] = {
                "input": {
                    "representation_compact": (
                        options["input_video_representation_compact"]
                    ),
                },
            }
        session.update(kwargs)
        if self._protocol == "livetranslate_v38":
            transcription = session.get("input_audio_transcription", _UNSET)
            if transcription is None or (
                isinstance(transcription, dict)
                and transcription.get("model", _UNSET) is None
            ):
                raise ValueError("Livetranslate 3.8 ASR cannot be disabled")
            if "same_language_skip_options" in session:
                raise ValueError(
                    "Livetranslate 3.8 does not support "
                    "same_language_skip_options",
                )
        if self._protocol in {"livetranslate_v38", "omni_v38"}:
            session = self._nested_audio_session(session)
        return session

    def _nested_audio_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        if self._protocol == "livetranslate_v38" and "modalities" in session:
            session["output_modalities"] = session.pop("modalities")
        audio: Dict[str, Any] = {}
        for field, direction, nested_key in (
            ("input_audio_format", "input", "type"),
            ("sample_rate", "input", "sample_rate"),
            ("output_audio_format", "output", "type"),
        ):
            if field in session:
                audio.setdefault(direction, {}).setdefault("format", {})[
                    nested_key
                ] = session.pop(field)
        if "voice" in session:
            audio.setdefault("output", {})["voice"] = session.pop("voice")
        if (
            self._protocol == "livetranslate_v38"
            and "turn_detection" in session
        ):
            audio.setdefault("input", {})["turn_detection"] = session.pop(
                "turn_detection",
            )
        raw_audio = session.pop("audio", _UNSET)
        if isinstance(raw_audio, dict):
            for direction, config in raw_audio.items():
                if isinstance(config, dict):
                    merged = {**audio.get(direction, {}), **config}
                    if isinstance(config.get("format"), dict):
                        merged["format"] = {
                            **audio.get(direction, {}).get("format", {}),
                            **config["format"],
                        }
                    audio[direction] = merged
                else:
                    audio[direction] = config
        if raw_audio is not _UNSET and not isinstance(raw_audio, dict):
            session["audio"] = raw_audio
        elif audio or raw_audio is not _UNSET:
            session["audio"] = audio
        return session

    def append_audio(self, audio_b64: Union[str, bytes]) -> None:
        """Append base64 audio, or encode raw bytes before appending."""
        self.send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": _as_base64(audio_b64),
            },
        )

    def append_video(self, video_b64: Union[str, bytes]) -> None:
        """Append a base64 image frame, or encode raw image bytes."""
        self.send_event(
            {
                "type": "input_image_buffer.append",
                "image": _as_base64(video_b64),
            },
        )

    def commit(self) -> None:
        """Commit the appended audio and video."""
        self.send_event({"type": "input_audio_buffer.commit"})

    def clear_appended_audio(self) -> None:
        self.send_event({"type": "input_audio_buffer.clear"})

    def create_item(
        self,
        item: Dict[str, Any],
        *,
        previous_item_id: Optional[str] = None,
    ) -> None:
        event = {"type": "conversation.item.create", "item": item}
        if previous_item_id is not None:
            event["previous_item_id"] = previous_item_id
        self.send_event(event)

    def retrieve_item(self, item_id: str) -> None:
        self.send_event(
            {
                "type": "conversation.item.retrieve",
                "item_id": item_id,
            },
        )

    def delete_item(self, item_id: str) -> None:
        self.send_event(
            {
                "type": "conversation.item.delete",
                "item_id": item_id,
            },
        )

    def create_response(
        self,
        modalities: Optional[List[str]] = cast(Optional[List[str]], _UNSET),
        instructions: Optional[str] = cast(Optional[str], _UNSET),
        **kwargs: Any,
    ) -> None:
        """Send ``response.create``.

        ``**kwargs`` merges extra keys into the wire ``response`` object
        (e.g. ``temperature``), applied after ``modalities``/
        ``instructions`` and preserved verbatim, including falsey values.
        """
        response: Dict[str, Any] = {}
        if instructions is not _UNSET:
            response["instructions"] = instructions
        elif self._protocol == "omni":
            response["instructions"] = None
        if modalities is not _UNSET and (
            modalities is not None or self._protocol != "omni"
        ):
            response["modalities"] = modalities
        response.update(kwargs)
        event: Dict[str, Any] = {"type": "response.create"}
        if response or self._protocol != "omni_v38":
            event["response"] = response
        self.send_event(event)

    def cancel_response(self) -> None:
        self.send_event({"type": "response.cancel"})

    def end_session(self) -> None:
        """Request finish for Omni/Live; Qwen-Audio must drain then close()."""
        if self._protocol == "qwen_audio":
            raise NotImplementedError(
                "Qwen-Audio does not define session.finish; "
                "read through response.done, then call close()",
            )
        self.send_event({"type": "session.finish"})

    # pylint: disable-next=too-many-branches,too-many-statements
    def recv_event(self) -> Optional[Dict[str, Any]]:
        """Receive a raw event; unfinished responses cannot end silently."""
        from websocket import (  # pylint: disable=import-outside-toplevel
            WebSocketConnectionClosedException,
            WebSocketTimeoutException,
        )

        # Network reads must not hold the lock needed by send_event() or close().
        with self._recv_lock:
            with self._lock:
                ws, generation = self._ws, self._generation
                if ws is None:
                    self._check_unfinished()
                    if self._closed:
                        return None
                    raise RuntimeError("realtime session is not open")
            failure = None
            closing_ws = None
            received_on_current_connection = False
            try:
                try:
                    raw = self._recv_raw(ws, generation)
                except WebSocketConnectionClosedException:
                    raw = None
                except WebSocketTimeoutException as exc:
                    raise APITimeoutError(detail=str(exc)) from exc
                with self._lock:
                    if generation != self._generation:
                        return None
                    if ws is not self._ws:
                        self._check_unfinished()
                        return None
                    received_on_current_connection = True
                    if raw is None or raw == "" or raw == b"":
                        closing_ws = self._detach_socket()
                        self._check_unfinished()
                        _logger.debug(
                            "Realtime connection closed, request_id=%s",
                            self._request_id_for_log(),
                        )
                        return None
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8")
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        raise ValueError("expected a JSON event object")
                    self.last_message = event
                    self._track_event(event)
                    if event.get("type") == "session.finished":
                        self._finished = True
                        closing_ws = self._detach_socket()
                        self._check_unfinished()
                    elif event.get("type") == "error":
                        error = event.get("error")
                        if error is None:
                            error = {}
                        if not isinstance(error, dict):
                            raise ValueError("expected a JSON error object")
                        code = error.get("code")
                        if code is not None and not isinstance(code, str):
                            raise ValueError("expected a string error code")
                        failure = error_from_business_code(
                            message=error.get("message") or "realtime error",
                            code=code,
                            body=event,
                        )
                        # Only real IDs belong on exceptions; protocol fallbacks
                        # and "unknown" are strictly logging context.
                        explicit_request_id = _event_request_id(event)
                        if explicit_request_id is not None:
                            failure.request_id = explicit_request_id
                        _logger.error(
                            "Realtime business error, request_id=%s",
                            self._request_id_for_log(),
                        )
                        recoverable = (
                            error.get("type") == "invalid_request_error"
                        )
                        if recoverable:
                            # The nested event_id identifies the rejected request.
                            request_id = error.get("event_id")
                            if (
                                isinstance(request_id, str)
                                and request_id in self._pending_response_ids
                            ):
                                self._pending_response_ids.remove(request_id)
                        if not recoverable or self._finish_requested:
                            closing_ws = self._detach_socket()
            except BaseException as exc:
                with self._lock:
                    if generation == self._generation and (
                        ws is self._ws or received_on_current_connection
                    ):
                        _logger.error(
                            "Realtime receive failed: %s, request_id=%s",
                            type(exc).__name__,
                            self._request_id_for_log(),
                        )
                    if ws is self._ws and generation == self._generation:
                        closing_ws = self._detach_socket()
                raise
            finally:
                self._close_socket(closing_ws)
            # Malformed errors require cleanup; parsed request errors may recover.
            if failure is not None:
                raise failure
            return event

    def _check_unfinished(self) -> None:
        # Omni 3.8 documents session.finish without a session.finished ACK.
        if (
            self._finish_requested
            and not self._finished
            and self._protocol != "omni_v38"
        ):
            raise APIConnectionError(
                "realtime connection closed before session.finished",
            )
        # Unlike legacy protocols, Omni 3.8's optional ACK cannot finish a response.
        if (self._response_active or self._pending_response_ids) and (
            not self._finished or self._protocol == "omni_v38"
        ):
            raise APIConnectionError(
                "realtime connection closed before response.done",
            )

    def _track_request_context(self, event: Dict[str, Any]) -> None:
        """Keep request logging separate from response latency tracking."""
        kind = event.get("type")
        self._request_id = _event_request_id(event) or self._request_id
        protocol_id = event.get("response_id") or event.get("session_id")
        response = event.get("response")
        if kind in {"session.created", "session.updated"}:
            session = event.get("session")
            protocol_id = (
                session.get("id") if isinstance(session, dict) else None
            ) or protocol_id
        elif isinstance(kind, str) and kind.startswith("response."):
            protocol_id = (
                response.get("id") if isinstance(response, dict) else None
            ) or protocol_id
        if isinstance(protocol_id, str) and protocol_id.strip():
            self._protocol_request_id = protocol_id
        if kind == "conversation.item.input_audio_transcription.failed" or (
            kind == "response.done"
            and isinstance(response, dict)
            and response.get("status") == "failed"
        ):
            # These are raw terminal events, not exceptions in this API.
            _logger.error(
                "Realtime ASR/TTS response failed, request_id=%s",
                self._request_id_for_log(),
            )

    def _track_event(self, event: Dict[str, Any]) -> None:
        """Track first-output delays in monotonic milliseconds per response."""
        self._track_request_context(event)
        kind = event.get("type")
        if kind in {"session.created", "session.updated"}:
            self.session_id = (event.get("session") or {}).get(
                "id",
                self.session_id,
            )
        elif kind == "response.created":
            self.last_response_id = (event.get("response") or {}).get("id")
            self.last_response_create_time = time.monotonic() * 1000
            self.last_first_audio_delay = None
            self.last_first_text_delay = None
            self._response_active = True
            # response.created has no client ID; acknowledge in wire order.
            if self._pending_response_ids:
                self._pending_response_ids.pop(0)
        elif kind == "response.done":
            response_id = (event.get("response") or {}).get(
                "id",
                event.get("response_id"),
            )
            if response_id is None or response_id == self.last_response_id:
                self._response_active = False
        elif (
            self._response_active
            and self.last_response_create_time is not None
        ):
            response_id = event.get("response_id")
            if (
                response_id is not None
                and response_id != self.last_response_id
            ):
                return
            if (
                kind == "response.audio.delta"
                and self.last_first_audio_delay is None
            ):
                self.last_first_audio_delay = (
                    time.monotonic() * 1000 - self.last_response_create_time
                )
            elif (
                kind
                in {
                    "response.text.delta",
                    "response.audio_transcript.delta",
                    "response.text.text",
                    "response.audio_transcript.text",
                }
                and self.last_first_text_delay is None
            ):
                self.last_first_text_delay = (
                    time.monotonic() * 1000 - self.last_response_create_time
                )

    def events(self) -> Iterator[Dict[str, Any]]:
        """Yield through session.finished, or until a normal close."""
        while True:
            event = self.recv_event()
            if event is None:
                return
            yield event
            if event.get("type") == "session.finished":
                return


class Realtime:
    """``client.audio.realtime``: Omni, Qwen-Audio and Livetranslate."""

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
    ) -> RealtimeSession:
        """Create a session; enter its context or call session.connect()."""
        return RealtimeSession(
            self._client,
            model=model,
            ws_url=ws_url,
            connector=connector,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
