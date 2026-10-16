# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.audio.tts resource (design doc M3 3.5).

Speech synthesis over one duplex websocket task (task_group=audio,
task=tts, function=SpeechSynthesizer). ``Tts.connect`` opens an
incremental session so text pieces can be fed as they arrive — the v2
form of the 1.x SpeechSynthesizer ``streaming_call``/
``streaming_complete`` pair.

``Tts.http`` is the plain-HTTP sibling (no websocket): one-shot and
SSE-streaming synthesis for the sambert/cosyvoice and qwen-tts model
families, returning the audio as a URL reference.
"""

from __future__ import annotations

import threading
import time
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

from dashscope.v2.resources.audio.http_tts import HttpTts
from dashscope.v2.transport.ws import Connector, WsSession
from dashscope.v2.transport.ws_connection import WsConnection, WsTask

__all__ = [
    "SynthesisSession",
    "Tts",
]

MODULE = "audio"


def _merged_parameters(
    *,
    voice: str,
    format: str,  # pylint: disable=redefined-builtin
    sample_rate: int,
    volume: Optional[int],
    rate: Optional[float],
    pitch: Optional[float],
    language_hints: Optional[List[str]],
    enable_ssml: Optional[bool],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "voice": voice,
        "format": format,
        "sample_rate": sample_rate,
        "text_type": "PlainText",
    }
    if parameters:
        merged.update(parameters)
    # Named kwargs win over the ``parameters=`` escape hatch; the keys
    # match the 1.x SpeechSynthesizer wire names.
    for key, value in (
        ("volume", volume),
        ("rate", rate),
        ("pitch", pitch),
        ("language_hints", language_hints),
        ("enable_ssml", enable_ssml),
    ):
        if value is not None:
            merged[key] = value
    # 1.x tts_v2 rides enable_ssml as a plain parameter; text_type follows
    # it ("PlainText" -> "SSML") so markup in the text is actually parsed.
    if enable_ssml is not None:
        merged["text_type"] = "SSML" if enable_ssml else "PlainText"
    return merged


class SynthesisSession:
    """Duplex synthesis with concurrent text input and one result iterator."""

    def __init__(
        self,
        client,
        *,
        model: str = "cosyvoice-v1",
        voice: str = "longxiaochun",
        format: str = "mp3",  # pylint: disable=redefined-builtin
        sample_rate: int = 22050,
        volume: Optional[int] = None,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        language_hints: Optional[List[str]] = None,
        enable_ssml: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> None:
        config = client.config
        self._model = model
        self._parameters = _merged_parameters(
            voice=voice,
            format=format,
            sample_rate=sample_rate,
            volume=volume,
            rate=rate,
            pitch=pitch,
            language_hints=language_hints,
            enable_ssml=enable_ssml,
            parameters=parameters,
        )
        # A shared connection runs this synthesis as one task on it;
        # ws_url/connector/api_key/workspace/headers then belong to the
        # connection and the per-session values are ignored.
        self._connection = connection
        self._session_kwargs = {
            "url": ws_url,
            "api_key": api_key or config.api_key,
            "workspace": workspace or config.workspace,
            "region": config.region,
            "headers": dict(headers or {}),
            "connector": connector,
            "module": MODULE,
            "timeout": config.timeout if timeout is None else timeout,
        }
        self._session: Union[WsSession, WsTask, None] = None
        self._closed = False
        self._reader_lock = threading.Lock()
        self._first_text_at: Optional[float] = None
        self.last_response: Optional[Dict[str, Any]] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.first_package_delay: Optional[float] = None

    @property
    def session(self) -> Union[WsSession, WsTask]:
        """Underlying websocket session, including its ``task_id``."""
        if self._closed or self._session is None:
            raise RuntimeError("synthesis session is not open")
        return self._session

    def start(self) -> "SynthesisSession":
        """Open the connection and wait for the task-started handshake."""
        if self._closed:
            raise RuntimeError("synthesis session is closed")
        if self._session is not None:
            raise RuntimeError("synthesis session is already started")
        try:
            if self._connection is not None:
                self._session = self._connection.open_task(
                    timeout=self._session_kwargs["timeout"],
                )
                # Synthesis output is binary audio, which carries no task
                # id on the wire; claim the downlink before it arrives.
                self._session.claim_binary_downlink()
            else:
                self._session = WsSession(**self._session_kwargs)
            response = self._session.run_task(
                task_group="audio",
                task="tts",
                function="SpeechSynthesizer",
                model=self._model,
                parameters=self._parameters,
                input={},
            )
            self._record_response(response)
        except Exception as error:
            self._record_response(getattr(error, "body", None))
            self.close()
            raise
        return self

    def __enter__(self) -> "SynthesisSession":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def send_text(self, text: str) -> None:
        """Send another text fragment; not allowed after stop or cancel."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            session = self.session
            if self._first_text_at is None:
                self._first_text_at = time.monotonic()
            session.continue_task({"text": text})
        except Exception:
            self.close()
            raise

    def flush(self) -> None:
        """Request synthesis of buffered text without ending input."""
        try:
            self.session.continue_task({"flush": True})
        except Exception:
            self.close()
            raise

    def stop(self) -> None:
        """End input without blocking for results; keep reading afterward."""
        try:
            self.session.finish_task()
        except Exception:
            self.close()
            raise

    def cancel(self) -> None:
        """Cancel once, even after stop, before a terminal event arrives."""
        try:
            self.session.finish_task({"directive": "cancel"})
        except Exception:
            self.close()
            raise

    def _record_response(self, response: Any) -> None:
        if isinstance(response, dict):
            self.last_response = response
            payload = response.get("payload")
            if isinstance(payload, dict) and payload.get("usage") is not None:
                self.usage = payload["usage"]

    def events(self) -> Generator[Union[Dict[str, Any], bytes], None, None]:
        """Yield JSON and audio through task-finished, raising on failure."""
        session = self.session
        # The lock guards the iterator's whole lifetime and is released in
        # its finally block, so a ``with`` scope cannot express it.
        if not self._reader_lock.acquire(  # pylint: disable=consider-using-with
            blocking=False,
        ):
            raise RuntimeError(
                "only one events() or audio_chunks() iterator may read "
                "the synthesis session",
            )
        try:
            while not session.finished:
                frame = self.session.recv()
                if frame is None:
                    return
                if isinstance(frame, bytes):
                    if (
                        self.first_package_delay is None
                        and self._first_text_at is not None
                    ):
                        self.first_package_delay = (
                            time.monotonic() - self._first_text_at
                        ) * 1000
                else:
                    self._record_response(frame)
                yield frame
        except Exception as error:
            self._record_response(getattr(error, "body", None))
            self.close()
            raise
        finally:
            self._reader_lock.release()

    def audio_chunks(self) -> Iterator[bytes]:
        """Yield only audio, retaining JSON metadata on this session."""
        events = self.events()
        try:
            for frame in events:
                if isinstance(frame, bytes):
                    yield frame
        finally:
            events.close()

    def close(self) -> None:
        """Close the connection; a closed session cannot be restarted.

        With a shared ``WsConnection`` this only detaches the task; the
        socket belongs to the connection and stays open.
        """
        if not self._closed:
            self._closed = True
            if self._session is not None:
                self._session.close()


class Tts:
    """Speech synthesis (one-shot + streaming) over websocket."""

    def __init__(self, client) -> None:
        self._client = client
        self.http = HttpTts(client)

    def connect(
        self,
        *,
        model: str = "cosyvoice-v1",
        voice: str = "longxiaochun",
        format: str = "mp3",  # pylint: disable=redefined-builtin
        sample_rate: int = 22050,
        volume: Optional[int] = None,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        language_hints: Optional[List[str]] = None,
        enable_ssml: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> SynthesisSession:
        """Create a duplex session; enter its context or call start().

        With ``connection=`` the session runs as one task on the shared
        socket and claims its binary downlink (one active synthesis per
        connection).
        """
        return SynthesisSession(
            self._client,
            model=model,
            voice=voice,
            format=format,
            sample_rate=sample_rate,
            volume=volume,
            rate=rate,
            pitch=pitch,
            language_hints=language_hints,
            enable_ssml=enable_ssml,
            parameters=parameters,
            ws_url=ws_url,
            connector=connector,
            connection=connection,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )

    def synthesize(
        self,
        text: str,
        *,
        model: str = "cosyvoice-v1",
        voice: str = "longxiaochun",
        format: str = "mp3",  # pylint: disable=redefined-builtin
        sample_rate: int = 22050,
        volume: Optional[int] = None,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        language_hints: Optional[List[str]] = None,
        enable_ssml: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> bytes:
        """Synthesize the whole text; returns the concatenated audio."""
        return b"".join(
            self.stream(
                text,
                model=model,
                voice=voice,
                format=format,
                sample_rate=sample_rate,
                volume=volume,
                rate=rate,
                pitch=pitch,
                language_hints=language_hints,
                enable_ssml=enable_ssml,
                parameters=parameters,
                ws_url=ws_url,
                connector=connector,
                timeout=timeout,
                headers=headers,
                api_key=api_key,
                workspace=workspace,
            ),
        )

    def stream(
        self,
        text: str,
        *,
        model: str = "cosyvoice-v1",
        voice: str = "longxiaochun",
        format: str = "mp3",  # pylint: disable=redefined-builtin
        sample_rate: int = 22050,
        volume: Optional[int] = None,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        language_hints: Optional[List[str]] = None,
        enable_ssml: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Iterator[bytes]:
        """Yield audio chunks as they arrive."""
        with self.connect(
            model=model,
            voice=voice,
            format=format,
            sample_rate=sample_rate,
            volume=volume,
            rate=rate,
            pitch=pitch,
            language_hints=language_hints,
            enable_ssml=enable_ssml,
            parameters=parameters,
            ws_url=ws_url,
            connector=connector,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        ) as session:
            session.send_text(text)
            session.stop()
            yield from session.audio_chunks()
