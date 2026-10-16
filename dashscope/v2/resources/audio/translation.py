# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.audio.translation resource (Gummy realtime speech translation).

Same duplex wire as ``asr.recognize`` (task_group=audio, task=asr,
function=recognition) — the 1.x ``TranslationRecognizerRealtime`` and
``TranslationRecognizerChat`` pair rides it too
(translation_recognizer.py:451/987). This resource names the translation
parameters and adds the bilingual typed view; the chat models
(gummy-chat-v1) share the wire and use the same entry — 1.x Chat only
differs by dropping audio after the first sentence end, which callers
express through ``TranslationResult.is_sentence_end()``.

The newer qwen3-livetranslate realtime models are a different protocol
and live under ``client.audio.realtime.connect`` instead.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Mapping, Optional

from dashscope.v2.resources.audio.asr import (
    Recognition,
    RecognitionResult,
    _stream_file_frames,
)
from dashscope.v2.transport.ws import EVENT_GENERATED, Connector
from dashscope.v2.transport.ws_connection import WsConnection

__all__ = [
    "Translation",
    "TranslationResult",
    "TranslationSession",
]


class TranslationResult(RecognitionResult):
    """Typed view over one bilingual ``result-generated`` payload.

    Gummy frames carry ``output.transcription`` (source-language
    sentence) and/or ``output.translations`` (one dict per target
    language, ``lang``/``text``/``words``/...), unlike the plain ASR
    ``output.sentence`` shape ``RecognitionResult.get_sentence`` reads.
    """

    def get_transcription(self) -> Optional[Dict[str, Any]]:
        """The source-language sentence dict, or None on this frame."""
        transcription = self.output.get("transcription")
        return transcription if isinstance(transcription, dict) else None

    def get_translations(self) -> List[Dict[str, Any]]:
        """Per-target-language sentence dicts (possibly empty)."""
        translations = self.output.get("translations")
        return translations if isinstance(translations, list) else []

    def is_sentence_end(self) -> bool:
        """True when transcription or any translation ends its sentence."""
        transcription = self.get_transcription()
        if transcription and transcription.get("sentence_end"):
            return True
        return any(
            item.get("sentence_end") for item in self.get_translations()
        )


def _results_from_events(
    events: Iterator[Dict[str, Any]],
) -> Iterator[TranslationResult]:
    for event in events:
        if event["header"].get("event") != EVENT_GENERATED:
            continue
        data = dict(event.get("payload") or {})
        data.setdefault("request_id", event["header"].get("request_id"))
        result = TranslationResult.model_validate(data)
        transcription = result.get_transcription()
        if transcription and transcription.get("heartbeat"):
            continue
        yield result


class TranslationSession(Recognition):
    """Duplex translation session; lifecycle identical to ``Recognition``."""

    def __init__(
        self,
        client,
        *,
        model: str,
        format: str = "pcm",  # pylint: disable=redefined-builtin
        sample_rate: int = 16000,
        source_language: Optional[str] = None,
        transcription_enabled: Optional[bool] = None,
        translation_enabled: Optional[bool] = None,
        translation_target_languages: Optional[List[str]] = None,
        max_end_silence: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        # pylint: disable-next=redefined-builtin
        input: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> None:
        merged = dict(parameters or {})
        # Named kwargs win over the ``parameters=`` escape hatch; the keys
        # match the 1.x TranslationRecognizerRealtime wire names. Omitted
        # parameters fall to the service defaults (transcription on,
        # translation off, source language auto).
        for key, value in (
            ("source_language", source_language),
            ("transcription_enabled", transcription_enabled),
            ("translation_enabled", translation_enabled),
            ("translation_target_languages", translation_target_languages),
            ("max_end_silence", max_end_silence),
        ):
            if value is not None:
                merged[key] = value
        super().__init__(
            client,
            model=model,
            format=format,
            sample_rate=sample_rate,
            parameters=merged,
            input=input,
            phrase_id=phrase_id,
            ws_url=ws_url,
            connector=connector,
            connection=connection,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )

    def results(self) -> Iterator[TranslationResult]:
        """Yield typed bilingual results until the task finishes.

        Non-result events (task-finished, ...) are skipped. Frames whose
        ``transcription`` is only a heartbeat are dropped too — a v2
        convenience so callers see real sentences; unlike the plain-ASR
        path, the 1.x translation recognizer forwards such frames to
        ``on_event`` rather than filtering them.
        """
        return _results_from_events(self.events())


class Translation:
    """``client.audio.translation``: realtime speech translation (Gummy)."""

    def __init__(self, client) -> None:
        self._client = client

    def recognize(
        self,
        *,
        model: str,
        format: str = "pcm",  # pylint: disable=redefined-builtin
        sample_rate: int = 16000,
        source_language: Optional[str] = None,
        transcription_enabled: Optional[bool] = None,
        translation_enabled: Optional[bool] = None,
        translation_target_languages: Optional[List[str]] = None,
        max_end_silence: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        # pylint: disable-next=redefined-builtin
        input: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> TranslationSession:
        """Open translation; send audio and consume events concurrently."""
        return TranslationSession(
            self._client,
            model=model,
            format=format,
            sample_rate=sample_rate,
            source_language=source_language,
            transcription_enabled=transcription_enabled,
            translation_enabled=translation_enabled,
            translation_target_languages=translation_target_languages,
            max_end_silence=max_end_silence,
            parameters=parameters,
            input=input,
            phrase_id=phrase_id,
            ws_url=ws_url,
            connector=connector,
            connection=connection,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )

    def recognize_file(
        self,
        path: str,
        *,
        model: str,
        format: str = "pcm",  # pylint: disable=redefined-builtin
        sample_rate: int = 16000,
        source_language: Optional[str] = None,
        transcription_enabled: Optional[bool] = None,
        translation_enabled: Optional[bool] = None,
        translation_target_languages: Optional[List[str]] = None,
        max_end_silence: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Iterator[TranslationResult]:
        """Translate a local audio file, yielding typed bilingual results.

        The 1.x ``TranslationRecognizerRealtime.call(file)`` equivalent
        (its ``TranslationRecognizerResultPack`` items arrive one
        ``TranslationResult`` at a time): the file is auto-chunked and
        driven to completion in one session. Unlike
        ``asr.recognize_file`` this yields typed results — the bilingual
        nesting is the point of this facade. Validation happens eagerly;
        the session itself opens on first iteration.
        """
        if not os.path.isfile(path):
            raise ValueError(f"audio file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"audio file is empty: {path}")
        session = self.recognize(
            model=model,
            format=format,
            sample_rate=sample_rate,
            source_language=source_language,
            transcription_enabled=transcription_enabled,
            translation_enabled=translation_enabled,
            translation_target_languages=translation_target_languages,
            max_end_silence=max_end_silence,
            parameters=parameters,
            phrase_id=phrase_id,
            ws_url=ws_url,
            connector=connector,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        return _results_from_events(_stream_file_frames(session, path))
