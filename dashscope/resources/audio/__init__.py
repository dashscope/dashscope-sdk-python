# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.audio resource group (asr/tts/realtime, design doc M3 3.4/3.5).

Assembly only — each sub-resource lives in its own module, so the tree
here is two levels deep at most: ``client.audio.asr.transcribe``,
``client.audio.asr.recognize``, ``client.audio.tts.synthesize``,
``client.audio.tts.http.synthesize``, ``client.audio.realtime.connect``,
and the management plane ``client.audio.vocabularies`` / ``phrases`` /
``voices`` / ``voice_design``.
"""

import asyncio

from dashscope.resources._blocking import bridge_client
from dashscope.resources.audio.asr import (
    Asr,
    AsyncAsr,
    Recognition,
)
from dashscope.resources.audio.customization import (
    DesignedVoice,
    DesignedVoiceList,
    Vocabularies,
    Vocabulary,
    VocabularyList,
    Voice,
    VoiceDesign,
    VoiceList,
    Voices,
)
from dashscope.resources.audio.http_tts import (
    AsyncHttpTts,
    HttpSynthesis,
    HttpSynthesisChunk,
    HttpSynthesisStream,
    HttpTts,
)
from dashscope.resources.audio.phrases import (
    Phrase,
    PhraseList,
    Phrases,
)
from dashscope.resources.audio.realtime import Realtime, RealtimeSession
from dashscope.resources.audio.tts import SynthesisSession, Tts

__all__ = [
    "Asr",
    "AsyncAsr",
    "AsyncHttpTts",
    "AsyncAudio",
    "Audio",
    "DesignedVoice",
    "DesignedVoiceList",
    "HttpSynthesis",
    "HttpSynthesisChunk",
    "HttpSynthesisStream",
    "HttpTts",
    "Phrase",
    "PhraseList",
    "Phrases",
    "Realtime",
    "RealtimeSession",
    "Recognition",
    "SynthesisSession",
    "Tts",
    "Vocabularies",
    "Vocabulary",
    "VocabularyList",
    "Voice",
    "VoiceDesign",
    "VoiceList",
    "Voices",
]


class _AsyncResourceProxy:
    """Run sync response parsing in a worker while reusing AsyncTransport."""

    def __init__(self, client, resource_cls) -> None:
        self._client = client
        self._resource_cls = resource_cls

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            resource = self._resource_cls(bridge_client(self._client))
            return await asyncio.to_thread(
                getattr(resource, name),
                *args,
                **kwargs,
            )

        return call


class _AsyncTts:
    def __init__(self, client) -> None:
        self.http = AsyncHttpTts(client)


class Audio:
    """Sync audio surface.

    Preview: the 2.0 audio API surface is not final and may change in
    a future release.
    """

    def __init__(self, client) -> None:
        self.asr = Asr(client)
        self.tts = Tts(client)
        self.realtime = Realtime(client)
        self.vocabularies = Vocabularies(client)
        self.phrases = Phrases(client)
        self.voices = Voices(client)
        self.voice_design = VoiceDesign(client)


class AsyncAudio:
    """Async audio surface.

    Only batch transcription is exposed here; the ws realtime resources
    (``asr.recognize``, ``tts``, ``realtime``) are sync-only by design
    (design doc M3 3.5).

    Preview: the 2.0 audio API surface is not final and may change in
    a future release.
    """

    def __init__(self, client) -> None:
        self.asr = AsyncAsr(client)
        self.tts = _AsyncTts(client)
        self.vocabularies = _AsyncResourceProxy(client, Vocabularies)
        self.phrases = _AsyncResourceProxy(client, Phrases)
        self.voices = _AsyncResourceProxy(client, Voices)
        self.voice_design = _AsyncResourceProxy(client, VoiceDesign)
