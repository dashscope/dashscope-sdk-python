# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.audio resource group (asr/tts/realtime, design doc M3 3.4/3.5).

Assembly only — each sub-resource lives in its own module, so the tree
here is two levels deep at most: ``client.audio.asr.transcribe``,
``client.audio.asr.recognize``, ``client.audio.tts.synthesize``,
``client.audio.tts.http.synthesize``, ``client.audio.realtime.connect``,
``client.audio.live.connect``, and the management plane
``client.audio.vocabularies`` / ``phrases`` / ``voices`` /
``voice_design``.

The Qwen-Live wire event names (``EVENT_*``) are re-exported here too, so
callers can import them from the same package as ``LiveSession``.
"""

from dashscope.resources._blocking import run_blocking
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
from dashscope.resources.audio.live import (
    EVENT_ERROR,
    EVENT_RESPONSE_CREATE,
    EVENT_RESPONSE_EVENT,
    EVENT_RESPONSE_ITEM_CREATE,
    EVENT_SESSION_CLOSE,
    EVENT_SESSION_CLOSED,
    EVENT_SESSION_COMMENTARY_APPEND,
    EVENT_SESSION_COMMENTARY_APPENDED,
    EVENT_SESSION_CREATED,
    EVENT_SESSION_DELEGATION_CREATED,
    EVENT_SESSION_EVENT,
    EVENT_SESSION_INPUT_AUDIO_APPEND,
    EVENT_SESSION_INPUT_AUDIO_MUTE,
    EVENT_SESSION_INPUT_AUDIO_MUTED,
    EVENT_SESSION_INPUT_AUDIO_UNMUTE,
    EVENT_SESSION_INPUT_AUDIO_UNMUTED,
    EVENT_SESSION_INPUT_TRANSCRIPT_DELTA,
    EVENT_SESSION_INSTRUCTION_APPEND,
    EVENT_SESSION_INSTRUCTIONS_APPENDED,
    EVENT_SESSION_ITEM_CREATE,
    EVENT_SESSION_OUTPUT_AUDIO_DELTA,
    EVENT_SESSION_OUTPUT_AUDIO_INTERRUPTED,
    EVENT_SESSION_OUTPUT_TRANSCRIPT_DELTA,
    EVENT_SESSION_THINKING_APPEND,
    EVENT_SESSION_THINKING_APPENDED,
    EVENT_SESSION_UPDATE,
    EVENT_SESSION_UPDATED,
    EVENT_SESSION_USAGE_UPDATED,
    Live,
    LiveSession,
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
    "HttpSynthesis",
    "HttpSynthesisChunk",
    "HttpSynthesisStream",
    "HttpTts",
    "Live",
    "LiveSession",
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


def _call_resource(client, resource_cls, name, args, kwargs):
    resource = resource_cls(client)
    return getattr(resource, name)(*args, **kwargs)


class _AsyncResourceProxy:
    """Run sync response parsing in a worker while reusing AsyncTransport."""

    def __init__(self, client, resource_cls) -> None:
        self._client = client
        self._resource_cls = resource_cls

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            return await run_blocking(
                self._client,
                _call_resource,
                self._resource_cls,
                name,
                args,
                kwargs,
            )

        return call


class _AsyncTts:
    def __init__(self, client) -> None:
        self.http = AsyncHttpTts(client)


class Audio:
    """Sync audio surface."""

    def __init__(self, client) -> None:
        self.asr = Asr(client)
        self.tts = Tts(client)
        self.realtime = Realtime(client)
        self.live = Live(client)
        self.vocabularies = Vocabularies(client)
        self.phrases = Phrases(client)
        self.voices = Voices(client)
        self.voice_design = VoiceDesign(client)


class AsyncAudio:
    """Async audio surface.

    Only batch transcription is exposed here; the ws realtime resources
    (``asr.recognize``, ``tts``, ``realtime``) are sync-only by design
    (design doc M3 3.5).
    """

    def __init__(self, client) -> None:
        self.asr = AsyncAsr(client)
        self.tts = _AsyncTts(client)
        self.vocabularies = _AsyncResourceProxy(client, Vocabularies)
        self.phrases = _AsyncResourceProxy(client, Phrases)
        self.voices = _AsyncResourceProxy(client, Voices)
        self.voice_design = _AsyncResourceProxy(client, VoiceDesign)
