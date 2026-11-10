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

from dashscope.v2.resources.audio.asr import (
    Asr,
    AsyncAsr,
    Recognition,
)
from dashscope.v2.resources.audio.customization import (
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
from dashscope.v2.resources.audio.http_tts import (
    HttpSynthesis,
    HttpSynthesisChunk,
    HttpSynthesisStream,
    HttpTts,
)
from dashscope.v2.resources.audio.live import (
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
from dashscope.v2.resources.audio.phrases import (
    Phrase,
    PhraseList,
    Phrases,
)
from dashscope.v2.resources.audio.realtime import Realtime, RealtimeSession
from dashscope.v2.resources.audio.tts import SynthesisSession, Tts

__all__ = [
    "Asr",
    "AsyncAsr",
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


class Audio:
    """Sync audio surface.

    Preview: the 2.0 audio API surface is not final and may change in
    a future release.
    """

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

    Preview: the 2.0 audio API surface is not final and may change in
    a future release.
    """

    def __init__(self, client) -> None:
        self.asr = AsyncAsr(client)
