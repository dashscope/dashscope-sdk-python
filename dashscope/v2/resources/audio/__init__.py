# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.audio resource group (asr/tts/realtime/translation, design doc
M3 3.4/3.5).

Assembly only — each sub-resource lives in its own module, so the tree
here is two levels deep at most: ``client.audio.asr.transcribe``,
``client.audio.asr.recognize``, ``client.audio.tts.synthesize``,
``client.audio.tts.http.synthesize``, ``client.audio.realtime.connect``,
``client.audio.translation.recognize``, and the management plane
``client.audio.vocabularies`` / ``phrases`` / ``voices`` /
``voice_design``.
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
from dashscope.v2.resources.audio.phrases import (
    Phrase,
    PhraseList,
    Phrases,
)
from dashscope.v2.resources.audio.realtime import Realtime, RealtimeSession
from dashscope.v2.resources.audio.translation import (
    Translation,
    TranslationSession,
)
from dashscope.v2.resources.audio.tts import SynthesisSession, Tts

__all__ = [
    "Asr",
    "AsyncAsr",
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
    "Translation",
    "TranslationSession",
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
    def __init__(self, client) -> None:
        self.asr = Asr(client)
        self.tts = Tts(client)
        self.realtime = Realtime(client)
        self.translation = Translation(client)
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
