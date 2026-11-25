---
description: "语音合成、识别、实时会话与音色定制 API 参考。"
---
# 语音（Audio）

`client.audio` 汇集语音合成（TTS）、语音识别（ASR）、实时 WebSocket
会话与 Qwen Live，以及音色/热词定制能力。

## Audio 入口

::: dashscope.resources.audio.Audio

::: dashscope.resources.audio.AsyncAudio

## 语音合成（TTS）

::: dashscope.resources.audio.tts.Tts

::: dashscope.resources.audio.tts.SynthesisSession

::: dashscope.resources.audio.http_tts.HttpTts

::: dashscope.resources.audio.http_tts.AsyncHttpTts

::: dashscope.resources.audio.http_tts.HttpSynthesis

::: dashscope.resources.audio.http_tts.HttpSynthesisChunk

::: dashscope.resources.audio.http_tts.HttpSynthesisStream

::: dashscope.resources.audio.http_tts.AsyncHttpSynthesisStream

## 语音识别（ASR）

::: dashscope.resources.audio.asr.Asr

::: dashscope.resources.audio.asr.AsyncAsr

::: dashscope.resources.audio.asr.Recognition

::: dashscope.resources.audio.asr.TranscriptionJobs

::: dashscope.resources.audio.asr.AsyncTranscriptionJobs

::: dashscope.resources.audio.asr.RecognitionResult

## 实时与 Qwen Live

`client.audio.live` 的完整接入流程参见
[Qwen Live 指南](../guides/qwen-live-python-sdk.md)。

::: dashscope.resources.audio.realtime.Realtime

::: dashscope.resources.audio.realtime.RealtimeSession

::: dashscope.resources.audio.live.Live

::: dashscope.resources.audio.live.LiveSession

## 音色与热词定制

::: dashscope.resources.audio.customization.Voices

::: dashscope.resources.audio.customization.Voice

::: dashscope.resources.audio.customization.VoiceList

::: dashscope.resources.audio.customization.VoiceDesign

::: dashscope.resources.audio.customization.DesignedVoice

::: dashscope.resources.audio.customization.DesignedVoiceList

::: dashscope.resources.audio.customization.Vocabularies

::: dashscope.resources.audio.customization.Vocabulary

::: dashscope.resources.audio.customization.VocabularyList

::: dashscope.resources.audio.phrases.Phrases

::: dashscope.resources.audio.phrases.Phrase

::: dashscope.resources.audio.phrases.PhraseList
