---
description: "API reference for speech synthesis, recognition, realtime sessions and voice customization."
---
# Audio

`client.audio` groups speech synthesis (TTS), recognition (ASR), realtime
WebSocket sessions and Qwen Live, plus voice/vocabulary customization.

## Audio surface

::: dashscope.resources.audio.Audio

::: dashscope.resources.audio.AsyncAudio

## Speech synthesis (TTS)

::: dashscope.resources.audio.tts.Tts

::: dashscope.resources.audio.tts.SynthesisSession

::: dashscope.resources.audio.http_tts.HttpTts

::: dashscope.resources.audio.http_tts.AsyncHttpTts

::: dashscope.resources.audio.http_tts.HttpSynthesis

::: dashscope.resources.audio.http_tts.HttpSynthesisChunk

::: dashscope.resources.audio.http_tts.HttpSynthesisStream

::: dashscope.resources.audio.http_tts.AsyncHttpSynthesisStream

## Recognition (ASR)

::: dashscope.resources.audio.asr.Asr

::: dashscope.resources.audio.asr.AsyncAsr

::: dashscope.resources.audio.asr.Recognition

::: dashscope.resources.audio.asr.TranscriptionJobs

::: dashscope.resources.audio.asr.AsyncTranscriptionJobs

::: dashscope.resources.audio.asr.RecognitionResult

## Realtime & Qwen Live

See the [Qwen Live guide](../guides/qwen-live-python-sdk.md) for a full
walkthrough of `client.audio.live`.

::: dashscope.resources.audio.realtime.Realtime

::: dashscope.resources.audio.realtime.RealtimeSession

::: dashscope.resources.audio.live.Live

::: dashscope.resources.audio.live.LiveSession

## Voice & vocabulary customization

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
