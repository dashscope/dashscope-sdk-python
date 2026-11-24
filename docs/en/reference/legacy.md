---
description: "Frozen 1.x API surface archived under dashscope.legacy_1x, kept for backward compatibility."
status: deprecated
---
# Legacy 1.x API

The 1.x API classes are archived under `dashscope.legacy_1x` and remain
importable from the `dashscope` root module for backward compatibility
(`from dashscope import Generation` still works). The layer is frozen: it
receives no new features. New code should use the new [Client](client.md).

!!! warning "Frozen layer"
    The 1.x surface receives no new features. `Assistants` / `Threads`
    additionally emit `DeprecationWarning` on access.

## Text generation

::: dashscope.legacy_1x.aigc.generation.Generation

::: dashscope.legacy_1x.aigc.generation.AioGeneration

::: dashscope.legacy_1x.aigc.code_generation.CodeGeneration

::: dashscope.legacy_1x.aigc.conversation.Conversation

::: dashscope.legacy_1x.aigc.conversation.History

::: dashscope.legacy_1x.aigc.conversation.HistoryItem

## Multimodal conversation

::: dashscope.legacy_1x.aigc.multimodal_conversation.MultiModalConversation

::: dashscope.legacy_1x.aigc.multimodal_conversation.AioMultiModalConversation

## Image & video synthesis

::: dashscope.legacy_1x.aigc.image_synthesis.ImageSynthesis

::: dashscope.legacy_1x.aigc.image_synthesis.AioImageSynthesis

::: dashscope.legacy_1x.aigc.image_generation.ImageGeneration

::: dashscope.legacy_1x.aigc.image_generation.AioImageGeneration

::: dashscope.legacy_1x.aigc.video_synthesis.VideoSynthesis

::: dashscope.legacy_1x.aigc.video_synthesis.AioVideoSynthesis

## Embeddings

::: dashscope.legacy_1x.embeddings.text_embedding.TextEmbedding

::: dashscope.legacy_1x.embeddings.batch_text_embedding.BatchTextEmbedding

::: dashscope.legacy_1x.embeddings.batch_text_embedding_response.BatchTextEmbeddingResponse

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbedding

::: dashscope.legacy_1x.embeddings.multimodal_embedding.AioMultiModalEmbedding

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbeddingItemText

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbeddingItemImage

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbeddingItemAudio

## Rerank

::: dashscope.legacy_1x.rerank.TextReRank

::: dashscope.legacy_1x.rerank.AioTextReRank

## NLP understanding

::: dashscope.legacy_1x.nlp.understanding.Understanding

## Application calls

::: dashscope.legacy_1x.app.application.Application

::: dashscope.legacy_1x.app.application.ApplicationResponse

## Speech synthesis (1.x)

::: dashscope.legacy_1x.audio.tts.speech_synthesizer.SpeechSynthesizer

::: dashscope.legacy_1x.audio.tts.speech_synthesizer.ResultCallback

::: dashscope.legacy_1x.audio.tts.speech_synthesizer.SpeechSynthesisResult

::: dashscope.legacy_1x.audio.http_tts.http_speech_synthesizer.HttpSpeechSynthesizer

::: dashscope.legacy_1x.audio.tts_v2.speech_synthesizer.SpeechSynthesizer

::: dashscope.legacy_1x.audio.tts_v2.speech_synthesizer.ResultCallback

::: dashscope.legacy_1x.audio.tts_v2.enrollment.VoiceEnrollmentService

::: dashscope.legacy_1x.audio.qwen_tts.speech_synthesizer.SpeechSynthesizer

::: dashscope.legacy_1x.audio.qwen_tts_realtime.qwen_tts_realtime.QwenTtsRealtime

::: dashscope.legacy_1x.audio.qwen_tts_realtime.qwen_tts_realtime.QwenTtsRealtimeCallback

## Speech recognition (1.x)

::: dashscope.legacy_1x.audio.asr.recognition.Recognition

::: dashscope.legacy_1x.audio.asr.recognition.RecognitionCallback

::: dashscope.legacy_1x.audio.asr.recognition.RecognitionResult

::: dashscope.legacy_1x.audio.asr.transcription.Transcription

::: dashscope.legacy_1x.audio.asr.asr_phrase_manager.AsrPhraseManager

::: dashscope.legacy_1x.audio.asr.translation_recognizer.TranslationRecognizerRealtime

::: dashscope.legacy_1x.audio.asr.translation_recognizer.TranslationRecognizerChat

::: dashscope.legacy_1x.audio.asr.translation_recognizer.TranslationRecognizerCallback

::: dashscope.legacy_1x.audio.qwen_asr.qwen_transcription.QwenTranscription

## Omni realtime (1.x)

::: dashscope.legacy_1x.audio.qwen_omni.omni_realtime.OmniRealtimeConversation

::: dashscope.legacy_1x.audio.qwen_omni.omni_realtime.OmniRealtimeCallback

## Files & models (1.x)

::: dashscope.legacy_1x.files.Files

::: dashscope.legacy_1x.models.Models

## Fine-tuning & deployments (1.x)

::: dashscope.legacy_1x.finetune.finetunes.FineTunes

::: dashscope.legacy_1x.finetune.deployments.Deployments

## Tokenizers

::: dashscope.legacy_1x.tokenizers.Tokenization

::: dashscope.legacy_1x.tokenizers.Tokenizer

::: dashscope.legacy_1x.tokenizers.get_tokenizer

::: dashscope.legacy_1x.tokenizers.list_tokenizers

## Assistants & Threads (deprecated)

The Assistants API is deprecated; accessing these names emits a
`DeprecationWarning`.

::: dashscope.legacy_1x.assistants.assistants.Assistants

::: dashscope.legacy_1x.assistants.assistant_types.Assistant

::: dashscope.legacy_1x.threads.threads.Threads

::: dashscope.legacy_1x.threads.thread_types.Thread

::: dashscope.legacy_1x.threads.thread_types.ThreadMessage

::: dashscope.legacy_1x.threads.messages.messages.Messages

::: dashscope.legacy_1x.threads.runs.runs.Runs

::: dashscope.legacy_1x.threads.thread_types.Run

::: dashscope.legacy_1x.threads.runs.steps.Steps

::: dashscope.legacy_1x.threads.thread_types.RunStep

## Shared session helpers

::: dashscope.legacy_1x.api_entities.http_request.close_shared_sync_session

::: dashscope.legacy_1x.api_entities.aio_session.close_shared_aio_session