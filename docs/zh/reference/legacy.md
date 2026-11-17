---
description: "归档于 dashscope.legacy_1x 的冻结 1.x 接口，为向后兼容保留。"
status: deprecated
---
# 旧版 1.x API

1.x API 类已归档至 `dashscope.legacy_1x`，为保持向后兼容仍可从
`dashscope` 根模块导入（`from dashscope import Generation` 仍可用）。
兼容层已冻结：不再新增能力。新代码请使用 2.0 [客户端](client.md)。

!!! warning "冻结层"
    1.x 接口不再新增能力。访问 `Assistants` / `Threads`
    还会触发 `DeprecationWarning`。

## 文本生成

::: dashscope.legacy_1x.aigc.generation.Generation

::: dashscope.legacy_1x.aigc.generation.AioGeneration

::: dashscope.legacy_1x.aigc.code_generation.CodeGeneration

::: dashscope.legacy_1x.aigc.conversation.Conversation

::: dashscope.legacy_1x.aigc.conversation.History

::: dashscope.legacy_1x.aigc.conversation.HistoryItem

## 多模态对话

::: dashscope.legacy_1x.aigc.multimodal_conversation.MultiModalConversation

::: dashscope.legacy_1x.aigc.multimodal_conversation.AioMultiModalConversation

## 图像与视频生成

::: dashscope.legacy_1x.aigc.image_synthesis.ImageSynthesis

::: dashscope.legacy_1x.aigc.image_synthesis.AioImageSynthesis

::: dashscope.legacy_1x.aigc.image_generation.ImageGeneration

::: dashscope.legacy_1x.aigc.image_generation.AioImageGeneration

::: dashscope.legacy_1x.aigc.video_synthesis.VideoSynthesis

::: dashscope.legacy_1x.aigc.video_synthesis.AioVideoSynthesis

## 向量嵌入

::: dashscope.legacy_1x.embeddings.text_embedding.TextEmbedding

::: dashscope.legacy_1x.embeddings.batch_text_embedding.BatchTextEmbedding

::: dashscope.legacy_1x.embeddings.batch_text_embedding_response.BatchTextEmbeddingResponse

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbedding

::: dashscope.legacy_1x.embeddings.multimodal_embedding.AioMultiModalEmbedding

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbeddingItemText

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbeddingItemImage

::: dashscope.legacy_1x.embeddings.multimodal_embedding.MultiModalEmbeddingItemAudio

## 重排序

::: dashscope.legacy_1x.rerank.TextReRank

::: dashscope.legacy_1x.rerank.AioTextReRank

## NLP 理解

::: dashscope.legacy_1x.nlp.understanding.Understanding

## 应用调用

::: dashscope.legacy_1x.app.application.Application

::: dashscope.legacy_1x.app.application.ApplicationResponse

## 语音合成（1.x）

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

## 语音识别（1.x）

::: dashscope.legacy_1x.audio.asr.recognition.Recognition

::: dashscope.legacy_1x.audio.asr.recognition.RecognitionCallback

::: dashscope.legacy_1x.audio.asr.recognition.RecognitionResult

::: dashscope.legacy_1x.audio.asr.transcription.Transcription

::: dashscope.legacy_1x.audio.asr.asr_phrase_manager.AsrPhraseManager

::: dashscope.legacy_1x.audio.asr.translation_recognizer.TranslationRecognizerRealtime

::: dashscope.legacy_1x.audio.asr.translation_recognizer.TranslationRecognizerChat

::: dashscope.legacy_1x.audio.asr.translation_recognizer.TranslationRecognizerCallback

::: dashscope.legacy_1x.audio.qwen_asr.qwen_transcription.QwenTranscription

## Omni 实时（1.x）

::: dashscope.legacy_1x.audio.qwen_omni.omni_realtime.OmniRealtimeConversation

::: dashscope.legacy_1x.audio.qwen_omni.omni_realtime.OmniRealtimeCallback

## 文件与模型（1.x）

::: dashscope.legacy_1x.files.Files

::: dashscope.legacy_1x.models.Models

## 微调与部署（1.x）

::: dashscope.legacy_1x.finetune.finetunes.FineTunes

::: dashscope.legacy_1x.finetune.deployments.Deployments

## 分词器

::: dashscope.legacy_1x.tokenizers.Tokenization

::: dashscope.legacy_1x.tokenizers.Tokenizer

::: dashscope.legacy_1x.tokenizers.get_tokenizer

::: dashscope.legacy_1x.tokenizers.list_tokenizers

## Assistants 与 Threads（已弃用）

Assistants API 已弃用，访问这些名称会触发 `DeprecationWarning`。

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

## 共享会话辅助函数

::: dashscope.legacy_1x.api_entities.http_request.close_shared_sync_session

::: dashscope.legacy_1x.api_entities.aio_session.close_shared_aio_session