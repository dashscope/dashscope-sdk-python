---
description: "API reference for client resource namespaces: embeddings, images, videos, files, fine-tunes and more."
---
# Client Resources

Resource namespaces mounted on `Client` and `AsyncClient`.
Each section lists the sync and async classes and their response models.

## Embeddings

`client.embeddings` — text embedding calls and batch jobs.

::: dashscope.resources.embeddings.Embeddings

::: dashscope.resources.embeddings.AsyncEmbeddings

::: dashscope.resources.embeddings.EmbeddingJobs

::: dashscope.resources.embeddings.AsyncEmbeddingJobs

::: dashscope.resources.embeddings.Embedding

::: dashscope.resources.embeddings.EmbeddingResponse

## Multimodal

`client.multimodal` — multimodal chat and embeddings.

::: dashscope.resources.multimodal.Multimodal

::: dashscope.resources.multimodal.AsyncMultimodal

::: dashscope.resources.multimodal.chat.MultimodalChat

::: dashscope.resources.multimodal.chat.AsyncMultimodalChat

::: dashscope.resources.multimodal.embeddings.MultimodalEmbeddings

::: dashscope.resources.multimodal.embeddings.AsyncMultimodalEmbeddings

::: dashscope.resources.multimodal.embeddings.EmbeddingItem

## Rerank

`client.rerank` — document reranking.

::: dashscope.resources.rerank.Rerank

::: dashscope.resources.rerank.AsyncRerank

::: dashscope.resources.rerank.RerankResponse

::: dashscope.resources.rerank.RerankResult

::: dashscope.resources.rerank.RerankDocument

## Images

`client.images` — image generation jobs.

::: dashscope.resources.images.Images

::: dashscope.resources.images.AsyncImages

::: dashscope.resources.images.ImageJobs

::: dashscope.resources.images.AsyncImageJobs

## Videos

`client.videos` — video generation jobs.

::: dashscope.resources.videos.Videos

::: dashscope.resources.videos.AsyncVideos

::: dashscope.resources.videos.VideoJobs

::: dashscope.resources.videos.AsyncVideoJobs

## Files

`client.files` — file upload, listing and deletion.

::: dashscope.resources.files.Files

::: dashscope.resources.files.AsyncFiles

::: dashscope.resources.files.File

::: dashscope.resources.files.FileList

::: dashscope.resources.files.DeletionStatus

## Models

`client.models` — the model catalog.

::: dashscope.resources.models.Models

::: dashscope.resources.models.AsyncModels

::: dashscope.resources.models.Model

::: dashscope.resources.models.ModelList

::: dashscope.resources.models.ModelPrice

::: dashscope.resources.models.ModelPriceRange

::: dashscope.resources.models.InferenceMetadata

::: dashscope.resources.models.ModelInfo

::: dashscope.resources.models.RateLimit

::: dashscope.resources.models.ModelQuota

::: dashscope.resources.models.QuotaList

::: dashscope.resources.models.PermissionDetail

::: dashscope.resources.models.ModelPermission

::: dashscope.resources.models.PermissionList

::: dashscope.resources.models.ModelLimits

::: dashscope.resources.models.AsyncModelLimits

::: dashscope.resources.models.ModelPermissions

::: dashscope.resources.models.AsyncModelPermissions

## Fine-tunes

`client.fine_tunes` — fine-tuning jobs, logs and events.

::: dashscope.resources.fine_tunes.FineTunes

::: dashscope.resources.fine_tunes.AsyncFineTunes

::: dashscope.resources.fine_tunes.FineTune

::: dashscope.resources.fine_tunes.FineTuneList

::: dashscope.resources.fine_tunes.JobLogs

::: dashscope.resources.fine_tunes.FineTuneEvent

## Deployments

`client.deployments` — model deployments.

::: dashscope.resources.deployments.Deployments

::: dashscope.resources.deployments.AsyncDeployments

::: dashscope.resources.deployments.Deployment

::: dashscope.resources.deployments.DeploymentList

::: dashscope.resources.deployments.DeploymentEvent

## Apps

`client.apps` — application completions (agents / workflows).

::: dashscope.resources.apps.Apps

::: dashscope.resources.apps.AsyncApps

::: dashscope.resources.apps.AppCompletions

::: dashscope.resources.apps.AsyncAppCompletions

::: dashscope.resources.apps.AppCompletion

::: dashscope.resources.apps.AppStream

::: dashscope.resources.apps.AsyncAppStream

::: dashscope.resources.apps.AppUsage

::: dashscope.resources.apps.Thought

::: dashscope.resources.apps.DocReference

::: dashscope.resources.apps.WorkflowMessage

::: dashscope.resources.apps.AppModelUsage

## TingWu

`client.tingwu` — meeting transcription and analysis.

::: dashscope.resources.tingwu.TingWu

::: dashscope.resources.tingwu.AsyncTingWu

::: dashscope.resources.tingwu.TingWuRealtime

::: dashscope.resources.tingwu.TingWuResult

## Decision

`client.decision` — structured decision / judging calls.

::: dashscope.resources.decision.Decision

::: dashscope.resources.decision.AsyncDecision

::: dashscope.resources.decision.DecisionResponse

::: dashscope.resources.decision.DecisionAnswer

::: dashscope.resources.decision.ChoiceQuestion

::: dashscope.resources.decision.NoulQuestion

::: dashscope.resources.decision.ScoreQuestion

## Tokenization

`client.tokenization` — token counting.

::: dashscope.resources.tokenization.Tokenization

::: dashscope.resources.tokenization.AsyncTokenization

::: dashscope.resources.tokenization.TokenCount

## Reinforcement learning

`client.rl` — agentic RL datasets, functions and jobs.

::: dashscope.resources.reinforcement_learning.Rl

::: dashscope.resources.reinforcement_learning.AsyncRl

::: dashscope.resources.reinforcement_learning.RlJobs

::: dashscope.resources.reinforcement_learning.AsyncRlJobs

::: dashscope.resources.reinforcement_learning.RegisteredFunctions

::: dashscope.resources.reinforcement_learning.UploadedData

## Shared async-task models

Long-running services (images, videos, transcription, batch embeddings)
share one task contract: `jobs.create(...)` submits and returns a
`PENDING`/`RUNNING` `Job`; `jobs.fetch(task_id)` reads one snapshot;
`jobs.wait(job)` polls with exponential backoff until a terminal status
(`SUCCEEDED` / `FAILED` / `CANCELED` / `UNKNOWN`) or the wait `timeout`
(default 24 hours) expires; `jobs.cancel(task_id)` cancels a `PENDING`
task. The one-call `generate()` helpers submit and wait in one step.

A `FAILED` task raises by default; with `Client(raise_on_error=False)`
it comes back as the `Job` envelope instead — check `job.status` /
`job.is_error`. `UNKNOWN` is terminal too: the gateway reports it for a
task id that does not exist, so check `job.succeeded` rather than
relying on `wait()` to raise. See
[Types — raw responses & parsing](types.md#raw-responses-parsing) for
how `strict_validation`, `raise_on_error` and `raw=True` interact.

::: dashscope.resources._jobs.Job

::: dashscope.resources._jobs.JobResult

::: dashscope.resources._jobs.JobList

::: dashscope.resources._jobs.Jobs

::: dashscope.resources._jobs.AsyncJobs

## Batch Chat

::: dashscope.resources.batch_chat.BatchChatCompletions

::: dashscope.resources.batch_chat.AsyncBatchChatCompletions

::: dashscope.resources.batch_chat.BatchChat

::: dashscope.resources.batch_chat.AsyncBatchChat

## Batches

::: dashscope.resources.batches.BatchFile

::: dashscope.resources.batches.RequestCounts

::: dashscope.resources.batches.BatchErrorItem

::: dashscope.resources.batches.BatchErrors

::: dashscope.resources.batches.Batch

::: dashscope.resources.batches.BatchFileList

::: dashscope.resources.batches.BatchList

::: dashscope.resources.batches.FileDeletion

::: dashscope.resources.batches.FileContent

::: dashscope.resources.batches.BatchFiles

::: dashscope.resources.batches.AsyncBatchFiles

::: dashscope.resources.batches.Batches

::: dashscope.resources.batches.AsyncBatches

## Conversations

::: dashscope.resources.conversations.ConversationItemList

::: dashscope.resources.conversations.ConversationItems

::: dashscope.resources.conversations.AsyncConversationItems

::: dashscope.resources.conversations.Conversations

::: dashscope.resources.conversations.AsyncConversations

## Fine Tunes Compress

::: dashscope.resources.fine_tunes_compress.CompressError

::: dashscope.resources.fine_tunes_compress.CompressJob

::: dashscope.resources.fine_tunes_compress.CompressJobList

::: dashscope.resources.fine_tunes_compress.CompressHyperParameter

::: dashscope.resources.fine_tunes_compress.CompressTemplate

::: dashscope.resources.fine_tunes_compress.CompressCustomModel

::: dashscope.resources.fine_tunes_compress.CompressTemplateList

::: dashscope.resources.fine_tunes_compress.CompressTemplates

::: dashscope.resources.fine_tunes_compress.CompressJobs

::: dashscope.resources.fine_tunes_compress.Compress

::: dashscope.resources.fine_tunes_compress.AsyncCompressTemplates

::: dashscope.resources.fine_tunes_compress.AsyncCompressJobs

::: dashscope.resources.fine_tunes_compress.AsyncCompress

## Messages

::: dashscope.resources.messages.MessageStream

::: dashscope.resources.messages.AsyncMessageStream

::: dashscope.resources.messages.Messages

::: dashscope.resources.messages.AsyncMessages

## Responses

::: dashscope.resources.responses.InputItemList

::: dashscope.resources.responses.ResponseStream

::: dashscope.resources.responses.AsyncResponseStream

::: dashscope.resources.responses.InputItems

::: dashscope.resources.responses.AsyncInputItems

::: dashscope.resources.responses.Responses

::: dashscope.resources.responses.AsyncResponses
