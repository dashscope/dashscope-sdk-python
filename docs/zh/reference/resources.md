---
description: "客户端资源命名空间 API 参考：向量嵌入、图像、视频、文件、微调等。"
---
# 客户端资源（Resources）

挂载在 `Client` 与 `AsyncClient` 上的资源命名空间。
各节列出同步与异步类及其响应模型。

## 向量嵌入（Embeddings）

`client.embeddings` —— 文本向量调用与批量任务。

::: dashscope.resources.embeddings.Embeddings

::: dashscope.resources.embeddings.AsyncEmbeddings

::: dashscope.resources.embeddings.EmbeddingJobs

::: dashscope.resources.embeddings.AsyncEmbeddingJobs

::: dashscope.resources.embeddings.Embedding

::: dashscope.resources.embeddings.EmbeddingResponse

## 多模态（Multimodal）

`client.multimodal` —— 多模态对话与向量嵌入。

::: dashscope.resources.multimodal.Multimodal

::: dashscope.resources.multimodal.AsyncMultimodal

::: dashscope.resources.multimodal.chat.MultimodalChat

::: dashscope.resources.multimodal.chat.AsyncMultimodalChat

::: dashscope.resources.multimodal.embeddings.MultimodalEmbeddings

::: dashscope.resources.multimodal.embeddings.AsyncMultimodalEmbeddings

::: dashscope.resources.multimodal.embeddings.EmbeddingItem

## 重排序（Rerank）

`client.rerank` —— 文档重排序。

::: dashscope.resources.rerank.Rerank

::: dashscope.resources.rerank.AsyncRerank

::: dashscope.resources.rerank.RerankResponse

::: dashscope.resources.rerank.RerankResult

::: dashscope.resources.rerank.RerankDocument

## 图像（Images）

`client.images` —— 图像生成任务。

::: dashscope.resources.images.Images

::: dashscope.resources.images.AsyncImages

::: dashscope.resources.images.ImageJobs

::: dashscope.resources.images.AsyncImageJobs

## 视频（Videos）

`client.videos` —— 视频生成任务。

::: dashscope.resources.videos.Videos

::: dashscope.resources.videos.AsyncVideos

::: dashscope.resources.videos.VideoJobs

::: dashscope.resources.videos.AsyncVideoJobs

## 文件（Files）

`client.files` —— 文件上传、列表与删除。

::: dashscope.resources.files.Files

::: dashscope.resources.files.AsyncFiles

::: dashscope.resources.files.File

::: dashscope.resources.files.FileList

::: dashscope.resources.files.DeletionStatus

## 模型（Models）

`client.models` —— 模型目录。

::: dashscope.resources.models.Models

::: dashscope.resources.models.AsyncModels

::: dashscope.resources.models.Model

::: dashscope.resources.models.ModelList

## 微调（Fine-tunes）

`client.fine_tunes` —— 微调任务、日志与事件。

::: dashscope.resources.fine_tunes.FineTunes

::: dashscope.resources.fine_tunes.AsyncFineTunes

::: dashscope.resources.fine_tunes.FineTune

::: dashscope.resources.fine_tunes.FineTuneList

::: dashscope.resources.fine_tunes.JobLogs

::: dashscope.resources.fine_tunes.FineTuneEvent

## 部署（Deployments）

`client.deployments` —— 模型部署。

::: dashscope.resources.deployments.Deployments

::: dashscope.resources.deployments.AsyncDeployments

::: dashscope.resources.deployments.Deployment

::: dashscope.resources.deployments.DeploymentList

::: dashscope.resources.deployments.DeploymentEvent

## 应用（Apps）

`client.apps` —— 应用补全（智能体 / 工作流）。

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

## 听悟（TingWu）

`client.tingwu` —— 会议转写与分析。

::: dashscope.resources.tingwu.TingWu

::: dashscope.resources.tingwu.AsyncTingWu

::: dashscope.resources.tingwu.TingWuRealtime

::: dashscope.resources.tingwu.TingWuResult

## 决策（Decision）

`client.decision` —— 结构化决策 / 评判调用。

::: dashscope.resources.decision.Decision

::: dashscope.resources.decision.AsyncDecision

::: dashscope.resources.decision.DecisionResponse

::: dashscope.resources.decision.DecisionAnswer

::: dashscope.resources.decision.ChoiceQuestion

::: dashscope.resources.decision.NoulQuestion

::: dashscope.resources.decision.ScoreQuestion

## 分词（Tokenization）

`client.tokenization` —— Token 计数。

::: dashscope.resources.tokenization.Tokenization

::: dashscope.resources.tokenization.AsyncTokenization

::: dashscope.resources.tokenization.TokenCount

## 强化学习（Reinforcement learning）

`client.rl` —— 智能体 RL 数据集、函数与任务。

::: dashscope.resources.reinforcement_learning.Rl

::: dashscope.resources.reinforcement_learning.AsyncRl

::: dashscope.resources.reinforcement_learning.RlJobs

::: dashscope.resources.reinforcement_learning.AsyncRlJobs

::: dashscope.resources.reinforcement_learning.RegisteredFunctions

::: dashscope.resources.reinforcement_learning.UploadedData

## 共享异步任务模型

长耗时服务（图像、视频、转写、批量向量）共享同一套任务契约：
`jobs.create(...)` 提交并返回 `PENDING`/`RUNNING` 的 `Job`；
`jobs.fetch(task_id)` 读取一次快照；`jobs.wait(job)` 以指数退避轮询，
直到终态（`SUCCEEDED` / `FAILED` / `CANCELED` / `UNKNOWN`）或等待
`timeout`（默认 24 小时）到期；`jobs.cancel(task_id)` 取消 `PENDING`
任务。一步式 `generate()` 则提交并等待完成。

`FAILED` 任务默认抛出异常；`Client(raise_on_error=False)` 时以 `Job`
信封返回——检查 `job.status` / `job.is_error`。`UNKNOWN` 也是终态：
网关对不存在的任务 ID 报告该状态，因此请检查 `job.succeeded`，
不要依赖 `wait()` 抛异常来区分。`strict_validation`、`raise_on_error`
与 `raw=True` 的关系见
[类型——原始响应与解析](types.md#raw-responses-parsing)。

::: dashscope.resources._jobs.Job

::: dashscope.resources._jobs.JobResult

::: dashscope.resources._jobs.JobList

::: dashscope.resources._jobs.Jobs

::: dashscope.resources._jobs.AsyncJobs
