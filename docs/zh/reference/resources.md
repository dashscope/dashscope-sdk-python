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