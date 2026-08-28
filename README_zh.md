# DashScope Python SDK

> [English](README.md) | **中文**

DashScope Python SDK 提供了访问[阿里云百炼（Model Studio）](https://help.aliyun.com/zh/model-studio/) API 的完整接口，覆盖文本生成、多模态理解、向量（Embedding）、重排（Rerank）、图像/视频生成、语音合成与识别等能力。

## 最新动态

**v1.27.0 内置交互式 AI 助手 —— [DashScope SDK Expert](#ai-助手dashscope-sdk-expert)。** 直接运行 `dashscope`（不带任何参数），或直接提问（如 `dashscope "如何流式输出 Generation 结果"`），即可在终端中获得 SDK/API 答疑、可运行示例、CLI 用法和错误诊断。助手基于按领域划分的速查技能（文本、多模态、语音、检索、微调、Agent、CLI），这些技能构建在 SDK 的公开接口——参数、输出、错误码——之上，让你直接提问，无需翻文档。在助手内输入 `/help` 可查看可用命令。

## 安装

安装 DashScope Python SDK，只需运行：
```shell
pip install dashscope
```

如果从 GitHub 克隆了源码，可以通过源码安装：
```shell
pip install -e .
```

如需在本地模式下使用 tokenizer 且不下载任何文件，运行：
```shell
pip install dashscope[tokenizer]
```

## 快速开始

```python
from http import HTTPStatus
from dashscope import Generation

responses = Generation.call(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"},
    ],
    result_format="message",
)

if responses.status_code == HTTPStatus.OK:
    print(responses.output.choices[0].message.content)
else:
    print(f"Error: {responses.code} - {responses.message}")
```

## API Key 鉴权

SDK 使用 API Key 进行鉴权。获取 API Key 的方法请参考[阿里云百炼官方文档（国内站）](https://help.aliyun.com/zh/model-studio/)和[阿里云百炼官方文档（国际站）](https://www.alibabacloud.com/help/en/model-studio/)。

### 使用 API Key

1. 通过代码设置 API Key
```python
import dashscope

dashscope.api_key = 'YOUR-DASHSCOPE-API-KEY'
# 或者通过代码指定 API Key 文件路径
# dashscope.api_key_file_path='~/.dashscope/api_key'

```

2. 通过环境变量设置 API Key

a. 直接使用以下环境变量设置 API Key

```shell
export DASHSCOPE_API_KEY='YOUR-DASHSCOPE-API-KEY'
```

b. 通过环境变量指定 API Key 文件路径

```shell
export DASHSCOPE_API_KEY_FILE_PATH='~/.dashscope/api_key'
```

3. 将 API Key 保存到文件
```python
from dashscope import save_api_key

save_api_key(api_key='YOUR-DASHSCOPE-API-KEY',
             api_key_file_path='api_key_file_location or (None, will save to default location "~/.dashscope/api_key"')

```

## AI 助手：DashScope SDK Expert

SDK 内置了交互式 AI 助手 **DashScope SDK Expert**，基于随包提供的 Agentic CLI（`dashscope/acli`）框架构建。对于 DashScope SDK/CLI 用户，它是获取开发咨询和 AI 编码帮助的推荐方式——直接在终端中解答 SDK/API 问题、生成可运行示例、展示 CLI 用法、诊断错误。

- 直接运行 `dashscope`（不带参数）即可启动助手。首次运行时会提示安装 SDK Expert 知识包（按领域划分的速查技能：文本、多模态、语音、检索、微调、Agent、CLI），使助手的指导来自 SDK 的公开接口——参数、输出、错误码——而无需阅读源码
- 直接提问代替翻文档——如 `dashscope "如何流式输出 Generation 结果"` 或 `dashscope "取消微调任务的 CLI 命令"`。在助手内输入 `/help` 可列出可用命令（`/setup`、`/skill`、`/stats` 等）；经典 SDK 子命令依然可用，无法识别的命令会自动转给助手处理

## 支持的模型

| 类别 | 推荐模型 | SDK 类 |
|----------|-------------------|-----------|
| 文本生成 | qwen3.8-max、qwen3.7-max、qwen3.7-plus、qwen3.6-flash | `Generation` |
| 多模态理解 | qwen3.5-omni-plus、qwen3.7-plus（视觉） | `MultiModalConversation` |
| 文本向量 | text-embedding-v4、text-embedding-v3 | `TextEmbedding` |
| 多模态向量 | tongyi-embedding-vision-plus、qwen3-vl-embedding | `MultiModalEmbedding` |
| 文本重排 | qwen3-rerank、gte-rerank-v2 | `TextReRank` |
| 图像生成 | wan2.7-image-pro、qwen-image-2.0-pro | `ImageSynthesis` |
| 视频生成 | wan2.7-t2v、wan2.7-i2v、happyhorse-1.0-t2v/i2v | `VideoSynthesis` |
| 语音合成（TTS） | cosyvoice-v3.5-plus、cosyvoice-v1 | `SpeechSynthesizer`、`HttpSpeechSynthesizer` |
| 语音识别（ASR） | fun-asr-realtime、fun-asr、paraformer-v1 | `Transcription` |
| 全模态（实时） | qwen3.5-omni-plus-realtime | `MultiModalConversation` |

最新模型列表请访问[百炼模型广场](https://bailian.console.aliyun.com/)。

## Shell 命令补全

运行对应命令一次，然后重启 Shell（或重新 source 配置文件）：

| Shell | 安装命令 |
|-------|-----------------|
| **bash** | `dashscope --install-completion bash` |
| **zsh** | `dashscope --install-completion zsh` |
| **fish** | `dashscope --install-completion fish` |

如需预览补全脚本而不安装：
```shell
dashscope --show-completion bash
```

## 日志

如需输出 DashScope 日志，请配置日志级别：
```shell
export DASHSCOPE_LOGGING_LEVEL='info'

```

## 输出

输出包含以下字段：
```
     request_id (str): 请求 ID。
     status_code (int): HTTP 状态码，200 表示请求成功，其他值表示错误。
     code (str): 出错时的错误码，否则为空字符串。
     message (str): 出错时设置为错误信息。
     output (Any): 请求输出。
     usage (Any): 请求用量信息。
```

## 许可证

本项目采用 Apache License (Version 2.0) 许可证。
