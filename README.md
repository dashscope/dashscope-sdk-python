# DashScope Python SDK

> **English** | [中文](README_zh.md)

The DashScope Python SDK provides a comprehensive interface to [Alibaba Cloud Model Studio (Bailian)](https://www.alibabacloud.com/help/en/model-studio/) APIs, covering text generation, multi-modal understanding, embeddings, reranking, image/video generation, speech synthesis & recognition, and more.

## What is New

**v1.27.0 ships an interactive AI assistant — [DashScope SDK Expert](#ai-assistant-dashscope-sdk-expert).** Run `dashscope` with no arguments (or ask directly, e.g. `dashscope "how do I stream Generation output"`) to get SDK/API answers, runnable examples, CLI usage, and error diagnosis right in your terminal. Guidance is drawn from per-domain quick-reference skills (text, multimodal, speech, retrieval, fine-tuning, agent, cli) built on the SDK's public interfaces — parameters, outputs, and error codes — so you can ask instead of reading the docs. Type `/help` inside the assistant to view available commands.

## Installation
To install the DashScope Python SDK, simply run:
```shell
pip install dashscope
```

The base install covers SDK API calls only. Optional feature groups are
available as extras:

| Extra | Provides | Install |
|-------|----------|---------|
| `cli` | `dashscope` command subcommands (generation, files, ...) | `pip install "dashscope[cli]"` |
| `acli` | Interactive AI assistant (DashScope SDK Expert) | `pip install "dashscope[acli]"` |
| `rl` | Agentic RL fine-tuning | `pip install "dashscope[rl]"` |
| `tokenizer` | Local tokenizer without downloads | `pip install "dashscope[tokenizer]"` |

If you clone the code from github, you can install from  source by running:
```shell
pip install -e .
```


## Quick Start

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

## API Key Authentication

The SDK uses API key for authentication. Please refer to [official documentation for alibabacloud china](https://www.alibabacloud.com/help/en/model-studio/) and [official documentation for alibabacloud international](https://www.alibabacloud.com/help/en/model-studio/) regarding how to obtain your api-key.

### Using the API Key

1. Set the API key via code
```python
import dashscope

dashscope.api_key = 'YOUR-DASHSCOPE-API-KEY'
# Or specify the API key file path via code
# dashscope.api_key_file_path='~/.dashscope/api_key'

```

2. Set the API key via environment variables

a. Set the API key directly using the environment variable below

```shell
export DASHSCOPE_API_KEY='YOUR-DASHSCOPE-API-KEY'
```

b. Specify the API key file path via an environment variable

```shell
export DASHSCOPE_API_KEY_FILE_PATH='~/.dashscope/api_key'
```

3. Save the API key to a file
```python
from dashscope import save_api_key

save_api_key(api_key='YOUR-DASHSCOPE-API-KEY',
             api_key_file_path='api_key_file_location or (None, will save to default location "~/.dashscope/api_key"')

```

## AI Assistant: DashScope SDK Expert

The SDK ships with an interactive AI assistant, **DashScope SDK Expert**, built on the bundled Agentic CLI (`dashscope/acli`) framework. For DashScope SDK/CLI users it is the recommended way to get development consultation and AI coding help — answering SDK/API questions, generating runnable examples, showing CLI usage, and diagnosing errors, right in your terminal.

- Run `dashscope` with no arguments to start the assistant. On first run it offers to install the SDK Expert knowledge pack (per-domain quick-reference skills: text, multimodal, speech, retrieval, fine-tuning, agent, cli), so guidance comes from the SDK's public interfaces — parameters, outputs, error codes — without reading the source
- Ask it instead of reading docs — e.g. `dashscope "how do I stream Generation output"` or `dashscope "CLI command to cancel a fine-tuning job"`. Type `/help` inside the assistant to list available commands (`/setup`, `/skill`, `/stats`, ...); classic SDK subcommands still work, and unrecognized commands are routed to the assistant

## Supported Models

| Category | Recommended Models | SDK Class |
|----------|-------------------|-----------|
| Text Generation | qwen3.8-max, qwen3.7-max, qwen3.7-plus, qwen3.6-flash | `Generation` |
| Multi-Modal Understanding | qwen3.5-omni-plus, qwen3.7-plus (vision) | `MultiModalConversation` |
| Text Embedding | text-embedding-v4, text-embedding-v3 | `TextEmbedding` |
| Multi-Modal Embedding | tongyi-embedding-vision-plus, qwen3-vl-embedding | `MultiModalEmbedding` |
| Text ReRank | qwen3-rerank, gte-rerank-v2 | `TextReRank` |
| Image Generation | wan2.7-image-pro, qwen-image-2.0-pro | `ImageSynthesis` |
| Video Generation | wan2.7-t2v, wan2.7-i2v, happyhorse-1.0-t2v/i2v | `VideoSynthesis` |
| Speech Synthesis (TTS) | cosyvoice-v3.5-plus, cosyvoice-v1 | `SpeechSynthesizer`, `HttpSpeechSynthesizer` |
| Speech Recognition (ASR) | fun-asr-realtime, fun-asr, paraformer-v1 | `Transcription` |
| Omni (Real-time) | qwen3.5-omni-plus-realtime | `MultiModalConversation` |

For the latest model list, visit [Bailian Model Plaza](https://bailian.console.aliyun.com/).

## Shell Completion

Run the appropriate command once, then restart your shell (or re-source your config file):

| Shell | Install command |
|-------|-----------------|
| **bash** | `dashscope --install-completion bash` |
| **zsh** | `dashscope --install-completion zsh` |
| **fish** | `dashscope --install-completion fish` |

To preview the completion script without installing:
```shell
dashscope --show-completion bash
```

## Logging
To output Dashscope logs, you need to configure the logger.
```shell
export DASHSCOPE_LOGGING_LEVEL='info'

```

## Output
The output contains the following fields:
```
     request_id (str): The request id.
     status_code (int): HTTP status code, 200 indicates that the
         request was successful, others indicate an error.
     code (str): Error code if error occurs, otherwise empty str.
     message (str): Set to error message on error.
     output (Any): The request output.
     usage (Any): The request usage information.
```

## License
This project is licensed under the Apache License (Version 2.0).
