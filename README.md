# DashScope Python SDK

The DashScope Python SDK provides a comprehensive interface to [Alibaba Cloud Model Studio (Bailian)](https://www.alibabacloud.com/help/en/model-studio/) APIs, covering text generation, multi-modal understanding, embeddings, reranking, image/video generation, speech synthesis & recognition, and more.

## Installation
To install the DashScope Python SDK, simply run:
```shell
pip install dashscope
```

If you clone the code from github, you can install from  source by running:
```shell
pip install -e .
```

To use tokenizer in local mode without downloading any files, run:
```shell
pip install dashscope[tokenizer]
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

## Supported Models

| Category | Recommended Models | SDK Class |
|----------|-------------------|-----------|
| Text Generation | qwen3.7-max, qwen3.7-plus, qwen3.6-flash | `Generation` |
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

## SDK Features

### Text Generation

Synchronous call with messages:

```python
from http import HTTPStatus
from dashscope import Generation

response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    result_format="message",
)
```

Streaming:

```python
responses = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Write a short poem"}],
    result_format="message",
    stream=True,
)
for response in responses:
    if response.status_code == HTTPStatus.OK:
        print(response.output.choices[0].message.content, end="")
```

Async (non-blocking):

```python
from dashscope import AioGeneration

response = await AioGeneration.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
    result_format="message",
)
```

Key parameters: `stream`, `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`, `stop`, `repetition_penalty`, `tools`, `tool_choice`, `enable_thinking`.

### Multi-Modal Conversation

Vision (image + text), video, and audio understanding:

```python
from dashscope import MultiModalConversation

response = MultiModalConversation.call(
    model="qwen-vl-max",
    messages=[{
        "role": "user",
        "content": [
            {"image": "https://example.com/image.jpg"},
            {"text": "What is in this image?"},
        ],
    }],
)
```

### Text Embedding

```python
from dashscope import TextEmbedding

response = TextEmbedding.call(
    model="text-embedding-v3",
    input="Hello world",
    dimension=1024,
    text_type="document",
)
```

Parameters: `text_type` ("query" / "document"), `dimension`, `output_type` ("dense" / "sparse" / "dense&sparse"), `instruct`.

### Multi-Modal Embedding

```python
from dashscope import (
    MultiModalEmbedding,
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
)

response = MultiModalEmbedding.call(
    model="multimodal-embedding-v1",
    input=[
        MultiModalEmbeddingItemImage("https://example.com/image.jpg", factor=1),
        MultiModalEmbeddingItemText("a cat", factor=1),
    ],
)
```

### Batch Text Embedding

```python
from dashscope import BatchTextEmbedding

# Sync: submit and wait for result
result = BatchTextEmbedding.call(
    model="text-embedding-async-v2",
    url="https://your-file-url/texts.txt",
)

# Or async: submit task, then poll
task = BatchTextEmbedding.async_call(
    model="text-embedding-async-v2",
    url="https://your-file-url/texts.txt",
)
result = BatchTextEmbedding.wait(task)
```

### Text ReRank

```python
from dashscope import TextReRank

response = TextReRank.call(
    model="gte-rerank-v2",
    query="What is deep learning?",
    documents=[
        "Deep learning is a subset of machine learning.",
        "The weather is nice today.",
        "Neural networks are the foundation of deep learning.",
    ],
    top_n=2,
    return_documents=True,
)
```

### Image Generation

```python
from dashscope import ImageSynthesis

# Async task pattern
response = ImageSynthesis.async_call(
    model="wanx-v1",
    prompt="A serene mountain landscape at sunset",
)

# Wait for result
result = ImageSynthesis.wait(response)

# Sync call (for wan2.2-t2i-flash/plus)
result = ImageSynthesis.sync_call(
    model="wan2.2-t2i-flash",
    prompt="A serene mountain landscape at sunset",
)
```

### Video Generation

```python
from dashscope import VideoSynthesis

# Text-to-video
response = VideoSynthesis.async_call(
    model="wan2.7-t2v",
    prompt="A cat playing with a ball of yarn",
)

result = VideoSynthesis.wait(response)
```

### Speech Synthesis (TTS)

WebSocket streaming (real-time):

```python
from dashscope.audio.tts_v2 import SpeechSynthesizer

synthesizer = SpeechSynthesizer(model="cosyvoice-v1", voice="longxiaochun")
audio = synthesizer.call("Hello, welcome to DashScope!")
```

HTTP (one-shot):

```python
from dashscope import HttpSpeechSynthesizer

result = HttpSpeechSynthesizer.call(
    model="cosyvoice-v3-flash",
    text="Hello, welcome to DashScope!",
    voice="longxiaochun",
)
```

### Speech Recognition (ASR)

File transcription:

```python
from dashscope.audio.asr import Transcription

# Submit transcription task
response = Transcription.async_call(
    model="paraformer-v1",
    file_urls=["https://example.com/audio.wav"],
)

# Wait for result
result = Transcription.wait(response)
```

### Tokenization

```python
from dashscope import Tokenization

response = Tokenization.call(model="qwen-turbo", prompt="Hello world")
print(response.usage)
```

Local tokenizer (requires `pip install dashscope[tokenizer]`):

```python
from dashscope import get_tokenizer

tokenizer = get_tokenizer("qwen-turbo")
tokens = tokenizer.encode("Hello world")
```

### Application

Call a Bailian application:

```python
from dashscope import Application

response = Application.call(app_id="YOUR_APP_ID", prompt="Hello")
```

### Fine-tuning

```python
from dashscope import FineTunes

# Create a fine-tuning job
response = FineTunes.call(
    model="qwen-turbo",
    training_file_ids=["file-xxx"],
    hyper_parameters={"n_epochs": 3},
)

# Check status
status = FineTunes.get("ft-xxx")
```

### Deployments

```python
from dashscope import Deployments

# Deploy a fine-tuned model
response = Deployments.call(model="your-finetuned-model", capacity=1)

# List / get / scale / delete
deployments = Deployments.list()
info = Deployments.get("deployed-model-name")
Deployments.scale("deployed-model-name", capacity=2)
Deployments.delete("deployed-model-name")
```

### Assistants & Threads (Beta)

```python
from dashscope import Assistants, Threads, Messages, Runs

# Create an assistant
assistant = Assistants.create(
    model="qwen-max",
    name="Math Tutor",
    instructions="You are a math tutor.",
)

# Create a thread and send a message
thread = Threads.create()
Messages.create(thread.id, content="What is 2+2?")

# Run the assistant and wait for completion
run = Runs.create(thread.id, assistant_id=assistant.id)
result = Runs.wait(run.id, thread_id=thread.id)
```

### OpenAI-Compatible Interface

```python
from dashscope.aigc.chat_completion import Completions

response = Completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
)
```

## CLI Reference

The DashScope CLI is installed automatically with the SDK. All commands support `-k / --api-key` for authentication (or use the `DASHSCOPE_API_KEY` environment variable).

### Text Generation

```shell
dashscope generation create -m qwen-plus -p "Hello, who are you?"
dashscope generation create -m qwen-plus -p "Tell me a story" --stream
```

### File Management

```shell
dashscope files upload --file training_data.jsonl --purpose fine-tune
dashscope files list
dashscope files get file-xxx
dashscope files delete file-xxx
```

### Fine-tuning

```shell
# Create a fine-tuning job
dashscope ft create -m qwen-turbo -t file-xxx

# List jobs
dashscope ft list

# Get job status
dashscope ft get ft-xxx

# Stream training events
dashscope ft stream ft-xxx

# Cancel / delete a job
dashscope ft cancel ft-xxx
dashscope ft delete ft-xxx
```

### Deployments

```shell
dashscope deployments create -m your-finetuned-model --capacity 1
dashscope deployments list
dashscope deployments get dm-xxx
dashscope deployments scale dm-xxx --capacity 2
dashscope deployments delete dm-xxx
```

### OSS Upload

```shell
dashscope oss upload -m qwen-plus --file data.jsonl
```

### Agentic RL (Reinforcement Learning)

```shell
# Register reward/rollout functions
dashscope rl register_functions \
    --rollout-classpaths rollout.py:MyRollout \
    --reward-classpaths reward.py:MyReward

# Test registered functions
dashscope rl test_functions instance-xxx --type reward --input '{"key": "value"}'

# Upload training data
dashscope rl upload_data --training-files train.jsonl

# Run RL training workflow
dashscope rl run -c config.yaml

# Monitor jobs
dashscope rl list
dashscope rl get job-xxx
dashscope rl logs job-xxx
dashscope rl cancel job-xxx
```

### Legacy Command Support

The CLI supports legacy argparse-style commands for backward compatibility:

```shell
# Legacy format (still works)
dashscope fine_tunes.call --training_file_ids file-xxx

# New format (recommended)
dashscope ft create --training-file-ids file-xxx
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

## Error Handling
Currently, errors are thrown as exceptions.


## Contributing
Coming soon.


## License
This project is licensed under the Apache License (Version 2.0).
