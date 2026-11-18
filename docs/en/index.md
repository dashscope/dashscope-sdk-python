---
description: "Official Python SDK for Alibaba Cloud Model Studio (DashScope): typed sync/async client, examples and feature overview."
---
# DashScope Python SDK

The `dashscope` SDK is the official Python client for
[Alibaba Cloud Model Studio (Bailian)](https://www.alibabacloud.com/help/en/model-studio/):
chat completions, multimodal understanding, embeddings, reranking, image and
video generation, speech synthesis & recognition, realtime voice, fine-tuning,
agentic RL, managed agents and applications.

Version 2.0 is a typed, resource-style client — `dashscope.Client()` — modeled
on the ergonomics of the OpenAI SDK: one client holds a shared connection pool,
every capability hangs off it as a typed resource, responses are typed objects,
and failures raise exceptions by default.

!!! tip "Where to start"
    New here? Follow the [Quickstart](quickstart.md). Migrating existing
    1.x code? See [Migrating from 1.x](migration.md).

## Features

- **Typed resources** — `client.chat`, `client.images`, `client.audio`, …
- **Sync & async** — `AsyncClient` mirrors the sync surface.
- **Three protocols** — `native`, `openai`, `anthropic` on one client.
- **Text-like streaming** — `.text_stream`, `.reasoning_stream`.
- **Typed errors + self-diagnosis** and an offline **dev mock**.
- **A real CLI** and an optional interactive AI assistant.

The full list: [Features](features.md).

## Requirements

- Python 3.9+ (the optional interactive assistant requires 3.10+).
- A Model Studio API key — create one in the
  [Model Studio console](https://bailian.console.aliyun.com/), then export it:

```shell
export DASHSCOPE_API_KEY="sk-..."
```

## Installation

```shell
pip install -U dashscope
```

The base install covers SDK API calls, the `dashscope` CLI, Agentic RL
fine-tuning and the local tokenizer. Optional extras configure the interactive
AI assistant, e.g. `pip install "dashscope[acli]"` (see [CLI](cli.md)).

## Examples

### Create it

```python
import dashscope

client = dashscope.Client()  # reads DASHSCOPE_API_KEY
```

### Run it

```python
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Tell me a joke"}],
)
print(resp.output_text)
print(resp.usage.total_tokens)
```

Or the one-liner: `dashscope.chat("qwen-plus", "Tell me a joke")`.

### Stream it

```python
with client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Write a haiku about the sea"}],
    stream=True,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()
```

### Async it

```python
import asyncio
import dashscope

async def main():
    async with dashscope.AsyncClient() as client:
        resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        print(resp.output_text)

asyncio.run(main())
```

### Talk to it (realtime voice)

`client.audio.live` runs full-duplex voice sessions over WebSocket — see the
[Qwen Live guide](guides/qwen-live-python-sdk.md) for a complete walkthrough.

## Command line (CLI 2.0)

Installing the SDK also installs the `dashscope` command. The 2.0 CLI
mirrors the SDK resources, so anything you can do with `Client` you can
also script from a shell:

```bash
dashscope version                 # print the SDK version
dashscope auth login              # save an API key
dashscope chat create --model qwen-plus --prompt "Hello!"
dashscope -o json ft list         # machine-readable output for scripting
```

It covers chat, multimodal, embeddings, images/videos, audio, files,
fine-tuning, agentic RL and more, with global options for protocol,
region and output format. Running `dashscope` with no arguments starts
the optional interactive AI assistant (`pip install "dashscope[acli]"`).
The frozen 1.x command tree remains available as `dashscope1`.
See [Command Line Interface](cli.md).

## Protocols

The same call can speak three wire protocols:

```python
client = dashscope.Client(protocol="openai")  # or "native", "anthropic"
```

See [Protocols](reference/protocols.md).

## Errors and diagnosis

Failures raise typed `DashScopeError` subclasses carrying `code`,
`request_id` and the response body; opt-in local diagnosis adds structured
fix guidance on `error.diagnosis`. See [Errors](reference/errors.md) and
[Devkit](reference/devkit.md).

## Legacy 1.x

The complete 1.x surface still works from its historical import paths
(`from dashscope import Generation`), implemented by frozen facades.
See [Legacy 1.x](reference/legacy.md).

## Dependencies

The base install pulls in `requests`, `aiohttp`, `httpx` + `httpx-sse`,
`websocket-client`, `pydantic` (v2), `tenacity`, `tiktoken`, and `typer` +
`rich` for the CLI. Optional assistant extras are listed under
[Installation](#installation).

## License

This project is licensed under the
[Apache License 2.0](https://github.com/dashscope/dashscope-sdk-python/blob/main/LICENSE).