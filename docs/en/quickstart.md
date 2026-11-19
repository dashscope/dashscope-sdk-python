---
description: "Install dashscope, configure your API key and make your first chat completion in minutes."
---
# Quickstart

The dashscope API centers on a typed [Client](reference/client.md) (and its
async twin `AsyncClient`) with resource namespaces such as `client.chat`,
`client.images` and `client.audio`. This page gets you from install to your
first chat completion.

## Installation

Python 3.9 or newer is required.

```bash
python -m pip install -U dashscope
```

## Configure your API key

The recommended way is the `DASHSCOPE_API_KEY` environment variable — never
hard-code keys in source files:

```bash
export DASHSCOPE_API_KEY="<your-api-key>"
```

Alternatively, persist the key once with `dashscope.save_api_key("sk-xxx")`
(writes `~/.dashscope/api_key`), or assign it at runtime through
`dashscope.settings.api_key`.

Configuration is resolved in this order, first match wins:

1. Explicit `Client(...)` arguments
2. Process-level `dashscope.settings`
3. Environment variables (`DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, ...)
4. `config.toml`
5. The API key file

!!! tip "CI and containers"
    Inject `DASHSCOPE_API_KEY` as a secret environment variable in CI;
    `save_api_key` is meant for interactive local machines only.

## One-call chat

For scripts and notebooks, `dashscope.chat` manages a client for you:

```python
import dashscope

resp = dashscope.chat("qwen-plus", "Tell me a joke")
print(resp.output_text)
```

Extra keyword arguments are forwarded to `client.chat.completions.create`.

## Use the Client

```python
import dashscope

client = dashscope.Client()  # reads DASHSCOPE_API_KEY
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.output_text)
```

`Client` owns an HTTP connection pool; close it when done, or use it as a
context manager:

```python
with dashscope.Client() as client:
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

### Streaming

Pass `stream=True` and iterate over the chunks:

```python
with dashscope.Client() as client:
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Write a haiku about the sea."}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Async

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

## Error handling

Failed calls (connection failures, HTTP errors and server-side business
errors) raise subclasses of
[DashScopeError][dashscope.errors.DashScopeError] carrying `code`,
`status_code`, `request_id` and the response `body`:

```python
from dashscope import Client, DashScopeError

try:
    with Client() as client:
        client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Hi"}],
        )
except DashScopeError as exc:
    print(exc.code, exc.status_code, exc.request_id, exc.message)
```

!!! note "Two boundaries"
    - Local configuration and argument validation may raise `ValueError`
      (e.g. an invalid region or an empty `file_id`); those are not
      `DashScopeError`s.
    - Streaming has two phases: a failure to establish the stream raises
      `DashScopeError` as above; once the stream is established, low-level
      transport errors during iteration (e.g. `httpx.ReadTimeout`) propagate
      unchanged. The SDK does not reconnect automatically — catch and
      handle them in the caller.

See [Errors](reference/errors.md) for the full hierarchy (rate limits,
authentication, timeouts, WebSocket errors, ...).

## Next steps

- [Client & Configuration](reference/client.md) — `Client` / `AsyncClient`
  constructor options, `Config` / `Settings`, region switching.
- [Chat](reference/chat.md) — chat completions, streaming and response models.
- [Audio](reference/audio.md) — TTS, ASR, realtime and Qwen Live sessions; see
  the [Qwen Live guide](guides/qwen-live-python-sdk.md) for a full walkthrough.
- [Resources](reference/resources.md) — embeddings, rerank, images, videos,
  files, models, fine-tunes, deployments, apps and more.
- [Types](reference/types.md) — shared message / response / pagination models.