---
description: "Compare 1.x and 2.0 side by side and migrate existing code to the resource-style dashscope client."
---
# Migrating from 1.x

The 2.0 client replaces the 1.x module-level static classes, but the complete
1.x surface keeps working from its historical import paths. Migrate at your
own pace — this page maps old habits to new ones.

## Side-by-side comparison

| | 1.x | 2.0 |
|---|---|---|
| Entry point | Module-level static classes: `Generation.call(...)` | A resource-style client: `client.chat.completions.create(...)` |
| One-liner | — | `dashscope.chat("qwen-plus", "Tell me a joke")` |
| Response | Raw dict-like; check `status_code == HTTPStatus.OK` | Typed object; read `resp.output_text`, `resp.usage.total_tokens` |
| Errors | Inspect `resp.code` / `resp.message` by hand | Typed `DashScopeError` exceptions raised by default |
| Streaming | Generator of raw frames | `ChatStream` with `.text_stream`, `.reasoning_stream`, `.get_final_message()` |
| Protocols | DashScope native + a separate compatible-mode entry | One client speaks `native`, `openai`, or `anthropic` via `protocol=` |
| Async | Separate `Aio*` classes | `dashscope.AsyncClient()` mirrors the sync surface |
| Connections | A fresh session per call | One keep-alive pool per client, shared across resources |
| Config | Module globals (`dashscope.api_key`, `set_region`) | Layered: args → `dashscope.settings` → env → `~/.dashscope/config.toml` |

## Call mapping

| 1.x class | 2.0 resource |
| --- | --- |
| `Generation` / `AioGeneration` | `client.chat.completions` |
| `MultiModalConversation` | `client.multimodal` |
| `ImageSynthesis` / `ImageGeneration` | `client.images` |
| `VideoSynthesis` | `client.videos` |
| `TextEmbedding` / `BatchTextEmbedding` | `client.embeddings` |
| `MultiModalEmbedding` | `client.multimodal` (embeddings) |
| `TextReRank` | `client.rerank` |
| `SpeechSynthesizer` (tts / tts_v2) | `client.audio.tts` |
| `Recognition` / `Transcription` | `client.audio.asr` |
| `Files` | `client.files` |
| `Models` | `client.models` |
| `FineTunes` | `client.fine_tunes` |
| `Deployments` | `client.deployments` |
| `Application` | `client.apps` |

Every 1.x class remains documented under [Legacy 1.x](reference/legacy.md).

## Before / after

1.x — call, then inspect the status by hand:

```python
from http import HTTPStatus
from dashscope import Generation

resp = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello!"}],
)
if resp.status_code != HTTPStatus.OK:
    print(resp.code, resp.message)
else:
    print(resp.output.text)
```

2.0 — typed response, exceptions on failure:

```python
from dashscope import Client, DashScopeError

try:
    with Client() as client:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    print(resp.output_text)
except DashScopeError as exc:
    print(exc.code, exc.message)
```

## What stays the same

- Import paths: `from dashscope import Generation` still works (frozen facade).
- The `DASHSCOPE_API_KEY` environment variable and the console-issued keys.
- Model names and request payloads — the wire contract is unchanged for the
  native protocol.

!!! tip "Migrate incrementally"
    Both surfaces coexist in one process. Move call sites one by one; there
    is no flag day. See [Release Notes](release-notes.md) for what each
    2.x release changed.