---
description: "Copy-paste recipes for every dashscope capability: multimodal, images, videos, embeddings, rerank, speech, apps, files and models."
---
# Capability recipes

Short runnable snippets, one per capability. All of them assume an exported
`DASHSCOPE_API_KEY` (see [Quickstart](../quickstart.md)) and use the sync
client. `AsyncClient` mirrors the HTTP calls with `await`; the WebSocket
speech synthesis and file recognition examples below are sync-only.
Response fields are typed — the [API Reference](../reference/client.md)
documents each one.

## Multimodal understanding

```python
from dashscope import Client

with Client() as client:
    resp = client.multimodal.chat.create(
        model="qwen-vl-plus",
        messages=[{
            "role": "user",
            "content": [
                {"image": "https://example.com/dog_and_girl.jpeg"},
                {"text": "What is in this image?"},
            ],
        }],
    )
    print(resp.output_text)
```

## Image generation

`generate()` submits the task and waits for it; the finished job carries
the result URLs.

```python
job = client.images.generate(
    model="wan2.2-t2i-flash",
    prompt="A cat astronaut floating in space",
)
print(job.image_urls)
```

## Video generation

```python
job = client.videos.generate(
    model="wan2.2-t2v-plus",
    prompt="A cat astronaut floating in space",
)
print(job.video_url)
```

## Text embeddings

```python
resp = client.embeddings.create(
    model="text-embedding-v4",
    input=["hello world", "你好世界"],
)
print(len(resp.embeddings[0].embedding))
```

## Rerank

```python
resp = client.rerank.create(
    model="gte-rerank-v2",
    query="What is DashScope?",
    documents=["DashScope is a model service.", "The weather is fine."],
    top_n=2,
)
print(resp.results)
```

## Speech synthesis

!!! note "Sync-only WebSocket API"
    Use `Client` for this example. `AsyncClient.audio.tts` does not expose
    `synthesize()`; adding `await` does not make this call async.

`synthesize()` returns the concatenated audio bytes.

```python
audio = client.audio.tts.synthesize(
    "Hello DashScope",
    model="cosyvoice-v2",
    voice="longxiaochun",
)
with open("out.mp3", "wb") as f:
    f.write(audio)
```

## File recognition

!!! note "Sync-only WebSocket API"
    Use `Client` for this example. `AsyncClient.audio.asr` does not expose
    `recognize_file()`; adding `await` does not make this call async.

```python
for event in client.audio.asr.recognize_file(
    "audio.pcm",
    model="paraformer-realtime-v2",
):
    print(event)
```

See [Audio](../reference/audio.md) and the
[Qwen Live guide](qwen-live-python-sdk.md) for realtime sessions.

## Application call

```python
resp = client.apps.create(app_id="your-app-id", prompt="Introduce yourself")
print(resp.text)
```

## File upload

```python
file = client.files.upload(file_path="train.jsonl", purpose="fine-tune")
print(file.id)
```

## Model list

```python
page = client.models.list()
print(len(page.items), "models available")
```

See [Client Resources](../reference/resources.md) for the full resource
surface, including batch jobs and fine-tunes.
