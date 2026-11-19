---
description: "Copy-paste recipes for every dashscope capability: multimodal, images, videos, embeddings, rerank, decision, tokenization, speech, TingWu, apps, files and models."
---
# Capability recipes

Short runnable snippets, one per capability. Each one is self-contained —
it creates and closes its own client, so any block can be copied and run
on its own. All of them assume an exported `DASHSCOPE_API_KEY` (see
[Quickstart](../quickstart.md)) and use the sync client. `AsyncClient`
mirrors the HTTP calls with `await`; the WebSocket speech synthesis and
file recognition examples below are sync-only. Response fields are typed —
the [API Reference](../reference/client.md) documents each one.

## Multimodal understanding

```python
from dashscope import Client

with Client() as client:
    resp = client.multimodal.chat.create(
        model="qwen-vl-plus",
        messages=[{
            "role": "user",
            "content": [
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/"
                          "file-manage-files/zh-CN/20241022/emyrja/"
                          "dog_and_girl.jpeg"},
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
from dashscope import Client

with Client() as client:
    job = client.images.generate(
        model="wan2.2-t2i-flash",
        prompt="A cat astronaut floating in space",
    )
    print(job.image_urls)
```

## Video generation

```python
from dashscope import Client

with Client() as client:
    job = client.videos.generate(
        model="wan2.2-t2v-plus",
        prompt="A cat astronaut floating in space",
    )
    print(job.video_url)
```

## Text embeddings

```python
from dashscope import Client

with Client() as client:
    resp = client.embeddings.create(
        model="text-embedding-v4",
        input=["hello world", "你好世界"],
    )
    print(len(resp.embeddings[0].embedding))
```

## Rerank

```python
from dashscope import Client

with Client() as client:
    resp = client.rerank.create(
        model="gte-rerank-v2",
        query="What is DashScope?",
        documents=["DashScope is a model service.", "The weather is fine."],
        top_n=2,
    )
    print(resp.results)
```

## Decision model

`client.decision` calls the System One decision model: it answers typed
questions about a business `state` in one forward pass — multi-choice
(`choice`), yes/no (`noul`) and ordinal scale (`score`) — returning
probability distributions without generating text. See
[Client Resources](../reference/resources.md) for details.

```python
from dashscope import Client
from dashscope.resources.decision import ChoiceQuestion, NoulQuestion

with Client() as client:
    resp = client.decision.create(
        model="decision-model-preview",
        state={"ticket_id": "T-1001", "content": "I was charged twice"},
        questions={
            "department": ChoiceQuestion(
                instructions="Which team should handle this ticket?",
                criteria={
                    "billing": "Billing and refund issues",
                    "technical": "Product faults and usage issues",
                },
            ),
            "escalate": NoulQuestion(
                instructions="Page the on-call engineer?",
            ),
        },
    )
    print(resp.answers["department"].choice)
    print(resp.answers["escalate"].noul)  # P(yes), 0=no 1=yes
```

## Token counting

Pass `messages=[...]` instead of `prompt` to count a chat message list.

```python
from dashscope import Client

with Client() as client:
    resp = client.tokenization.count(
        model="qwen3.7-plus",
        prompt="Introduce Hangzhou in one sentence",
    )
    print(resp.input_tokens)
```

## Speech synthesis

!!! note "Sync-only WebSocket API"
    Use `Client` for this example. `AsyncClient.audio.tts` does not expose
    `synthesize()`; adding `await` does not make this call async.

`synthesize()` returns the concatenated audio bytes.

```python
from dashscope import Client

with Client() as client:
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
from dashscope import Client

with Client() as client:
    for event in client.audio.asr.recognize_file(
        "audio.pcm",
        model="paraformer-realtime-v2",
    ):
        print(event)
```

See [Audio](../reference/audio.md) and the
[Qwen Live guide](qwen-live-python-sdk.md) for realtime sessions.

## TingWu (offline analysis)

`call()` returns the parsed result in one POST. The `fileUrl` and
`appid` below are placeholders — replace them with your own TingWu app
ID and a publicly reachable audio/video URL.

```python
from dashscope import Client

with Client() as client:
    result = client.tingwu.call(
        model="tingwu-automotive-service-inspection",
        input={
            "fileUrl": "https://your-bucket.oss-cn-beijing.aliyuncs.com/meeting.mp3",
            "appid": "your-tingwu-app-id",
        },
    )
    print(result.output)
```

## Application call

```python
from dashscope import Client

with Client() as client:
    resp = client.apps.create(app_id="your-app-id", prompt="Introduce yourself")
    print(resp.text)
```

## File upload

```python
from dashscope import Client

with Client() as client:
    file = client.files.upload(file_path="train.jsonl", purpose="fine-tune")
    print(file.file_id)
```

## Model list

```python
from dashscope import Client

with Client() as client:
    page = client.models.list()
    print(len(page.items), "models available")
```

The full interface for the remaining capabilities — batch jobs,
fine-tunes, model deployments, agentic RL and multimodal embeddings —
lives in [Client Resources](../reference/resources.md); realtime voice
in the [Qwen Live guide](qwen-live-python-sdk.md), voice & vocabulary
customization in [Audio](../reference/audio.md), and managed agents in
[AgentStudio](../reference/agentstudio.md). Chat completion examples are
in the [Quickstart](../quickstart.md).