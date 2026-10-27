---
name: multimodal
description: Quick reference for multimodal (MultiModalConversation/ImageSynthesis/VideoSynthesis) models, API parameters, inputs/outputs, and error codes
---
# Multimodal

For new Python code, target **DashScope SDK 2.0.3** and use `dashscope.Client` / `dashscope.AsyncClient`. The 2.0 resources live in the root `dashscope` package; **`dashscope.v2` does not exist**. Set `DASHSCOPE_API_KEY` before running the examples. Use the explicitly marked 1.x section only for legacy maintenance, and do not mix 2.0 flattened responses with 1.x `.output` responses.

Authoritative sources: `dashscope/resources/multimodal/chat.py`, `images.py`, `videos.py`, and `_jobs.py`.

## 2.0.3 Resources and Models

- Multimodal understanding: `client.multimodal.chat.create`. Models include current Qwen VL/OCR model names; pass the model as a string and verify service-side availability.
- Image generation/editing: `client.images.generate` for submit-and-wait, or `client.images.jobs.create/fetch/wait` for explicit task control.
- Video generation: `client.videos.generate` or `client.videos.jobs.create/fetch/wait`.
- `AsyncClient` mounts the same resources. `await` each operation, and use `async with dashscope.AsyncClient() as client` to close connections.

`client.chat.completions.create` automatically routes messages containing image/audio/video parts to the multimodal endpoint. `client.multimodal.chat.create` always selects that endpoint in the native protocol, including text-only calls to VL models. Compatible protocols translate media on their own chat path.

## Multimodal Chat

`create` has the same keyword-only signature and `ChatCompletion` / `ChatStream` return types as `client.chat.completions.create`:

```python
create(*, model: str, messages: str | list[MessageInput], stream: bool = False,
    tools=None, temperature=None, top_p=None, max_tokens=None, stop=None,
    seed=None, repetition_penalty=None, presence_penalty=None,
    enable_thinking=None, response_format=None, top_k=None, tool_choice=None,
    n=None, logprobs=None, top_logprobs=None, thinking_budget=None,
    enable_search=None, search_options=None, parallel_tool_calls=None,
    incremental_output=None, parameters=None, timeout=None,
    api_key=None, workspace=None)
```

Use `parameters={...}` for model-specific options not present in the signature. Media content is a list of parts. With the native protocol, local paths and `file://` URIs are uploaded to OSS before the request.

```python
import dashscope

messages = [{
    "role": "user",
    "content": [
        {"image": "https://example.com/cat.jpg"},
        {"text": "Describe this image"},
    ],
}]

with dashscope.Client() as client:
    response = client.multimodal.chat.create(
        model="qwen-vl-plus",
        messages=messages,
    )
    print(response.output_text)

    stream = client.multimodal.chat.create(
        model="qwen-vl-plus",
        messages=messages,
        stream=True,
    )
    try:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
    finally:
        stream.close()
```

The 2.0 response flattens list-form multimodal text into a string. Read `response.choices[0].message.content` or `response.output_text`, plus flat `usage` / `request_id`; **do not read `.output`**. Streaming chunks use `choices[i].delta`, with the same incremental/cumulative `incremental_output` semantics as text chat.

## Images and Videos

### Actual resource signatures

Both `generate` methods return a finished `Job`. Their `jobs.create` methods accept the same model/input/generation fields but only submit; their `timeout` is the submit HTTP timeout. `generate` additionally has `submit_timeout`, `poll_interval`, and `max_interval`, and its `timeout` is the total wait budget.

```python
client.images.generate(
    *, model: str, prompt: str, negative_prompt=None, n=None, size=None,
    image_urls=None, extra_input=None, function=None, seed=None, style=None,
    prompt_extend=None, watermark=None, parameters=None,
    timeout=86400.0, submit_timeout=None, poll_interval=1.0,
    max_interval=8.0, api_key=None, workspace=None,
) -> Job

client.videos.generate(
    *, model: str, prompt=None, image_url=None, media=None, extra_input=None,
    function=None, seed=None, size=None, duration=None, resolution=None,
    ratio=None, prompt_extend=None, watermark=None, parameters=None,
    timeout=86400.0, submit_timeout=None, poll_interval=1.0,
    max_interval=8.0, api_key=None, workspace=None,
) -> Job

client.images.jobs.fetch(task_id: str, *, timeout=None, api_key=None, workspace=None)
client.videos.jobs.fetch(task_id: str, *, timeout=None, api_key=None, workspace=None)
client.images.jobs.wait(job_or_task_id, *, poll_interval=1.0, max_interval=8.0,
                        timeout=86400.0, api_key=None, workspace=None)
client.videos.jobs.wait(job_or_task_id, *, poll_interval=1.0, max_interval=8.0,
                        timeout=86400.0, api_key=None, workspace=None)
```

- Image `size` accepts `1024*1024` and normalizes `1024x1024` to the wire form. `image_urls` supplies reference images for supported models; use `extra_input` for model-specific input fields and `parameters` for model-specific generation options.
- Image protocol selection is automatic by model name. Legacy text2image models use asynchronous tasks; `wan2.6-t2i*` / `wan2.6-image*` use the multimodal-generation protocol. A synchronous multimodal `generate` is capped by `submit_timeout`, not polling settings.
- Video `image_url` is the public name for the legacy wire field `img_url`. Older models accept `prompt` and/or `image_url`. `wan2.7-i2v*` requires `media` or `image_url`; media items use asset types `first_frame`, `last_frame`, `driving_audio`, and `first_clip`. `image_url` is shorthand for one `first_frame` item and must not duplicate one already in `media`.
- Unlike 1.x video behavior, explicitly pass `prompt_extend=True` when prompt extension is wanted.

### Runnable examples

```python
import dashscope

with dashscope.Client() as client:
    image_job = client.images.generate(
        model="wanx2.1-t2i-turbo",
        prompt="一只在月球上弹吉他的猫，水彩风格",
        size="1024*1024",
        timeout=600.0,
    )
    print(image_job.status, image_job.image_urls)

    video_job = client.videos.generate(
        model="wan2.7-t2v",
        prompt="一只小猫在草地上奔跑",
        size="1280*720",
        timeout=900.0,
    )
    print(video_job.status, video_job.video_url)
```

Explicit submission, polling, and waiting:

```python
import dashscope

with dashscope.Client() as client:
    submitted = client.images.jobs.create(
        model="wanx2.1-t2i-turbo",
        prompt="一只猫",
        size="1024*1024",
    )
    snapshot = client.images.jobs.fetch(submitted.task_id)
    print(snapshot.status)

    done = client.images.jobs.wait(submitted, timeout=600.0)
    print(done.image_urls)
```

`Job` exposes flat `task_id`, `status`, `results`, `result_url`, `image_urls`, `video_url`, `usage`, and the shared response envelope. Each item in `results` can expose `url` (and service-specific extra fields). Status values are `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELED`, and `UNKNOWN`; use `job.succeeded` / `job.finished` for checks. `wait` polls with exponential backoff. With default error handling, `create` / `fetch` raise on `FAILED`, and `wait` raises on `FAILED` or `CANCELED`; `UNKNOWN` is terminal and is returned.

## Errors (brief)

- In 2.0, catch `dashscope.errors.DashScopeError` or a more specific subclass. If using `raise_on_error=False`, inspect `response.is_error` / `code` / `message`; for jobs also inspect `status`.
- In 1.x, keep checking `status_code`, `code`, and `message` on the response, including task failure information under the 1.x `output` contract.

## 1.x Compatibility and Maintenance Guide

These APIs retain their old response shapes; do not rewrite only the response access while leaving the old entry point in place.

| 1.x entry point | Maintenance notes |
| --- | --- |
| `MultiModalConversation.call` / `AioMultiModalConversation.call` | `stream=True` returns a generator/async generator; otherwise returns `MultiModalConversationResponse` |
| `ImageSynthesis.call/async_call/fetch/wait/cancel/list` | `call` submits and waits; `async_call` only submits; `sync_call` is limited to supported wan2.2 t2i models |
| `VideoSynthesis.call/async_call/fetch/wait/cancel/list` | Same asynchronous task pattern for video |

For `MultiModalConversation.call`, key parameters remain `model`, `messages`, `text`, `voice`, `language_type`, `stream`, sampling options, `result_format`, `incremental_output`, `enable_search`, tools, thinking options, `ocr_options`, and logprobs. Empty `model` raises `ModelRequired`; empty `messages` and `text` raises `ValueError`.

For `ImageSynthesis`, old inputs include `prompt` (required), `negative_prompt`, `size`, `n`, `seed`, `style`, `prompt_extend`, `watermark`, `ref_img`, `sketch_image_url`, `base_image_url`, `mask_image_url`, `ref_strength`, and `ref_mode`. For `VideoSynthesis`, old inputs include `prompt`, `negative_prompt`, `img_url`, `audio_url`, `first_frame_url` / `last_frame_url` (or `head_frame` / `tail_frame`), `media`, `reference_urls`, `template`, `size`, `duration`, `resolution`, `ratio`, `seed`, `prompt_extend`, `watermark`, `shot_type`, and `audio_setting`. Keep model-specific validation when repairing these calls.

The 1.x task flow is `async_call(...)` → `fetch(task)` for one snapshot or `wait(task)` until terminal; `task` can be a task ID or previous response. `cancel` applies only to `PENDING`. Old response contracts are:

- `MultiModalConversationResponse.output.choices[0].message.content` (a content-part list), plus `output.text` where supplied.
- `ImageSynthesisResponse.output.task_id/task_status/results`; each result has `url`.
- `VideoSynthesisResponse.output.task_id/task_status/video_url`.
- All expose `status_code`, `request_id`, `code`, `message`, and `usage`; check `status_code == HTTPStatus.OK` before reading `output`.

Do not use a context manager for these 1.x classmethod calls:

```python
from http import HTTPStatus
from dashscope import ImageSynthesis, MultiModalConversation

response = MultiModalConversation.call(
    model="qwen-vl-plus",
    messages=[{"role": "user", "content": [
        {"image": "https://example.com/cat.jpg"},
        {"text": "Describe this image"},
    ]}],
)
if response.status_code == HTTPStatus.OK:
    print(response.output.choices[0].message.content)

job = ImageSynthesis.async_call(
    model="qwen-image-3.0",
    prompt="A cat running under the moonlight",
    n=2,
    size="1024*1024",
)
job = ImageSynthesis.wait(job)
if job.status_code == HTTPStatus.OK and job.output.task_status == "SUCCEEDED":
    for result in job.output.results:
        print(result.url)
```

## Java SDK

Java SDK v2.22.23 entry points:

- `MultiModalConversation`: `call(param)` / `call(param, ResultCallback)` / `streamCall(param)` / `streamCall(param, ResultCallback)` — input `MultiModalConversationParam` with `MultiModalMessage` content lists (image/audio/video + text)
- `ImageSynthesis`: `call(ImageSynthesisParam)` / `asyncCall` / `syncCall` (also `SketchImageSynthesisParam` variants) — async task pattern: submit then `fetch`/`wait`
- `ImageGeneration`: `call` / `streamCall` / `asyncCall` (`ImageGenerationParam`)
- `VideoSynthesis`: `call(param)` / `asyncCall(param)` / `fetch(String taskId, String apiKey)` / `list(...)`

```java
import com.alibaba.dashscope.aigc.multimodalconversation.*;

MultiModalConversation conv = new MultiModalConversation();
MultiModalConversationParam param = MultiModalConversationParam.builder()
        .model("qwen3-vl-flash")
        .messages(java.util.Arrays.asList(userMessage))
        .build();
MultiModalConversationResult result = conv.call(param);
```

Samples: `MultiModalDialogUsage.java`, `ImageSynthesisUsage.java`, `VideoSynthesisUsage.java`, `MultiModalEmbeddingUsage.java`.
