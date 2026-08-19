---
name: multimodal
description: Quick reference for multimodal (MultiModalConversation/ImageSynthesis/VideoSynthesis) models, API parameters, inputs/outputs, and error codes
---
# Multimodal

Locate before answering: understanding tasks use `MultiModalConversation` (sync/streaming); generation tasks use `ImageSynthesis`/`VideoSynthesis` (async tasks). Parameter names must follow the source code in `dashscope/aigc/`; never answer from memory.

## Applicable Models and Scenarios

- Understanding (image/video understanding, OCR, function calling): `qwen-vl-max`, `qwen-vl-plus`, qwen-ocr series, etc. (the `Models` class in source only has the constant `qwen_vl_chat_v1`; pass others as strings).
- Text-to-image / image editing (`ImageSynthesis`, task=`text2image`): `wanx-v1`, `wanx-sketch-to-image-v1` (sketch coloring), `wanx2.1-imageedit` (image editing, automatically switches to image2image task).
- Video generation (`VideoSynthesis`, task=`video-generation`): `wanx2.1-t2v-turbo/plus` (text-to-video), `wanx2.1-i2v-turbo/plus` (image-to-video), `wanx2.1-kf2v-plus`/`wanx-kf2v` (first/last frame). `wanx-txt2video-pro` and `wanx-img2video-pro` are deprecated.

## SDK Interfaces

| Class | Key Methods | Notes |
|---|---|---|
| `MultiModalConversation` | `call(model, messages, ...)` | `stream=True` returns a Generator, otherwise returns a response object; async version `AioMultiModalConversation.call` |
| `ImageSynthesis` | `call / async_call / fetch(task) / wait(task) / cancel(task) / list(...)` | `call` blocks until results are ready; `sync_call` only supports wan2.2-t2i-flash/plus |
| `VideoSynthesis` | Same as above (no `list` difference, parameters below) | Same async task flow |

Async task flow: `async_call(...)` returns a task (`output.task_id`, `task_status=PENDING`) → `fetch(task)` for a single query or `wait(task)` to poll until a terminal state; `cancel(task)` can only cancel `PENDING`. The `task` parameter accepts a task_id string or the response object from async_call. `task_status` values: `PENDING/RUNNING/SUCCEEDED/FAILED/CANCELED/UNKNOWN`.

Key parameters of `MultiModalConversation.call` (all optional, in source signature order): `messages`, `text`, `voice`, `language_type`, `stream`, `temperature`[0,2), `top_p`(0,1], `top_k`, `max_tokens`, `seed`, `stop`, `repetition_penalty`, `presence_penalty`, `result_format`("message"/"text"), `incremental_output`, `enable_search`, `tools`, `tool_choice`, `enable_thinking`, `n`(1-4), `ocr_options`, `logprobs`, `top_logprobs`; plus `api_key`, `workspace`. **Empty `model` raises `ModelRequired`; both messages and text empty raises `ValueError`.**

Key parameters of `ImageSynthesis.call/async_call`: `prompt` (required, empty raises `InputRequired`), `negative_prompt`, `size`("width*height" e.g. "1024*1024"), `n`, `seed`, `style` (auto/photography/anime etc.), `prompt_extend`, `watermark`, `ref_img`, `sketch_image_url` (sketch), `base_image_url`+`mask_image_url` (imageedit), `ref_strength`[0,1], `ref_mode`(repaint/refonly).

Key parameters of `VideoSynthesis.call/async_call`: `prompt`, `negative_prompt`, `img_url` (image-to-video), `audio_url`, `first_frame_url`/`last_frame_url` (or `head_frame`/`tail_frame`), `media`, `reference_urls`, `template`, `size`("1280*720"), `duration` (seconds, default 5), `resolution`("720P"/"1080P"), `ratio`("16:9"), `seed`, `prompt_extend`, `watermark`, `shot_type`, `audio_setting`.

## Input/Output

Multimodal messages (content is an array of dicts; local paths or URLs both work, the SDK auto-uploads to OSS and adds header `X-DashScope-OssResourceResolve: enable`):

```python
messages = [{"role": "user", "content": [
    {"image": "https://.../a.jpg"},   # or {"video": [...]} / {"audio": ...}
    {"text": "Describe this image"},
]}]
```

Responses uniformly contain `status_code / request_id / code / message / output / usage` (on error `output` is None; first check `status_code==200` i.e. `HTTPStatus.OK`):

- `MultiModalConversationResponse.output`: `choices[0].message.content` (content array isomorphic to the input), `finish_reason`, `text`; `usage`: `input_tokens`/`output_tokens`/`characters`.
- `ImageSynthesisResponse.output`: `task_id`, `task_status`, `results` (each contains `url`); `usage.image_count`.
- `VideoSynthesisResponse.output`: `task_id`, `task_status`, `video_url`; `usage`: `video_count`/`video_duration`/`video_ratio`.

## Minimal Examples

Image-text understanding:

```python
from http import HTTPStatus
from dashscope import MultiModalConversation

rsp = MultiModalConversation.call(model="qwen-vl-max", messages=messages)
if rsp.status_code == HTTPStatus.OK:
    print(rsp.output.choices[0].message.content[0]["text"])
```

Text-to-image async task:

```python
from dashscope import ImageSynthesis

rsp = ImageSynthesis.async_call(
    model=ImageSynthesis.Models.wanx_v1,
    prompt="A cat running under the moonlight", n=2, size="1024*1024",
)
rsp = ImageSynthesis.wait(rsp)          # poll until SUCCEEDED/FAILED
if rsp.output.task_status == "SUCCEEDED":
    for r in rsp.output.results:
        print(r.url)
```

## Common Error Codes

| Error Code | HTTP Status | Meaning | Handling |
|---|---|---|---|
| InvalidApiKey | 401 | API Key missing/invalid | Check `dashscope.api_key` or the `DASHSCOPE_API_KEY` environment variable |
| InvalidParameter | 400 | Invalid parameter (size/format/field name) | Check each item against the parameter tables in this file; locally also raises `InvalidParameter` exception |
| DataInspectionFailed | 400 | prompt/image failed content moderation | Rewrite the prompt or change the input image |
| Throttling | 429 | Rate limit triggered | Retry with exponential backoff, reduce concurrency, confirm quota |
| InternalError | 500 | Server-side error | Retry with `request_id` or file a ticket |
| Task FAILED | 200 | Async task execution failed | Read `output.task_status` and the failure info in `output`, then adjust input and retry |

Local validation exceptions (no request sent): `ModelRequired` (missing model), `InputRequired` (missing prompt), `ValueError` (messages/text both empty). Network/server exceptions may raise `RequestFailure` (contains `http_code`).

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
        .model("qwen-vl-max")
        .messages(java.util.Arrays.asList(userMessage))
        .build();
MultiModalConversationResult result = conv.call(param);
```

Samples: `MultiModalDialogUsage.java`, `ImageSynthesisUsage.java`, `VideoSynthesisUsage.java`, `MultiModalEmbeddingUsage.java`.
