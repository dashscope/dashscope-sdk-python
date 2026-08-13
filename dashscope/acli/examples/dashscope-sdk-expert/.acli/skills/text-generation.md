---
name: text-generation
description: Quick reference for text generation (Generation / OpenAI-compatible API) models, parameters, input/output, and error codes
---

# Text Generation

Before answering, verify the user's SDK version and signature: `python -c "import dashscope,inspect; print(dashscope.__version__); print(inspect.signature(dashscope.Generation.call))"`. Based on SDK source `dashscope/aigc/generation.py`.

## Applicable Models and Scenarios

- Entry point `dashscope.Generation` (sync) / `dashscope.AioGeneration` (async, both top-level exports in `dashscope/__init__.py`), `task = "text-generation"`.
- Plain text chat/continuation: qwen series. Built-in constants in `Generation.Models`: `qwen_turbo="qwen-turbo"`, `qwen_plus="qwen-plus"`, `qwen_max="qwen-max"`; `qwen_v1` and `qwen_plus_v1` are deprecated (calls only warn).
- The `model` parameter is just a `str`; the SDK does not validate against a whitelist. For new models like qwen3, pass the model name string directly; refer to the console/docs for exact names.
- `enable_search` is only supported when `model` starts with `"qwen"`; when it starts with `"bailian"` you must pass `customized_model_id`, otherwise `InputRequired` is raised.
- Do not use this API for multimodal; use `MultiModalConversation`.

## SDK Interface

Signature (`Generation.call`, classmethod; `AioGeneration.call` is the async version with the same parameters):

```python
Generation.call(model, prompt=None, history=None, api_key=None, messages=None,
    plugins=None, workspace=None, stream=None, temperature=None, top_p=None,
    top_k=None, max_tokens=None, seed=None, stop=None, repetition_penalty=None,
    presence_penalty=None, result_format=None, incremental_output=None,
    enable_search=None, tools=None, tool_choice=None, enable_thinking=None,
    thinking_budget=None, n=None, logprobs=None, top_logprobs=None,
    search_options=None, parallel_tool_calls=None, response_format=None,
    output_format=None, **kwargs)
```

Returns: `Generator[GenerationResponse, None, None]` when `stream=True` (AsyncGenerator for async), otherwise a single `GenerationResponse`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| model | str | required | Model name; raises `ModelRequired` if empty |
| prompt | Any | None | Text input; at least one of prompt/messages required, otherwise raises `InputRequired` |
| messages | List[Message] | None | `[{"role":..., "content":...}]`; role constants in `Role`: user/system/assistant (plus bot/attachment) |
| api_key | str | None | Defaults to `dashscope.api_key` (i.e. env var `DASHSCOPE_API_KEY`) |
| stream | bool | None | Whether to stream |
| incremental_output | bool | None | Streaming only: True=each chunk has only new tokens; False=cumulative full text (for some models the SDK requests increments internally and merges them into full text) |
| result_format | str | None | `"message"` (recommended, via output.choices) or `"text"` (via output.text) |
| temperature / top_p / top_k | float/float/int | None | Sampling: [0,2) / (0,1.0] / candidate set size |
| max_tokens / seed / stop | int/int/str\|list | None | Max output tokens / random seed / stop words |
| repetition_penalty / presence_penalty | float | None | Repetition penalty, 1.0 = no penalty / range [-2.0, 2.0] |
| enable_search / search_options | bool/dict | None | Web search (only for qwen*) and its options |
| tools / tool_choice / parallel_tool_calls | list/str\|dict/bool | None | function calling definitions, selection strategy, parallel calls |
| enable_thinking / thinking_budget | bool/int | None | Hybrid thinking models: enable thinking / thinking token budget |
| n / logprobs / top_logprobs | int/bool/int | None | Number of generations 1-4 / return log probabilities and candidates per position |
| response_format | dict | None | e.g. `{"type": "json_object"}` (JSON mode) |
| output_format | str | None | qwen-deep-research only: "model_detailed_report" (default) / "model_summary_report" |
| workspace / plugins | str/str\|dict | None | Workspace id / plugins (sent via `X-DashScope-Plugin` header) |

## Input/Output

- Input is one of two: `messages` (chat) or `prompt` (`history` is deprecated, do not recommend).
- Response `GenerationResponse` (DictMixin, accessible via attributes or subscript): `status_code`, `request_id`, `code`, `message`, `output`, `usage`.
- `output` (GenerationOutput): `text`+`finish_reason` (text format); `choices[i].finish_reason`, `choices[i].message.role/content` (message format). Common `finish_reason` values: `stop`/`length`/`null` (mid-stream).
- `usage` (GenerationUsage): `input_tokens`, `output_tokens` (sum them yourself for the total).
- Streaming: each chunk is also a `GenerationResponse`; with `incremental_output=True`, concatenate each chunk's content for the full text.

## OpenAI-Compatible API

- Use the openai library with base_url from `dashscope.base_compatible_api_url`, default `https://dashscope.aliyuncs.com/compatible-mode/v1` (overridable via env var `DASHSCOPE_COMPATIBLE_BASE_URL`).
- Differences: returns native OpenAI objects (`choices[0].message.content`, `usage.prompt_tokens/completion_tokens/total_tokens`); errors are raised as openai exceptions (AuthenticationError/BadRequestError/RateLimitError etc.) instead of a code in the response body.

## Minimal Example

```python
from http import HTTPStatus
import dashscope
from dashscope import Generation

dashscope.api_key = "sk-xxx"  # or set env var DASHSCOPE_API_KEY

# Non-streaming
resp = Generation.call(model="qwen-plus",
                       messages=[{"role": "user", "content": "Hello"}],
                       result_format="message")
if resp.status_code == HTTPStatus.OK:
    print(resp.output.choices[0].message.content)
else:
    print(resp.request_id, resp.code, resp.message)

# Streaming (incremental output)
for r in Generation.call(model="qwen-plus",
                         messages=[{"role": "user", "content": "Write a short poem"}],
                         result_format="message", stream=True,
                         incremental_output=True):
    if r.status_code == HTTPStatus.OK:
        print(r.output.choices[0].message.content, end="")
    else:
        print(r.code, r.message); break

# OpenAI-compatible
from openai import OpenAI
client = OpenAI(api_key=dashscope.api_key, base_url=dashscope.base_compatible_api_url)
print(client.chat.completions.create(model="qwen-plus",
      messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
```

## Common Error Codes

Key point: Generation.call does **not** raise on HTTP failure; it returns a response with `status_code != 200`, with the error in `code`/`message`. Only local parameter validation raises `dashscope.common.error` exceptions.

| Error code/exception | HTTP | Meaning | Handling |
| --- | --- | --- | --- |
| InputRequired (exception) | local | Neither prompt nor messages passed | Supply the parameter before calling |
| ModelRequired (exception) | local | model not passed | Pass a model name |
| InvalidApiKey | 401 | API Key missing/invalid | Check DASHSCOPE_API_KEY or dashscope.api_key |
| InvalidParameter | 400 | Invalid parameters (result_format, messages structure, etc.) | Fix per the message |
| Model.NotExist | 400 | Wrong model name or no access | Verify model name and activation status |
| DataInspectionFailed | 400 | Blocked by content safety review | Adjust input/output prompts |
| Throttling / Throttling.RateQuota | 429 | Throttling / QPS exceeded | Lower concurrency, retry with exponential backoff |
| InternalError | 500 | Server-side error | Retry; file a ticket with request_id |
| ServiceUnavailable etc. | 503/504 | Service unavailable / gateway timeout | SDK auto-retries (REPEATABLE_STATUS); back off if it still fails |

Actual code strings are as returned by the server in `response.code`; in OpenAI-compatible mode they map to openai library exception types.

## Java SDK

Entry point `com.alibaba.dashscope.aigc.generation.Generation` (Java SDK v2.22.23):

- `GenerationResult call(HalfDuplexServiceParam param)` — sync
- `void call(HalfDuplexServiceParam param, ResultCallback<GenerationResult> callback)` — async callback
- `Flowable<GenerationResult> streamCall(HalfDuplexServiceParam param)` — streaming (RxJava)
- `void streamCall(HalfDuplexServiceParam param, ResultCallback<GenerationResult> callback)`

```java
import com.alibaba.dashscope.aigc.generation.*;

Generation gen = new Generation();
GenerationParam param = GenerationParam.builder()
        .model("qwen-plus")
        .messages(java.util.Arrays.asList(
            Message.builder().role("user").content("Hello").build()))
        .resultFormat("message")
        .build();
GenerationResult result = gen.call(param);
String text = result.getOutput().getChoices().get(0).getMessage().getContent();
// usage: result.getUsage().getInputTokens() / getOutputTokens()
```

Params are built with `GenerationParam.builder()` (model/messages/prompt/temperature/topP/maxTokens/resultFormat/enableSearch/tools...); errors come back on the result (`result.getCode()`/`getMessage()`) or as `ApiException`/`NoApiKeyException`. Samples: `ConversationQuickStart.java`, `ConversationStreamCall.java`.
