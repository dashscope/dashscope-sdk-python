---
name: text-generation
description: Quick reference for text generation (Generation / OpenAI-compatible API) models, parameters, input/output, and error codes
---

# Text Generation

For new Python code, target **DashScope SDK 2.0.3** and use `dashscope.Client` / `dashscope.AsyncClient`. The 2.0 implementation is in the root `dashscope` package; **`dashscope.v2` does not exist**. Check the installed version with `python -c "import dashscope; print(dashscope.__version__)"`. Keep 1.x repair advice in the compatibility section below; do not mix its `.output` response contract with 2.0.

Authoritative sources: `dashscope/resources/chat/completions.py`, `dashscope/resources/decision.py`, `dashscope/types/completions.py`, and `dashscope/_quickstart.py`.

## 2.0.3 Chat Completions

Use `client.chat.completions.create` for text generation, continuation, tool calling, structured output, and chat-style understanding. Pass current model names such as `qwen-plus` as strings; the client does not maintain a model whitelist. The caller owns multi-turn `messages` history. Set `DASHSCOPE_API_KEY`, or pass `api_key=` to the client.

All parameters are keyword-only:

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

- `messages` accepts a text string or message dictionaries such as `{"role": "user", "content": "Hello"}`.
- Use named sampling, thinking, search, and tool parameters directly. Put additional model-specific parameters in `parameters={...}` rather than inventing SDK keyword arguments.
- `response_format={"type": "json_object"}` requests JSON mode where supported. Support for each option remains model-dependent.
- `AsyncClient` exposes the same signature with `await`; use `async with dashscope.AsyncClient() as client` and `await client.aclose()` when not using a context manager.

### Non-streaming and streaming delta

```python
import dashscope

with dashscope.Client() as client:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "用一句话介绍杭州"}],
        temperature=0.7,
        max_tokens=256,
    )
    print(response.choices[0].message.content)
    # Equivalent convenience view for the first choice:
    print(response.output_text)
    print(response.usage.total_tokens, response.request_id)

    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="写一首短诗",
        stream=True,
    )
    try:
        for chunk in stream:
            if not chunk.choices:  # a trailing usage-only chunk is valid
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
    finally:
        stream.close()
```

A non-streaming `ChatCompletion` exposes flat `choices`, `usage`, `search_info`, `model`, `request_id`, and the shared response envelope. Read `choices[i].message.content`, `finish_reason`, `reasoning_content`, and `tool_calls`; **do not read `.output`**. `usage` has `input_tokens`, `output_tokens`, and `total_tokens`.

Streaming returns `ChatStream` (`AsyncChatStream` for async). Each `ChatCompletionChunk` contains `choices[i].delta`, not `message`. By default frames are incremental. `incremental_output=False` requests cumulative snapshots, with client-side merging where necessary; in that mode do not concatenate every snapshot. Helpers include `stream.text_stream`, `reasoning_stream`, `tool_call_stream`, and `get_final_message()`; `headers` is on the stream. Async streaming uses `stream = await client.chat.completions.create(..., stream=True)`, `async for`, and `await stream.aclose()`.

### One-call shortcut

For a one-off script, `dashscope.chat(model, prompt, **kwargs)` forwards extra keyword arguments to `chat.completions.create`:

```python
import dashscope

response = dashscope.chat("qwen-plus", "给我讲个笑话")
print(response.output_text)
```

This shortcut uses a lazy process-lifetime singleton client. Prefer an explicitly managed `Client` / `AsyncClient` in applications and tests.

## 2.0.3 Decision (not text generation)

`client.decision.create` is the System One typed-decision API, currently using `model="decision-model-preview"`. It answers questions about a business `state`; it **does not generate text and has no streaming API**. It also does not provide function calling or web search. There is no 1.x Decision wrapper.

```python
create(*, model: str, state: str | dict | list,
    questions: dict[str, QuestionInput], timeout=None,
    api_key=None, workspace=None) -> DecisionResponse
```

`questions` maps caller-chosen question IDs to typed questions or raw wire dictionaries. The typed models are imported from `dashscope.resources.decision`:

| Question | Request fields | Corresponding answer fields |
| --- | --- | --- |
| `ChoiceQuestion` | `type="choice"`, optional `instructions`, required `criteria: Dict[str, str]` with 1–255 option descriptions | `choice` selected option, `probabilities`, `confidence` |
| `NoulQuestion` | `type="noul"`, optional `instructions` and optional `criteria: Dict[str, str]` | `noul` = P(yes), where 0=no and 1=yes; `probabilities` |
| `ScoreQuestion` | `type="score"`, optional `instructions`, required low-to-high `criteria: List[str]` with 2–255 levels | `score` probability-weighted level index, `probabilities`, `confidence`, `legend` |

Every answer also has optional `type`; the service may add extra fields. `probabilities` covers all options/levels. `DecisionResponse` exposes flat `answers`, `model`, `usage`, and `latency_ms`; answer keys mirror the request question IDs.

```python
from dashscope import Client
from dashscope.resources.decision import ChoiceQuestion, NoulQuestion, ScoreQuestion

with Client() as client:
    response = client.decision.create(
        model="decision-model-preview",
        state={"ticket_id": "T-1001", "content": "支付后仍未到账"},
        questions={
            "department": ChoiceQuestion(
                instructions="应该由哪个团队处理？",
                criteria={"billing": "支付和账单", "technical": "产品故障"},
            ),
            "escalate": NoulQuestion(instructions="是否通知值班人员？"),
            "severity": ScoreQuestion(
                instructions="严重程度如何？",
                criteria=["轻微", "中等", "严重"],
            ),
        },
    )

print(response.answers["department"].choice)
print(response.answers["department"].probabilities)
print(response.answers["escalate"].noul)
print(response.answers["severity"].score)
print(response.answers["severity"].probabilities)
```

An empty `questions` mapping raises `ValueError`; typed question fields are validated locally by pydantic. Raw dictionaries pass through for forward-compatible fields. Async code uses `await client.decision.create(...)` and returns the same `DecisionResponse` contract.

## Errors (brief)

- In 2.0, catch exceptions from `dashscope.errors` (normally `DashScopeError` or a more specific subclass). With `client.with_options(raise_on_error=False)`, check `response.is_error`, then `code` / `message` / `request_id`.
- In 1.x, continue checking `response.status_code`, `response.code`, and `response.message`; local validation can still raise 1.x exceptions.

## 1.x Compatibility and Repair Guide

Use this section only to maintain `Generation` / `AioGeneration` applications. These remain 1.x APIs even though they are still exported from the root package in SDK 2.0.3.

- Entry points: `dashscope.Generation.call` (sync) and `dashscope.AioGeneration.call` (async), `task="text-generation"`. `Generation.Models` contains only legacy constants; pass current model names as strings.
- Signature: `call(model, prompt=None, history=None, api_key=None, messages=None, plugins=None, workspace=None, stream=None, temperature=None, top_p=None, top_k=None, max_tokens=None, seed=None, stop=None, repetition_penalty=None, presence_penalty=None, result_format=None, incremental_output=None, enable_search=None, tools=None, tool_choice=None, enable_thinking=None, thinking_budget=None, n=None, logprobs=None, top_logprobs=None, search_options=None, parallel_tool_calls=None, response_format=None, output_format=None, **kwargs)`.
- Supply `prompt` or `messages`; `history` is deprecated. `result_format="message"` reads `output.choices`; `"text"` reads `output.text`. `enable_search` is supported only for model names beginning with `qwen`; `bailian*` requires `customized_model_id`.
- `stream=True` returns a generator (`AsyncGenerator` for `AioGeneration`). With `incremental_output=True`, concatenate chunk content; otherwise chunks may be cumulative.
- The 1.x `GenerationResponse` contract is `status_code`, `request_id`, `code`, `message`, `output`, and `usage`. In message mode use `output.choices[0].message.content`; usage exposes `input_tokens` and `output_tokens`.

```python
from http import HTTPStatus
from dashscope import Generation

response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
    result_format="message",
)
if response.status_code == HTTPStatus.OK:
    print(response.output.choices[0].message.content)
else:
    print(response.request_id, response.code, response.message)
```

HTTP failures in this 1.x API normally return an error response rather than raising. Missing model/input can raise `ModelRequired` / `InputRequired`; fix those locally before retrying.

## OpenAI-Compatible Library (independent API line)

This is separate from both the 2.0 resource client and 1.x `Generation`. Use the `openai` package with `base_url=dashscope.base_compatible_api_url` (default `https://dashscope.aliyuncs.com/compatible-mode/v1`, configurable through `DASHSCOPE_COMPATIBLE_BASE_URL`). It returns native OpenAI objects (`choices[0].message.content`, `usage.prompt_tokens/completion_tokens/total_tokens`) and raises `openai` exceptions.

```python
import os
from openai import OpenAI

with OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
) as client:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)
```

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
        .model("qwen3.7-plus")
        .messages(java.util.Arrays.asList(
            Message.builder().role("user").content("Hello").build()))
        .resultFormat("message")
        .build();
GenerationResult result = gen.call(param);
String text = result.getOutput().getChoices().get(0).getMessage().getContent();
// usage: result.getUsage().getInputTokens() / getOutputTokens()
```

Params are built with `GenerationParam.builder()` (model/messages/prompt/temperature/topP/maxTokens/resultFormat/enableSearch/tools...); errors come back on the result (`result.getCode()`/`getMessage()`) or as `ApiException`/`NoApiKeyException`. Samples: `ConversationQuickStart.java`, `ConversationStreamCall.java`.
