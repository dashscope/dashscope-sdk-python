---
name: retrieval
description: Quick reference for retrieval (Embedding / Batch Embedding / Multimodal Embedding / TextReRank) models, API parameters, inputs/outputs, and error codes
---

# Retrieval

For new Python code, target **DashScope SDK 2.0.3** and use `dashscope.Client` / `dashscope.AsyncClient`. The implementation is in the root `dashscope` package; **`dashscope.v2` does not exist**. Set `DASHSCOPE_API_KEY` before running the network examples. Keep the 1.x `.output` contract confined to the compatibility section.

Authoritative sources: `dashscope/resources/embeddings.py`, `multimodal/embeddings.py`, `rerank.py`, `tokenization.py`, and `_jobs.py`.

## Applicable Models and Scenarios

- Text vectors: `text-embedding-v1/v2` (fixed dimension), `text-embedding-v3/v4` (configurable dimensions and dense/sparse output where supported).
- Offline batch text vectors: `text-embedding-async-v1/v2`, using a file containing one text per line.
- Multimodal vectors: `multimodal-embedding-one-peace-v1`, `multimodal-embedding-v1`, `qwen3-vl-embedding`, `qwen2.5-vl-embedding`, `tongyi-embedding-vision-plus`, and `tongyi-embedding-vision-flash`.
- Relevance ranking: `gte-rerank`, `gte-rerank-v2`, `qwen3-rerank`, and `qwen3-vl-rerank` where supported by the selected rerank resource.
- Token counting: `client.tokenization.count`; local tokenization is separately available under `dashscope.tokenizers` and sends no request.

Model support and allowed parameter values are service/model-specific; the SDK passes model names as strings rather than enforcing a whitelist.

## 2.0.3 Text and Multimodal Embeddings

### Text embeddings

```python
client.embeddings.create(
    *, model: str, input: str | list[str], text_type=None, dimension=None,
    output_type=None, instruct=None, parameters=None, timeout=None,
    api_key=None, workspace=None,
) -> EmbeddingResponse
```

- `text_type` is typically `query` for retrieval queries or `document` for corpus/symmetric inputs.
- v3/v4 support model-dependent `dimension` and `output_type` (`dense`, `sparse`, or `dense&sparse`). Do not pass these to v1/v2. `instruct` supplies a task instruction where supported; additional options belong in `parameters`.
- Empty input raises `ValueError` locally. Streaming is not supported.

### Multimodal embeddings

The actual 2.0 entry point is **`client.multimodal.embeddings.create`** (there is no method named `create_multimodal`):

```python
client.multimodal.embeddings.create(
    *, model: str, input: dict | EmbeddingItem | list,
    dimension=None, output_type=None, fps=None, instruct=None,
    enable_fusion=None, res_level=None, max_video_frames=None,
    parameters=None, timeout=None, api_key=None, workspace=None,
) -> EmbeddingResponse
```

Each input item sets exactly one of `text`, `image`, `audio`, or `video`, plus optional `factor`. Use dictionaries or `EmbeddingItem` from `dashscope.resources.multimodal.embeddings`. `fps`, `res_level`, and `max_video_frames` apply only to supported snapshot models; `enable_fusion` is model-specific (including qwen3-vl-embedding). Local paths and `file://` URIs upload to OSS first. Empty input or an item without a modality raises `ValueError`.

```python
import dashscope

with dashscope.Client() as client:
    embedded = client.embeddings.create(
        model="text-embedding-v3",
        input=["今天天气不错", "明天会下雨吗"],
        text_type="document",
        dimension=1024,
    )
    for item in embedded.embeddings:
        print(item.index, len(item.embedding))

    multimodal = client.multimodal.embeddings.create(
        model="multimodal-embedding-v1",
        input=[
            {"text": "一只猫"},
            {"image": "https://example.com/cat.jpg"},
        ],
    )
    for item in multimodal.embeddings:
        print(item.index, item.type, len(item.embedding))
```

Both return the same flat `EmbeddingResponse`: `embeddings`, `usage`, and the shared response envelope. Each `Embedding` has `index`, `embedding`, and optional `type`; the 1.x wire name `text_index` is normalized to `index`. The old one-peace singular vector is normalized to item 0. **Do not read `.output["embeddings"]` in 2.0.**

## 2.0.3 Batch Embedding Jobs

```python
client.embeddings.jobs.create(
    *, model: str, file_url: str, text_type=None, parameters=None,
    timeout=None, api_key=None, workspace=None,
) -> Job
client.embeddings.jobs.fetch(task_id: str, *, timeout=None,
                             api_key=None, workspace=None) -> Job
client.embeddings.jobs.wait(job_or_task_id, *, poll_interval=1.0,
    max_interval=8.0, timeout=86400.0, api_key=None, workspace=None) -> Job
```

`create` accepts exactly one `file_url`, not a list; an empty URL raises `ValueError`. `create` only submits, `fetch` obtains one snapshot, and `wait` polls to a terminal state. `timeout` on create/fetch is an HTTP timeout; on wait it is the total wait budget.

```python
import dashscope

with dashscope.Client() as client:
    submitted = client.embeddings.jobs.create(
        model="text-embedding-async-v2",
        file_url="https://example.com/embedding-input.txt",
        text_type="document",
    )
    snapshot = client.embeddings.jobs.fetch(submitted.task_id)
    print(snapshot.status)
    done = client.embeddings.jobs.wait(submitted, timeout=3600.0)
    print(done.result_url)  # download and parse this result file
```

Read flat `task_id`, `status`, `result_url`, `usage`, `request_id`, and `job.succeeded` / `job.finished`. Status values are `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELED`, and `UNKNOWN`. By default, `create` / `fetch` raise on `FAILED`, and `wait` raises on `FAILED` or `CANCELED`; `UNKNOWN` is terminal and returned.

## 2.0.3 Rerank

```python
client.rerank.create(
    *, model: str, query: str, documents: list[str], top_n=None,
    return_documents=None, instruct=None, parameters=None, timeout=None,
    api_key=None, workspace=None,
) -> RerankResponse
```

`top_n` defaults to service behavior (normally all documents); `return_documents=True` includes source text. `instruct` supplies a ranking instruction where supported. Empty `query` or `documents` raises `ValueError`.

```python
import dashscope

with dashscope.Client() as client:
    ranked = client.rerank.create(
        model="gte-rerank-v2",
        query="什么是文本向量？",
        documents=["文本向量把文本映射到向量空间。", "今天天气晴朗。"],
        top_n=1,
        return_documents=True,
    )
    for result in ranked.results:
        document_text = result.document.text if result.document else None
        print(result.index, result.relevance_score, document_text)
```

`RerankResponse.results` is flat. Each result has `index`, `relevance_score`, and optional `document`; `document` is an object whose text is read as `.text`, not a plain string. `index` refers to the original `documents` position.

## 2.0.3 Tokenization

```python
client.tokenization.count(
    *, model: str, prompt=None, messages=None, parameters=None,
    timeout=None, api_key=None, workspace=None,
) -> TokenCount
```

Supply `prompt` or `messages`; if both are supplied, `prompt` is appended as the latest user message. With neither, the SDK raises `ValueError`.

```python
import dashscope
from dashscope.tokenizers import get_tokenizer

with dashscope.Client() as client:
    count = client.tokenization.count(model="qwen-plus", prompt="你好，世界")
    print(count.input_tokens)
    print(count.token_ids, count.tokens)  # present when returned by the service

# Local tokenizer: no HTTP request and no Client lifecycle to manage.
tokenizer = get_tokenizer("qwen-plus")
token_ids = tokenizer.encode("你好，世界")
print(tokenizer.decode(token_ids))
```

`TokenCount` exposes `usage`, convenience `input_tokens`, and optional flat `token_ids` / `tokens`. Async clients use `await` for every network operation above and `await client.aclose()` when not using `async with`.

## RAG Integration Guide

- Self-built pipeline: call `client.embeddings.create(..., text_type="document")` for the corpus and `text_type="query"` for queries. After vector-store recall, pass candidates to `client.rerank.create`, then send the top results to `client.chat.completions.create`.
- Managed Bailian knowledge-base applications: configure the knowledge base in the console and call `client.apps.completions.create(app_id=..., prompt=...)`; do not manually duplicate platform-managed retrieval.

## Errors (brief)

- In 2.0, catch `dashscope.errors.DashScopeError` or a specific subclass. With `raise_on_error=False`, check `response.is_error`, `code`, `message`, and `request_id`; for jobs also check `status`.
- In 1.x, continue checking `status_code`, `code`, and `message`; do not assume HTTP failures raise as they do in the default 2.0 mode.

## 1.x Compatibility and Maintenance Guide

Use these contracts only for existing 1.x calls. They remain available as legacy exports in SDK 2.0.3 but are not the recommended surface for new code.

| 1.x entry point | Maintenance contract |
| --- | --- |
| `TextEmbedding.call(model, input, workspace=None, api_key=None, text_type=None, dimension=None, output_type=None, instruct=None, **kwargs)` | Returns `DashScopeAPIResponse`; read `output["embeddings"]`, whose items use `text_index` and `embedding` |
| `MultiModalEmbedding.call(...)` / `AioMultiModalEmbedding.call(...)` | Input item classes include text/image/audio variants with `factor`; read `output["embeddings"]` (or old one-peace `output["embedding"]`) |
| `BatchTextEmbedding.call(model, url, ...)` / `async_call` / `fetch` / `wait` / `cancel` / `list` | `call` submits and waits, `async_call` only submits; completed `output.url` points to the result file |
| `TextReRank.call(model, query, documents, return_documents=None, top_n=None, api_key=None, instruct=None, **kwargs)` | Read `output.results`; `document` follows the 1.x response representation |
| `Tokenization.call` | Keep its legacy response access, including `output.token_ids` / `output.tokens`, when maintaining old code |

For multimodal 1.x calls, `dimension`, `output_type`, `fps`, `instruct`, `enable_fusion`, `res_level`, and `max_video_frames` remain model-dependent. `auto_truncation` is a 1.x extra option for supported models. Empty model/input validation raises 1.x `ModelRequired` / `InputRequired`; batch `url` is required.

```python
from http import HTTPStatus
from dashscope import TextEmbedding, TextReRank

response = TextEmbedding.call(
    model=TextEmbedding.Models.text_embedding_v3,
    input=["hello", "world"],
    text_type="document",
    dimension=1024,
)
if response.status_code == HTTPStatus.OK:
    for item in response.output["embeddings"]:
        print(item["text_index"], item["embedding"][:3])
else:
    print(response.request_id, response.code, response.message)

ranked = TextReRank.call(
    model=TextReRank.Models.gte_rerank,
    query="What is a text embedding?",
    documents=["Text embeddings map text into vectors", "The weather is nice"],
    top_n=1,
    return_documents=True,
)
if ranked.status_code == HTTPStatus.OK:
    for result in ranked.output.results:
        print(result["index"], result["relevance_score"], result["document"])
```

All synchronous 1.x responses above expose `status_code`, `request_id`, `code`, `message`, `output`, and `usage`. Batch `output` includes `task_id`, `task_status`, and `url`; rerank results include `index`, `relevance_score`, and optionally `document`.

## Java SDK

Java SDK v2.22.23 entry points:

- `TextEmbedding`: `TextEmbeddingResult call(TextEmbeddingParam param)` / `call(param, ResultCallback<TextEmbeddingResult>)`
- `MultiModalEmbedding`: `MultiModalEmbeddingResult call(MultiModalEmbeddingParam param)` (text/image/audio items)

```java
import com.alibaba.dashscope.embeddings.*;

TextEmbedding embedding = new TextEmbedding();
TextEmbeddingParam param = TextEmbeddingParam.builder()
        .model("text-embedding-v3")
        .texts(java.util.Arrays.asList("hello", "world"))
        .build();
TextEmbeddingResult result = embedding.call(param);
double[] vector = result.getOutput().getEmbeddings().get(0).getEmbedding().stream()
        .mapToDouble(Double::doubleValue).toArray();
```

Samples: `TextEmbeddingUsage.java`, `BatchTextEmbeddingUsage.java`, `MultiModalEmbeddingUsage.java`. (TextReRank / BatchTextEmbedding have no dedicated Java wrapper in the index — use the HTTP API.)
