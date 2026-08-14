---
name: retrieval
description: Quick reference for retrieval (Embedding / Batch Embedding / Multimodal Embedding / TextReRank) models, API parameters, inputs/outputs, and error codes
---

# Retrieval

> Compiled from the SDK source code (embeddings/text_embedding.py, batch_text_embedding.py, multimodal_embedding.py, rerank/text_rerank.py); parameters and return fields are based on these sources.

## Applicable Models and Scenarios

**TextEmbedding** (task="text-embedding", `TextEmbedding.Models`):
- `text-embedding-v1` / `text-embedding-v2`: basic general-purpose vectors; only text_type is supported; fixed dimension.
- `text-embedding-v3`: supports text_type, dimension (default 1024), output_type (dense/sparse/dense&sparse).
- `text-embedding-v4`: everything v3 offers, plus exclusive dimension options 2048/1536.

**BatchTextEmbedding** (offline batch): `text-embedding-async-v1` / `text-embedding-async-v2`; submits a file url containing line-by-line texts for async batch processing.

**MultiModalEmbedding** (task="multimodal-embedding"): mixed text/image/audio/video vectors. Models: `multimodal-embedding-one-peace-v1`, `multimodal-embedding-v1`, `qwen3-vl-embedding`, `qwen2.5-vl-embedding`, `tongyi-embedding-vision-plus`, `tongyi-embedding-vision-flash`; coroutine version `AioMultiModalEmbedding`.

**TextReRank** (task="text-rerank"): reranks recalled results by relevance. Models: `gte-rerank`, `gte-rerank-v2`, `qwen3-rerank`, `qwen3-vl-rerank`; coroutine version `AioTextReRank`.

## SDK Interfaces

`TextEmbedding.call(model, input, workspace=None, api_key=None, text_type=None, dimension=None, output_type=None, instruct=None, **kwargs) -> DashScopeAPIResponse`

| Parameter | Description (based on source docstrings) |
| --- | --- |
| input | str or List[str], internally assembled as `{"texts": [...]}`; streaming not supported |
| text_type | `"query"` (retrieval query) / `"document"` (default; corpus and symmetric tasks) |
| dimension | v3/v4 only: 2048, 1536 (v4 only), 1024 (default), 768, 512, 256, 128, 64 |
| output_type | v3/v4 only: `"dense"` (default) / `"sparse"` / `"dense&sparse"` |
| instruct | Custom task instruction to guide the model in understanding query intent |

`MultiModalEmbedding.call(model, input, api_key=None, workspace=None, dimension=None, output_type=None, fps=None, instruct=None, enable_fusion=None, res_level=None, max_video_frames=None, **kwargs)`

- input is a List; elements use `MultiModalEmbeddingItemText(text, factor)` / `...ItemImage(image, factor)` / `...ItemAudio(audio, factor)` (factor: float, required), or equivalent dicts.
- dimension: vector dimension (supported values vary by model); output_type: currently only `"dense"`.
- fps: video frame-sampling ratio in [0,1], default 1.0; instruct: task instruction.
- enable_fusion: qwen3-vl-embedding only; when True, all content is fused into a single vector.
- res_level (0/1/2/3) and max_video_frames (<=64): snapshot-type models only.
- kwargs supports `auto_truncation`: automatically truncates audio >15s / text >70 words; default False (over-length input raises an error).
- Local files are automatically uploaded via OSS (header `X-DashScope-OssResourceResolve: enable`); empty input raises `InputRequired`; empty model raises `ModelRequired`.

**BatchTextEmbedding** (BaseAsyncApi): `call(model, url, ...)` submits and waits; `async_call(model, url, ...)` only creates the task; `fetch(task)` checks status; `wait(task)` waits for completion; `cancel(task)` cancels only PENDING tasks; `list(...)` lists tasks. url is the address of a file with line-by-line texts and is required (empty raises InputRequired); kwargs supports text_type (query/document, default document).

`TextReRank.call(model, query, documents, return_documents=None, top_n=None, api_key=None, instruct=None, **kwargs) -> ReRankResponse`

- return_documents: default False (results do not include original text); top_n: returns all documents by default.
- instruct: ranking instruction; English recommended. Empty query/documents raises InputRequired; empty model raises ModelRequired.
- AioTextReRank.call is the coroutine version, with an additional workspace parameter.

## Input / Output

Synchronous interfaces return `DashScopeAPIResponse`: status_code (200 = success) / request_id / code / message / output / usage.
- Embedding: `output["embeddings"]` is `[{"text_index": i, "embedding": [...]}]` (each multimodal item contains `embedding`); `usage` reports token usage (e.g. `input_tokens`).
- Batch: `output` = `{task_id, task_status, url}`; after SUCCEEDED, url points to the result file; `usage.total_tokens`; task_status in PENDING / RUNNING / SUCCEEDED / FAILED / CANCELED.
- ReRank: input `input={"query": str, "documents": List[str]}`; `output.results` is `[{index, relevance_score, document}]`, where index is the position in documents; document is returned only when return_documents=True; `usage.total_tokens`.

## Minimal Examples

```python
import dashscope
from dashscope import TextEmbedding

resp = TextEmbedding.call(
    model=TextEmbedding.Models.text_embedding_v3,
    input=["The wind is swift, the sky is high, the gibbons wail", "The islets are clear, the sand is white, the birds wheel"],
    text_type="document", dimension=1024,
)
if resp.status_code == 200:  # HTTPStatus.OK
    for e in resp.output["embeddings"]:
        print(e["text_index"], e["embedding"][:3])
```

```python
from dashscope import TextReRank

resp = TextReRank.call(
    model=TextReRank.Models.gte_rerank,
    query="What is a text embedding?",
    documents=["Text embeddings map text into a dense vector space", "The weather is nice today", "Embeddings are often used for semantic retrieval"],
    top_n=2, return_documents=True,
)
if resp.status_code == 200:
    for r in resp.output.results:
        print(r["index"], r["relevance_score"], r["document"])
```

## RAG Integration Guide

- Self-built pipeline: use TextEmbedding with `text_type="document"` to vectorize the corpus into a vector store; query with `text_type="query"`; after coarse ranking by the vector store, use TextReRank for fine ranking, then feed top_n results to the generation model.
- Managed solution: create a knowledge base in the Bailian console and bind it to an application, then call `dashscope.Application.call(app_id=...)` directly; vectors and retrieval are managed by the platform. See the Application-related documentation.

## Common Error Codes

HTTP errors do not raise exceptions: check `resp.status_code != 200`, then read `resp.code` / `resp.message`.

| Error Code | HTTP Status | Meaning | Handling |
| --- | --- | --- | --- |
| InvalidParameter | 400 | Missing/invalid parameter (e.g. empty documents, unsupported dimension value) | Verify per message; do not pass dimension/output_type for v1/v2 |
| InvalidApiKey | 401 | Invalid API Key | Check api_key or DASHSCOPE_API_KEY |
| AccessDenied | 403 | Model not activated or no permission | Activate the model service; check workspace |
| Model.NotExist / InvalidModel | 404 | Wrong model name | Verify model name (distinguish -v4 from -async-v2) |
| Throttling | 429 | Rate limit hit / quota exhausted | Reduce concurrency, retry with backoff, request quota increase |
| InternalError | 500 | Server internal error | Record request_id and retry, or file a ticket |
| ServiceUnavailable | 503 | Service temporarily unavailable | Retry later (SDK treats 503/504 as retryable) |

SDK local pre-validation (raises directly without sending a request): `InputRequired` (empty query/documents for rerank, empty url for batch, empty input for multimodal), `ModelRequired` (empty model).

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
