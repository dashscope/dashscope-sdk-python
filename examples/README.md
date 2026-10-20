# DashScope SDK Examples — Production Best Practices for Model Studio

> **English** | [中文](README_zh.md)

This is not an "API call reference". In the age of AI-assisted development, anyone can generate a one-line `call(model="xxx")` in seconds — the value of this directory is that it **writes those few hundred lines of "grubby code" needed to push a Model Studio application to production, correctly, in advance**: every advanced file is a miniature production-grade harness blueprint with rigid engineering defenses, targeting a specific development or business scenario. Everything is built on the 2.0 resource-style client `dashscope.Client()` (consistent with `dashscope.v2` inside the package); legacy 1.x examples are preserved intact under `legacy_1x/`.

A qualified best-practice case must demonstrate three dimensions at once:

1. **Solving real-world business/engineering pain points** (enterprise knowledge-base Q&A, high-concurrency realtime meeting transcription, online intelligent customer service, and so on).
2. **Cloud LLM × local SDK engineering synergy** (local token-watermark trimming, graceful streaming incremental merge, transcript assembly across reconnects).
3. **Production-grade rigid defense and resilience governance** (graceful 429/500 handling, timeout backoff and retry, multi-region circuit breaking and disaster recovery, state-machine management, per-room fault isolation).

## Flagship cases (read these three first)

| Case | In one line | Runs offline |
| --- | --- | --- |
| [`04_real_world_scenarios/production_ready_rag_harness.py`](./04_real_world_scenarios/production_ready_rag_harness.py) | Industrial-grade RAG: streaming semantic retrieval + local token watermarks + streaming backoff retry (the SDK does not retry streams) + multi-region circuit breaking and failover + graceful degradation | ✅ |
| [`04_real_world_scenarios/realtime_audio_meeting_harness.py`](./04_real_world_scenarios/realtime_audio_meeting_harness.py) | High-concurrency realtime meeting transcription: one dedicated websocket per room (realtime cannot be multiplexed) + meeting state machine + reconnects + transcript merge across reconnects + per-room isolation and degradation | ✅ |
| [`04_real_world_scenarios/customer_service_assistant.py`](./04_real_world_scenarios/customer_service_assistant.py) | Online intelligent customer service: guardrailed persona + business tools + sliding-window multi-turn memory (with orphan tool-message trimming) + managed tool loop | Needs a key |

> All three flagship cases embed source-level insights into *why* the code is written the way it is (e.g. **the SDK does not auto-retry streaming requests**, **the realtime protocol cannot be multiplexed**); see [`04_real_world_scenarios/README.md`](./04_real_world_scenarios/README.md).

## Directory layout

```
examples/
├── 01_atomic_capabilities/   Atomic capability quickref (minimal runnable; also the authoritative wire-shape seeds for AI)
├── 02_advanced_engineering/  Engineering patterns: local mock debugging, semantic cache, circuit breaker + multi-region DR
├── 03_agent_recipes/         Agent recipes: tool-call loop, RAG, MCP server
├── 04_real_world_scenarios/  Flagship best practices: stitch the three tiers above into end-to-end production harnesses
└── legacy_1x/                1.x module-level API examples (dashscope.Generation etc.), for reference only
```

What the tiers mean (an advancement path from "quickref" to "production blueprint"):

- **01 Atomic capabilities (quickref seeds)**: one file per capability, minimal but runnable. Deliberately kept simple — it is the authoritative seed for looking up wire shapes, and for feeding Tongyi Lingma / Cursor so they generate correct code. Each file ends with a "production springboard" footer pointing at the 02/03/04 cases that harden it.
- **02 Advanced engineering**: wraps the SDK in the patterns real production cares about (saving money, resisting pressure, offline debugging).
- **03 Agent recipes**: multi-step, composable end-to-end agent flows.
- **04 Real-world scenarios**: not "how to call some capability", but a whole real business built end to end into a production-grade harness (with guardrails, context budgets, fault tolerance, disaster recovery, state machines and other production tradeoffs); each case ships its own `README.md` explaining "why it is written this way".
- **legacy_1x**: the old 1.x module-level style, subdirectory layout kept exactly as-is, for reference only; new projects should prefer the four tiers above.

## Getting ready

1. Install the SDK:

   ```bash
   pip install -U dashscope
   ```

2. Configure your Model Studio API key ([how to get one](https://help.aliyun.com/zh/model-studio/get-api-key)):

   ```bash
   export DASHSCOPE_API_KEY="sk-xxx"
   ```

3. Run any example file directly (from the repository root):

   ```bash
   python3 examples/01_atomic_capabilities/text_generation/chat.py
   ```

> **Runs offline (no API key needed)**: `02_advanced_engineering/local_mock_debugging.py`, `02_advanced_engineering/circuit_breaker_retry.py`, `03_agent_recipes/mcp_server_demo/client_demo.py`, `04_real_world_scenarios/production_ready_rag_harness.py`, `04_real_world_scenarios/realtime_audio_meeting_harness.py` and `04_real_world_scenarios/ticket_triage_harness.py` use mock transports / scripted seams / subprocesses — fully offline and deterministic, ideal for a first hands-on look at the call shapes.

## Common environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | Model Studio API key (required for networked examples) | none |
| `DASHSCOPE_MODEL` | Overrides the chat/generation model (chat, cache, RAG, tool loop, etc.) | per-file default (mostly `qwen-plus`) |
| `DASHSCOPE_EMBED_MODEL` | Overrides the embedding model (`client_side_cache.py`, `rag_expert_system.py`) | `text-embedding-v4` |
| `DASHSCOPE_RERANK_MODEL` | Overrides the rerank model (`rag_expert_system.py`) | `gte-rerank-v2` |
| `DASHSCOPE_WORKSPACE` | AgentStudio workspace | `trial` |
| `DASHSCOPE_APP_ID` | Model Studio application ID (application.py) | none |
| `DASHSCOPE_FILE_ID` | File ID (assistant_api_file.py) | none |
| `DASHSCOPE_DEMO_AUDIO` | Local audio path (tingwu_realtime.py, tingwu.py) | `tests/data/asr_example.wav` |
| `DASHSCOPE_TINGWU_APP_ID` | TingWu application ID (tingwu_realtime.py, tingwu.py) | none |
| `DASHSCOPE_TTS_VOICE` | TTS voice (audio_tts.py) | `zhichu` |
| `DASHSCOPE_TRAINING_FILE_ID` | Fine-tuning training file ID (fine_tunes.py) | none |
| `DASHSCOPE_DEMO_IMAGE` | Local image path (cv_models.py) | `~/Downloads/cat.png` |
| `DASHSCOPE_DEMO_VIDEO_URL` | Video URL (multimodal_conversation/video.py) | sample video |
| `RAG_HARNESS_LIVE` | Set to `1` for `production_ready_rag_harness.py` to hit the real gateway (otherwise offline mock) | unset (offline) |
| `MEETING_HARNESS_LIVE` | Set to `1` for `realtime_audio_meeting_harness.py` to hit the real gateway (otherwise offline mock) | unset (offline) |
| `TRIAGE_HARNESS_LIVE` | Set to `1` for `ticket_triage_harness.py` to hit the real gateway (otherwise scripted decision backend) | unset (offline) |
| `DASHSCOPE_TINGWU_MODEL` | TingWu realtime model (`realtime_audio_meeting_harness.py`) | `tingwu-turbo` |
| `MEETING_MAX_CONCURRENT` | Realtime meeting concurrency cap (i.e. the active-socket cap) | `4` |

## 01_atomic_capabilities — Atomic capabilities

Uniform style: `client = dashscope.Client()` (reads the key from the `DASHSCOPE_API_KEY` environment variable by default).

| Path | Contents |
| --- | --- |
| `text_generation/chat.py` | Text generation: one-line `dashscope.chat()`, non-streaming, streaming (text_stream) |
| `text_generation/tokenization.py` | Token counting |
| `multimodal_vision/multimodal_chat.py` | Multimodal chat (qwen-vl image understanding) |
| `audio/` | Speech samples: ASR/TTS/realtime/voice CLIs (see `audio/README.md`; `direct_call/` holds the no-`with` variants) |
| `audio/tingwu.py` | TingWu offline tasks and realtime sessions |
| `image_video_synthesis/images.py` | Image generation: one-call generate-and-wait + the manual jobs flow |
| `image_video_synthesis/videos.py` | Video generation: one-call generate-and-wait + the manual jobs flow |
| `embeddings/embeddings.py` | Text embedding (with batch-task entry notes) |
| `rerank/rerank.py` | Document reranking |
| `decision/decision.py` | Decision model (System One): typed choice / noul / score questions over a business state in one call |
| `files/files.py` | File management: upload / list / get / delete |
| `applications/apps.py` | Model Studio application invocation (non-streaming + streaming) |
| `fine_tuning/fine_tunes.py` | Fine-tune jobs: list / get, create / wait / delete (uncomment to run) |
| `fine_tuning/reinforcement_learning.py` | Agentic RL: list / get jobs; register functions -> upload data -> create / wait / delete (uncomment to run) |
| `models/models.py` | Model list and get |

## 02_advanced_engineering — Advanced engineering

| Path | Contents | Network |
| --- | --- | --- |
| `local_mock_debugging.py` | Injects `httpx.MockTransport` to run the success path and the error paths (400 / throttling / content safety) offline at zero cost; swap `_mock_client()` for `dashscope.Client()` to hit the real gateway | No |
| `client_side_cache.py` | Client-side semantic cache: embeddings + cosine similarity catch repeat / near-repeat questions; a hit returns immediately without calling the model, saving tokens directly | Yes (only on cache miss) |
| `circuit_breaker_retry.py` | Circuit breaker + multi-region disaster recovery: on top of the SDK's built-in Retry-After backoff, per-region CLOSED/OPEN/HALF_OPEN breaking with automatic failover; the demo uses mock transports (primary region 503, secondary 200) and runs offline | No |

## 03_agent_recipes — Agent recipes

| Path | Contents | Network |
| --- | --- | --- |
| `tool_call_loop.py` | Managed multi-turn tool-call state machine: declare `tools` schemas, execute locally when the model requests a call, feed the `role="tool"` result back, and loop until a final answer (with a `MAX_TURNS` backstop) | Yes |
| `rag_expert_system.py` | Standard RAG: corpus embedding -> cosine top-k recall -> `client.rerank` refinement -> an "answer only from the sources" chat prompt -> an answer with citations | Yes |
| `mcp_server_demo/` | Dependency-free MCP (Model Context Protocol) stdio server + client demo: `server.py` speaks JSON-RPC 2.0; `client_demo.py` spawns the subprocess and runs initialize -> tools/list -> tools/call. See its `README.md` | No |
| `agentstudio.py` | AgentStudio end-to-end quickstart via `client.agentstudio`: environment -> agent -> session -> streamed reply (plus an opt-in memory-store demo); AgentStudio is shared by the 1.x/2.0 generations and mounts on the 2.0 client | Yes |
| `agentstudio_deployments.py` | Managed Agent deployments: create a cron-scheduled deployment, trigger a run, list runs | Yes |

## 04_real_world_scenarios — Real-world scenarios

Composes `01`–`03` into complete, directly comparable business cases with production tradeoffs such as guardrails, context budgets and fault tolerance; every case ships its own `README.md` covering the design tradeoffs and how to adapt it.

| Path | Contents | Network |
| --- | --- | --- |
| `customer_service_assistant.py` | E-commerce online customer service: guardrailed persona + business tools (orders / logistics / refunds / tickets) + sliding-window multi-turn memory (with orphan tool-message trimming) + managed tool loop; the demo runs a three-turn dialogue that depends on context | Yes |
| `production_ready_rag_harness.py` | Industrial-grade RAG engine: streaming semantic retrieval + local token watermarks (measured segment by segment via tokenization) + streaming backoff retry (the SDK does not retry streams) + multi-region circuit breaking and failover + graceful degradation; runs offline with mocks by default, demonstrating scenarios A/B/C/D | No |
| `realtime_audio_meeting_harness.py` | High-concurrency realtime meeting transcription base: one dedicated websocket per meeting (realtime cannot be multiplexed) + meeting state machine + reconnect backoff + local transcript merge across reconnects + per-room fault isolation and degradation; runs offline with mocks by default, with 4 concurrent rooms demonstrating reconnect and degradation | No |
| `ticket_triage_harness.py` | Decision-model ticket triage: one decision round per ticket (choice / noul / score in a single forward pass) + local threshold policy (confidence floor, escalation cut, severity bands) + fail-closed degradation to the human queue + per-ticket isolation and an exact input-token cost ledger; runs offline with a scripted decision backend by default | No |

See [`04_real_world_scenarios/README.md`](./04_real_world_scenarios/README.md).

## legacy_1x — 1.x module-level API (reference only)

The following use the old 1.x style (`dashscope.Generation`, `dashscope.MultiModalConversation`, etc.). The internal subdirectory layout is kept exactly as-is and the content is untouched. New projects should use the 2.0 tiered examples above.

| Path | Contents |
| --- | --- |
| `generation/` | Generation text generation: basics, deep thinking, tool calls, deep research, custom session |
| `aio_generation/` | Async versions of the above |
| `multimodal_conversation/` | MultiModalConversation multimodal chat: image, video, tool calls, OCR, ASR, deep thinking |
| `aio_multimodal_conversation/` | Async versions of the above |
| `application.py` | Model Studio application (Application) invocation |
| `assistant_api*.py` | Assistant agents: create / update, file attachment, end-to-end dialogue |
| `image_generation.py` / `image_synthesis.py` / `aio_image_synthesis.py` | Image generation (wanx / wan2.x) |
| `video_synthesis.py` / `aio_video_synthesis.py` | Video generation (wan series) |
| `multimodal_embedding.py` | Multimodal embedding |
| `text_rerank.py` | Document reranking |
| `qwen_tts.py` / `qwen_asr.py` | Qwen speech synthesis and recognition |
| `tingwu_realtime.py` / `tingwu_usages.py` | TingWu realtime and offline tasks |
| `cv_models.py` | CV models (portrait style repainting, with local upload) |
| `set_region_example.py` | Multi-region (region) configuration |
