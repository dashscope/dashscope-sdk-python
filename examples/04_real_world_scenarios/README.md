# 04 Real-world Scenarios

> **English** | [中文](README_zh.md)

This tier is not "how to call some capability" — it is **building a real business end to end**: composing the atomic capabilities of `01`, the engineering wrappers of `02` and the agent recipes of `03` into complete, directly comparable cases. Every case carries the tradeoffs production actually brings (guardrails, context budgets, fault tolerance, observability), plus a "why it is written this way" walkthrough.

| Case | Scenario | Network |
| --- | --- | --- |
| [`customer_service_assistant.py`](./customer_service_assistant.py) | E-commerce online customer service: guardrailed persona + business tools + multi-turn memory + managed tool loop | Yes |
| [`production_ready_rag_harness.py`](./production_ready_rag_harness.py) | Industrial-grade RAG Q&A engine: streaming semantic retrieval + local token watermarks + streaming backoff retry + multi-region circuit breaking and failover + graceful degradation | No (offline mock by default; `RAG_HARNESS_LIVE=1` hits the real gateway) |
| [`realtime_audio_meeting_harness.py`](./realtime_audio_meeting_harness.py) | High-concurrency realtime meeting transcription base: one dedicated websocket per meeting + meeting state machine + reconnect backoff + local transcript merge across reconnects + per-room fault isolation and degradation | No (offline mock by default; `MEETING_HARNESS_LIVE=1` hits the real gateway) |
| [`ticket_triage_harness.py`](./ticket_triage_harness.py) | Decision-model ticket triage: one decision round per ticket (choice / noul / score in a single forward pass) + local threshold policy + fail-closed degradation to the human queue + per-ticket isolation and an exact input-token cost ledger | No (scripted decision backend by default; `TRIAGE_HARNESS_LIVE=1` hits the real gateway) |

---

## Case: e-commerce online customer-service assistant

`customer_service_assistant.py` demonstrates an after-sales service bot for a "sample mall". Users can ask about orders, track shipments, ask whether a refund is possible, and request a human agent in natural language; the bot calls business tools to fetch **real data** before answering.

### The real problems it solves

| Production concern | What the example does |
| --- | --- |
| **Guardrails** | The system prompt scopes answers to after-sales topics, requires tool calls for order facts, forbids fabrication, and defines when to escalate to a human |
| **Business tool integration** | `query_order` / `track_shipment` / `check_refund_eligible` / `create_ticket`, all over one in-memory "order database" — swap in your own services |
| **Context budget** | `ConversationMemory` uses a sliding window: the system message is pinned and only the last `MAX_HISTORY_TURNS` turns are kept, so long dialogues never blow up the context |
| **Wire integrity** | Truncating from the head can leave "orphan" `role="tool"` messages (whose assistant tool-call turn was dropped), which the API rejects; `_trim` discards leading tool messages to keep the request valid |
| **Runaway backstop** | `MAX_TURNS` stops the model from calling tools forever; tool execution uses defensive JSON parsing, returning structured errors for unknown tools / bad arguments instead of raising |

### Architecture at a glance

```text
user input
  │
  ▼
ConversationMemory ──(system + recent turns)──► client.chat.completions.create(tools=TOOLS)
  ▲                                                      │
  │                                   ┌──────────────────┴──────────────────┐
  │                                   │ tool_calls present?               │
  │                            yes ◄──┤                                   ├──► no
  │                                   └───────────────────────────────────┘
  │ execute tools locally → feed role="tool" results back
  └──────────────────────────────────────────────────────────────────────┐
                                                         final answer ◄──┘
```

### Why the tool loop is non-streaming

A tool-call turn must **read the complete `message.tool_calls`** before local execution, so streaming deltas are meaningless inside the loop; the loop uses non-streaming `create`. If you want the pure-chat typewriter effect, see `stream=True` + `text_stream` in [`01_atomic_capabilities/text_generation/chat.py`](../01_atomic_capabilities/text_generation/chat.py); when you need both streaming and merged `tool_calls`, use `stream.get_final_message()`.

### Run

```bash
export DASHSCOPE_API_KEY="sk-xxx"
python3 examples/04_real_world_scenarios/customer_service_assistant.py
```

The demo runs a three-turn dialogue whose second turn — "can this order be refunded now?" — never repeats the order id: it only resolves because `ConversationMemory` carries turn one's `SO2024001` into the context. That is exactly what the memory layer is for.

### Adapt it into your own customer service

1. **Swap the data source**: replace `ORDERS` and the four `*_tool` functions with real order / logistics / ticket service calls (keep returning JSON-serializable dicts).
2. **Change the persona and policies**: edit `SYSTEM_PROMPT` and business constants like `REFUND_WINDOW_DAYS`.
3. **Add tools**: write the function → add its schema to `TOOLS` → register it in `TOOL_IMPL`; the loop picks it up automatically.
4. **Plug in persistent memory**: `ConversationMemory` is currently an in-process list; in production swap in Redis / a database keyed by session id, keeping the interface (`add` / `extend` / `messages`) unchanged.

---

## Case: industrial-grade production-ready RAG harness

`production_ready_rag_harness.py` is not "how to string-concatenate a RAG call" — it stitches the four things an enterprise knowledge-base Q&A must get right for production into one runnable blueprint: **streaming semantic retrieval -> local token watermarks -> rigid resilience governance -> graceful degradation**. It runs offline by default (`httpx.MockTransport` serves canned responses and injects faults on demand), so you can **watch** retries, failover and watermark trimming actually happen without an API key.

### The four production dimensions

| Dimension | Approach | Why it matters |
| --- | --- | --- |
| **Streaming semantic retrieval** | `LocalVectorStore` embeds once + cosine top-k, then `client.rerank` cross-encoder refinement | Fast recall and precise ranking — the foundation of RAG quality |
| **Local token watermark** | `TokenWatermark` measures segment by segment via `client.tokenization.count` and sends only what fits | A fat context never hits the window limit — or the bill |
| **Rigid resilience** | Non-streaming calls rely on the SDK's built-in `max_retries` (Retry-After-aware backoff); **the SDK does not retry streaming calls**, so the harness ships its own `retry_with_backoff`; on top, `ResilientGateway` breaks per region and fails over across regions | 429/5xx / whole-region outages cannot take the app down |
| **Graceful degradation** | `answer()` wraps retrieval / rerank / generation in one guard; `RateLimitError` / `ContentFilterError` / all-regions-down all become clean degraded answers | Upper layers receive a result, not a stack trace |

### A real pit you will step into without reading the source

> **Streaming requests do not go through the SDK's auto-retry.** `open_stream` in `dashscope/v2/transport/http.py` checks the initial status once and returns (source comment: "only the initial status is checked here, no retries"), while the plain `request` runs the `max_retries` backoff loop. So a streaming RAG that does not handle 429/503 itself throws the exception straight at the user. The harness's `retry_with_backoff` exists precisely for this gap.

Likewise, a stream is a **one-shot iterator**: consuming `text_stream` and then calling `get_final_message()` yields an empty tail, while the usage counters live in the **last chunk**, not the first. `_consume_stream` therefore iterates exactly once, printing the deltas and folding in usage at the same time.

### Run

```bash
# Offline (default, no key needed, deterministic output, demos scenarios A/B/C/D in turn)
python3 examples/04_real_world_scenarios/production_ready_rag_harness.py

# Hit the real gateway (needs DASHSCOPE_API_KEY; runs scenario A only, no fault injection)
RAG_HARNESS_LIVE=1 python3 examples/04_real_world_scenarios/production_ready_rag_harness.py
```

The offline demo runs in order:

- **Scenario A** standard Q&A: recall -> watermark trimming (watch `dropped`) -> rerank -> streaming generation with citations.
- **Scenario B** one injected 429: the harness backs off exponentially and then succeeds (because streams are not retried by the SDK).
- **Scenario C** primary-region 503: mid-answer the breaker flips CLOSED -> OPEN; later steps skip the dead region outright and fail over to the secondary.
- **Scenario D** every region down: no exception — a degraded answer is returned.

### Adapt it into your own knowledge base

1. **Swap the vector store**: `LocalVectorStore` is in-memory cosine; in production switch to Milvus / AnalyticDB / pgvector, keeping the `index` / `search` interface.
2. **Hit the real gateway**: drop `MockGateway`; `build_engine` already has the `RAG_HARNESS_LIVE=1` branch (single region); true multi-region disaster recovery needs a key + workspace per region (see `legacy_1x/set_region_example.py`).
3. **Tune the watermark and backoff**: derive `CONTEXT_TOKEN_BUDGET` from the model's max context minus instructions and the reserved answer length; for `GEN_RETRIES` / `base_delay`, production suggests `base_delay≈0.5` while honoring the server's Retry-After.
4. **Batch the counting**: the demo counts segment by segment for observability; high-frequency paths should batch counting or estimate with a local tokenizer.

---

## Case: high-concurrency realtime meeting transcription base

`realtime_audio_meeting_harness.py` rebuilds the legacy TingWu realtime script (`legacy_1x/tingwu_realtime.py`, a test-flavored style of callbacks + `time.sleep` polling) into a **meeting-platform-grade concurrency supervisor**: transcribe N meeting rooms in realtime at the same time, each room with its own state machine and transcript, and no single room's disconnect or moderation block can drag the others down. It also runs offline by default — a scripted duplex socket is injected through TingWu realtime's `connector=` (the same seam used by `tests/v2/test_tingwu.py`), so without a microphone or a key you can watch concurrency, reconnects and degradation actually happen.

### The four production dimensions

| Dimension | Approach | Why it matters |
| --- | --- | --- |
| **Protocol-respecting concurrency** | One meeting = one dedicated websocket + one worker; `ThreadPoolExecutor(max_workers=MAX_CONCURRENT)` does both concurrency and admission | See the pit below: realtime cannot be multiplexed, so the pool size is the cap on active sockets |
| **Local meeting state machine** | Each room walks `PENDING→CONNECTING→LISTENING→TRANSCRIBING→COMPLETED`, with `RECONNECTING`/`DEGRADED` as managed side branches | Hundreds of rooms become observable and reportable through one shared state vocabulary |
| **Cloud recognition / local assembly** | `TranscriptAssembler` merges incremental sentences by `sentenceId` (partials overwritten by finals) | TingWu re-emits the same `sentenceId` incrementally; only local merging yields the full transcript — and lets transcription resume across reconnects |
| **Rigid resilience + isolation** | A drop (`APIConnectionError`) triggers exponential-backoff reconnect; a business `task-failed` degrades that room instead of blindly reconnecting; exceptions stay sealed inside each room's worker | Transport faults deserve reconnects and business faults deserve degradation — misclassify and you do wasted work; isolation keeps one room's failure from spreading |

### A real pit you will step into without reading the source

> **The realtime family cannot be multiplexed.** The header comment of `dashscope/v2/transport/ws_connection.py` states it: the realtime protocol has no task id — one socket is one session, closed when the session ends — so it always opens a dedicated connection per session. (The multiplexable `WsConnection` only serves the JSON-framed asr/tts/translation tasks, and binary uplink is single-owner.) Therefore the correct shape of "high-concurrency realtime audio" is **one dedicated connection per meeting plus a supervising thread pool**, not stuffing multiple audio streams into one socket — the protocol simply cannot do that. The harness's `MeetingSupervisor` uses a bounded thread pool to tie "concurrency" to "active sockets / blast radius" as admission control.

Likewise, when a socket drops mid-stream, `WsSession.recv` raises `APIConnectionError` ("closed before task-finished"), while a `task-failed` event is mapped through the error catalog into a business exception with `status_code=200` (e.g. `ContentFilterError`). `_is_reconnectable` distinguishes exactly on this: the former reconnects, the latter degrades — the offline demo's `room-gamma` (after a drop-and-reconnect, the half sentence `今天我们讨论` is completed into `今天我们讨论第三季度的路线图。`) and `room-delta` (a content-moderation `task-failed` that degrades immediately) demonstrate these two paths respectively.

### Run

```bash
# Offline (default, no key needed, deterministic output, 4 rooms concurrent + reconnect + single-room degradation)
python3 examples/04_real_world_scenarios/realtime_audio_meeting_harness.py

# Hit the real gateway (needs DASHSCOPE_API_KEY + DASHSCOPE_TINGWU_APP_ID, runs one real-audio room)
MEETING_HARNESS_LIVE=1 python3 examples/04_real_world_scenarios/realtime_audio_meeting_harness.py
```

The offline demo runs 4 rooms concurrently, reports `completed` / `degraded` separately, and shows each room's attempts / reconnects / segments plus a transcript preview — `room-delta` is degraded by a moderation block while the other 3 complete, visibly proving "one room's failure does not spread".

### Adapt it into your own meeting platform

1. **Swap the audio source**: `_read_pcm` currently reads local 16 kHz PCM/WAV; in production feed `_pump_audio` frames from a microphone / RTC downlink / message-queue byte stream, keeping `pace=True` for realtime pacing.
2. **True speak-while-transcribing**: the demo is "send all audio → stop → read events" (consistent with `01_atomic_capabilities/audio_speech/tingwu.py`, for offline determinism); for live-caption style, drain `realtime.events()` on a reader thread concurrently while sending — the socket itself is full-duplex.
3. **Tune concurrency and admission**: `MAX_CONCURRENT` is the active-socket cap — set it by your machine's file-descriptor / thread budget; `RECONNECT_ATTEMPTS` and `BASE_DELAY` (production suggests ≈0.5s) control reconnect aggressiveness.
4. **Plug in persistence and observability**: `MeetingSession.transition` is the single exit for state; in production wire it to logging / metrics / meeting-record storage, persisting transcripts and states by `meeting_id`.

---

## Case: decision-model ticket triage harness

`ticket_triage_harness.py` turns the decision model (`decision-model-preview`) into the front door of a ticket system: every incoming ticket gets ONE decision round — which team owns it (`choice`), whether to page the on-call engineer (`noul`), how severe it is (`score`) — and a local policy engine turns the returned probability distributions into routing actions. It runs offline by default: the mock seam is the one-method decision backend, so `MockDecisionBackend` serves scripted answers (a vague ticket that falls below the confidence floor, plus an injected backend outage) while the policy, degradation and ledger logic under test is the production code, unchanged.

### The four production dimensions

| Dimension | Approach | Why it matters |
| --- | --- | --- |
| **One round, typed questions** | `client.decision.create` answers choice + noul + score about the same `state` in a single forward pass; no text is generated | Latency and cost stay flat per ticket — unlike "ask an LLM for JSON and parse it" |
| **Local threshold policy** | `TriagePolicy` owns the confidence floor, the P(yes) escalation cut and the severity bands | The model returns distributions, never decisions; thresholds are auditable business decisions, tunable without redeploying anything |
| **Fail-closed degradation** | Low routing confidence sends the ticket to the human queue; an exception OR an `is_error` envelope degrades it there with the error attached | No ticket is ever auto-routed on a guess — see the pit below |
| **Per-ticket isolation + token ledger** | A backend failure degrades that ticket alone; `usage.input_tokens` is booked per ticket into a running ledger | One bad ticket never stalls the batch, and the ledger is the exact bill — input tokens are the only quantity the decision model bills |

### A real pit you will step into without reading the source

> **DashScope can report a failure with HTTP 200.** `ResponseBase` in `dashscope/v2/types/common.py` carries `code` / `message` / `is_error` on every response, and with `raise_on_error=False` the SDK fills that envelope instead of raising. A triage loop that only wraps `try/except` will treat a failed call as a successful one with empty answers — and auto-route the ticket on nothing. The harness therefore checks `response.is_error` even in the default raising mode: belt and suspenders, because the day someone flips `raise_on_error=False` the guard must already be in place.

Likewise, **`score` answers are probability-weighted level indices, not integer classes** — `2.3` means "between level 2 and 3, leaning 2", and `legend` maps an index back to your level text. Cut lines like `P1_SCORE = 2.5` are therefore thresholds on a continuous axis, which is exactly what severity bands want.

And a routing note: the System One endpoint lives under the **compatible-mode path on every gateway** (derived by `compatible_mode_base_url` in `dashscope/v2/_config.py`), so changing the client's chat `protocol` does not move it.

### Run

```bash
# Offline (default, no key needed, deterministic: 5 tickets, one low-confidence, one injected outage)
python3 examples/04_real_world_scenarios/ticket_triage_harness.py

# Hit the real gateway (needs DASHSCOPE_API_KEY; the same harness, no fault injection)
TRIAGE_HARNESS_LIVE=1 python3 examples/04_real_world_scenarios/ticket_triage_harness.py
```

The offline demo triages 5 tickets: T-2001 / T-2002 route to their teams, T-2003 ("你们这个系统到底怎么回事") falls below the confidence floor and goes to human review, T-2004 pages the on-call engineer as P1, and T-2005 suffers an injected backend outage and degrades to the human queue — then the summary prints the state distribution and the exact input-token ledger.

### Adapt it into your own ticket system

1. **Swap the intake queue**: `TICKETS` is a stand-in list; stream real tickets in, keeping whatever `state` shape the model should judge (`ticket_id` / `content` here).
2. **Tune the policy, not the model**: `CONFIDENCE_FLOOR` / `ESCALATE_THRESHOLD` / `P1_SCORE` / `P2_SCORE` are business decisions — calibrate them from observed routing quality.
3. **Wire the exits**: verdicts are printed today; in production push `triaged` to the team queue, `escalated` to paging, and `human_review` / `degraded` to the manual triage inbox — the verdict object already carries everything a sink needs.
4. **Scale out**: decision rounds are independent per ticket, so `run` can be parallelized; keep a bounded pool and let the SDK's `max_retries` absorb 429/5xx inside each non-streaming call.
