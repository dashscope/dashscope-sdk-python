# DashScope SDK Examples — Production Best Practices for Model Studio

> **English** | [中文](README_zh.md)

This is not an "API call reference". In the age of AI-assisted development, anyone can generate a one-line `call(model="xxx")` in seconds — the value of this directory is that it **writes those few hundred lines of "grubby code" needed to push a Model Studio application to production, correctly, in advance**: every advanced file is a miniature production-grade harness blueprint with rigid engineering defenses, targeting a specific development or business scenario. Everything is built on the 2.0 resource-style client `dashscope.Client()` (consistent with `dashscope` inside the package); legacy 1.x examples are preserved intact under `legacy_1x/`.

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
├── 05_cli/                   The `dashscope` command line: output formats, exit codes, configuration and CI recipes
└── legacy_1x/                1.x module-level API examples (dashscope.Generation etc.), for reference only
```

What the tiers mean (an advancement path from "quickref" to "production blueprint"):

- **01 Atomic capabilities (quickref seeds)**: one file per capability, minimal but runnable. Deliberately kept simple — it is the authoritative seed for looking up wire shapes, and for feeding Tongyi Lingma / Cursor so they generate correct code. Each file ends with a "production springboard" footer pointing at the 02/03/04 cases that harden it.
- **02 Advanced engineering**: wraps the SDK in the patterns real production cares about (saving money, resisting pressure, offline debugging).
- **03 Agent recipes**: multi-step, composable end-to-end agent flows.
- **04 Real-world scenarios**: not "how to call some capability", but a whole real business built end to end into a production-grade harness (with guardrails, context budgets, fault tolerance, disaster recovery, state machines and other production tradeoffs); each case ships its own `README.md` explaining "why it is written this way".
- **05 Command line**: the same capabilities behind a shell/CI contract rather than a Python one — machine-readable output, layered exit codes, a fixed stderr error line and a documented configuration precedence; both files run offline.
- **legacy_1x**: the old 1.x module-level style, subdirectory layout kept exactly as-is, for reference only; new projects should prefer the five tiers above.

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
   python examples/01_atomic_capabilities/text_generation/chat.py
   ```

> **Runs offline (no API key needed)**: `02_advanced_engineering/local_mock_debugging.py`, `02_advanced_engineering/circuit_breaker_retry.py`, `03_agent_recipes/mcp_server_demo/client_demo.py`, `04_real_world_scenarios/production_ready_rag_harness.py`, `04_real_world_scenarios/realtime_audio_meeting_harness.py`, `04_real_world_scenarios/ticket_triage_harness.py`, `05_cli/cli_output_and_errors.py` and `05_cli/cli_ci_harness.py` use mock transports / a local mock gateway / scripted seams / subprocesses — fully offline and deterministic, ideal for a first hands-on look at the call shapes.

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
| `CLI_HARNESS_LIVE` | Set to `1` for `cli_ci_harness.py` to hit the real gateway, read-only scenarios only (otherwise a local mock gateway) | unset (offline) |
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

## 05_cli — Command line

The same capabilities behind a **shell** contract rather than a Python one: three output formats with a documented stdout/stderr split, layered exit codes (0 ok / 1 server / 2 usage / 3 auth / 4 not found), one fixed `Error [category]: message` line on stderr, and four configuration layers with a stated precedence. Both files run offline and deterministically — no key, no network.

| Path | Contents | Network |
| --- | --- | --- |
| `cli_output_and_errors.py` | The contract from the inside: one fake client driven in-process through `typer.testing.CliRunner`, printing stdout / stderr / exit code side by side for table vs json vs yaml, NDJSON streaming, the `@file` / stdin / `--messages` input shapes, and every failure tier including the destructive-command refusal | No |
| `cli_ci_harness.py` | The contract from the outside: a stdlib mock gateway on an ephemeral port driving the real CLI in real subprocesses — parse `-o json`, branch and retry on exit codes, consume an NDJSON stream line by line, isolate configuration with `DASHSCOPE_CONFIG`, and pass `--yes` where CI has no terminal | No |

Read the two examples in table order. The first shows *what* comes out of each channel; the second shows *how a pipeline should consume it*.

### Install and first call

The package registers a console script, so after `pip install -U dashscope` there is a `dashscope` command on your `PATH`:

```bash
dashscope --help
dashscope version
```

From a source checkout where the script is not installed, the module entry point is identical:

```bash
python -m dashscope.cli --help
python -m dashscope.cli version
```

> Existing 1.x command forms use this same `dashscope` entry point. Legacy-only names run against the frozen 1.x implementation; shared names accept their old option spellings through the 2.0 tree. See [`legacy_1x/`](./legacy_1x/) for the 1.x module-level API examples.

Set the key once, either way:

```bash
export DASHSCOPE_API_KEY="sk-xxx"
dashscope chat create -m qwen-plus --prompt "用一句话介绍杭州"
```

**Global options must come before the subcommand.** They belong to the app, not to `chat create`:

```bash
dashscope -k sk-xxx --output json chat create -m qwen-plus --prompt "hi"   # correct
dashscope chat create -m qwen-plus --prompt "hi" --output json             # exit 2, usage
```

That last one is not a quirk — it is the single most common way to get a confusing failure out of this CLI, and `cli_output_and_errors.py` demonstrates it explicitly.

### Command map

| Command | Purpose |
| --- | --- |
| `dashscope version` | Print the installed SDK version. Touches nothing, so it is the cheapest smoke test you have |
| `dashscope auth login` / `whoami` / `logout` | Save, verify, or remove the API key in `~/.dashscope/api_key`. Top-level `dashscope login` / `whoami` / `logout` are shortcuts |
| `dashscope chat create` | One chat completion: `--prompt` / `--messages`, `--stream`, `--tools`, sampling flags, `--parameters` for anything else |
| `dashscope ft create` | Start a fine-tune job; `--wait` blocks with a spinner until a terminal status, `--timeout` bounds the wait |
| `dashscope ft get` | Show one job |
| `dashscope ft wait` | Block on a job this process did not create — the second half of `create --wait`, for a pipeline that submits in one step and collects in the next. A failed or cancelled job exits 1 |
| `dashscope ft list` | Page through jobs: `--page`, `--size` |
| `dashscope ft stream` | Follow a job's server-side event feed until the server closes it |
| `dashscope ft logs` | Read a job's training logs: `--offset`, `--lines`. Table mode prints the log text verbatim — no table — so `> train.log` captures exactly what the trainer printed |
| `dashscope ft cancel` | Cancel a running job. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope ft delete` | Delete a job and its output model. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope files upload` | Upload a file for fine-tuning; returns the `file_id` that `ft create -t` consumes |
| `dashscope files list` | Page through uploaded files: `--page`, `--size` |
| `dashscope files get` | Show one file by id |
| `dashscope files delete` | Delete a file. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope models list` | Page through the model catalog: `--page`, `--size`, `--provider` |
| `dashscope models get` | Look up one model by name; exits 4 if it does not exist |
| `dashscope embeddings create` | Generate embeddings synchronously: `--input` (text or JSON array), `--text-type`, `--dimension` |
| `dashscope embeddings jobs create` | Submit an async batch embedding task over a file URL; `--wait` blocks until done |
| `dashscope embeddings jobs get` | Check the state of a batch task |
| `dashscope embeddings jobs wait` | Block on a batch task id from an earlier step instead of polling `jobs get` in a loop |
| `dashscope embeddings jobs list` | Page through batch tasks: `--page`, `--size`, `--status` |
| `dashscope embeddings jobs cancel` | Cancel a pending batch task. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope multimodal chat create` | `chat create` against the multimodal endpoint, for a conversation that has to hit it even without media parts |
| `dashscope multimodal embeddings create` | Embed text, image, audio and video into one vector space: repeatable `--text`/`--image`/`--audio`/`--video`, or `--input` JSON |
| `dashscope rerank create` | Rerank documents by relevance: `--query`, `--documents` (JSON array) or repeatable `--document` |
| `dashscope tokenization count` | Count tokens for a prompt or message list |
| `dashscope apps completions create` | Call a Bailian **application** by `--app-id`: `--prompt`/`--messages`, repeatable `--image`/`--file`, `--session-id`/`--memory-id`, `--rag-options`, `--biz-params`. `--stream` also asks for incremental frames, because the service otherwise resends the whole answer every time |
| `dashscope images generate` | Draw an image and wait for it; prints one URL per line so `$(...)` and `xargs wget` work |
| `dashscope images jobs create` | Submit an image task and return its id now; `--wait` blocks, `jobs get`/`jobs list` collect later |
| `dashscope images jobs get` / `list` / `wait` | Collect a task from an earlier step: read one back, page through them, or block until it settles and print its URLs the way `generate` does |
| `dashscope images jobs cancel` | Cancel a pending image task. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope videos generate` | Generate a video and wait for it; prints the video URL, so `$(...)` and `xargs wget` work. `--timeout` defaults to an hour |
| `dashscope videos jobs create` | Submit a video task and return its id now; `--wait` blocks, `jobs get`/`jobs list` collect later |
| `dashscope videos jobs get` / `list` / `wait` | Collect a task from an earlier step: read one back, page through them, or block until it settles and print the video URL the way `generate` does |
| `dashscope videos jobs cancel` | Cancel a pending video task. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope deployments create` | Provision capacity for a model: `--capacity`, plus `--deploy-spec`/`--billing-method` for MU deployments |
| `dashscope deployments get` / `list` | Inspect one deployment, or page through them |
| `dashscope deployments stream` | Follow a deployment's event feed until the server closes it |
| `dashscope deployments scale` | Resize a deployment. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope deployments delete` | Tear a deployment down. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope rl jobs create` | Run the whole agentic-RL workflow (register, upload, submit); `--set key=value` overrides one config key, `--wait` blocks |
| `dashscope rl jobs get` / `list` / `logs` | Inspect one RL job, page through them, or read its training logs |
| `dashscope rl jobs wait` | Block on an RL job id from an earlier step; a failed or cancelled job exits 1 |
| `dashscope rl jobs cancel` / `delete` | Stop or remove an RL job. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope rl functions register` | Upload rollout/reward code; returns the entity and instance ids |
| `dashscope rl functions test` | Run one registered instance against a sample `--input` payload |
| `dashscope rl functions upload-data` | Upload datasets; returns file ids for `rl jobs create` |
| `dashscope rl functions logs` | Read a function instance's logs; `--all` aggregates every page |
| `dashscope rl functions delete` | Delete a function instance. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope audio tts synthesize` | Speak `TEXT` and write the audio: `--out speech.mp3`, or stdout when `--out` is omitted so `dashscope audio tts synthesize hi > hi.mp3` works. `--voice`, `--format`, `--sample-rate`, `--volume`, `--rate`, `--pitch` |
| `dashscope audio tts http` | The same request over the non-streaming endpoint, printing the `audio_url` instead of the bytes. `-m` and `--voice` are **required** here — the streaming endpoint's cosyvoice defaults do not apply |
| `dashscope audio asr transcribe` | Transcribe one or more audio/video URLs and **wait** for the result; `--no-wait` submits and returns the task id |
| `dashscope audio asr get` / `wait` | Read a transcription task now, or block on it with a spinner until it settles |
| `dashscope audio asr recognize` | Transcribe a **local file** through the realtime recognizer; table mode prints sentences, `-o json` emits one raw event per line |
| `dashscope audio realtime run` | One turn over the duplex Omni socket: feed a whole `--audio` file (path, `@path` or piped stdin), print the answer text and the input transcript, write the spoken reply to `--out`. `-o json` is an NDJSON trace of every event in the turn |
| `dashscope audio vocabularies create` / `list` / `get` / `update` / `delete` | Hot-word tables, consumed by `asr recognize --vocabulary-id`. `--vocabulary` is a JSON **array** of entries. `delete` is **destructive** |
| `dashscope audio phrases create` / `list` / `get` / `update` / `delete` | Compiled phrase sets, consumed by `asr transcribe --phrase-id`. `--phrases` is a JSON **object** of phrase lists. `delete` is **destructive** |
| `dashscope audio voices create` / `list` / `get` / `update` / `delete` | Voices cloned from a sample recording at `--url`. `delete` is **destructive** |
| `dashscope audio voice-design create` / `list` / `get` / `delete` | Voices described in words (`--voice-prompt` + `--preview-text`) instead of audio — the same registry, id namespace and quota as `voices`. There is no `update` because the service has none. `delete` is **destructive** |
| `dashscope tingwu call` | Run one offline TingWu task: `-m`, plus a business-defined payload either as `--input @task.json` or through the `--file-url` / `--app-id` pair, which win over the JSON. The realtime websocket is SDK-only |
| `dashscope decision create` | Answer **typed** questions about a `--state` (free text, a dialogue, or JSON): `--questions` declares `choice` / `noul` / `score` per id, and one row comes back per answer with its confidence. The model generates no text |
| `dashscope agentstudio agents create` / `get` / `list` / `versions` / `update` / `archive` | Manage a **managed agent**: `--name`, `-m`, `--system-prompt`, and `--tools` / `--mcp-servers` / `--skills` as JSON arrays. `versions` lists the numbers that `get --version` and `update --version` ask for. `update` requires `--version` — a mismatch is a 409. `archive` is **destructive** |
| `dashscope agentstudio sessions create` / `get` / `list` / `update` / `archive` / `delete` | Open a session on an agent (`--agent`, `--environment-id`, `--vault-ids`, `--env`, `--mcp-configs`) and inspect it. `update` patches one and keeps every field you omit; its `--agent-config` is a session-level agent override (`system` / `tools` / `mcp_servers` / `skills`), **not** an agent id. `archive` and `delete` are **destructive** |
| `dashscope agentstudio sessions events send` | Post one turn: `--message` is wrapped into a user message for you, `--events` sends a raw JSON array as-is and wins, `--thread-id` picks the thread |
| `dashscope agentstudio sessions events list` | Read a session's persisted events back: `--type` (repeatable), `--order`, `--limit`, `--page` |
| `dashscope agentstudio sessions events stream` | Follow the run until it settles: table mode prints the answer text, `--deltas message` opts into incremental frames, `-o json` is one event per line |
| `dashscope config set` | Write one key to `~/.dashscope/config.toml` |
| `dashscope config get` | Read one key back (api_key masked unless `--reveal`) |
| `dashscope config list` | Every key, its effective value, and the layer that supplied it |

One `dashscope` entry point serves both generations. Legacy-only groups (`generation`, `application`, `image-synthesis`, `multimodal-conversation`, `speech-synthesis`, `transcription`, `video-synthesis`, `image-generation`, `multimodal-embedding`) and pre-Typer dotted spellings keep their original argv and run against the frozen 1.x implementation. Shared groups such as `ft`/`fine-tunes`, `files`, `deployments`, `models`, `rl`/`agentic-rl`, `auth`, and `oss` run through 2.0; the compatibility layer accepts the old option spellings and preserves old defaults where they differ. New 2.0 commands follow the SDK resource tree, so application invocation is `apps completions create`, not a shortened `apps call`. A bare `dashscope` with no subcommand opens the bundled `expert` agent interactively.

What is *not* here is a decision rather than a backlog, and the gaps all have the same shape: a long-lived bidirectional socket, or a long tail of CRUD that a shell adds nothing to. `audio realtime run` drives **one turn** of the duplex Omni socket and hangs up; a live microphone — which stays open across turns and expects audio at realtime pace — is `client.audio.realtime.connect()` in Python, and `tingwu realtime` is the same story. `agentstudio` covers the loop you actually script (agents, sessions, session events); its other nine top-level resources (deployments and their runs, environments, files, skills, vaults, memory stores, webhook endpoints, security) plus session threads and session resources stay SDK-only. Everything else the SDK exposes has a command.

### Output formats

`-o table` (default) is for a human; `-o json` and `-o yaml` are for a machine. The split is deliberate and it is enforced in three ways:

| Behaviour | table | json | yaml |
| --- | --- | --- | --- |
| Renderer | Rich, wrapped and coloured | `json.dumps(indent=2, ensure_ascii=False)` on stdout | `yaml.safe_dump(allow_unicode=True, sort_keys=False)` on stdout |
| Answer text | stdout | inside the document | inside the document |
| `reasoning_content`, `usage:`, pagination | **stderr only** | part of the document | part of the document |
| Streaming | incremental text on stdout | **NDJSON** | **NDJSON** (not a YAML document) |
| Transport `headers` | never | stripped, unless `-v` | stripped, unless `-v` |

Two consequences worth internalising:

1. In table mode `dashscope chat create ... > answer.txt` captures the answer and nothing else. Reasoning and token counts go to stderr, so a redirect never picks up decoration.
2. `--stream -o yaml` is a trap for the reader: it emits one **JSON** object per line, exactly like `--stream -o json`. Streaming output is line-oriented by nature, and a multi-document YAML stream would be hostile to `while read`.

Pipe recipes:

```bash
dashscope -o json chat create -m qwen-plus --prompt "hi" | jq -r '.choices[0].message.content'
dashscope -o json chat create -m qwen-plus --prompt "hi" | jq -r '.usage.total_tokens'
dashscope -o json ft list --size 50 | jq -r '.jobs[] | "\(.job_id)\t\(.status)"'
dashscope -o json chat create -m qwen-plus --prompt "hi" --stream \
  | jq -j '.choices[0].delta.content // empty'
```

Add `-v` / `--verbose` when you need the raw HTTP response headers (for a support ticket) and tracebacks on failure. Leave it off in a pipeline: `headers` is transport metadata, not part of the result.

### Exit codes and error format

The exit code is the contract. Do not grep the message.

| Exit | Meaning | Retryable? |
| --- | --- | --- |
| `0` | Success | — |
| `1` | `server`, `ratelimit`, `network` — the service or the link failed | Yes, with backoff. For `ratelimit`, honour `Retry-After` |
| `2` | `usage` — your command line, your input file, or a missing `--yes` | No. Fix the invocation |
| `3` | `auth` — credentials rejected | No. Fix the key |
| `4` | `notfound` — the id does not exist | No. Fix the id |

Every failure prints one line on stderr in a fixed shape, then optional dim context:

```text
Error [ratelimit]: Rate limit exceeded.
  request_id: 7574ee8f-38a3-4b1e-9280-11c33ab46e51
  solution: Reduce your request frequency.
```

`Error [category]: message` is stable; the two dim lines appear only when there is something to say. `request_id` is what a support ticket needs, and `solution` is the registry's own guidance — kept on a separate line so you can always tell which words came from the gateway.

How an HTTP status becomes a category (`cli/errors.py`, `_STATUS_CATEGORY`):

| Status | Category | Exit |
| --- | --- | --- |
| 400, 413 | `usage` | 2 |
| 401, 403 | `auth` | 3 |
| 404 | `notfound` | 4 |
| 429 | `ratelimit` | 1 |
| 500, 503, 504 | `server` | 1 |

SDK-side failures have no HTTP status, so they are categorised by their catalog code instead: `sdk.ConnectionError` and `sdk.Timeout` → `network` (1), `sdk.InvalidApiKey` → `auth` (3), `sdk.IoError` → `usage` (2, because "the file you named is not readable" is your mistake). Anything unattributable lands in `server`.

The retry wrapper in a shell:

```bash
attempt=1
until dashscope -o json chat create -m qwen-plus --prompt "hi" > out.json; do
  code=$?
  case $code in
    1) [ $attempt -ge 5 ] && exit 1
       sleep $(( attempt * 2 )); attempt=$(( attempt + 1 )) ;;
    *) exit $code ;;   # 2 / 3 / 4 will not fix themselves
  esac
done
```

`cli_ci_harness.py` scenario 2 is the same logic in Python, with a mock gateway that throttles the first attempt and serves the second so you can watch it recover.

### Configuration

Precedence, highest first:

```text
command-line flag  >  -c KEY=VALUE  >  environment variable  >  config file  >  built-in default
```

So `-k sk-a -c api_key=sk-b` uses `sk-a`: the dedicated flag is the more specific spelling and always wins.

The file is `~/.dashscope/config.toml`. Redirect it with `DASHSCOPE_CONFIG`, which is how you give one CI job its own configuration without touching anyone's home directory:

```bash
DASHSCOPE_CONFIG=/tmp/job-42/config.toml dashscope config set default_model qwen-plus
```

Top-level keys are SDK keys and are passed straight to `Client(...)`: `api_key`, `protocol`, `base_url`, `workspace`, `region`, `timeout`, `max_retries`, `http_max_connections`, `http_max_keepalive_connections`, `http_keepalive_expiry`. The `[cli]` table holds exactly two CLI-only defaults: `output` and `default_model`. Write them as `cli.output` / `cli.default_model`, or plainly — `config set output json` knows where they go.

```toml
api_key = "sk-xxx"
protocol = "native"

[cli]
output = "json"
default_model = "qwen-plus"
```

Three details that matter in practice:

- `config set api_key` **never echoes the value**, and the file is chmod `0600` the moment a secret lands in it. A terminal recording or a build log cannot capture what was just typed.
- `config get api_key` masks: first four characters, stars, last two (`sk-l*******mo`). `--reveal` prints it whole, for the one time you genuinely need to check.
- `config get` reads **the file alone**; `config list` reports the **effective** value plus the layer that won. They can disagree — that disagreement *is* the precedence lesson, and `cli_ci_harness.py` scenario 4 narrates it rather than letting it look like a bug.

`-c` is validated immediately, on every command, including ones that never touch the network:

```bash
dashscope -c defualt_model=qwen-max version
# Error [usage]: unknown configuration key 'defualt_model'; known keys: api_key, base_url, ...
# exit 2
```

That is deliberate. A typo in a pipeline should surface on the cheapest command there is, not three stages later as a mysteriously wrong model.

### Input conventions

Every text option accepts three forms:

| Form | Meaning |
| --- | --- |
| `--prompt "text"` | Literal value |
| `--prompt @path/to/file.txt` | Read the file (`~` expanded). The `@` form is what keeps a long payload or a tool schema out of your shell history and your `ps` output |
| `--prompt -` | Read stdin to EOF |

With neither flag nor `@file`, a **piped** stdin is picked up automatically, so this works:

```bash
cat question.txt | dashscope chat create -m qwen-plus
```

Source precedence for the conversation: `--messages` > `--prompt` (+ `--system`) > piped stdin. If none of the three yields anything, you get `exit 2` with `no input given: pass --prompt, --messages, or pipe text on stdin`.

`--messages` accepts three shapes and figures out which one you meant:

```bash
# a JSON array of message objects
dashscope chat create -m qwen-plus \
  --messages '[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"},{"role":"user","content":"继续"}]'

# JSONL -- one message object per line, easiest to build with a loop
dashscope chat create -m qwen-plus --messages @turns.jsonl

# an object with a messages key, i.e. a captured request body replayed as-is
dashscope chat create -m qwen-plus --messages @last_request.json
```

Bad JSON is reported with a position, not a shrug: `--messages: invalid JSON (line 1 column 23)`.

`--tools` and `--parameters` are JSON too and accept `@file`. `--tools` must be a **JSON array**; handing it an object is a `usage` error (exit 2). Local image and video paths inside a message are passed through to the SDK unchanged, which uploads them itself — you do not need to pre-upload.

### Recipes

Copy-pasteable. Everything marked 🔑 needs a real `DASHSCOPE_API_KEY`; the rest runs against nothing at all.

**One-shot answer into a variable** 🔑

```bash
answer=$(dashscope -o json chat create -m qwen-plus --prompt "用一句话介绍杭州" \
         | jq -r '.choices[0].message.content')
```

**Stream to the terminal as it arrives** 🔑

```bash
dashscope chat create -m qwen-plus --prompt "写一首关于春天的短诗" --stream
```

**A long prompt from a file, never from argv** 🔑

```bash
dashscope chat create -m qwen-plus --prompt @brief.md --system "你是资深编辑"
```

**Tools from a file** 🔑

```bash
dashscope chat create -m qwen-plus --prompt @question.txt --tools @tools.json -o json
```

**Multi-turn from JSONL built in the loop** 🔑

```bash
: > turns.jsonl
while read -r line; do
  printf '{"role":"user","content":"%s"}\n' "$line" >> turns.jsonl
done < questions.txt
dashscope chat create -m qwen-plus --messages @turns.jsonl
```

**Retry only what can succeed later** 🔑 — see the shell wrapper in the exit-code section above.

**Fine-tune and block until it settles** 🔑

```bash
dashscope ft create -m qwen-plus -t "$FILE_ID" --n-epochs 2 --wait --timeout 1800
```

`ft create -m` **does not** fall back to `default_model`. That is intentional: a chat default is not a fine-tune base, and silently fine-tuning the wrong model is worse than an error.

**Follow a job's events** 🔑

```bash
dashscope ft stream "$JOB_ID"
```

**Capture a job's training log** 🔑

```bash
dashscope ft logs "$JOB_ID" --offset 1 --lines 500 > train.log
```

**Collect a generated video, then download it** 🔑

```bash
url=$(dashscope videos generate -m wan2.2-t2v-plus --prompt "海浪拍打礁石" --timeout 1800)
curl -o clip.mp4 "$url"
```

Submit-only, for a batch you collect later:

```bash
dashscope -o json videos jobs create -m wan2.2-t2v-plus --prompt @brief.md \
  | jq -r '.task_id' >> task_ids.txt
while read -r id; do dashscope videos jobs get "$id"; done < task_ids.txt
```

**Submit in one CI step, block in the next** 🔑

```bash
# step 1 -- submit and keep the id
dashscope -o json images jobs create -m wan2.2-t2i-flash --prompt @brief.md \
  | jq -r '.task_id' > task_id.txt

# step 2 -- possibly another runner, possibly an hour later
dashscope images jobs wait "$(cat task_id.txt)" --timeout 1800
```

`wait` takes an **id**, not a job object, so nothing has to survive between the two steps but that string. It exits 1 on a task that failed or was cancelled, and 2 on a `--timeout` it could never honour — `0` and `nan` included, because `nan` is false against every comparison and the poll would spin forever. The same command exists for `ft wait`, `videos jobs wait`, `embeddings jobs wait`, `rl jobs wait` and `audio asr wait`.

**Speech to a file, or straight into a pipe** 🔑

```bash
dashscope audio tts synthesize "你好，世界" --out hello.mp3
dashscope audio tts synthesize @script.txt --voice longxiaochun | ffplay -i - -autoexit -nodisp
```

Omit `--out` and the raw bytes go to stdout. With `--out`, the file gets the audio and a `wrote N bytes to ...` note goes to stderr — so neither form pollutes a redirect.

**Transcribe a recording and block for the result** 🔑

```bash
dashscope audio asr transcribe "https://example.com/meeting.mp3" --timeout 1800
```

**One turn over the realtime socket** 🔑

```bash
dashscope audio realtime run -m qwen3.8-omni-flash-realtime \
  --audio question.pcm --out reply.wav
```

The answer text goes to stdout, the input transcript and a `wrote N bytes` note go to stderr, the spoken reply goes to `--out`. Omit `--out` and the reply audio is **dropped** — the command will not spray binary into your terminal just because you forgot a flag. `-o json` turns the whole thing into an NDJSON trace of every event in the turn, which is what you want when the turn does not behave.

**Hot words, then recognize with them** 🔑

```bash
vid=$(dashscope -o json audio vocabularies create \
        --prefix demo --target-model paraformer-realtime-v2 \
        --vocabulary @hotwords.json \
      | jq -r '.vocabulary_id')
dashscope audio asr recognize meeting.pcm -m paraformer-realtime-v2 --vocabulary-id "$vid"
```

Note where `-o json` sits: before `audio`, because it is a global option. Inside this group the short form is already taken — `tts synthesize` and `realtime run` both spell `--out` as `-o`.

**Call a Bailian application, keeping the session** 🔑

```bash
dashscope -o json apps completions create --app-id "$APP_ID" --prompt "帮我查一下订单" \
  | jq -r '.session_id' > session.txt
dashscope apps completions create --app-id "$APP_ID" \
  --prompt "那第二个呢" --session-id "$(cat session.txt)"
```

**Drive a managed agent from a shell** 🔑

```bash
sid=$(dashscope -o json agentstudio sessions create --agent "$AGENT_ID" \
      | jq -r '.id')
dashscope agentstudio sessions events send "$sid" --message "帮我查一下订单"
dashscope agentstudio sessions events stream "$sid" --deltas message
```

`events send` posts the turn, `events stream` follows the run until it settles. They are separate commands, not one, because a pipeline usually wants to post N turns and then collect — and because a stream that broke mid-run should exit non-zero without having swallowed the send.

**Answer typed questions about a ticket** 🔑

```bash
dashscope decision create --state @ticket.txt --questions @questions.json
```

`--questions` declares a `type` per id — `choice` names an option, `noul` is a yes/no probability, `score` is a weighted level index. One row comes back per answer with its confidence, and `-o json` gives you `.answers.<id>.choice` to branch on. The model generates no prose, so there is nothing to parse out of a sentence. `-m` defaults to `decision-model-preview` and does **not** read `default_model`: this is a decision model, not a chat model.

**CI: destructive commands need `--yes`** 🔑

```bash
dashscope ft delete "$JOB_ID" --yes
```

Without `--yes` and without a terminal, it refuses at exit 2 rather than guessing.

**Per-job configuration isolation** 🔑

```bash
export DASHSCOPE_CONFIG="$CI_JOB_DIR/config.toml"
dashscope config set default_model qwen-plus
dashscope config set api_key "$DASHSCOPE_API_KEY"
dashscope chat create --prompt "hi"     # -m omitted: comes from default_model
```

**Clean logs** — no colour:

```bash
export NO_COLOR=1
```

`NO_COLOR=1` and `--no-color` mean the same thing. Rich already detects a pipe, so this matters mainly for runners that allocate a pseudo-terminal and then dump it into a log.

### Run the examples

Both run offline and deterministically; neither needs a key. From the repository root, with the checkout importable:

```bash
PYTHONPATH=. python examples/05_cli/cli_output_and_errors.py
PYTHONPATH=. python examples/05_cli/cli_ci_harness.py
```

(`PYTHONPATH=.` is only needed when the package is not pip-installed. If it is, `python examples/05_cli/cli_output_and_errors.py` is enough. The harness sets it for its own child processes, so it works either way.)

Both print a summary at the end and exit `0` only if every check passed — `cli_output_and_errors.py` asserts the exit code of every command it runs, and `cli_ci_harness.py` asserts all seven scenarios. Neither one reads or writes your real `~/.dashscope/config.toml`: the harness points `DASHSCOPE_CONFIG` at a temporary directory that is removed on exit.

To run the harness's read-only scenarios against the real gateway:

```bash
CLI_HARNESS_LIVE=1 DASHSCOPE_API_KEY=sk-xxx python examples/05_cli/cli_ci_harness.py
```

Live mode skips every fault-injection scenario and **never** runs `ft cancel` or `ft delete`. It will run `ft stream` only if you also set `DASHSCOPE_JOB_ID` to a job you own — guessing an id against a real account is how a demo deletes something important.

### Common pitfalls

1. **Global options cannot go after the subcommand.** `dashscope chat create --prompt "hi" -o json` is exit 2, not a formatted answer. Everything in the `--help` block at the top level is positional-relative to the app.
2. **`--stream -o yaml` is NDJSON, not YAML.** One JSON object per line, same as `-o json`. A YAML parser will not read it.
3. **Destructive commands refuse in CI.** No terminal + no `--yes` = exit 2. Your pipeline needs `--yes` on every `cancel`, `delete`, `scale` and `archive` — `ft`, `files`, `embeddings jobs`, `images jobs`, `videos jobs`, `deployments`, `rl`, `audio vocabularies`/`phrases`/`voices`/`voice-design`, `agentstudio agents`/`sessions` — and it needs it explicitly, because the whole point is that the command will not decide for you.
4. **`ft create -m` never falls back to `default_model`.** Only `chat create` and `multimodal chat create` do. The source comment says why: a chat default is not a fine-tune base.
5. **`config get` and `config list` can disagree.** `get` reads the file; `list` resolves the layers. When they differ, the `source` column in `list` is telling you which layer won — usually an environment variable you forgot you exported.
6. **In table mode the useful metadata is on stderr.** `usage:`, reasoning text and `page x/y, total n` are deliberately not on stdout, so a redirect stays clean. If you want them as data, ask for `-o json` instead of capturing stderr.
7. **`audio tts synthesize` writes raw bytes to stdout unless you pass `--out`.** That is what makes `> hello.mp3` work, and it is also what makes a forgotten `--out` spray binary into your terminal. There is no table for this command; `-o json` changes nothing about where the audio goes.
8. **`apps completions create --stream` also switches the service to incremental frames.** Without that, every event carries the whole answer so far and the terminal repeats the same growing sentence. Pass `--no-incremental-output` if you genuinely want the cumulative form.
9. **`multimodal chat create` exists even though `chat create` already routes media.** Reach for it when the conversation has to hit the multimodal endpoint without any media parts in it — the two commands take the same flags and render the same way.
10. **Inside `audio`, `-o` means `--out`, not `--output`.** `dashscope audio tts synthesize hi -o json` writes a file called `json` and exits 0. The format flag is global, so it goes before the group: `dashscope -o json audio ...`. This is the one place the CLI reuses a short option with a different meaning, and it is deliberate — `--out` is the option a shell recipe actually reaches for in this group.
11. **The `audio` registries do not paginate the same way.** `vocabularies`, `voices` and `voice-design` take a **0-based** `--page-index` plus `--page-size`; `phrases` is fine-tune-shaped and takes a **1-based** `--page` plus `--size`. That is not an oversight in the CLI — the two services behind them disagree, and the commands spell each one the way its own service does rather than inventing a translation layer that would silently drop a page.
12. **`agentstudio agents update` will not run without `--version`.** The service does optimistic concurrency on the agent's current version; exit 2 before any call if you left it out. A stale version is a 409, which the CLI has no specific tier for and therefore reports as `Error [server]` at exit 1 — retryable, but the retry that works is "re-read the agent, take its new `version`, update again", not "sleep and resend the same thing".
13. **`agentstudio sessions events stream` prints whole messages unless you ask for deltas.** Table mode reads the stream's `text_stream`, which a reconnect can rebuild; `--deltas message` (repeatable, also `reasoning`) opts into incremental frames and switches the command to `text_deltas`. `-o json` is unaffected — it is one raw event per line either way.
14. **`--timeout 0` is an error, not "check once and return".** Every `wait` command and every `--wait` flag rejects a timeout that is not a finite positive number of seconds — exit 2, before any API call, so a pipeline with no key configured still gets the usage error rather than an auth one. `0`, `-1`, `nan` and `inf` are all refused. `nan` is the one that matters: it is false against every comparison, so the SDK's deadline check never fires and the poll spins forever instead of timing out. If you want the current state without blocking, that is `get`.

For the SDK-side equivalent of the mock gateway, see [`02_advanced_engineering/local_mock_debugging.py`](./02_advanced_engineering/local_mock_debugging.py).

## legacy_1x — 1.x module-level API (reference only)

The following use the old 1.x style (`dashscope.Generation`, `dashscope.MultiModalConversation`, etc.). The internal subdirectory layout is kept exactly as-is and the content is untouched. New projects should use the 2.0 tiered examples above.

**1.x CLI:** existing 1.x command forms remain available through the same `dashscope` entry point as the 2.0 commands documented in [`05_cli/`](./05_cli/). Legacy-only names run against the frozen 1.x implementation; shared names are served by 2.0 with compatible option spellings.

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
