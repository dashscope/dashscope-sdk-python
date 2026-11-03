# 05 Command Line

> **English** | [中文](README_zh.md)

The other tiers drive the SDK from Python. This one drives it from a **shell**: the `dashscope` command is the same capabilities behind a different contract, and the contract is what makes it scriptable. Three output formats, a layered exit code, one fixed stderr error line, four configuration layers with a documented precedence, and a refusal to guess when a destructive command meets a terminal-less CI job. Every guarantee below is a promise a build script can rely on — and every one of them is invisible if you only ever read the Python examples.

| File | What it shows | Network |
| --- | --- | --- |
| [`cli_output_and_errors.py`](./cli_output_and_errors.py) | The contract from the inside: one fake client, driven in-process through `typer.testing.CliRunner`, printing stdout / stderr / exit code side by side for table vs json vs yaml, NDJSON streaming, `@file` / stdin / `--messages` input shapes, and all four failure tiers | No |
| [`cli_ci_harness.py`](./cli_ci_harness.py) | The contract from the outside: a stdlib mock gateway on an ephemeral port, driving the **real CLI in a real subprocess** — parse `-o json`, branch and retry on exit codes, consume an NDJSON stream line by line, isolate configuration per job, and survive the destructive-command prompt | No (offline mock by default; `CLI_HARNESS_LIVE=1` hits the real gateway, read-only scenarios only) |

Read them in that order. The first one shows *what* comes out of each channel; the second shows *how a pipeline should consume it*.

---

## Install and first call

The package registers a console script, so after `pip install -U dashscope` there is a `dashscope` command on your `PATH`:

```bash
dashscope --help
dashscope version
```

From a source checkout where the script is not installed, the module entry point is identical:

```bash
python3 -m dashscope.cli2 --help
python3 -m dashscope.cli2 version
```

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

---

## Command map

| Command | Purpose |
| --- | --- |
| `dashscope version` | Print the installed SDK version. Touches nothing, so it is the cheapest smoke test you have |
| `dashscope chat create` | One chat completion: `--prompt` / `--messages`, `--stream`, `--tools`, sampling flags, `--parameters` for anything else |
| `dashscope ft create` | Start a fine-tune job; `--wait` blocks with a spinner until a terminal status, `--timeout` bounds the wait |
| `dashscope ft get` | Show one job |
| `dashscope ft list` | Page through jobs: `--page`, `--size` |
| `dashscope ft stream` | Follow a job's server-side event feed until the server closes it |
| `dashscope ft cancel` | Cancel a running job. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope ft delete` | Delete a job and its output model. **Destructive** — prompts, needs `--yes` in CI |
| `dashscope config set` | Write one key to `~/.dashscope/config.toml` |
| `dashscope config get` | Read one key back (api_key masked unless `--reveal`) |
| `dashscope config list` | Every key, its effective value, and the layer that supplied it |

The same entry point still runs the 1.x commands, so old shell scripts keep working. `main.py` sorts argv into three buckets: the v2 tree above, a **deprecated** set (`generation`, `application`, `image-synthesis`, `multimodal-conversation`, `speech-synthesis`, `transcription`, `video-synthesis`, `image-generation`, `multimodal-embedding`, `agentic-rl`) that prints a notice before delegating, and an **awaiting-v2** set (`auth`, `deployments`, `embeddings`, `files`, `models`, `oss`, `rerank`, `rl`, `tokenization`) that delegates silently. Set `DASHSCOPE_CLI_NO_DEPRECATION=1` to silence the notice in a build log. A bare `dashscope` with no subcommand opens the bundled `expert` agent interactively.

---

## Output formats

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

---

## Exit codes and error format

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

How an HTTP status becomes a category (`cli2/errors.py`, `_STATUS_CATEGORY`):

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

---

## Configuration

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

---

## Input conventions

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

---

## Recipes

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

**Clean logs** — no colour, no deprecation notice:

```bash
export NO_COLOR=1 DASHSCOPE_CLI_NO_DEPRECATION=1
```

`NO_COLOR=1` and `--no-color` mean the same thing. Rich already detects a pipe, so this matters mainly for runners that allocate a pseudo-terminal and then dump it into a log.

---

## Run the examples

Both run offline and deterministically; neither needs a key. From the repository root, with the checkout importable:

```bash
PYTHONPATH=. python3 examples/05_cli/cli_output_and_errors.py
PYTHONPATH=. python3 examples/05_cli/cli_ci_harness.py
```

(`PYTHONPATH=.` is only needed when the package is not pip-installed. If it is, `python3 examples/05_cli/cli_output_and_errors.py` is enough. The harness sets it for its own child processes, so it works either way.)

Both print a summary at the end and exit `0` only if every check passed — `cli_output_and_errors.py` asserts the exit code of every command it runs, and `cli_ci_harness.py` asserts all seven scenarios. Neither one reads or writes your real `~/.dashscope/config.toml`: the harness points `DASHSCOPE_CONFIG` at a temporary directory that is removed on exit.

To run the harness's read-only scenarios against the real gateway:

```bash
CLI_HARNESS_LIVE=1 DASHSCOPE_API_KEY=sk-xxx python3 examples/05_cli/cli_ci_harness.py
```

Live mode skips every fault-injection scenario and **never** runs `ft cancel` or `ft delete`. It will run `ft stream` only if you also set `DASHSCOPE_JOB_ID` to a job you own — guessing an id against a real account is how a demo deletes something important.

---

## A pit you will step into without reading the source

1. **Global options cannot go after the subcommand.** `dashscope chat create --prompt "hi" -o json` is exit 2, not a formatted answer. Everything in the `--help` block at the top level is positional-relative to the app.
2. **`--stream -o yaml` is NDJSON, not YAML.** One JSON object per line, same as `-o json`. A YAML parser will not read it.
3. **Destructive commands refuse in CI.** No terminal + no `--yes` = exit 2. Your pipeline needs `--yes` on `ft cancel` and `ft delete`, and it needs it explicitly, because the whole point is that the command will not decide for you.
4. **`ft create -m` never falls back to `default_model`.** Only `chat create` does. The source comment says why: a chat default is not a fine-tune base.
5. **`config get` and `config list` can disagree.** `get` reads the file; `list` resolves the layers. When they differ, the `source` column in `list` is telling you which layer won — usually an environment variable you forgot you exported.
6. **In table mode the useful metadata is on stderr.** `usage:`, reasoning text and `page x/y, total n` are deliberately not on stdout, so a redirect stays clean. If you want them as data, ask for `-o json` instead of capturing stderr.

See also [`../README.md`](../README.md) for how this tier fits the rest, and [`../02_advanced_engineering/local_mock_debugging.py`](../02_advanced_engineering/local_mock_debugging.py) for the SDK-side equivalent of the mock gateway the harness builds here.
