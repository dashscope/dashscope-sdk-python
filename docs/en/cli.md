---
description: "The dashscope command line: global options, exit codes and every command group."
---
# Command Line Interface

Installing the SDK also installs the `dashscope` command (and `dashscope1`
for the archived 1.x command tree). The CLI mirrors the SDK resources, so
anything you can do with `Client` you can also script from a shell.

```bash
dashscope version                 # print the SDK version (a subcommand, not --version)
dashscope auth login              # save an API key (prompted interactively)
dashscope chat create --model qwen-plus --prompt "Hello!"
dashscope -k sk-xxx --output json ft list
```

## Global options

Global options come **before** the subcommand:

| Option | Meaning |
| --- | --- |
| `-k`, `--api-key` | DashScope API key; overrides environment and config file. |
| `--protocol` | Wire protocol: `native`, `openai` or `anthropic`. |
| `--base-url` | Override the service base URL. |
| `--workspace` | Workspace (business space) ID. |
| `--region` | Service region, e.g. `cn-beijing`, `ap-southeast-1`. |
| `--timeout` | Per-request timeout in seconds. |
| `-o`, `--output` | Output format; machine formats keep stdout parseable (e.g. `json`). |

Errors go to stderr as `Error [category]: message`. Exit codes: `0` ok,
`1` server / rate-limit / network, `2` bad command line, `3` auth,
`4` not found.

## Command groups

| Group | Purpose |
| --- | --- |
| `auth` | `login` / `whoami` / `logout` — manage the stored API key. |
| `config` | `set` / `get` / `list` — persistent CLI configuration. |
| `chat` | Chat completions (sync and streaming). |
| `multimodal` | Multimodal understanding / generation calls. |
| `embeddings` | Text embedding calls and batch jobs. |
| `rerank` | Document reranking. |
| `tokenization` | Token counting. |
| `images` | Image generation jobs. |
| `videos` | Video generation jobs. |
| `audio` | TTS / ASR / realtime audio commands. |
| `files` | Upload, list, get and delete files. |
| `models` | Browse the model catalog. |
| `ft` | Fine-tuning jobs (`fine-tunes` is a hidden alias). |
| `deployments` | Manage model deployments. |
| `rl` | Agentic RL datasets, functions and jobs (`agentic-rl` alias). |
| `apps` | Model Studio application completions. |
| `tingwu` | Meeting transcription and analysis. |
| `decision` | Structured decision / judging calls. |
| `agentstudio` | AgentStudio managed-agent resources. |
| `oss` | OSS upload helpers for large media inputs. |

Run `dashscope <group> --help` (or `dashscope <group> <command> --help`) for
the exact arguments of each command.

## The agent entry point

Invoking `dashscope` with **no arguments** starts the interactive AgenticX
agent instead of the command tree. It requires the optional extra and
Python 3.10+:

```bash
pip install -U "dashscope[acli]"
dashscope                         # launches the agent
```

## The 1.x CLI

The legacy command tree (`generation create`, `application create`,
`files.upload`, `login`, ...) is installed separately as `dashscope1` and is
frozen. Running a 1.x command name under `dashscope` only prints a migration
hint pointing at `dashscope1` or the 2.0 equivalent.