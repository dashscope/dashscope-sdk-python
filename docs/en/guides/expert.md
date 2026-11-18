---
description: "DashScope SDK Expert: the AgenticX AI assistant in your terminal — install, launch modes, one-shot questions and the first-run expert profile."
---
# SDK Expert (AgenticX)

The SDK ships an interactive AI assistant, **DashScope SDK Expert**, powered
by the optional `dashscope-agenticx` (AgenticX) dependency. It answers
SDK/API questions, generates runnable examples, shows CLI usage and
diagnoses errors right in your terminal.

The assistant requires **Python 3.10+** (the SDK itself supports 3.9+) and
is not installed by default.

## Install

```bash
pip install -U "dashscope[acli]"
```

Optional extras add assistant capabilities on top of `acli`:

| Extra | Provides |
| --- | --- |
| `acli` | Interactive AI assistant (DashScope SDK Expert). |
| `acli-openai` | Run the assistant on an OpenAI-compatible model provider. |
| `acli-anthropic` | Run the assistant on an Anthropic model provider. |
| `acli-voice` | Voice input and spoken replies in the assistant. |
| `acli-camera` | Camera capture for image input in the assistant. |
| `acli-all` | All of the above (includes `acli`). |

## Launch

```bash
dashscope                     # interactive assistant (TUI by default)
dashscope expert              # the same, spelled explicitly
dashscope expert --cli        # interactive, plain REPL instead of the TUI
dashscope expert --tui        # interactive, force the TUI
dashscope expert "question"   # ask once and exit
dashscope expert --help       # show all forms
```

With no argument and piped (non-terminal) stdin, the input is read as a
one-shot prompt:

```bash
echo "how do I stream chat output?" | dashscope
```

An unrecognized command is reported as an error rather than treated as a
question — ask questions with `dashscope expert "question"`. Inside the
assistant, type `/help` for the available commands.

The assistant uses the same API key as the SDK: export `DASHSCOPE_API_KEY`
or pass it on the command line:

```bash
dashscope -k sk-xxx expert "how do I upload a file?"
```

## First run: the expert profile

On the first interactive run in a directory without `./.agenticx/`, the
assistant offers to download the `dashscope-sdk-expert` profile into
`./.agenticx/`: an SDK Q&A expert persona, skill templates and an SDK
knowledge index. It is pure configuration — editable anytime.

- The download needs `git` and access to `github.com`. If it fails, the
  assistant still starts; retry later with:

  ```bash
  agenticx expert download dashscope-sdk-expert
  ```

- Declining leaves a marker in `./.agenticx/` so the offer is not repeated;
  the same command installs the profile later.

## Troubleshooting

- **`dashscope` prints an install tip instead of starting** — the `acli`
  extra is missing (or Python is older than 3.10). Run
  `pip install -U "dashscope[acli]"` under Python 3.10+.
- **The tip says `dashscope-agenticx` is installed but incompatible** —
  the runtime does not provide the interface this SDK expects. Upgrade
  with `pip install -U "dashscope[acli]"`; if that does not help, report
  the mismatch.

## Full walkthrough

The [DashScope SDK Expert guide](https://help.aliyun.com/en/model-studio/dashscope-sdk-expert)
on Alibaba Cloud Model Studio covers the assistant end to end. For the
command tree behind `dashscope <group> ...`, see [CLI](../cli.md).