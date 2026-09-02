# Basic Chat — Minimal Working acli Example

Shows how to start a general-purpose chat agent with web search capability using minimal configuration. **All the intelligence lives in the `.acli/` configuration — there is no Python startup code** — download and run `acli` directly.

## Directory Structure

```
basic-chat/
└── .acli/
    ├── config.toml                   # Default provider/model/user_name
    ├── custom-extensions.toml        # tongyi provider declaration + capability/skill/shell_tool comment templates
    ├── hooks.toml                    # Event hook (before/after_tool_call, on_error, etc.) comment templates
    ├── system-prompt.md              # Agent persona and behavior rules
    └── skills/
        ├── research-topic.md         # Web-search a topic and produce a briefing (calls web_search)
        ├── explain-code.md           # Explain code logic
        ├── translate.md              # Chinese-English translation
        └── write-poem.md             # Write a seven-character quatrain (demonstrates a pure prompt template)
```

## Quick Start

```bash
pip install acli
export DASHSCOPE_API_KEY="sk-xxx"

# Merge the example into ./.acli/ (same-name files are auto-backed up to .acli/backup/; undo with example restore)
acli example download basic-chat

# Edit .acli/custom-extensions.toml to add the providers you need
# Edit .acli/system-prompt.md to define the agent persona
# Add your own skill templates under .acli/skills/

# Start (no cd needed; the config is already in the current directory)
acli
acli --tui
acli -c "hello"
```

> Want it in a brand-new directory? `mkdir my-agent && cd my-agent && acli example download basic-chat`,
> or use `acli example download basic-chat --target my-agent`.

## Configuration as Program

### custom-extensions.toml — Provider Declarations

Declares which LLM providers acli can use. A minimal config needs just one `[[providers]]` block:

```toml
[[providers]]
name = "tongyi"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"      # ← stores only the env var name; the shell provides sk-xxx
default_model = "qwen3.7-max"
models = ["qwen3.7-max", "qwen3.7-plus", "qwen-turbo", "qwen-vl-max"]
vision_models = ["qwen-vl-max"]        # ← tells acli these models accept image input
protocol = "openai"                     # ← openai / anthropic / dashscope
```

Want Claude / GPT / GLM? Just uncomment the corresponding `[[providers]]` block in the toml.

**Three ways to provide an API key** (in decreasing order of recommendation):

1. `api_key_env = "FOO_API_KEY"` — `export FOO_API_KEY=sk-xxx` in the shell; the toml is safe to commit to git
2. `/provider` wizard interactive entry — writes `api_key = "ENC:..."` (machine-bound encryption)
3. Plaintext `api_key = "sk-xxx"` — rejected by the loader; placeholder illustration only

### system-prompt.md — Agent Persona

Defines who the agent "is". acli automatically loads `.acli/system-prompt.md` at startup (workspace takes precedence over `~/.acli/system-prompt.md`).

### skills/*.md — Prompt Templates

Each `.md` file is a reusable prompt with YAML frontmatter:

```yaml
---
name: research-topic
description: Web-search a topic and produce a briefing with source URLs
arguments: [topic]
---

Use the web_search tool to research "{topic}":
...
```

How to invoke:
- `/skill research-topic quantum computing` — explicit invocation
- Natural language: "help me research the latest progress in quantum computing" — the LLM decides whether to use it

`research-topic` demonstrates how to use a prompt to guide the LLM to call the built-in `web_search` tool for online information gathering.

### config.toml — Defaults

```toml
user_name = "dashscope"
provider = "tongyi"
model = "qwen3.7-max"
memory_user_id = "acli-basic"
```

## Next Steps

- **Add more providers**: add `[[providers]]` blocks in `custom-extensions.toml`
- **Add HTTP tools**: add `[[capabilities]]` + `[[capabilities.tools]]` blocks (e.g. calling Coze, Zhipu image generation, etc.)
- **Add vision capability**: add a capability tool with `type = "vision"` so the text agent can call a vision LLM on demand
- **Add shell tools**: add `[[shell_tools]]` blocks to wrap common local commands
- **Add hooks**: configure pre/post tool-call hooks in `.acli/hooks.toml` (e.g. auto `py_compile` after writing a `.py` file, confirm before `pip install`, block file deletion). See the template in `.acli/hooks.toml`, covering all 5 events (`before_tool_call` / `after_tool_call` / `on_error` / `on_message` / `on_response`) × 6 actions (run/block/confirm/warn/alert/log).
- **Add persistent knowledge**: put documents that must **always** appear in the system prompt (e.g. an API index) into `.acli/references/*.md`
- **Change the persona**: edit `system-prompt.md` — e.g. turn it into a "code reviewer", "data analyst", or "customer support agent"

See the project root `README.md` for the full feature documentation.
