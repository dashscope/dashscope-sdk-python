# DashScope SDK Expert — acli Configuration-Driven Example

**Online guide**: https://help.aliyun.com/en/model-studio/dashscope-sdk-expert

This example shows how to build a scenario-specific AI expert agent using **AgenticCLI (acli)**'s native configuration mechanisms.

**Core idea: configuration-driven, zero Python glue.** The agent's identity, capabilities, skills, and knowledge index are all defined by files under `.acli/`; download the example and run `acli` directly to start.

**SDK baseline:** new Python examples use DashScope **2.0.3** (`dashscope.Client` / `AsyncClient`) and `dashscope.errors`. The skills also retain 1.x guidance for maintaining existing `Generation.call`-style code; installing 2.0 does not require rewriting working 1.x applications.

## Directory Structure

```
dashscope-sdk-expert/
└── .acli/                      # Agent configuration directory
    ├── config.toml              # Model and user configuration
    ├── custom-extensions.toml   # Provider declaration (tongyi)
    ├── hooks.toml               # Event hooks
    ├── system-prompt.md         # System prompt (agent persona and behavior rules)
    └── skills/                  # Skill templates (loaded by the model on demand via use_skill)
        ├── text-generation.md   # Chat, decision & 1.x Generation (Python+Java)
        ├── multimodal.md        # Multimodal chat, images & videos
        ├── speech.md            # 2.0 audio, TingWu & 1.x speech compatibility
        ├── retrieval.md         # Embeddings, rerank & tokenization
        ├── fine-tuning.md       # Files, fine-tunes, deployments & Agentic RL
        ├── agent.md             # Apps, AgentStudio & 1.x Assistants compatibility
        ├── cli.md               # dashscope CLI command reference
        ├── sdk-example.md       # Generate SDK code examples
        ├── api-doc.md           # View API parameter docs
        ├── diagnose.md          # Diagnose SDK call errors
        ├── error-code.md        # Explain error codes
        ├── explain-code.md      # Explain code logic
        └── translate.md         # Chinese-English translation
```

## Quick Start

```bash
pip install acli
export DASHSCOPE_API_KEY="sk-xxx"

# Merge the example into ./.acli/ (same-name files are auto-backed up to .acli/backup/; undo with example restore)
acli example download dashscope-sdk-expert

# Start — no cd, no Python launcher script needed
acli
acli --tui
acli -c "Show a DashScope 2.0 streaming chat example with error handling"
```

For legacy maintenance, ask for example: `acli -c "Fix this Generation.call code without migrating it to 2.0"`.

## The Configuration-Driven Approach

### 1. system-prompt.md — Agent Persona

Defines the agent's identity, knowledge scope, and behavior rules. This is the core of who the agent "is":

```markdown
You are DashScope SDK Expert, an intelligent assistant for the DashScope Python SDK...

### Choose 1.x vs 2.0 from the user's context
For new Python code, recommend 2.0 resource APIs and `dashscope.errors`...
```

### 2. skills/ — Domain Knowledge Base (Loaded on Demand)

The SDK/CLI's public interface knowledge lives directly in domain skills: one file per domain, with current 2.0.3 resource signatures, 1.x compatibility notes, Java references, input/output structures, and error handling. When answering API questions, the model loads the matching skill via `use_skill` on demand — **nothing stays permanently in the system prompt** — cutting first-turn input tokens by about 16k characters; details not covered by a skill fall back to `inspect.signature` / `help()` against the installed package.

### 3. skills/ — Task Templates

Each `.md` file is a reusable prompt template with frontmatter metadata:

```yaml
---
name: sdk-example
description: Generate runnable DashScope SDK code examples
arguments: [api_name]
---

Before generating code, first verify the user's installed SDK version and API signature:
1. `run_command("python -c 'import dashscope; ...'")`
...
```

- **name**: skill identifier, invoked via the `/skill` command
- **description**: short description; the agent uses it to decide when to apply the skill
- **arguments**: template variables, replaced with actual values on invocation

### 4. config.toml — Runtime Configuration

```toml
user_name = "dashscope"
provider = "tongyi"
model = "qwen3.8-max"
memory_user_id = "acli-dashscope"
```

## Reuse This Pattern

To create an AI expert for your own scenario:

1. `acli example download dashscope-sdk-expert` (merges into your project's `./.acli/`)
2. Edit `.acli/system-prompt.md` — define your agent persona
3. Edit `.acli/skills/` — add your domain knowledge and skill templates (loaded by the model on demand)
4. Edit `.acli/config.toml` — choose a suitable model
5. Run `acli`

## Design Takeaways

| Traditional Approach | acli Configuration Approach |
|---------|---------------|
| Prompts hardcoded in code | `system-prompt.md` file |
| if-else branches for different scenarios | `skills/*.md` template library |
| Entire large docs stuffed into the prompt | `skills/` domain knowledge loaded on demand |
| Code changes required to adjust behavior | Just edit Markdown |
| Hard to share and reuse | Entire `.acli/` directory is portable |
