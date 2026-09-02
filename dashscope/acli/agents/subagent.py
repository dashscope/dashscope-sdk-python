# -*- coding: utf-8 -*-
"""local.subagent: spawn an isolated Agent loop sharing the parent's
provider + executor + tool registry, but with its own messages list so
the sub-task doesn't pollute the parent's conversation context.

Use cases the parent agent reaches for this tool:
  - "review file X end-to-end" — fan out a long read+analyze loop whose
    intermediate tool calls should NOT clog the main thread
  - "summarize this 50-file directory" — sub-agent does the scanning,
    returns one paragraph
  - "explore an unfamiliar API" — speculative reads in isolation; if it
    finds nothing useful, parent context stays clean

What it is NOT:
  - A different LLM provider (it inherits whatever the parent is using).
    For "use Claude for hard reasoning while main uses tongyi" you want a
    different provider per call, not subagent.
  - A long-running daemon. Each invocation is one synchronous call.
"""

from __future__ import annotations

from dashscope.acli.tools.registry import registry

# Set by cli._run_loop after Agent construction. We pin a reference here
# so the registered subagent_invoke tool can spin up a child Agent from
# whatever provider/executor the main one currently holds — without a
# circular import to cli.py.
_parent_agent = None  # type: ignore[var-annotated]
_config = None  # type: ignore[var-annotated]


def set_parent_agent(agent) -> None:
    global _parent_agent
    _parent_agent = agent


def set_config(config) -> None:
    """Store a reference to the Config so subagent_invoke can look up
    per-agent overrides (max_turns, model, temperature)."""
    global _config
    _config = config


def _has_parent() -> bool:
    return _parent_agent is not None


async def _subagent_invoke(
    prompt: str,
    system_prompt: str = "",
    max_turns: int = 10,
) -> str:
    """Spawn a one-shot sub-agent with isolated message history.

    prompt          The task the sub-agent should handle. Treat it as the
                    user message it'll receive.
    system_prompt   Optional extra system prompt segment to specialize the
                    sub-agent (e.g. "You are a strict code reviewer; only
                    use read_file / list_directory; report findings as a
                    bulleted list."). If empty, the sub-agent uses the
                    parent's default system prompt.
    max_turns       Bound on the sub-agent loop length. Default 10 covers
                    short focused tasks; bump for deeper exploration.
                    Hard cap at 50 to prevent runaway sub-loops.

    Returns the sub-agent's final assistant content (text only — tool call
    intermediates are NOT included; that's the whole point).
    """
    if _parent_agent is None:
        return (
            "Error: subagent not initialized "
            "(missing parent agent reference)"
        )

    from dashscope.acli.agent import (
        Agent,
    )  # local import to avoid module-load cycle
    from dashscope.acli.memory.manager import MemoryManager

    # Look up per-agent config overrides (max_turns, model, temperature)
    capped_turns = min(max(1, max_turns), 50)
    if _config is not None:
        agent_cfg = _config.subagents.get("local.subagent")
        if agent_cfg and agent_cfg.max_turns:
            capped_turns = min(max(1, agent_cfg.max_turns), 50)

    # Compose a single system prompt: parent's base prompt with the caller's
    # specialization appended. Without a specialization, pass None so the
    # child ctor resolves the default prompt itself — either way the child
    # ends up with exactly one system message.
    composed_prompt = None
    if system_prompt:
        base = getattr(_parent_agent, "system_prompt", None) or ""
        composed_prompt = (
            f"{base}\n\n{system_prompt}" if base else system_prompt
        )

    sub = Agent(
        provider=_parent_agent.provider,
        executor=_parent_agent.executor,
        max_turns=capped_turns,
        memory=None,  # don't recall main user's profile
        user_name=_parent_agent.user_name,
        provider_name=_parent_agent.provider_name,
        model_name=_parent_agent.model_name,
        session_path=None,  # never persist sub-agent runs
        # disabled_caps + directives DO inherit — they express user intent
        # that should apply to any agent acting on the user's behalf.
        disabled_caps_provider=_parent_agent.disabled_caps_provider,
        directives_provider=_parent_agent.directives_provider,
        system_prompt=composed_prompt,
        allow_delegate=False,  # no nested spawning from a sub-agent
        hook_bus=getattr(_parent_agent, "hook_bus", None),
        # Isolated session tier: child plans/failures must not move the
        # parent's plan tracker or reflection counters.
        memory_manager=(
            MemoryManager.derive_child(_parent_agent.memory_manager)
            if _parent_agent.memory_manager is not None
            else None
        ),
    )

    # Drain run_stream into a single result string. Strip the "[tool] →"
    # trail lines that the agent inserts for the UI — for the parent agent
    # they're noise, not signal.
    buffer: list[str] = []
    async for chunk in sub.run_stream(prompt):
        if chunk.startswith("\n[") and "] →" in chunk:
            continue  # skip tool-call trail markers
        buffer.append(chunk)
    return ("".join(buffer)).strip() or "(subagent returned no content)"


def register_subagent_tool() -> None:
    """Wire subagent_invoke into the tool registry. Called from
    tools/platform.register_one_capability when local.subagent is enabled."""
    registry.register_mcp_tool(
        name="subagent_invoke",
        description=(
            "Spawn an isolated sub-agent (same provider, same tools, fresh "
            "context) to handle a self-contained task whose intermediate "
            "reasoning shouldn't pollute the main conversation. Returns "
            "only the sub-agent's final answer as text. Good for: long "
            "code reviews, multi-file scans, speculative exploration. "
            "Pass system_prompt to specialize role (e.g. strict reviewer, "
            "researcher)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The task description the sub-agent acts on."
                    ),
                },
                "system_prompt": {
                    "type": "string",
                    "description": (
                        "Optional system prompt to specialize the sub-agent's "
                        "role. Leave empty to inherit parent's default."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": (
                        "Cap on sub-agent loop iterations "
                        "(default 10, max 50)."
                    ),
                    "default": 10,
                },
            },
            "required": ["prompt"],
        },
        call_fn=_subagent_invoke,
    )

    # Register specialist subagent tools
    _register_specialist_tools()


# Specialist subagent system prompts
CODE_REVIEWER_PROMPT = """You are a professional code reviewer.
When reviewing code, focus on:
1. Potential bugs and logic errors
2. Security issues (SQL injection, XSS, permission bypass, etc.)
3. Performance problems (inefficient algorithms, unnecessary loops)
4. Code style (naming, comments, readability)

Use the read_file and search_files tools to read code. Output format:
- Critical issues (must fix)
- Suggested improvements (optional)
- Code strengths

Report only concrete findings; avoid generic remarks."""

TEST_WRITER_PROMPT = """You are a professional test engineer.
Generate comprehensive test cases for the given code. Focus on:
1. Happy-path tests
2. Boundary conditions (empty, extreme, and special-character values)
3. Error handling (exceptions, failure scenarios)
4. Performance tests (where applicable)

Use read_file to read source code and write_file to create test files.
Prefer pytest as the test framework.
Output test code only, without extra explanation."""

DOC_GENERATOR_PROMPT = """You are a professional technical writer.
Generate clear documentation for code. Include:
1. Overview of the module/class purpose
2. Detailed public API reference (parameters, return values, examples)
3. Usage examples
4. Caveats and limitations

Use read_file to read source code and write_file to generate docs
(Markdown format or code comments).
Keep documentation concise and practical; avoid redundancy."""


async def _specialist_invoke(specialist_type: str, task: str) -> str:
    """Invoke a specialist subagent with pre-defined system prompt."""
    prompts = {
        "code_reviewer": CODE_REVIEWER_PROMPT,
        "test_writer": TEST_WRITER_PROMPT,
        "doc_generator": DOC_GENERATOR_PROMPT,
    }
    system_prompt = prompts.get(specialist_type, "")
    if not system_prompt:
        return (
            f"Error: unknown specialist type '{specialist_type}'; "
            f"options: {', '.join(prompts.keys())}"
        )

    return await _subagent_invoke(
        prompt=task,
        system_prompt=system_prompt,
        max_turns=15,
    )


def _register_specialist_tools() -> None:
    """Register specialist subagent tools."""

    async def review_code(task: str) -> str:
        """Review code for bugs, security issues, performance, and style."""
        return await _specialist_invoke("code_reviewer", task)

    async def write_tests(task: str) -> str:
        """Generate comprehensive test cases for the given code."""
        return await _specialist_invoke("test_writer", task)

    async def generate_docs(task: str) -> str:
        """Generate clear documentation for the given code."""
        return await _specialist_invoke("doc_generator", task)

    registry.register_mcp_tool(
        name="review_code",
        description=(
            "Invoke a specialist code reviewer to find bugs, security "
            "issues, performance problems, and style issues"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "review task, e.g. 'check src/auth.py for "
                        "security issues'"
                    ),
                },
            },
            "required": ["task"],
        },
        call_fn=review_code,
    )
    registry.register_mcp_tool(
        name="write_tests",
        description=(
            "Invoke a specialist test engineer to generate "
            "comprehensive test cases for code"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "test task, e.g. 'generate tests for " "src/utils.py'"
                    ),
                },
            },
            "required": ["task"],
        },
        call_fn=write_tests,
    )
    registry.register_mcp_tool(
        name="generate_docs",
        description=(
            "Invoke a specialist doc writer to generate clear "
            "documentation for code"
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "doc task, e.g. 'generate API docs for " "src/api.py'"
                    ),
                },
            },
            "required": ["task"],
        },
        call_fn=generate_docs,
    )
