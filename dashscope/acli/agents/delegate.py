# -*- coding: utf-8 -*-
"""local.delegate: spawn parallel sub-agents for fan-out tasks.

A `delegate` tool lets the main agent break a request into independent
sub-tasks, run them in parallel, and collect structured results. Child
agents share the parent's provider and executor but have isolated message
histories and cannot delegate further (no nested spawning).
"""
# pylint: disable=too-many-branches

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from dashscope.acli.tools.registry import registry

# Set by cli._run_loop / _run_tui_mode after Agent construction so the
# delegate tool can spawn children of the live main agent.
_parent_agent = None  # type: ignore[var-annotated]
_config = None  # type: ignore[var-annotated]

# Concurrency control across all delegate calls in the process.
# asyncio.Semaphore binds to the event loop that first uses it, so a single
# module-level instance breaks when delegates run on different loops (e.g.
# consecutive asyncio.run() calls or REPL vs TUI loops). Keep one semaphore
# per running loop, created lazily inside the async entry point.
_delegate_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def set_parent_agent(agent) -> None:
    global _parent_agent
    _parent_agent = agent


def set_config(config) -> None:
    global _config
    _config = config


def _get_delegate_semaphore() -> asyncio.Semaphore:
    """Return the delegate semaphore bound to the current running loop."""
    loop = asyncio.get_running_loop()
    sem = _delegate_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(max(1, _delegation_config().max_concurrent))
        _delegate_semaphores[loop] = sem
    return sem


def _has_parent() -> bool:
    return _parent_agent is not None


def _delegation_config():
    from dashscope.acli.config import Config

    if _config is None:
        return Config().delegation
    return getattr(_config, "delegation", _default_delegation_config())


def _default_delegation_config():
    """Fallback when config object has no delegation field (e.g. tests)."""

    class _Fallback:
        max_concurrent = 5
        default_timeout = 120
        allow_nested = False

    return _Fallback()


def _child_system_prompt(
    tools: list[str] | None,
    override: str | None = None,
) -> str:
    if override:
        base = override
    else:
        base = (
            "You are an acli subagent focused on completing the "
            "assigned sub-task."
        )
    if tools:
        base += f"\nYou may only use these tools: {', '.join(tools)}"
    else:
        base += "\nYou may use any currently available tool."
    base += (
        "\nDo not call delegate, delegate_parallel, or subagent_invoke "
        "(subagents may not delegate further)."
    )
    return base


def _build_prompt(task: str, context_files: list[str] | None) -> str:
    parts = []
    if context_files:
        parts.append("Context files:")
        for path in context_files:
            parts.append(f"- {path}")
            try:
                content = Path(path).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
                parts.append(f"```\n{content}\n```")
            except Exception as e:
                parts.append(f"(read failed: {e})")
        parts.append("")
    parts.append(f"Task: {task}")
    return "\n".join(parts)


async def _drain_stream(stream, chunks: list[str]) -> None:
    async for chunk in stream:
        if chunk:
            chunks.append(chunk)


async def _run_child(
    task: str,
    tools: list[str] | None,
    context_files: list[str] | None,
    timeout: int,
    model: str | None,
    system_prompt: str | None = None,
    max_turns: int | None = None,
) -> dict[str, Any]:
    if _parent_agent is None:
        return {
            "task_id": str(uuid.uuid4())[:8],
            "status": "failed",
            "result": (
                "Error: delegate not initialized "
                "(missing parent agent reference)"
            ),
        }

    from dashscope.acli.agent import Agent
    from dashscope.acli.memory.manager import MemoryManager

    task_id = str(uuid.uuid4())[:8]
    cfg = _delegation_config()
    effective_timeout = timeout or cfg.default_timeout
    effective_timeout = max(1, effective_timeout)

    # Capped max_turns: explicit param > subagents config > default 10,
    # hard cap 50.
    if max_turns is not None:
        capped_turns = min(max(1, max_turns), 50)
    else:
        capped_turns = 10
        if _config is not None:
            agent_cfg = _config.subagents.get("local.delegate")
            if agent_cfg and agent_cfg.max_turns:
                capped_turns = min(max(1, agent_cfg.max_turns), 50)

    sub = Agent(
        provider=_parent_agent.provider,
        executor=_parent_agent.executor,
        max_turns=capped_turns,
        memory=None,
        user_name=_parent_agent.user_name,
        provider_name=_parent_agent.provider_name,
        model_name=model or _parent_agent.model_name,
        session_path=None,
        disabled_caps_provider=_parent_agent.disabled_caps_provider,
        directives_provider=_parent_agent.directives_provider,
        system_prompt=_child_system_prompt(tools, override=system_prompt),
        allow_delegate=False,
        allowed_tools=tools,
        hook_bus=getattr(_parent_agent, "hook_bus", None),
        # Isolated session tier: child plans/failures must not move the
        # parent's plan tracker or reflection counters.
        memory_manager=(
            MemoryManager.derive_child(_parent_agent.memory_manager)
            if _parent_agent.memory_manager is not None
            else None
        ),
    )

    prompt = _build_prompt(task, context_files)
    chunks: list[str] = []
    stream_task = asyncio.create_task(
        _drain_stream(sub.run_stream(prompt), chunks),
    )
    status = "failed"

    try:
        try:
            await asyncio.wait_for(
                asyncio.shield(stream_task),
                timeout=effective_timeout,
            )
            status = "completed"
        except asyncio.TimeoutError:
            status = "timeout"
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            # Cancelled by the caller — the finally below stops the child
            # stream first, then the cancellation propagates.
            raise
        except Exception:
            status = "failed"
    finally:
        # Never abandon the child stream: cancel it on timeout/failure/
        # cancellation and wait for it to actually finish (suppress errors).
        if not stream_task.done():
            stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass
        except Exception:
            status = "failed"

    # Strip tool-call trail markers from the child output.
    cleaned: list[str] = []
    for chunk in chunks:
        if chunk.startswith("\n[") and "] →" in chunk:
            continue
        cleaned.append(chunk)

    result = ("".join(cleaned)).strip() or "(subagent returned no content)"
    return {"task_id": task_id, "status": status, "result": result}


async def _delegate(
    task: str,
    tools: list[str] | None = None,
    context_files: list[str] | None = None,
    timeout: int | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    max_turns: int | None = None,
) -> str:
    """Spawn a single sub-agent and return a JSON result."""
    async with _get_delegate_semaphore():
        result = await _run_child(
            task,
            tools,
            context_files,
            timeout,
            model,
            system_prompt,
            max_turns,
        )
    return json.dumps(result, ensure_ascii=False)


async def _delegate_parallel(
    tasks: list[dict[str, Any]],
    max_concurrent: int | None = None,
) -> str:
    """Spawn multiple sub-agents in parallel with controlled concurrency.

    Each entry in `tasks` is a dict with optional keys:
    task, tools, context_files, timeout, model, system_prompt, max_turns.
    """
    cfg = _delegation_config()
    limit = max_concurrent or cfg.max_concurrent
    limit = max(1, limit)
    semaphore = asyncio.Semaphore(limit)

    async def _run_one(task_def: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _run_child(
                task=task_def.get("task", ""),
                tools=task_def.get("tools"),
                context_files=task_def.get("context_files"),
                timeout=task_def.get("timeout"),
                model=task_def.get("model"),
                system_prompt=task_def.get("system_prompt"),
                max_turns=task_def.get("max_turns"),
            )

    results = await asyncio.gather(*[_run_one(t) for t in tasks])
    return json.dumps(results, ensure_ascii=False)


def register_delegate_tools() -> None:
    """Wire delegate tools into the registry."""
    registry.register_mcp_tool(
        name="delegate",
        description=(
            "Delegate a sub-task to an independent subagent. The "
            "subagent shares the current provider/executor but has an "
            "isolated message history that won't pollute the main "
            "conversation. You may restrict its tool whitelist, pass "
            "context files, and set timeout, model, system_prompt, "
            "and max_turns. Returns JSON with {task_id, status, "
            "result}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "sub-task description; the subagent treats it "
                        "as a user message."
                    ),
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "tool whitelist the subagent may use "
                        "(optional; default is all)."
                    ),
                },
                "context_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "file paths whose contents are prepended to "
                        "the sub-task."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "subagent timeout in seconds (default 120)."
                    ),
                    "default": 120,
                },
                "model": {
                    "type": "string",
                    "description": (
                        "model name for the subagent (optional; "
                        "defaults to the main agent's)."
                    ),
                },
                "system_prompt": {
                    "type": "string",
                    "description": (
                        "override the subagent's system prompt to "
                        "specialize its role (optional)."
                    ),
                },
                "max_turns": {
                    "type": "integer",
                    "description": (
                        "max subagent turns (default 10, cap 50)."
                    ),
                },
            },
            "required": ["task"],
        },
        call_fn=_delegate,
    )

    registry.register_mcp_tool(
        name="delegate_parallel",
        description=(
            "Delegate multiple sub-tasks in parallel with a cap on "
            "max concurrency. Each element is a task config supporting "
            "task/tools/context_files/timeout/model/system_prompt/"
            "max_turns. Returns a JSON array of task results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "list of sub-task configs.",
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": (
                        "max concurrency (optional; defaults to config "
                        "delegation.max_concurrent)."
                    ),
                },
            },
            "required": ["tasks"],
        },
        call_fn=_delegate_parallel,
    )
