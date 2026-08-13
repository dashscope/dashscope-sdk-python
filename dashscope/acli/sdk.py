# -*- coding: utf-8 -*-
"""Lightweight programmatic SDK for embedding acli in other applications.

This module exposes a stable, typed API surface on top of the lower-level
``Agent``, ``Executor``, and ``ProviderChain`` internals.  It is intended for
scripts, CI pipelines, IDE plugins, and other Python programs that want to
reuse acli's agent stack without invoking the full interactive CLI.

Example::

    import asyncio
    from dashscope.acli.sdk import run_once

    result = asyncio.run(run_once("Summarize README.md"))
    print(result)
"""
# pylint: disable=unused-import

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from dashscope.acli.agent import Agent
from dashscope.acli.config import Config
from dashscope.acli.executor import Executor
from dashscope.acli.providers import get_provider_chain


def _ensure_tools_imported() -> None:
    """Import tool modules so the registry is populated."""
    import dashscope.acli.tools.filesystem  # noqa: F401
    import dashscope.acli.tools.shell  # noqa: F401
    import dashscope.acli.tools.web_search  # noqa: F401


def load_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Config:
    """Load acli configuration with caller-supplied overrides.

    Workspace/global config and environment variables still take their normal
    priority; explicit arguments only fill in missing defaults.
    """
    config = Config.load(
        default_provider=provider,
        default_model=model,
    )
    if api_key and not config.api_key:
        config.api_key = api_key
    if base_url and not config.base_url:
        config.base_url = base_url
    return config


def create_agent(
    config: Optional[Config] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Agent:
    """Create an ``Agent`` instance ready for programmatic use.

    Args:
        config: acli configuration.  If None, loads default config.
        system_prompt: Optional system prompt override.
        session_id: Optional session identifier.  If provided, the agent's
            in-memory conversation is persisted under
            ``.acli/session/{session_id}/history.json``.
    """
    _ensure_tools_imported()
    config = config or load_config()

    from dashscope.acli.cli import _compose_system_prompt, _load_system_prompt

    resolved_prompt = _compose_system_prompt(
        system_prompt or _load_system_prompt(),
    )
    provider = get_provider_chain(config)
    executor = Executor()

    session_path: Optional[Path] = None
    if session_id:
        session_path = Path(".acli") / "session" / session_id / "history.json"

    return Agent(
        provider=provider,
        executor=executor,
        max_turns=config.max_turns,
        provider_name=config.provider,
        model_name=config.model,
        system_prompt=resolved_prompt,
        session_path=session_path,
    )


async def run_stream(
    prompt: str,
    config: Optional[Config] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream the agent's response for ``prompt`` as an async iterator."""
    agent = create_agent(
        config=config,
        system_prompt=system_prompt,
        session_id=session_id,
    )
    async for chunk in agent.run_stream(prompt):
        yield chunk


async def run_once(
    prompt: str,
    config: Optional[Config] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Run one non-interactive turn and return the full response text.

    This is the easiest entry point for scripts and CI: it streams the full
    agent turn, executes any tool calls, and returns the final assistant text.
    """
    parts: list[str] = []
    async for chunk in run_stream(
        prompt,
        config=config,
        system_prompt=system_prompt,
        session_id=session_id or str(uuid.uuid4())[:8],
    ):
        parts.append(chunk)
    return "".join(parts)


def run_once_sync(
    prompt: str,
    config: Optional[Config] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Synchronous wrapper around :func:`run_once`."""
    return asyncio.run(run_once(prompt, config, system_prompt, session_id))


def run_interactive(
    system_prompt: Optional[str] = None,
    app_name: str = "Agent",
    default_model: str = "qwen3.7-plus",
    default_provider: str = "tongyi",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    command: Optional[str] = None,
    prompt_symbol: str = "You> ",
    sdk_index: Optional[list[str]] = None,
    tui: Optional[bool] = None,
) -> None:
    """Run the full acli interactive loop with a custom identity.

    This is a thin SDK wrapper around the existing embedded mode; see
    ``acli.ui.embedded.run`` for details.
    """
    from dashscope.acli.ui.embedded import run as _embedded_run

    _embedded_run(
        system_prompt=system_prompt,
        app_name=app_name,
        default_model=default_model,
        default_provider=default_provider,
        api_key=api_key,
        base_url=base_url,
        command=command,
        prompt_symbol=prompt_symbol,
        sdk_index=sdk_index,
        tui=tui,
    )


__all__ = [
    "load_config",
    "create_agent",
    "run_stream",
    "run_once",
    "run_once_sync",
    "run_interactive",
]
