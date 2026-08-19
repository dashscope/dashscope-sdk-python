# -*- coding: utf-8 -*-
"""Embedded mode — run acli as a library inside another CLI application.

Usage:
    from dashscope.acli.ui.embedded import run

    run(
        system_prompt="You are ...",
        app_name="My App",
        default_model="qwen3.7-plus",
        default_provider="tongyi",
        api_key="sk-...",
    )

The full acli interactive loop is reused (completions, slash commands,
keybindings, /skill menus, etc.). Only identity and defaults are overridden.
"""
# pylint: disable=protected-access,unused-import
from __future__ import annotations

import asyncio
import sys
from typing import Optional


def run(
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
):
    """Run the full acli agent loop with a custom identity.

    Constructs an acli Config with the caller's overrides, then delegates
    to acli's complete interactive loop (slash commands, completions,
    keybindings, /skill, /provider sub-menus — everything).

    Args:
        system_prompt: The system prompt. If None, loads from
            .acli/system-prompt.md.
        app_name: Display name shown in the banner.
        default_model: Default LLM model name.
        default_provider: Default provider (tongyi/anthropic/openai/...).
        api_key: API key. If None, loads from acli config or env.
        base_url: Custom base URL for the provider.
        command: If set, run one-shot mode with this prompt then exit.
        prompt_symbol: The input prompt symbol shown to the user.
        sdk_index: List of SDK index files loaded (e.g., ["python-sdk",
            "python-cli"]).
        tui: If set, override config.tui. None = use config value.
    """
    from dashscope.acli.config import Config

    # Pass caller's defaults as initial values — Config.load() will let
    # workspace/global config and env vars override them naturally.
    config = Config.load(
        default_provider=default_provider,
        default_model=default_model,
    )

    # Explicit api_key from caller fills in MISSING key; saved config
    # takes priority.
    if api_key and not config.api_key:
        config.api_key = api_key
    if base_url and not config.base_url:
        config.base_url = base_url

    # Inject custom system prompt and display settings into config
    # so _run_loop can pick them up.
    if system_prompt:
        config._embedded_system_prompt = system_prompt
    config._embedded_app_name = app_name
    config._embedded_prompt_symbol = prompt_symbol
    if sdk_index:
        config._embedded_sdk_index = sdk_index
    if tui is not None:
        config.tui = tui

    try:
        if command:
            asyncio.run(_run_oneshot_embedded(config, command, system_prompt))
        elif config.tui:
            from dashscope.acli.cli import _run_tui_mode

            _run_tui_mode(config)
        else:
            asyncio.run(_run_loop_embedded(config))
    except (KeyboardInterrupt, SystemExit):
        pass


async def _run_oneshot_embedded(
    config,
    prompt: str,
    system_prompt: str | None,
):
    """One-shot mode using acli's full agent stack."""
    import dashscope.acli.tools.filesystem  # noqa: F401
    import dashscope.acli.tools.shell  # noqa: F401
    from dashscope.acli.agent import Agent
    from dashscope.acli.cli import _compose_system_prompt, _load_system_prompt
    from dashscope.acli.executor import Executor
    from dashscope.acli.providers import get_provider_chain

    resolved_prompt = _compose_system_prompt(
        system_prompt or _load_system_prompt(),
    )
    provider = get_provider_chain(config)
    executor = Executor()
    agent = Agent(
        provider=provider,
        executor=executor,
        max_turns=config.max_turns,
        provider_name=config.provider,
        model_name=config.model,
        system_prompt=resolved_prompt,
    )

    async for chunk in agent.run_stream(prompt):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")


async def _run_loop_embedded(config):
    """Delegate to acli's full interactive loop."""
    from dashscope.acli.cli import _run_loop

    await _run_loop(config)
