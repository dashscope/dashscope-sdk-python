# -*- coding: utf-8 -*-
"""Shared runtime wiring for programmatic and embedded Agent entrypoints.

The interactive CLI owns presentation and lifecycle, but every entrypoint must
construct the same capability surface. Keeping this wiring here prevents SDK
and embedded one-shot modes from silently losing memory, session tools, vision,
skills, evolution, or delegation.
"""
from __future__ import annotations

# pylint: disable=unused-import,protected-access

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dashscope.acli.agent import Agent
    from dashscope.acli.config import Config
    from dashscope.acli.hooks import HookBus
    from dashscope.acli.platforms.base import MemoryProvider


def import_builtin_tools() -> None:
    """Populate the global registry with all built-in static tools."""
    import dashscope.acli.tools.browser  # noqa: F401
    import dashscope.acli.tools.camera  # noqa: F401
    import dashscope.acli.tools.filesystem  # noqa: F401
    import dashscope.acli.tools.shell  # noqa: F401
    import dashscope.acli.tools.web_search  # noqa: F401


def prepare_extensions(config: Config) -> None:
    """Load declarative providers, capabilities, shell tools, and skills."""
    from dashscope.acli.config import PROVIDER_MODELS
    from dashscope.acli.extensions import apply_extensions

    del config  # kept in the signature for a stable runtime-prep contract
    apply_extensions(PROVIDER_MODELS)


def create_runtime_memory(config: Config) -> MemoryProvider | None:
    """Create and publish the configured memory provider, if enabled."""
    if not config.memory_enabled:
        return None
    from dashscope.acli.cli.handlers_profile import set_memory_client
    from dashscope.acli.platforms import get_memory_provider

    memory = get_memory_provider(config)
    if memory is not None:
        set_memory_client(memory)
    return memory


def wire_agent_runtime(
    config: Config,
    agent: Agent,
    hook_bus: HookBus | None = None,
) -> None:
    """Attach every dynamic tool/capability family to an existing Agent."""
    import_builtin_tools()

    from dashscope.acli.agents.delegate import (
        set_config as set_delegate_config,
    )
    from dashscope.acli.agents.delegate import (
        set_parent_agent as set_delegate_parent,
    )
    from dashscope.acli.agents.subagent import (
        set_config as set_subagent_config,
    )
    from dashscope.acli.agents.subagent import (
        set_parent_agent as set_subagent_parent,
    )

    set_subagent_parent(agent)
    set_subagent_config(config)
    set_delegate_parent(agent)
    set_delegate_config(config)

    from dashscope.acli.cli.mcp import (
        _connect_mcp,
        _disconnect_mcp,
        _mcp_clients,
    )
    from dashscope.acli.providers import get_provider
    from dashscope.acli.skills import get_skill_manager, list_known_services
    from dashscope.acli.tools.platform import register_platform_tools

    register_platform_tools(config, connect_mcp_fn=_connect_mcp)

    skill_manager = get_skill_manager()
    skill_manager._registry_url = getattr(config, "skill_registry", "")
    skill_manager._global = False
    skill_manager.load(hook_bus=hook_bus or agent.hook_bus)

    from dashscope.acli.tools.session import register_session_tools

    register_session_tools(
        config,
        get_agent=lambda: agent,
        get_provider_fn=get_provider,
        connect_mcp_fn=_connect_mcp,
        disconnect_mcp_fn=_disconnect_mcp,
        list_mcp_services_fn=list_known_services,
        get_mcp_clients_fn=lambda: _mcp_clients,
    )

    from dashscope.acli.tools.vision import register_vision_tools

    register_vision_tools(config, get_provider_fn=get_provider)

    from dashscope.acli.tools.evolution import register_evolution_tools
    from dashscope.acli.tools.skills import register_skill_tools

    register_evolution_tools(get_agent=lambda: agent)
    register_skill_tools(get_agent=lambda: agent)
