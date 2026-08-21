# -*- coding: utf-8 -*-
"""Session/config tools — let the agent switch models, manage capabilities,
and connect MCP services on behalf of the user."""
# pylint: disable=too-many-statements

from __future__ import annotations

from typing import Callable

from dashscope.acli.config import PROVIDER_MODELS, Config, normalize_model_name
from dashscope.acli.tools.registry import (
    PermissionLevel,
    ToolDefinition,
    registry,
)


def register_session_tools(
    config: Config,
    *,
    get_agent: Callable,
    get_provider_fn: Callable,
    connect_mcp_fn: Callable,
    disconnect_mcp_fn: Callable,
    list_mcp_services_fn: Callable,
    get_mcp_clients_fn: Callable,
):
    """Register tools that let the LLM manage session state.

    Called once after agent construction. Uses closures over runtime objects.
    """

    # ===== Model / Provider =====

    async def switch_model(model_name: str) -> str:
        agent = get_agent()
        config.model = normalize_model_name(model_name)
        agent.provider = get_provider_fn(
            config.provider,
            config.model,
            config.api_key,
            base_url=config.base_url or None,
        )
        agent.model_name = config.model
        config.save_workspace()
        return f"Switched model: {config.provider}/{config.model}"

    async def switch_provider(provider_name: str) -> str:
        if provider_name not in PROVIDER_MODELS:
            return (
                f"Unknown provider: {provider_name}; "
                f"options: {', '.join(PROVIDER_MODELS)}"
            )
        agent = get_agent()
        config.provider = provider_name
        config.model = PROVIDER_MODELS[provider_name][0]
        agent.provider = get_provider_fn(
            config.provider,
            config.model,
            config.api_key,
            base_url=config.base_url or None,
        )
        agent.provider_name = config.provider
        agent.model_name = config.model
        config.save_global()
        config.save_workspace()
        return f"Switched: {provider_name}/{config.model}"

    async def list_models() -> str:
        lines = []
        for provider, models in PROVIDER_MODELS.items():
            marker = " ← current" if provider == config.provider else ""
            lines.append(f"{provider}{marker}: {', '.join(models)}")
        return "\n".join(lines)

    registry.register(
        ToolDefinition(
            name="switch_model",
            description=(
                "Switch the current AI model "
                "(e.g. qwen-plus, claude-sonnet-4-20250514)"
            ),
            permission=PermissionLevel.AUTO,
            func=switch_model,
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name",
                    },
                },
                "required": ["model_name"],
            },
        ),
    )
    registry.register(
        ToolDefinition(
            name="switch_provider",
            description=(
                "Switch AI provider "
                "(tongyi/anthropic/openai/deepseek/zhipu)"
            ),
            permission=PermissionLevel.AUTO,
            func=switch_provider,
            parameters={
                "type": "object",
                "properties": {
                    "provider_name": {
                        "type": "string",
                        "description": "Provider name",
                    },
                },
                "required": ["provider_name"],
            },
        ),
    )
    registry.register_mcp_tool(
        name="list_models",
        description="List all available AI models and providers",
        parameters={"type": "object", "properties": {}},
        call_fn=list_models,
    )

    # ===== Capability =====

    async def capability_list() -> str:
        from dashscope.acli.tools.platform import all_capability_keys

        all_keys = all_capability_keys()
        enabled = (
            set(config.enabled_capabilities)
            if config.enabled_capabilities is not None
            else set(all_keys)
        )
        lines = []
        for k in all_keys:
            status = "✓ enabled" if k in enabled else "✗ disabled"
            lines.append(f"  {k} — {status}")
        return "Capabilities:\n" + "\n".join(lines)

    async def capability_enable(key: str) -> str:
        from dashscope.acli.tools.platform import (
            _is_cloud_capability,
            all_capability_keys,
            register_one_capability,
        )

        all_keys = all_capability_keys()
        if key not in all_keys:
            return (
                f"Unknown capability: {key}; "
                f"options: {', '.join(all_keys)}"
            )
        if config.privacy_mode and _is_cloud_capability(key):
            return (
                f"Privacy mode is on; cannot enable cloud capability "
                f"{key}. Run /privacy off first to use it."
            )
        enabled = (
            set(config.enabled_capabilities)
            if config.enabled_capabilities is not None
            else set(all_keys)
        )
        if key in enabled:
            return f"{key} is already enabled"
        enabled.add(key)
        config.enabled_capabilities = sorted(enabled)
        config.save_workspace()
        count = register_one_capability(config, key, connect_mcp_fn)
        if count == 0:
            return (
                f"Enabled {key} but registered 0 tools — credentials "
                f"or runtime requirements are likely missing (e.g. bl "
                f"not in PATH). The agent cannot add credentials "
                f"interactively; tell the user to run "
                f"/capability enable {key} to finish setup."
            )
        return f"Enabled {key} (registered {count} tools)"

    async def capability_disable(key: str) -> str:
        from dashscope.acli.tools.platform import (
            all_capability_keys,
            unregister_capability_tools,
        )

        all_keys = all_capability_keys()
        if key not in all_keys:
            return (
                f"Unknown capability: {key}; "
                f"options: {', '.join(all_keys)}"
            )
        enabled = (
            set(config.enabled_capabilities)
            if config.enabled_capabilities is not None
            else set(all_keys)
        )
        if key not in enabled:
            return f"{key} is already disabled"
        enabled.discard(key)
        config.enabled_capabilities = sorted(enabled)
        config.save_workspace()
        unregister_capability_tools(key)
        return f"Disabled {key}"

    registry.register_mcp_tool(
        name="capability_list",
        description="List all capabilities and their enabled status",
        parameters={"type": "object", "properties": {}},
        call_fn=capability_list,
    )
    registry.register(
        ToolDefinition(
            name="capability_enable",
            description=(
                "Enable a cloud capability "
                "(e.g. knowledge base, web search, MCP)"
            ),
            permission=PermissionLevel.CONFIRM,
            func=capability_enable,
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Capability key, e.g. bailian.mcp, bailian.cli"
                        ),
                    },
                },
                "required": ["key"],
            },
        ),
    )
    registry.register(
        ToolDefinition(
            name="capability_disable",
            description="Disable a cloud capability",
            permission=PermissionLevel.CONFIRM,
            func=capability_disable,
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Capability key",
                    },
                },
                "required": ["key"],
            },
        ),
    )

    # ===== MCP =====

    async def mcp_list_services() -> str:
        return list_mcp_services_fn()

    async def mcp_connect(service: str) -> str:
        clients = get_mcp_clients_fn()
        if service in clients:
            return (
                f"{service} already connected "
                f"({len(clients[service].tools)} tools)"
            )
        error = await connect_mcp_fn(service, config)
        if error:
            return f"Failed to connect {service}: {error}"
        clients = get_mcp_clients_fn()
        client = clients.get(service)
        tool_count = len(client.tools) if client else 0
        from dashscope.acli.config import MCPServerConfig

        if not any(m.service == service for m in config.mcp_servers):
            config.mcp_servers.append(MCPServerConfig(service=service))
            config.save_workspace()
        return f"Connected {service}; found {tool_count} tools"

    async def mcp_disconnect(service: str) -> str:
        clients = get_mcp_clients_fn()
        if service not in clients:
            return f"{service} is not connected"
        await disconnect_mcp_fn(service)
        config.mcp_servers = [
            m for m in config.mcp_servers if m.service != service
        ]
        config.save_workspace()
        return f"Disconnected {service}"

    registry.register_mcp_tool(
        name="mcp_list_services",
        description=(
            "List available MCP cloud services "
            "(time, code execution, doc parsing, etc.)"
        ),
        parameters={"type": "object", "properties": {}},
        call_fn=mcp_list_services,
    )
    registry.register(
        ToolDefinition(
            name="mcp_connect",
            description=(
                "Connect an MCP cloud service "
                "(its tools become available once connected)"
            ),
            permission=PermissionLevel.CONFIRM,
            func=mcp_connect,
            parameters={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": (
                            "Service name, e.g. time, "
                            "code-interpreter, doc-analysis"
                        ),
                    },
                },
                "required": ["service"],
            },
        ),
    )
    registry.register(
        ToolDefinition(
            name="mcp_disconnect",
            description="Disconnect an MCP cloud service",
            permission=PermissionLevel.CONFIRM,
            func=mcp_disconnect,
            parameters={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service name",
                    },
                },
                "required": ["service"],
            },
        ),
    )

    # ===== Plan Management =====

    async def create_plan(goal: str, steps: list[str]) -> str:
        """Create a multi-step execution plan for complex tasks."""
        return get_agent().memory_manager.session.plan.create_plan(goal, steps)

    async def complete_step(step_index: int) -> str:
        """Mark a plan step as completed."""
        tracker = get_agent().memory_manager.session.plan
        tracker.mark_step_complete(step_index)
        if tracker.is_complete():
            tracker.clear_plan()
            return "All steps completed"
        return f"Step {step_index + 1} completed"

    async def get_plan_status() -> str:
        """Get current plan progress."""
        section = get_agent().memory_manager.session.plan.get_plan_section()
        return section if section else "No plan in progress"

    registry.register(
        ToolDefinition(
            name="create_plan",
            description=(
                "Create a multi-step execution plan for complex "
                "tasks (e.g. refactors, designs, new features)."
            ),
            permission=PermissionLevel.AUTO,
            func=create_plan,
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Overall goal of the plan",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of step descriptions",
                    },
                },
                "required": ["goal", "steps"],
            },
        ),
    )
    registry.register(
        ToolDefinition(
            name="complete_step",
            description="Mark a plan step as completed",
            permission=PermissionLevel.AUTO,
            func=complete_step,
            parameters={
                "type": "object",
                "properties": {
                    "step_index": {
                        "type": "integer",
                        "description": "Step index (0-based)",
                    },
                },
                "required": ["step_index"],
            },
        ),
    )
    registry.register(
        ToolDefinition(
            name="get_plan_status",
            description="Get progress of the current plan",
            permission=PermissionLevel.AUTO,
            func=get_plan_status,
            parameters={"type": "object", "properties": {}},
        ),
    )
