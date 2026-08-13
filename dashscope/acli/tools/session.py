"""Session/config tools — let the agent switch models, manage capabilities,
and connect MCP services on behalf of the user."""

from __future__ import annotations

from typing import Callable

from dashscope.acli.config import PROVIDER_MODELS, Config, normalize_model_name
from dashscope.acli.tools.registry import PermissionLevel, ToolDefinition, registry


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
        return f"已切换模型: {config.provider}/{config.model}"

    async def switch_provider(provider_name: str) -> str:
        if provider_name not in PROVIDER_MODELS:
            return f"未知 Provider: {provider_name}，可选: {', '.join(PROVIDER_MODELS)}"
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
        return f"已切换: {provider_name}/{config.model}"

    async def list_models() -> str:
        lines = []
        for provider, models in PROVIDER_MODELS.items():
            marker = " ← 当前" if provider == config.provider else ""
            lines.append(f"{provider}{marker}: {', '.join(models)}")
        return "\n".join(lines)

    registry.register(
        ToolDefinition(
            name="switch_model",
            description="切换当前使用的 AI 模型（如 qwen-plus、claude-sonnet-4-20250514）",
            permission=PermissionLevel.AUTO,
            func=switch_model,
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "模型名称"},
                },
                "required": ["model_name"],
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="switch_provider",
            description="切换 AI 服务商（tongyi/anthropic/openai/deepseek/zhipu）",
            permission=PermissionLevel.AUTO,
            func=switch_provider,
            parameters={
                "type": "object",
                "properties": {
                    "provider_name": {"type": "string", "description": "Provider 名称"},
                },
                "required": ["provider_name"],
            },
        )
    )
    registry.register_mcp_tool(
        name="list_models",
        description="列出所有可用的 AI 模型和服务商",
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
            status = "✓ 已启用" if k in enabled else "✗ 未启用"
            lines.append(f"  {k} — {status}")
        return "能力列表:\n" + "\n".join(lines)

    async def capability_enable(key: str) -> str:
        from dashscope.acli.tools.platform import (
            _is_cloud_capability,
            all_capability_keys,
            register_one_capability,
        )

        all_keys = all_capability_keys()
        if key not in all_keys:
            return f"未知能力: {key}，可选: {', '.join(all_keys)}"
        if config.privacy_mode and _is_cloud_capability(key):
            return f"隐私模式已启用，无法开启云端能力 {key}；如需使用请先 /privacy off"
        enabled = (
            set(config.enabled_capabilities)
            if config.enabled_capabilities is not None
            else set(all_keys)
        )
        if key in enabled:
            return f"{key} 已处于启用状态"
        enabled.add(key)
        config.enabled_capabilities = sorted(enabled)
        config.save_workspace()
        count = register_one_capability(config, key, connect_mcp_fn)
        if count == 0:
            return (
                f"已启用 {key}，但注册了 0 个工具——很可能缺少凭证或运行条件（如 bl 不在 PATH）。"
                f"agent 无法交互式补录凭证，请告知用户运行 /capability enable {key} 手动补全。"
            )
        return f"已启用 {key}（注册了 {count} 个工具）"

    async def capability_disable(key: str) -> str:
        from dashscope.acli.tools.platform import all_capability_keys, unregister_capability_tools

        all_keys = all_capability_keys()
        if key not in all_keys:
            return f"未知能力: {key}，可选: {', '.join(all_keys)}"
        enabled = (
            set(config.enabled_capabilities)
            if config.enabled_capabilities is not None
            else set(all_keys)
        )
        if key not in enabled:
            return f"{key} 已处于禁用状态"
        enabled.discard(key)
        config.enabled_capabilities = sorted(enabled)
        config.save_workspace()
        unregister_capability_tools(key)
        return f"已禁用 {key}"

    registry.register_mcp_tool(
        name="capability_list",
        description="列出所有可用能力及其启用状态",
        parameters={"type": "object", "properties": {}},
        call_fn=capability_list,
    )
    registry.register(
        ToolDefinition(
            name="capability_enable",
            description="启用一项云端能力（如知识库、联网搜索、MCP 等）",
            permission=PermissionLevel.CONFIRM,
            func=capability_enable,
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "能力标识，如 bailian.mcp、bailian.cli",
                    },
                },
                "required": ["key"],
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="capability_disable",
            description="禁用一项云端能力",
            permission=PermissionLevel.CONFIRM,
            func=capability_disable,
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "能力标识"},
                },
                "required": ["key"],
            },
        )
    )

    # ===== MCP =====

    async def mcp_list_services() -> str:
        return list_mcp_services_fn()

    async def mcp_connect(service: str) -> str:
        clients = get_mcp_clients_fn()
        if service in clients:
            return f"{service} 已连接（{len(clients[service].tools)} 个工具）"
        error = await connect_mcp_fn(service, config)
        if error:
            return f"连接 {service} 失败: {error}"
        clients = get_mcp_clients_fn()
        client = clients.get(service)
        tool_count = len(client.tools) if client else 0
        from dashscope.acli.config import MCPServerConfig

        if not any(m.service == service for m in config.mcp_servers):
            config.mcp_servers.append(MCPServerConfig(service=service))
            config.save_workspace()
        return f"已连接 {service}，发现 {tool_count} 个工具"

    async def mcp_disconnect(service: str) -> str:
        clients = get_mcp_clients_fn()
        if service not in clients:
            return f"{service} 未连接"
        await disconnect_mcp_fn(service)
        config.mcp_servers = [m for m in config.mcp_servers if m.service != service]
        config.save_workspace()
        return f"已断开 {service}"

    registry.register_mcp_tool(
        name="mcp_list_services",
        description="列出可用的 MCP 云端服务（时间、代码执行、文档解析等）",
        parameters={"type": "object", "properties": {}},
        call_fn=mcp_list_services,
    )
    registry.register(
        ToolDefinition(
            name="mcp_connect",
            description="连接一个 MCP 云端服务（连接后其工具自动可用）",
            permission=PermissionLevel.CONFIRM,
            func=mcp_connect,
            parameters={
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "服务名称，如 time、code-interpreter、doc-analysis",
                    },
                },
                "required": ["service"],
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="mcp_disconnect",
            description="断开一个 MCP 云端服务",
            permission=PermissionLevel.CONFIRM,
            func=mcp_disconnect,
            parameters={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "服务名称"},
                },
                "required": ["service"],
            },
        )
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
            return "所有步骤已完成"
        return f"步骤 {step_index + 1} 已完成"

    async def get_plan_status() -> str:
        """Get current plan progress."""
        section = get_agent().memory_manager.session.plan.get_plan_section()
        return section if section else "当前没有进行中的计划"

    registry.register(
        ToolDefinition(
            name="create_plan",
            description="为复杂任务创建多步骤执行计划。用于需要多步操作的任务（如重构、设计、实现功能）。",
            permission=PermissionLevel.AUTO,
            func=create_plan,
            parameters={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "计划的总体目标",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "步骤描述列表",
                    },
                },
                "required": ["goal", "steps"],
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="complete_step",
            description="标记计划中的一个步骤为已完成",
            permission=PermissionLevel.AUTO,
            func=complete_step,
            parameters={
                "type": "object",
                "properties": {
                    "step_index": {
                        "type": "integer",
                        "description": "步骤索引（从 0 开始）",
                    },
                },
                "required": ["step_index"],
            },
        )
    )
    registry.register(
        ToolDefinition(
            name="get_plan_status",
            description="获取当前计划的执行进度",
            permission=PermissionLevel.AUTO,
            func=get_plan_status,
            parameters={"type": "object", "properties": {}},
        )
    )
