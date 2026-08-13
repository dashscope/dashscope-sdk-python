# -*- coding: utf-8 -*-
"""Runner functions for different execution modes."""
# pylint: disable=protected-access,too-many-branches,too-many-statements

from __future__ import annotations

import sys

from rich.console import Console

from dashscope.acli.agent import Agent
from dashscope.acli.cli.handlers_key import ensure_provider_key
from dashscope.acli.cli.multimodal import (
    _expand_at_references,
    _to_multimodal_content,
)
from dashscope.acli.cli.startup import (
    _compose_system_prompt,
    _load_system_prompt,
)
from dashscope.acli.config import (
    PROVIDER_MODELS,
    Config,
    is_audio_model,
    is_vision_model,
)
from dashscope.acli.dev import _apply_custom_models
from dashscope.acli.executor import Executor
from dashscope.acli.providers import get_provider, get_provider_chain
from dashscope.acli.session import get_session_manager
from dashscope.acli.skills import list_known_services

console = Console()


async def _run_oneshot(config: Config, prompt: str):
    """Non-interactive single-shot mode: run one prompt and exit."""
    _apply_custom_models(config)
    # Re-arm audit redaction from persisted config (restart must not drop it).
    from dashscope.acli.audit import configure_audit_logger

    configure_audit_logger(config)
    from dashscope.acli.debuglog import configure_debug_log

    configure_debug_log(config)
    from dashscope.acli.permission import configure_permission_policy

    configure_permission_policy()
    from dashscope.acli.extensions import apply_extensions

    apply_extensions(PROVIDER_MODELS)

    if not config.api_key:
        from dashscope.acli.cli.handlers_key import all_key_targets

        targets = all_key_targets(config)
        key_info = targets.get(config.provider)
        env_hint = (
            key_info.get("env")
            if key_info
            else f"{config.provider.upper()}_API_KEY"
        )
        print(f"错误: 未找到 {config.provider} 的 API Key")
        print(f"  1) 设置环境变量: export {env_hint}=sk-xxx")
        print("  2) 启动后设置:   /provider")
        print("  3) 交互式设置:   /setup")
        sys.exit(1)

    provider = get_provider_chain(config)
    executor = Executor()
    agent = Agent(
        provider=provider,
        executor=executor,
        max_turns=config.max_turns,
        provider_name=config.provider,
        model_name=config.model,
    )

    # Pin parent agent ref for local.subagent / local.delegate BEFORE platform
    # tool registration so register_one_capability finds a parent to attach to
    # (same ordering as cli/repl.py).
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

    from dashscope.acli.cli.mcp import _connect_mcp
    from dashscope.acli.tools.platform import register_platform_tools

    register_platform_tools(config, connect_mcp_fn=_connect_mcp)

    expanded, images, audio_clips = _expand_at_references(prompt)
    if images and not is_vision_model(config.model):
        console.print(
            f"[yellow]当前模型 {config.model} 不支持图片，"
            f"{len(images)} 张图片已忽略。[/yellow]",
        )
        images = []
    if audio_clips and not is_audio_model(config.model):
        console.print(
            f"[yellow]当前模型 {config.model} 不支持音频，"
            f"{len(audio_clips)} 段音频已忽略。[/yellow]",
        )
        audio_clips = []
    agent_input = _to_multimodal_content(expanded, images, audio_clips)

    async for chunk in agent.run_stream(agent_input):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")


def _run_dry_run(config: Config):
    """Preview configuration without starting the agent.

    Shows what would be loaded: provider, model, skills, MCP services,
    tools, capabilities, and overall readiness status.
    """
    from rich.table import Table

    console.print("\n[bold cyan]acli --dry-run[/bold cyan]  配置预览\n")

    # 1. Provider & Model
    provider_table = Table(title="Provider 配置", show_header=False, box=None)
    provider_table.add_column("Key", style="cyan")
    provider_table.add_column("Value", style="green")
    provider_table.add_row("Provider", config.provider)
    provider_table.add_row("Model", config.model)
    provider_table.add_row("Protocol", config.protocol or "openai")
    provider_table.add_row("API Key", "✓ 已配置" if config.api_key else "✗ 未配置")
    if config.base_url:
        provider_table.add_row("Base URL", config.base_url)
    console.print(provider_table)
    console.print()

    # 2. Skills
    from dashscope.acli.skills.base import BUILTIN_SKILLS, load_skill_files

    load_skill_files()
    skills = list(BUILTIN_SKILLS.values())
    if skills:
        skill_table = Table(
            title=f"已加载 Skills ({len(skills)})",
            show_header=True,
        )
        skill_table.add_column("名称", style="cyan")
        skill_table.add_column("描述", style="dim")
        for skill in skills:
            skill_table.add_row(skill.name, skill.description or "")
        console.print(skill_table)
    else:
        console.print("[yellow]Skills:[/yellow] 无\n")
    console.print()

    # 3. MCP Services
    mcp_servers = getattr(config, "mcp_servers", None) or []
    if mcp_servers:
        mcp_table = Table(
            title=f"MCP 服务 ({len(mcp_servers)})",
            show_header=True,
        )
        mcp_table.add_column("名称", style="cyan")
        mcp_table.add_column("URL", style="dim")
        mcp_table.add_column("状态", style="green")
        for svc in mcp_servers:
            if isinstance(svc, dict):
                name, url = svc.get("service", "?"), svc.get("url", "?")
            else:
                name, url = svc.service, svc.url
            mcp_table.add_row(name or "?", url or "?", "待连接")
        console.print(mcp_table)
    else:
        console.print("[yellow]MCP 服务:[/yellow] 无\n")
    console.print()

    # 4. Tools
    from dashscope.acli.tools.registry import registry

    tools = registry.list_tools()
    tool_table = Table(title=f"可用 Tools ({len(tools)})", show_header=True)
    tool_table.add_column("名称", style="cyan")
    tool_table.add_column("描述", style="dim")
    for tool in tools[:20]:  # Show first 20 tools
        tool_table.add_row(tool.name, (tool.description or "")[:60])
    if len(tools) > 20:
        tool_table.add_row("...", f"(还有 {len(tools) - 20} 个工具)")
    console.print(tool_table)
    console.print()

    # 5. Capabilities
    caps_table = Table(title="Capability 状态", show_header=False, box=None)
    caps_table.add_column("Capability", style="cyan")
    caps_table.add_column("状态", style="green")
    caps_table.add_row(
        "Memory",
        "✓ 启用" if getattr(config, "memory_enabled", False) else "✗ 禁用",
    )
    caps_table.add_row(
        "Session Persist",
        "✓ 启用" if config.session_persist else "✗ 禁用",
    )
    caps_table.add_row(
        "Auto Approve",
        "✓ 启用" if config.auto_approve else "✗ 禁用",
    )
    console.print(caps_table)
    console.print()

    # 6. Readiness assessment
    issues = []
    if not config.api_key:
        issues.append("API Key 未配置")
    if not tools:
        issues.append("无可用工具")

    if issues:
        console.print("[bold red]⚠ Readiness: Warning[/bold red]")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("[bold green]✓ Readiness: Ready[/bold green]")

    console.print(
        "\n[dim]这是配置预览，agent 未启动。移除 --dry-run 以正常启动。[/dim]\n",
    )


def _run_tui_mode(config: Config):
    """Run AgenticCLI with Textual TUI (fixed input at bottom)."""
    _apply_custom_models(config)
    # Re-arm audit redaction from persisted config (restart must not drop it).
    from dashscope.acli.audit import configure_audit_logger

    configure_audit_logger(config)
    from dashscope.acli.debuglog import configure_debug_log

    configure_debug_log(config)
    from dashscope.acli.permission import configure_permission_policy

    configure_permission_policy()
    from dashscope.acli.extensions import apply_extensions

    _ext = apply_extensions(PROVIDER_MODELS)
    from dashscope.acli.cli.handlers_capability import (
        sync_extensions_into_catalog,
    )

    sync_extensions_into_catalog(_ext)

    # CLI mode uses the full provider chain so temporary failures can fall back
    # to configured alternatives instead of dying immediately.
    provider = get_provider_chain(config)
    executor = Executor(auto_approve=config.auto_approve)

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
    from dashscope.acli.hooks import create_hook_bus
    from dashscope.acli.platforms import get_memory_provider
    from dashscope.acli.tools.platform import disabled_capabilities_hint

    # Initialize memory only when enabled (same as CLI mode)
    memory = None
    if config.memory_enabled:
        memory = get_memory_provider(config)
    hook_bus = create_hook_bus()

    session_manager = get_session_manager()
    session_path = (
        session_manager.get_history_path() if config.session_persist else None
    )

    # user_name is display-only; no longer derived from memory_user_id.
    user_name = config.user_name

    agent = Agent(
        provider=provider,
        executor=executor,
        max_turns=config.max_turns,
        memory=memory,
        user_name=user_name,
        provider_name=config.provider,
        model_name=config.model,
        session_path=session_path,
        disabled_caps_provider=lambda: disabled_capabilities_hint(config),
        directives_provider=lambda: config.user_directives,
        system_prompt=_compose_system_prompt(
            getattr(config, "_embedded_system_prompt", None)
            or _load_system_prompt(),
        ),
        hook_bus=hook_bus,
    )

    set_subagent_parent(agent)
    set_subagent_config(config)
    set_delegate_parent(agent)
    set_delegate_config(config)

    from dashscope.acli.cli.mcp import _connect_mcp
    from dashscope.acli.tools.platform import register_platform_tools

    register_platform_tools(config, connect_mcp_fn=_connect_mcp)

    from dashscope.acli.skills import get_skill_manager

    skill_manager = get_skill_manager()
    skill_manager._registry_url = getattr(config, "skill_registry", "")
    skill_manager._global = False
    skill_manager.load(hook_bus=hook_bus)

    from dashscope.acli.cli.mcp import _disconnect_mcp, _mcp_clients
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

    from dashscope.acli.tools.evolution import register_evolution_tools

    register_evolution_tools(get_agent=lambda: agent)

    from dashscope.acli.tools.skills import register_skill_tools

    register_skill_tools(get_agent=lambda: agent)

    # Check API key before launching TUI
    ensure_provider_key(config, agent)

    try:
        from dashscope.acli.ui.tui import run_tui
    except ImportError:
        console.print(
            "[yellow]TUI 需要 textual 依赖，请先安装: pip install textual"
            "（或使用 CLI 模式: --cli）[/yellow]",
        )
        return

    # Run Textual TUI
    input_history_path = (
        session_manager.get_input_history_path()
        if config.session_persist
        else None
    )
    run_tui(config, agent, input_history_path=input_history_path)
