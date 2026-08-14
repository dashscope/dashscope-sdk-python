# -*- coding: utf-8 -*-
"""MCP (Model Context Protocol) client management for acli CLI."""

from __future__ import annotations

from rich.console import Console
from rich.status import Status

from dashscope.acli.config import Config, MCPServerConfig
from dashscope.acli.platforms.bailian import MCPClient, MCPError
from dashscope.acli.skills import list_known_services
from dashscope.acli.tools.registry import registry

console = Console()

# Active MCP clients - shared state
_mcp_clients: dict[str, MCPClient] = {}


async def _connect_mcp(service: str, config: Config, url: str = "") -> str:
    """Connect to an MCP service and register its tools.

    Returns empty string on success, error message on failure."""
    if service in _mcp_clients:
        return ""

    try:
        client = MCPClient(
            service=service,
            api_key=config.tongyi_api_key,
            url=url,
        )
    except MCPError as e:
        return str(e)

    if not await client.initialize():
        error = client.last_error
        await client.close()
        return error or "Initialization failed"

    tools = await client.list_tools()
    for tool_info in tools:
        tool_name = f"mcp_{service}_{tool_info['name']}"
        description = tool_info.get("description", "")
        input_schema = tool_info.get(
            "inputSchema",
            {"type": "object", "properties": {}},
        )

        # Create closure for tool call
        _tool_original_name = tool_info["name"]
        _client = client

        async def _call(_cn=_tool_original_name, _cl=_client, **kwargs) -> str:
            return await _cl.call_tool(_cn, kwargs)

        registry.register_mcp_tool(
            name=tool_name,
            description=f"[MCP:{service}] {description}",
            parameters=input_schema,
            call_fn=_call,
        )

    # Also discover prompts/skills (optional, server may not support)
    await client.list_prompts()

    _mcp_clients[service] = client
    return ""


async def _disconnect_mcp(service: str):
    """Disconnect an MCP service and unregister its tools."""
    client = _mcp_clients.pop(service, None)
    if not client:
        return
    for tool_info in client.tools:
        registry.unregister(f"mcp_{service}_{tool_info['name']}")
    await client.close()


async def _handle_mcp_command(cmd: str, config: Config):
    """Handle /mcp commands (async)."""
    parts = cmd.strip().split()
    if len(parts) == 1:
        # /mcp — list connected services
        if not _mcp_clients:
            console.print("[dim]No MCP services connected[/dim]")
            console.print(
                "[dim]Use /mcp list to see available services, "
                "/mcp add <service> to add one[/dim]",
            )
        else:
            console.print("[bold]Connected MCP services:[/bold]")
            for svc, client in _mcp_clients.items():
                tool_count = len(client.tools)
                console.print(f"  {svc} — {tool_count} tools")
                for t in client.tools:
                    console.print(
                        f"    • {t['name']}: {t.get('description', '')[:60]}",
                    )
    elif parts[1] == "add" and len(parts) >= 3:
        service = parts[2]
        url = parts[3] if len(parts) > 3 else ""
        with Status(
            f"[dim]Connecting to MCP service {service}...[/dim]",
            console=console,
            spinner="aesthetic",
        ):
            error = await _connect_mcp(service, config, url=url)
        if not error:
            client = _mcp_clients[service]
            console.print(
                f"[green]Connected to {service}, found "
                f"{len(client.tools)} tools[/green]",
            )
            if not any(m.service == service for m in config.mcp_servers):
                config.mcp_servers.append(
                    MCPServerConfig(service=service, url=url),
                )
                config.save_workspace()
        else:
            console.print(f"[red]Failed to connect {service}: {error}[/red]")
    elif parts[1] == "remove" and len(parts) >= 3:
        service = parts[2]
        await _disconnect_mcp(service)
        config.mcp_servers = [
            m for m in config.mcp_servers if m.service != service
        ]
        config.save_workspace()
        console.print(f"[dim]Removed {service}[/dim]")
    elif parts[1] == "list":
        console.print(list_known_services())
    else:
        console.print(
            "[dim]Usage:\n"
            "  /mcp              — list connected services\n"
            "  /mcp list         — show Bailian available services\n"
            "  /mcp add <svc>    — add an MCP service\n"
            "  /mcp remove <svc> — remove an MCP service[/dim]",
        )


async def _init_mcp_servers(config: Config):
    """Connect to configured MCP servers on startup."""
    for mcp_cfg in config.mcp_servers:
        error = await _connect_mcp(mcp_cfg.service, config, url=mcp_cfg.url)
        if not error:
            client = _mcp_clients[mcp_cfg.service]
            summary = f"{len(client.tools)} tools"
            if client.prompts:
                summary += f", {len(client.prompts)} skills"
            console.print(
                f"[dim]MCP: {mcp_cfg.service} connected ({summary})[/dim]",
            )
        else:
            console.print(
                f"[yellow]MCP: {mcp_cfg.service} connection failed - "
                f"{error}[/yellow]",
            )
