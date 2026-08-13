# -*- coding: utf-8 -*-
"""MCP Server mode: expose acli's tools to external MCP clients.

The design doc's Capability layer includes MCP as both a client (existing)
and a server. This module exposes acli's registered tools via the Model
Context Protocol so other MCP-compatible agents can call them.

Usage:
    acli mcp-server  # starts the MCP server on stdio

Implementation: converts ``ToolDefinition`` objects from the tool registry
into MCP tool schemas, and dispatches incoming MCP tool-call requests to
the registered handlers.
"""
# pylint: disable=unused-import

from __future__ import annotations

import json
import sys
from typing import Any

from dashscope.acli.tools.registry import registry


def tools_to_mcp_schema() -> list[dict[str, Any]]:
    """Convert registered tools to MCP tool schema format.

    Each tool becomes:
        {
            "name": "...",
            "description": "...",
            "inputSchema": { ... JSON schema ... }
        }
    """
    schemas: list[dict[str, Any]] = []
    for tool in registry.list_tools():
        schemas.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            },
        )
    return schemas


async def handle_mcp_request(
    request: dict[str, Any],
    executor: Any = None,
) -> dict[str, Any]:
    """Handle a single MCP JSON-RPC request.

    Supports:
      - ``tools/list``: return all registered tools
      - ``tools/call``: execute a tool by name

    Returns a JSON-RPC response dict.
    """
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools_to_mcp_schema()},
        }

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        tool_def = registry.get(tool_name)
        if not tool_def:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}",
                },
            }
        try:
            if executor:
                result = await executor.execute(tool_def, arguments)
            else:
                result = tool_def.func(**arguments)
                if hasattr(result, "__await__"):
                    result = await result
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": str(result)}]},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            }

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "acli-mcp-server", "version": "0.1.0"},
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


async def run_stdio_server(executor: Any = None) -> None:
    """Run the MCP server on stdio (JSON-RPC over stdin/stdout).

    Reads line-delimited JSON requests from stdin, writes responses to stdout.
    """
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue
        # JSON-RPC notifications (requests without an "id") MUST NOT be
        # answered per spec — skip them entirely.
        if "id" not in request:
            continue
        response = await handle_mcp_request(request, executor)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


def main() -> None:
    """Entry point for ``acli mcp-server``: run the MCP stdio server.

    Registers the builtin tools (same set the CLI imports at startup) and
    serves JSON-RPC over stdin/stdout. There is no interactive user at the
    console in this mode, so the executor auto-approves tool calls.
    """
    import asyncio

    import dashscope.acli.tools.browser  # noqa: F401
    import dashscope.acli.tools.camera  # noqa: F401
    import dashscope.acli.tools.filesystem  # noqa: F401
    import dashscope.acli.tools.shell  # noqa: F401
    import dashscope.acli.tools.web_search  # noqa: F401
    from dashscope.acli.executor import Executor

    asyncio.run(run_stdio_server(executor=Executor(auto_approve=True)))
