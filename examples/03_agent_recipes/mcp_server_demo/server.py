# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""A dependency-free MCP (Model Context Protocol) server over stdio.

MCP lets an agent discover and call external tools through a small
JSON-RPC 2.0 protocol. This server speaks the ``stdio`` transport: it
reads newline-delimited JSON-RPC requests on stdin and writes one JSON
response per line on stdout (stdout is reserved for protocol messages;
anything a tool prints must go to stderr).

It implements the minimum a tools-only server needs:

* ``initialize``               -> protocol version + capabilities
* ``notifications/initialized`` -> a notification, so no reply
* ``tools/list``               -> the tool schemas the agent may call
* ``tools/call``               -> run one tool and return its content

Only the Python standard library is used, so it runs anywhere with no
API key and no network. Run it directly and type JSON-RPC lines, or
drive it end to end with ``client_demo.py``.
"""

import json
import sys

# The protocol version the SDK's own MCP clients negotiate. Mirrors
# ``dashscope/acli/platforms/bailian/mcp.py`` (read-only reference); the
# constant is inlined here to keep this server dependency-free.
PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "dashscope-demo-mcp-server", "version": "0.1.0"}


# ---- tool implementations (pure stdlib, offline, deterministic) ---- #
def _add(a, b):
    return a + b


def _echo(text):
    return text


# Tool registry: name -> {"description", "inputSchema"} plus its impl.
TOOLS = {
    "add": {
        "description": "Add two numbers and return their sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "first addend"},
                "b": {"type": "number", "description": "second addend"},
            },
            "required": ["a", "b"],
        },
    },
    "echo": {
        "description": "Return the input text unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "text to echo"},
            },
            "required": ["text"],
        },
    },
}

IMPL = {"add": _add, "echo": _echo}


def _tool_list():
    """Build the ``tools/list`` result payload."""
    return [{"name": name, **spec} for name, spec in TOOLS.items()]


def _text_content(text):
    """Wrap a string as an MCP text content block."""
    return {"content": [{"type": "text", "text": text}]}


def _tool_error(message):
    """A tool-level failure: content plus ``isError`` (not JSON-RPC)."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _handle_initialize(_params):
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": SERVER_INFO,
    }


def _handle_tools_list(_params):
    return {"tools": _tool_list()}


def _handle_tools_call(params):
    name = params.get("name")
    arguments = params.get("arguments") or {}
    impl = IMPL.get(name)
    if impl is None:
        return _tool_error(f"unknown tool: {name}")
    try:
        value = impl(**arguments)
    except TypeError as exc:
        return _tool_error(f"bad arguments for {name}: {exc}")
    return _text_content(json.dumps(value, ensure_ascii=False))


HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
}


def handle(request):
    """Return a JSON-RPC response dict, or ``None`` for notifications."""
    req_id = request.get("id")
    if req_id is None:
        # Notifications (e.g. notifications/initialized) get no reply.
        return None

    method = request.get("method")
    params = request.get("params") or {}
    handler = HANDLERS.get(method)
    if handler is None:
        error = {"code": -32601, "message": f"method not found: {method}"}
        return {"jsonrpc": "2.0", "id": req_id, "error": error}
    return {"jsonrpc": "2.0", "id": req_id, "result": handler(params)}


def _write(response):
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    """Read JSON-RPC lines from stdin until EOF, replying on stdout."""
    while True:
        line = sys.stdin.readline()
        if not line:
            break  # EOF: the client closed the pipe
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue  # ignore malformed input; keep serving
        response = handle(request)
        if response is not None:
            _write(response)


if __name__ == "__main__":
    main()
