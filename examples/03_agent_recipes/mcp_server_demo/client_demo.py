# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Drive the local MCP demo server end to end over stdio.

``server.py`` is a real MCP server, but you normally reach it through an
MCP client (an agent host). This script *is* a minimal client: it spawns
``server.py`` as a subprocess and speaks newline-delimited JSON-RPC 2.0
to it, doing the standard handshake and then listing and calling tools::

    initialize -> notifications/initialized -> tools/list -> tools/call

That makes the MCP demo runnable and testable on its own -- no external
agent, no API key, no network. Run it from anywhere::

    python3 .../mcp_server_demo/client_demo.py
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "server.py")
PROTOCOL_VERSION = "2024-11-05"


def _send(proc, payload):
    """Write one newline-delimited JSON-RPC message to the server."""
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _recv(proc):
    """Read one JSON-RPC message back from the server."""
    line = proc.stdout.readline()
    return json.loads(line) if line.strip() else None


def _request(proc, req_id, method, params=None):
    """Send a request (it has an id) and return its response."""
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    }
    _send(proc, payload)
    return _recv(proc)


def _notify(proc, method):
    """Send a notification (no id): the server must not reply."""
    _send(proc, {"jsonrpc": "2.0", "method": method})


def _session(proc):
    """Run the full handshake + tools/list + tools/call conversation."""
    # 1. Handshake: negotiate protocol version and capabilities.
    resp = _request(
        proc,
        1,
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "client-demo", "version": "0.1.0"},
        },
    )
    server_info = resp["result"]["serverInfo"]
    print(f"initialize -> serverInfo={server_info}")

    # The client acknowledges the handshake with a notification.
    _notify(proc, "notifications/initialized")

    # 2. Discover the tools the server exposes.
    resp = _request(proc, 2, "tools/list")
    names = [tool["name"] for tool in resp["result"]["tools"]]
    print(f"tools/list -> {names}")

    # 3. Call a tool and read back its content blocks.
    resp = _request(
        proc,
        3,
        "tools/call",
        {"name": "add", "arguments": {"a": 2, "b": 3}},
    )
    content = resp["result"]["content"]
    print(f"tools/call add(2, 3) -> {content}")

    resp = _request(
        proc,
        4,
        "tools/call",
        {"name": "echo", "arguments": {"text": "hello MCP"}},
    )
    text = resp["result"]["content"][0]["text"]
    print(f"tools/call echo -> {text}")

    # 4. An unknown tool comes back as a tool-level error, not a crash.
    resp = _request(proc, 5, "tools/call", {"name": "nope", "arguments": {}})
    is_error = resp["result"]["isError"]
    print(f"tools/call unknown -> isError={is_error}")


def main():
    with subprocess.Popen(
        [sys.executable, SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        try:
            _session(proc)
        finally:
            proc.stdin.close()  # EOF ends the server's read loop
            proc.wait(timeout=5)
            err = proc.stderr.read().strip()
            if err:
                print(f"[server stderr]\n{err}", file=sys.stderr)


if __name__ == "__main__":
    main()
