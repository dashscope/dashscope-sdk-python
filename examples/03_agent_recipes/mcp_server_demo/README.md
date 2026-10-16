# MCP server demo (stdio, dependency-free)

A minimal **Model Context Protocol (MCP)** server that runs over `stdio`
using only the Python standard library -- no SDK, no API key, no network.

MCP is the protocol an agent host uses to discover and call external
tools. This folder shows both sides so the demo is self-contained:

| File            | Role                                                              |
| --------------- | ----------------------------------------------------------------- |
| `server.py`     | The MCP **server**: newline-delimited JSON-RPC 2.0 on stdin/stdout. |
| `client_demo.py`| A tiny MCP **client**: spawns `server.py` and drives a full session.|

`server.py` implements the minimum a tools-only server needs:

- `initialize` -> returns `protocolVersion`, `capabilities`, `serverInfo`
- `notifications/initialized` -> a notification, so the server stays silent
- `tools/list` -> the tool schemas the agent may call (`add`, `echo`)
- `tools/call` -> runs one tool and returns its `content` (or `isError`)

The protocol version (`2024-11-05`) matches the SDK's own MCP clients;
see `dashscope/acli/platforms/bailian/mcp.py` (read-only reference).

## Run it

Drive the whole conversation automatically (recommended):

```bash
python3 examples/03_agent_recipes/mcp_server_demo/client_demo.py
```

Expected output:

```text
initialize -> serverInfo={'name': 'dashscope-demo-mcp-server', 'version': '0.1.0'}
tools/list -> ['add', 'echo']
tools/call add(2, 3) -> [{'type': 'text', 'text': '5'}]
tools/call echo -> "hello MCP"
tools/call unknown -> isError=True
```

Or talk to the server by hand -- pipe JSON-RPC lines into it:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"you","version":"0.1.0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":2,"b":3}}}' \
  | python3 examples/03_agent_recipes/mcp_server_demo/server.py
```

## Sample session

One JSON object per line; the server replies to requests (they carry an
`id`) and stays silent for notifications (no `id`):

```jsonc
// -> initialize
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"you","version":"0.1.0"}}}
// <- result
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"dashscope-demo-mcp-server","version":"0.1.0"}}}

// -> notification (no reply)
{"jsonrpc":"2.0","method":"notifications/initialized"}

// -> tools/list
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
// <- result (schemas trimmed here; see server.py for the full text)
{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"add","description":"Add two numbers and return their sum.","inputSchema":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}},{"name":"echo","description":"Return the input text unchanged.","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}}

// -> tools/call
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add","arguments":{"a":2,"b":3}}}
// <- result
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"5"}]}}
```

## Wire it into a real MCP client

Any MCP-capable host (Claude Desktop, an IDE agent, etc.) can launch
this server. Use the `command` + `args` form and an absolute path:

```json
{
  "mcpServers": {
    "dashscope-demo": {
      "command": "python3",
      "args": [
        "/ABSOLUTE/PATH/TO/examples/03_agent_recipes/mcp_server_demo/server.py"
      ]
    }
  }
}
```

> `stdout` is reserved for JSON-RPC messages. If you add a tool that
> needs to log, write to `stderr` (e.g. `print(..., file=sys.stderr)`),
> never to `stdout`.

## Extend it

Add a tool in three steps inside `server.py`:

1. Write the function (pure stdlib, no side effects on `stdout`).
2. Register its schema in `TOOLS` (`description` + JSON `inputSchema`).
3. Map its name to the function in `IMPL`.

`tools/list` and `tools/call` pick it up automatically.
