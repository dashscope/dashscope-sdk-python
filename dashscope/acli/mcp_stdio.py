# -*- coding: utf-8 -*-
"""MCP stdio transport client.

Speaks newline-delimited JSON-RPC 2.0 to a local MCP server subprocess
over stdin/stdout. Mirrors the SSE ``MCPClient`` interface (initialize /
list_tools / list_prompts / call_tool / close plus ``tools``, ``prompts``
and ``last_error`` attributes) so ``cli/mcp.py`` can use either transport
interchangeably.
"""

from __future__ import annotations

import asyncio
import json

from dashscope.acli import __version__
from dashscope.acli.platforms.bailian.mcp import MCP_PROTOCOL_VERSION, MCPError

__all__ = ["StdioMCPClient", "MCPError"]

_CLIENT_INFO = {"name": "acli", "version": __version__}
_CLOSE_TIMEOUT = 3.0
_STDERR_TAIL = 500


class StdioMCPClient:
    """MCP client that drives a local server over stdio.

    The server process is spawned with ``command`` + ``args``; requests
    and responses are single-line JSON documents. A background reader
    matches response ids to pending request futures; server notifications
    (messages without an ``id``) are ignored.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        timeout: float = 30.0,
    ):
        if not command:
            raise MCPError("stdio MCP server requires a command")
        self.command = command
        self.args = list(args or [])
        self.timeout = timeout
        self.tools: list[dict] = []
        self.prompts: list[dict] = []
        self.last_error = ""
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._request_id = 0
        self._stderr_tail = ""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def initialize(self) -> bool:
        """Spawn the server, run the MCP initialize handshake."""
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            self.last_error = f"Failed to start MCP server: {exc}"
            return False

        self._reader_task = asyncio.create_task(self._read_loop())

        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": dict(_CLIENT_INFO),
        }
        try:
            resp = await self._request("initialize", params)
        except MCPError as exc:
            self.last_error = str(exc)
            await self.close()
            return False
        if "error" in resp:
            error = resp["error"]
            message = error.get("message", str(error))
            self.last_error = f"MCP initialize failed: {message}"
            await self.close()
            return False
        try:
            await self._notify("notifications/initialized")
        except MCPError as exc:
            self.last_error = str(exc)
            await self.close()
            return False
        return True

    async def close(self) -> None:
        """Stop the reader, close stdin, terminate the subprocess."""
        self._fail_all_pending("connection closed")
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        self._reader_task = None

        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin and not proc.stdin.is_closing():
            try:
                proc.stdin.close()
            except (OSError, RuntimeError):
                pass
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=_CLOSE_TIMEOUT)
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                    await asyncio.wait_for(
                        proc.wait(),
                        timeout=_CLOSE_TIMEOUT,
                    )
                except (asyncio.TimeoutError, ProcessLookupError, OSError):
                    try:
                        proc.kill()
                    except (ProcessLookupError, OSError):
                        pass
        await self._drain_stderr(proc)

    # ------------------------------------------------------------------ #
    # MCP operations
    # ------------------------------------------------------------------ #
    async def list_tools(self) -> list[dict]:
        resp = await self._request("tools/list")
        if "error" in resp:
            error = resp["error"]
            message = error.get("message", str(error))
            raise MCPError(f"tools/list failed: {message}")
        result = resp.get("result", {})
        self.tools = result.get("tools", [])
        return self.tools

    async def list_prompts(self) -> list[dict]:
        """Discover prompts/skills; returns [] when unsupported."""
        try:
            resp = await self._request("prompts/list")
        except MCPError:
            return []
        if "error" in resp:
            return []
        result = resp.get("result", {})
        self.prompts = result.get("prompts", [])
        return self.prompts

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        params = {"name": tool_name, "arguments": arguments}
        resp = await self._request("tools/call", params)
        if "error" in resp:
            error = resp["error"]
            message = error.get("message", str(error))
            return f"MCP Error: {message}"
        result = resp.get("result", {})
        content = result.get("content", [])
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts) if parts else str(result)

    # ------------------------------------------------------------------ #
    # JSON-RPC plumbing
    # ------------------------------------------------------------------ #
    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _request(
        self,
        method: str,
        params: dict | None = None,
    ) -> dict:
        """Send a request and await the matching response."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError("stdio MCP client is not connected")
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": req_id,
        }
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            await self._write(payload)
        except MCPError:
            self._pending.pop(req_id, None)
            raise
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            message = (
                f"MCP request '{method}' timed out " f"after {self.timeout:g}s"
            )
            raise MCPError(message) from None

    async def _notify(
        self,
        method: str,
        params: dict | None = None,
    ) -> None:
        """Send a notification (no id, no response expected)."""
        if self._proc is None or self._proc.stdin is None:
            raise MCPError("stdio MCP client is not connected")
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        await self._write(payload)

    async def _write(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError("stdio MCP client is not connected")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            proc.stdin.write(line.encode("utf-8"))
            await proc.stdin.drain()
        except (
            ConnectionResetError,
            BrokenPipeError,
            OSError,
            RuntimeError,
        ) as exc:
            raise MCPError(
                f"MCP server process is gone: {exc}",
            ) from exc

    async def _read_loop(self) -> None:
        """Read stdout lines and resolve pending futures by id."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    self._fail_all_pending(
                        "MCP server closed stdout" + self._stderr_suffix(),
                    )
                    return
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # stdout is reserved for JSON-RPC messages; anything
                    # else is a protocol violation — fail in-flight
                    # requests instead of hanging until timeout.
                    self._stderr_tail = (
                        self._stderr_tail + "non-JSON: " + line + "\n"
                    )[-_STDERR_TAIL:]
                    message = (
                        "MCP server sent non-JSON output: " f"{line[:100]}"
                    )
                    self._fail_all_pending(message)
                    return
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                if msg_id is None:
                    continue  # server notification — ignore
                future = self._pending.pop(msg_id, None)
                if future is not None and not future.done():
                    future.set_result(msg)
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as exc:  # stream broke unexpectedly
            self._fail_all_pending(f"MCP stdio stream error: {exc}")

    def _fail_all_pending(self, reason: str) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(MCPError(reason))

    async def _drain_stderr(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stderr is None:
            return
        try:
            data = await asyncio.wait_for(
                proc.stderr.read(_STDERR_TAIL),
                timeout=1.0,
            )
            if data:
                text = data.decode("utf-8", errors="replace").strip()
                self._stderr_tail = (self._stderr_tail + text)[-_STDERR_TAIL:]
        except (asyncio.TimeoutError, OSError):
            pass

    def _stderr_suffix(self) -> str:
        tail = self._stderr_tail.strip()
        return f" (stderr: {tail})" if tail else ""
