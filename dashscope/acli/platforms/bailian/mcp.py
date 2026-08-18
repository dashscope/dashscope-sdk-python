# -*- coding: utf-8 -*-
# pylint: disable=too-many-return-statements
from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urljoin

import httpx

BAILIAN_MCP_BASE = "https://dashscope.aliyuncs.com/api/v1/mcps"
MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    pass


class MCPClient:
    """MCP Client using SSE transport.

    Flow:
    1. GET /sse — opens SSE stream, server sends 'endpoint' event with POST URL
    2. POST to that endpoint — send JSON-RPC requests
    3. Server responds via SSE 'message' events on the GET stream
    """

    def __init__(self, service: str, api_key: str, url: str = ""):
        if not api_key:
            raise MCPError("API key not configured; cannot connect to MCP")
        self.service = service
        base_url = url or f"{BAILIAN_MCP_BASE}/{service}"
        self.sse_url = f"{base_url}/sse"
        self._base_url = base_url
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self._client: httpx.AsyncClient | None = None
        self._request_id = 0
        self._message_endpoint: str = ""
        self._pending: dict[int, asyncio.Future] = {}
        self._sse_task: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self.tools: list[dict] = []
        self.prompts: list[dict] = []
        self.last_error: str = ""

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60, follow_redirects=True)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _start_sse(self):
        """Open SSE connection and listen for events in background."""
        await self._ensure_client()
        self._sse_task = asyncio.create_task(self._sse_loop())
        # Wait for endpoint event
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=15)
        except asyncio.TimeoutError as exc:
            self._sse_task.cancel()
            raise MCPError("SSE connect timed out; no endpoint event") from exc

    async def _sse_loop(self):
        """Background task: read SSE stream."""
        try:
            async with self._client.stream(
                "GET",
                self.sse_url,
                headers=self._headers,
            ) as resp:
                if resp.status_code != 200:
                    body = ""
                    async for line in resp.aiter_lines():
                        body += line
                        break
                    self.last_error = (
                        "Service unavailable: "
                        f"{body or f'HTTP {resp.status_code}'}"
                    )
                    self._connected.set()
                    return

                event_type = ""
                data_buf = ""

                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_buf += line[5:].strip()
                    elif line == "":
                        # End of event
                        if event_type == "endpoint" and data_buf:
                            self._resolve_endpoint(data_buf)
                        elif event_type == "message" and data_buf:
                            self._handle_message(data_buf)
                        event_type = ""
                        data_buf = ""
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.last_error = f"SSE stream error: {e}"
            self._fail_all_pending(f"SSE stream interrupted: {e}")

    def _fail_all_pending(self, reason: str) -> None:
        """Resolve every in-flight request with an error so awaiters
        don't hang."""
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_result({"error": {"message": reason}})

    def _resolve_endpoint(self, data: str):
        """Handle the 'endpoint' event — resolve POST URL."""
        if data.startswith("http"):
            self._message_endpoint = data
        else:
            # Relative URL
            self._message_endpoint = urljoin(self._base_url + "/", data)
        self._connected.set()

    def _handle_message(self, data: str):
        """Handle a 'message' event — resolve pending futures."""
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return
        msg_id = msg.get("id")
        if msg_id and msg_id in self._pending:
            future = self._pending.pop(msg_id)
            if not future.done():
                future.set_result(msg)

    async def _send_jsonrpc(
        self,
        method: str,
        params: dict | None = None,
    ) -> Any:
        """Send JSON-RPC request via POST and wait for response on SSE
        stream."""
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": req_id,
        }

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future

        headers = {
            **self._headers,
            "Content-Type": "application/json",
        }
        try:
            resp = await self._client.post(
                self._message_endpoint,
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        except Exception:
            self._pending.pop(req_id, None)
            raise

        # Wait for response on SSE stream
        try:
            result = await asyncio.wait_for(future, timeout=30)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"error": {"message": "request timed out"}}

    async def _send_notification(
        self,
        method: str,
        params: dict | None = None,
    ):
        """Send JSON-RPC notification (no id, no response expected)."""
        payload = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        headers = {
            **self._headers,
            "Content-Type": "application/json",
        }
        try:
            await self._client.post(
                self._message_endpoint,
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError:
            pass

    async def initialize(self) -> bool:
        try:
            await self._start_sse()
        except MCPError as e:
            self.last_error = str(e)
            return False
        except httpx.HTTPStatusError as e:
            self.last_error = (
                f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
            return False
        except httpx.HTTPError as e:
            self.last_error = str(e)
            return False
        except Exception as e:
            self.last_error = str(e)
            return False

        if self.last_error:
            return False

        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "clientInfo": {"name": "acli", "version": "0.1.0"},
            "capabilities": {},
        }
        try:
            resp = await self._send_jsonrpc("initialize", params)
            if isinstance(resp, dict) and "error" in resp:
                self.last_error = resp["error"].get(
                    "message",
                    str(resp["error"]),
                )
                return False
            await self._send_notification("notifications/initialized")
            return True
        except httpx.HTTPStatusError as e:
            self.last_error = (
                f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
            return False
        except httpx.HTTPError as e:
            self.last_error = str(e)
            return False
        except Exception as e:
            self.last_error = str(e)
            return False

    async def list_tools(self) -> list[dict]:
        resp = await self._send_jsonrpc("tools/list")
        result = resp.get("result", {})
        self.tools = result.get("tools", [])
        return self.tools

    async def list_prompts(self) -> list[dict]:
        """Discover available prompts/skills. Returns [] if not supported."""
        try:
            resp = await self._send_jsonrpc("prompts/list")
            if isinstance(resp, dict) and "error" in resp:
                return []
            result = resp.get("result", {})
            self.prompts = result.get("prompts", [])
            return self.prompts
        except Exception:
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        params = {"name": tool_name, "arguments": arguments}
        resp = await self._send_jsonrpc("tools/call", params)

        if "error" in resp:
            error = resp["error"]
            return f"MCP Error: {error.get('message', str(error))}"

        result = resp.get("result", {})
        content = result.get("content", [])
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append(f"[image: {item.get('mimeType', 'image')}]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else str(result)

    async def close(self):
        self._fail_all_pending("connection closed")
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None
