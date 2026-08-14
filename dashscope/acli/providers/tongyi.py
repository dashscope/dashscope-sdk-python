# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from dashscope.acli.providers.base import LLMChunk, LLMResponse, ToolCall

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _safe_get(obj, key, default=None):
    """Get attribute from dict-like object or real dict."""
    try:
        return obj.get(key, default)
    except (AttributeError, TypeError):
        return getattr(obj, key, default)


def _extract_usage(response) -> dict | None:
    """Extract token usage from response dict."""
    usage = _safe_get(response, "usage")
    if not usage:
        return None
    details = _safe_get(usage, "prompt_tokens_details") or {}
    return {
        "input_tokens": _safe_get(usage, "prompt_tokens", 0) or 0,
        "output_tokens": _safe_get(usage, "completion_tokens", 0) or 0,
        "total_tokens": _safe_get(usage, "total_tokens", 0) or 0,
        "cached_tokens": (
            _safe_get(details, "cached_tokens", 0)
            or _safe_get(usage, "cached_tokens", 0)
            or 0
        ),
    }


class TongyiProvider:
    def __init__(
        self,
        model: str = "qwen3.7-plus",
        api_key: str | None = None,
        request_timeout: int = 60,
        protocol: str = "openai",
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.protocol = protocol
        self.base_url = (base_url or DASHSCOPE_BASE_URL).rstrip("/")

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        result = []
        for t in tools:
            # Already in OpenAI format (e.g. converted by
            # anthropic_to_openai_request)
            if t.get("type") == "function" and "function" in t:
                result.append(t)
            else:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["parameters"],
                        },
                    },
                )
        return result

    def _parse_tool_calls(self, raw_calls: list) -> list[ToolCall]:
        result = []
        for call in raw_calls:
            func = _safe_get(call, "function", {}) or {}
            args = _safe_get(func, "arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {}
            elif not isinstance(args, dict):
                args = {}
            name = _safe_get(func, "name", "") or ""
            if not name:
                continue
            result.append(
                ToolCall(
                    id=_safe_get(call, "id", "") or "",
                    name=name,
                    arguments=args,
                ),
            )
        return result

    def _build_request_body(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool = False,
        response_format: dict | None = None,
    ) -> dict:
        """Build OpenAI-compatible request body for DashScope."""
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        ds_tools = self._convert_tools(tools)
        if ds_tools:
            body["tools"] = ds_tools
        if response_format:
            body["response_format"] = response_format
        return body

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        # If protocol is anthropic, convert input from Anthropic to
        # OpenAI format
        if self.protocol == "anthropic":
            from dashscope.acli.providers.adapter import (
                anthropic_to_openai_request,
            )

            # Agent may send system as first message, extract it
            system_msg = None
            if messages and messages[0].get("role") == "system":
                system_msg = messages[0].get("content")
                messages = messages[1:]
            converted = anthropic_to_openai_request(
                messages,
                system=system_msg,
                tools=tools,
            )
            messages = converted["messages"]
            tools = converted["tools"]

        body = self._build_request_body(
            messages,
            tools,
            stream=False,
            response_format=response_format,
        )
        headers = self._get_headers()

        try:
            async with httpx.AsyncClient(
                timeout=self.request_timeout,
            ) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
        except httpx.TimeoutException as e:
            raise RuntimeError(
                "API request timed out; check network or retry later",
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Cannot connect to API server; check network",
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: {response.status_code} - "
                f"{response.text}",
            )

        data = response.json()

        # If protocol is anthropic, convert output from OpenAI to
        # Anthropic format
        if self.protocol == "anthropic":
            from dashscope.acli.providers.adapter import (
                anthropic_to_openai_response,
                openai_to_anthropic_response,
            )

            anthropic_resp = openai_to_anthropic_response(data)
            # Extract content and tool_calls from Anthropic format
            extracted = anthropic_to_openai_response(anthropic_resp)
            reasoning = ""  # Anthropic format doesn't have reasoning_content
            usage = anthropic_resp.get("usage", {})
            return LLMResponse(
                content=extracted["content"],
                tool_calls=self._parse_tool_calls(extracted["tool_calls"]),
                reasoning_content=reasoning,
                usage={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                    "cached_tokens": usage.get("cache_read_input_tokens", 0)
                    or 0,
                },
            )

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        raw_calls = msg.get("tool_calls", []) or []
        usage = _extract_usage(data)

        return LLMResponse(
            content=content,
            tool_calls=self._parse_tool_calls(raw_calls),
            reasoning_content=reasoning,
            usage=usage,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        # If protocol is anthropic, convert input from Anthropic to
        # OpenAI format
        if self.protocol == "anthropic":
            from dashscope.acli.providers.adapter import (
                anthropic_to_openai_request,
            )

            # Agent may send system as first message, extract it
            system_msg = None
            if messages and messages[0].get("role") == "system":
                system_msg = messages[0].get("content")
                messages = messages[1:]
            converted = anthropic_to_openai_request(
                messages,
                system=system_msg,
                tools=tools,
            )
            messages = converted["messages"]
            tools = converted["tools"]

        body = self._build_request_body(
            messages,
            tools,
            stream=True,
            response_format=response_format,
        )
        headers = self._get_headers()
        # Request incremental streaming for DashScope
        body["stream_options"] = {"include_usage": True}

        try:
            async with httpx.AsyncClient(
                timeout=self.request_timeout,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        raise RuntimeError(
                            f"DashScope API error: {response.status_code} - "
                            f"{error_body.decode()}",
                        )

                    pending_tools: dict[int, dict] = {}
                    last_usage: dict | None = None
                    usage_sent = False
                    _json_buf: str | None = None

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break

                        # Buffer incomplete JSON across SSE lines
                        # (server may split tool argument strings across
                        # multiple data: lines or send multi-line JSON)
                        if _json_buf is not None:
                            payload = _json_buf + payload
                            _json_buf = None

                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            _json_buf = payload
                            continue

                        usage = _extract_usage(chunk)
                        if usage:
                            last_usage = usage

                        if not chunk.get("choices"):
                            continue

                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        finish = choice.get("finish_reason")

                        content = delta.get("content", "") or ""
                        delta_reasoning = (
                            delta.get("reasoning_content", "") or ""
                        )
                        raw_calls = delta.get("tool_calls", []) or []

                        # Accumulate tool calls across chunks
                        for pos, call in enumerate(raw_calls):
                            slot = _safe_get(call, "index", pos)
                            func = _safe_get(call, "function", {}) or {}
                            if slot not in pending_tools:
                                pending_tools[slot] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            call_id = _safe_get(call, "id", "")
                            if call_id:
                                pending_tools[slot]["id"] = call_id
                            func_name = _safe_get(func, "name", "")
                            if func_name:
                                pending_tools[slot]["name"] = func_name
                            args = _safe_get(func, "arguments", "")
                            if args:
                                if isinstance(args, str):
                                    pending_tools[slot]["arguments"] += args
                                elif isinstance(args, dict):
                                    pending_tools[slot][
                                        "arguments"
                                    ] = json.dumps(
                                        args,
                                        ensure_ascii=False,
                                    )

                        if content:
                            yield LLMChunk(delta_content=content)
                        if delta_reasoning:
                            yield LLMChunk(
                                delta_reasoning_content=delta_reasoning,
                            )

                        if finish and finish != "null":
                            tool_calls = []
                            for tool_data in pending_tools.values():
                                if not tool_data["name"]:
                                    continue
                                raw_args = tool_data["arguments"]
                                try:
                                    args = (
                                        json.loads(raw_args)
                                        if raw_args
                                        else {}
                                    )
                                except json.JSONDecodeError:
                                    # Try to repair truncated JSON
                                    try:
                                        args = json.loads(raw_args + '"}')
                                    except json.JSONDecodeError:
                                        args = {}
                                tool_calls.append(
                                    ToolCall(
                                        id=tool_data["id"],
                                        name=tool_data["name"],
                                        arguments=args,
                                    ),
                                )
                            if tool_calls:
                                yield LLMChunk(
                                    tool_calls=tool_calls,
                                    finish_reason=finish,
                                    usage=last_usage,
                                )
                            else:
                                yield LLMChunk(
                                    finish_reason=finish,
                                    usage=last_usage,
                                )
                            usage_sent = usage_sent or last_usage is not None
                            # Prevent re-emission by later finish chunks or
                            # the orphan flush below (duplicate tool calls).
                            pending_tools.clear()

                    # Flush pending tool calls if stream ended without
                    # finish_reason (network drop, rate limit, truncated
                    # response).
                    if pending_tools:
                        orphan_calls = []
                        for tool_data in pending_tools.values():
                            if not tool_data["name"]:
                                continue
                            raw_args = tool_data["arguments"]
                            try:
                                args = json.loads(raw_args) if raw_args else {}
                            except json.JSONDecodeError:
                                try:
                                    args = json.loads(raw_args + '"}')
                                except json.JSONDecodeError:
                                    args = {}
                            orphan_calls.append(
                                ToolCall(
                                    id=tool_data["id"],
                                    name=tool_data["name"],
                                    arguments=args,
                                ),
                            )
                        if orphan_calls:
                            yield LLMChunk(
                                tool_calls=orphan_calls,
                                finish_reason="stop",
                                usage=last_usage,
                            )
                            usage_sent = usage_sent or last_usage is not None

                    # include_usage payload arrives as a separate tail
                    # chunk after finish; last_usage was still None when
                    # the finish block above yielded, so flush it here.
                    if last_usage and not usage_sent:
                        yield LLMChunk(usage=last_usage)
        except httpx.TimeoutException as e:
            raise RuntimeError(
                "API request timed out; check network or retry later",
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Cannot connect to API server; check network",
            ) from e
