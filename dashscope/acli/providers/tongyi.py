# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import json
import os
from typing import AsyncIterator

import httpx

from dashscope.acli import SDK_SESSION_ID, __version__
from dashscope.acli.providers.base import LLMChunk, LLMResponse, ToolCall

# Service root; the provider appends the native generation path. This is
# the DashScope native route (NOT the OpenAI-compatible one) so requests
# land in the native SLS logstore.
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com"
# The gateway binds each model to exactly one aigc service path (newer
# models like qwen3.8-max live on multimodal-generation, classic text
# models on text-generation). The provider tries them in order and
# remembers the one that answers 200 — see _is_url_error.
_GENERATION_PATHS = (
    "/api/v1/services/aigc/text-generation/generation",
    "/api/v1/services/aigc/multimodal-generation/generation",
)
# Historical/standard prefixes a persisted config may carry; the provider
# normalizes them back to the service root.
_LEGACY_PREFIXES = ("/compatible-mode/v1", "/api/v1")


def _safe_get(obj, key, default=None):
    """Get attribute from dict-like object or real dict."""
    try:
        return obj.get(key, default)
    except (AttributeError, TypeError):
        return getattr(obj, key, default)


def _is_url_error(status_code: int, body: str) -> bool:
    """The gateway's "model is not served on this service path" signal.

    The message text ("url error, please check url") is misleading — it
    really means the model is bound to the other aigc generation path.
    """
    return status_code == 400 and "url error" in body


class _StreamAPIError(RuntimeError):
    """Mid-stream native error (200 SSE body carrying {code, message})."""

    def __init__(self, code, message):
        super().__init__(f"DashScope API error: {code} - {message}")
        self.code = code
        self.message = message


def _is_url_error_message(message: str) -> bool:
    """url-error check for the SSE variant, which arrives on a 200."""
    return "url error" in (message or "")


def _to_native_content(content):
    """Convert OpenAI message content parts to the native shape.

    Text-only part lists flatten to a plain string (text-generation
    expects string content). Lists carrying images become native
    {"text"/"image"} parts — multimodal-generation rejects the OpenAI
    {"type": "image_url", "image_url": {...}} shape outright.
    """
    if not isinstance(content, list):
        return content
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            parts.append({"text": part.get("text", "")})
        elif ptype == "image_url":
            image_url = part.get("image_url") or {}
            url = (
                _safe_get(image_url, "url", "")
                if isinstance(image_url, dict)
                else image_url
            )
            parts.append({"image": url})
        elif "text" in part or "image" in part:
            parts.append(part)  # already native
    if parts and all("text" in p for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _to_native_messages(messages: list[dict]) -> list[dict]:
    result = []
    for m in messages:
        if isinstance(_safe_get(m, "content"), list):
            m = {**m, "content": _to_native_content(m["content"])}
        result.append(m)
    return result


def _flatten_content(message: dict) -> dict:
    """multimodal-generation returns content as [{"text": ...}] parts."""
    content = _safe_get(message, "content")
    if isinstance(content, list):
        message = dict(message)
        message["content"] = "".join(
            _safe_get(p, "text", "") or ""
            for p in content
            if isinstance(p, dict)
        )
    return message


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


def _native_usage_to_openai(usage) -> dict | None:
    """Map native usage names to the OpenAI names _extract_usage reads."""
    if not usage:
        return None
    result = {
        "prompt_tokens": _safe_get(usage, "input_tokens", 0) or 0,
        "completion_tokens": _safe_get(usage, "output_tokens", 0) or 0,
        "total_tokens": _safe_get(usage, "total_tokens", 0) or 0,
    }
    details = _safe_get(usage, "prompt_tokens_details")
    if details:
        result["prompt_tokens_details"] = details
    return result


def _native_to_openai_response(data) -> dict:
    """Reshape a native generation response to OpenAI chat.completion."""
    output = _safe_get(data, "output") or {}
    choices = []
    for choice in _safe_get(output, "choices") or []:
        c = dict(choice)
        if "message" in c:
            c["message"] = _flatten_content(c["message"])
        choices.append(c)
    result = {"choices": choices}
    usage = _native_usage_to_openai(_safe_get(data, "usage"))
    if usage:
        result["usage"] = usage
    return result


def _native_chunk_to_openai(chunk) -> dict:
    """Reshape a native SSE chunk to the OpenAI delta shape.

    Native streams choices[].message; OpenAI streams choices[].delta. With
    result_format=message + incremental_output=true the message holds only
    the incremental delta, so a key rename plus content-list flattening is
    the whole conversion.
    """
    output = _safe_get(chunk, "output") or {}
    choices = []
    for choice in _safe_get(output, "choices") or []:
        c = dict(choice)
        if "message" in c:
            c["delta"] = _flatten_content(c.pop("message"))
        choices.append(c)
    result = {"choices": choices}
    usage = _native_usage_to_openai(_safe_get(chunk, "usage"))
    if usage:
        result["usage"] = usage
    return result


class TongyiProvider:
    def __init__(
        self,
        model: str = "qwen3.8-max",
        api_key: str | None = None,
        request_timeout: int = 60,
        protocol: str = "openai",
        base_url: str | None = None,
        module: str = "app",
    ):
        self.model = model
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.protocol = protocol
        base = (base_url or DASHSCOPE_BASE_URL).rstrip("/")
        for prefix in _LEGACY_PREFIXES:
            if base.endswith(prefix):
                base = base[: -len(prefix)]
                break
        self.base_url = base
        self.module = module
        self._generation_path = _GENERATION_PATHS[0]

    def _candidate_paths(self) -> list[str]:
        others = [p for p in _GENERATION_PATHS if p != self._generation_path]
        return [self._generation_path, *others]

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
        """Build native DashScope generation request body.

        result_format="message" makes the native route return OpenAI-like
        output.choices[].message and is required for tool calling. The
        stream flag mirrors what the python SDK sends; the SSE headers in
        _get_headers are what actually switch the wire to SSE.
        """
        parameters: dict = {"result_format": "message"}
        if stream:
            parameters["stream"] = True
            parameters["incremental_output"] = True
        ds_tools = self._convert_tools(tools)
        if ds_tools:
            parameters["tools"] = ds_tools
        if response_format:
            parameters["response_format"] = response_format
        return {
            "model": self.model,
            "input": {"messages": _to_native_messages(messages)},
            "parameters": parameters,
        }

    def _get_headers(self, stream: bool = False) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if stream:
            headers["Accept"] = "text/event-stream"
            headers["X-Accel-Buffering"] = "no"
            headers["X-DashScope-SSE"] = "enable"
        if not os.environ.get("DASHSCOPE_DISABLE_SDK_HEADERS"):
            parts = ["acli", __version__]
            if self.module:
                parts.append(self.module)
            headers["x-dashscope-sdk-client"] = "/".join(parts)
            headers["x-dashscope-sdk-session-id"] = SDK_SESSION_ID
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
                paths = self._candidate_paths()
                for i, path in enumerate(paths):
                    response = await client.post(
                        f"{self.base_url}{path}",
                        json=body,
                        headers=headers,
                    )
                    if response.status_code == 200:
                        self._generation_path = path
                        break
                    if i + 1 < len(paths) and _is_url_error(
                        response.status_code,
                        response.text,
                    ):
                        continue  # model lives on the other service path
                    raise RuntimeError(
                        f"DashScope API error: {response.status_code} - "
                        f"{response.text}",
                    )
        except httpx.TimeoutException as e:
            raise RuntimeError(
                "API request timed out; check network or retry later",
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Cannot connect to API server; check network",
            ) from e

        raw = response.json()
        if raw.get("code"):
            raise RuntimeError(
                f"DashScope API error: {raw.get('code')} - "
                f"{raw.get('message')}",
            )
        data = _native_to_openai_response(raw)
        if not data["choices"]:
            raise RuntimeError(
                f"DashScope API error: unexpected response: {raw}",
            )

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
        headers = self._get_headers(stream=True)

        try:
            async with httpx.AsyncClient(
                timeout=self.request_timeout,
            ) as client:
                paths = self._candidate_paths()
                for i, path in enumerate(paths):
                    async with client.stream(
                        "POST",
                        f"{self.base_url}{path}",
                        json=body,
                        headers=headers,
                    ) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            text = error_body.decode()
                            if i + 1 < len(paths) and _is_url_error(
                                response.status_code,
                                text,
                            ):
                                continue  # model on the other service path
                            raise RuntimeError(
                                f"DashScope API error: "
                                f"{response.status_code} - {text}",
                            )
                        # The url error also arrives as a 200 SSE error
                        # event, so commit to (and cache) this path only
                        # after the first real chunk; before that, trying
                        # the other path is still safe.
                        stream = self._iter_stream(response)
                        try:
                            first = await stream.__anext__()
                        except StopAsyncIteration:
                            self._generation_path = path
                            return
                        except _StreamAPIError as e:
                            if i + 1 < len(paths) and _is_url_error_message(
                                e.message,
                            ):
                                continue
                            raise
                        self._generation_path = path
                        yield first
                        async for chunk in stream:
                            yield chunk
                        return
        except httpx.TimeoutException as e:
            raise RuntimeError(
                "API request timed out; check network or retry later",
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Cannot connect to API server; check network",
            ) from e

    async def _iter_stream(self, response) -> AsyncIterator[LLMChunk]:
        pending_tools: dict[int, dict] = {}
        last_usage: dict | None = None
        usage_sent = False
        _json_buf: str | None = None

        async for line in response.aiter_lines():
            # Native SSE sends "data:{...}" with no space;
            # compatible-mode sends "data: {...}".
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break

            # Buffer incomplete JSON across SSE lines
            # (server may split tool argument strings across
            # multiple data: lines or send multi-line JSON)
            if _json_buf is not None:
                payload = _json_buf + payload
                _json_buf = None

            try:
                raw_chunk = json.loads(payload)
            except json.JSONDecodeError:
                _json_buf = payload
                continue

            # Mid-stream native errors arrive as {code, message}
            # data lines on a 200 response.
            if raw_chunk.get("code"):
                raise _StreamAPIError(
                    raw_chunk.get("code"),
                    raw_chunk.get("message"),
                )

            chunk = _native_chunk_to_openai(raw_chunk)

            usage = _extract_usage(chunk)
            if usage:
                last_usage = usage

            if not chunk.get("choices"):
                continue

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            finish = choice.get("finish_reason")

            content = delta.get("content", "") or ""
            delta_reasoning = delta.get("reasoning_content", "") or ""
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
                        pending_tools[slot]["arguments"] = json.dumps(
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
                        args = json.loads(raw_args) if raw_args else {}
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
