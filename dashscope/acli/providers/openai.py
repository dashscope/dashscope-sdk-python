# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import json
from typing import AsyncIterator

from dashscope.acli.providers.base import LLMChunk, LLMResponse, ToolCall

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class OpenAIProvider:
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package not installed. "
                "Run: pip install 'acli[openai]'",
            )
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        converted = []
        for msg in messages:
            if msg["role"] == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "content": msg["content"],
                        "tool_call_id": msg.get("tool_use_id", ""),
                    },
                )
            elif msg["role"] == "assistant":
                out: dict = {
                    "role": "assistant",
                    "content": msg.get("content") or None,
                }
                # Reasoning models (deepseek-v4, qwen-thinking, etc.)
                # require the reasoning_content from prior assistant
                # turns to be echoed back.
                if msg.get("reasoning_content"):
                    out["reasoning_content"] = msg["reasoning_content"]
                if "tool_calls" in msg:
                    out["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": (
                                    tc["function"]["arguments"]
                                    if isinstance(
                                        tc["function"]["arguments"],
                                        str,
                                    )
                                    else json.dumps(
                                        tc["function"]["arguments"],
                                    )
                                ),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ]
                converted.append(out)
            else:
                converted.append(
                    {"role": msg["role"], "content": msg["content"]},
                )
        return converted

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        converted = self._convert_messages(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": converted,
        }
        oai_tools = self._convert_tools(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                raise RuntimeError(
                    "API endpoint not found (404): base_url may not match "
                    "protocol; run /provider to check config",
                ) from e
            raise

        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ),
                )

        usage = None
        resp_usage = getattr(response, "usage", None)
        if resp_usage is not None:
            details = getattr(resp_usage, "prompt_tokens_details", None)
            usage = {
                "input_tokens": getattr(resp_usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(resp_usage, "completion_tokens", 0)
                or 0,
                "total_tokens": getattr(resp_usage, "total_tokens", 0) or 0,
                "cached_tokens": (getattr(details, "cached_tokens", 0) or 0)
                if details
                else 0,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning,
            usage=usage,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        converted = self._convert_messages(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": converted,
            "stream": True,
        }
        oai_tools = self._convert_tools(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools
        if response_format:
            kwargs["response_format"] = response_format
        # Ask for a terminal usage-only chunk (choices=[]) so /stats works
        # in streaming mode; backends that don't support it simply omit it.
        kwargs["stream_options"] = {"include_usage": True}

        # Track tool calls being assembled across chunks
        pending_tools: dict[int, dict] = {}
        stream_usage: dict | None = None
        pending_finish: str | None = None

        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                raise RuntimeError(
                    "API endpoint not found (404): base_url may not match "
                    "protocol; run /provider to check config",
                ) from e
            raise
        async for chunk in stream:
            if not chunk.choices:
                # Terminal usage-only chunk (stream_options.include_usage).
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    details = getattr(usage, "prompt_tokens_details", None)
                    stream_usage = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0)
                        or 0,
                        "output_tokens": getattr(usage, "completion_tokens", 0)
                        or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                        "cached_tokens": (
                            getattr(details, "cached_tokens", 0) or 0
                        )
                        if details
                        else 0,
                    }
                continue

            delta = chunk.choices[0].delta
            if delta is None:
                continue
            finish_reason = chunk.choices[0].finish_reason

            if delta.content:
                yield LLMChunk(delta_content=delta.content)

            delta_reasoning = getattr(delta, "reasoning_content", None)
            if delta_reasoning:
                yield LLMChunk(delta_reasoning_content=delta_reasoning)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in pending_tools:
                        pending_tools[idx] = {
                            "id": tc_delta.id or "",
                            "name": (
                                tc_delta.function.name or ""
                                if tc_delta.function
                                else ""
                            ),
                            "arguments": "",
                        }
                    else:
                        if tc_delta.id:
                            pending_tools[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            pending_tools[idx]["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        pending_tools[idx][
                            "arguments"
                        ] += tc_delta.function.arguments

            if finish_reason == "tool_calls":
                for tool_data in pending_tools.values():
                    raw_args = tool_data["arguments"]
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        try:
                            args = json.loads(raw_args + '"}')
                        except json.JSONDecodeError:
                            args = {}
                    yield LLMChunk(
                        tool_calls=[
                            ToolCall(
                                id=tool_data["id"],
                                name=tool_data["name"],
                                arguments=args,
                            ),
                        ],
                    )
                pending_tools.clear()
                pending_finish = finish_reason
            elif finish_reason:
                # "stop" / "length" / ... — hold until stream end so the
                # usage-only chunk (arriving after finish) can be attached.
                pending_finish = finish_reason

        if pending_finish:
            yield LLMChunk(finish_reason=pending_finish, usage=stream_usage)
