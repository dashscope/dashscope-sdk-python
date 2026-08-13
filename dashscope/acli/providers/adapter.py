# -*- coding: utf-8 -*-
"""
Anthropic <-> OpenAI protocol adapter.

Converts Anthropic Messages API format to OpenAI Chat Completions format
and vice versa, allowing providers like Tongyi (Qwen) and Zhipu to be
accessed using Anthropic protocol.
"""
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

import json
from typing import Any


def _normalize_tool_call(call: dict) -> dict:
    """Normalize one tool call to OpenAI's wire shape.

    Accepts the agent's internal entry ({"id", "function": {...}}) and
    guarantees ``arguments`` is a JSON string.
    """
    func = call.get("function", {})
    args = func.get("arguments", "")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "id": call.get("id", ""),
        "type": "function",
        "function": {"name": func.get("name", ""), "arguments": args},
    }


def _tool_result_text(content) -> str:
    """Flatten an Anthropic tool_result content payload to a string."""
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content if isinstance(content, str) else str(content)


def anthropic_to_openai_request(
    messages: list[dict],
    system: str | None = None,
    tools: list[dict] | None = None,
    **kwargs,
) -> dict:
    """
    Convert Anthropic Messages API request to OpenAI Chat Completions format.

    Anthropic format:
    {
      "system": "...",
      "messages": [{"role": "user", "content": "..."}],
      "tools": [{"name": "foo", "input_schema": {...}}]
    }

    OpenAI format:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "tools": [{"type": "function", "function": {"name": "foo",
      "parameters": {...}}}]
    }

    Tool structures round-trip: assistant tool_use blocks (or the agent's
    internal ``tool_calls`` key) become OpenAI tool_calls, and tool_result
    blocks (or internal ``role: "tool"`` messages) become OpenAI tool
    messages correlated by tool_call_id.
    """
    openai_messages: list[dict[str, Any]] = []

    # Add system message if present
    if system:
        openai_messages.append({"role": "system", "content": system})

    # Convert messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Internal tool-result message → OpenAI tool message. The
        # tool_call_id correlation is mandatory for OpenAI-compatible
        # backends; without it the request is rejected.
        if role == "tool":
            openai_messages.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": (
                        msg.get("tool_call_id") or msg.get("tool_use_id", "")
                    ),
                },
            )
            continue

        # Assistant message carrying internal OpenAI-style tool_calls:
        # keep them so later tool messages can be correlated.
        if role == "assistant" and msg.get("tool_calls"):
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        _normalize_tool_call(tc) for tc in msg["tool_calls"]
                    ],
                },
            )
            continue

        # Handle content blocks. Preserve media blocks (image_url, input_audio)
        # that are already in OpenAI format; flatten text blocks; convert
        # tool_use / tool_result blocks to OpenAI tool structures.
        if isinstance(content, list):
            new_blocks = []
            text_buffer: list[str] = []
            has_media = False
            tool_calls = []
            tool_results = []

            def _flush_text(text_buffer=text_buffer, new_blocks=new_blocks):
                # pylint: disable=dangerous-default-value
                if text_buffer:
                    new_blocks.append(
                        {"type": "text", "text": "\n".join(text_buffer)},
                    )
                    text_buffer.clear()

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_buffer.append(block.get("text", ""))
                elif btype in ("image_url", "input_audio"):
                    _flush_text()
                    new_blocks.append(block)
                    has_media = True
                elif btype == "tool_use":
                    args = block.get("input", {})
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": args,
                            },
                        },
                    )
                elif btype == "tool_result":
                    tool_results.append(block)
            _flush_text()

            # Anthropic tool_result blocks become standalone OpenAI tool
            # messages; they must directly follow the assistant message
            # that carried the matching tool_calls.
            for tr in tool_results:
                openai_messages.append(
                    {
                        "role": "tool",
                        "content": _tool_result_text(tr.get("content", "")),
                        "tool_call_id": tr.get("tool_use_id", ""),
                    },
                )

            content = (
                new_blocks
                if has_media
                else "\n".join(
                    b.get("text", "")
                    for b in new_blocks
                    if b.get("type") == "text"
                )
            )
            if tool_results and not content:
                # Message carried only tool results — already emitted above.
                continue
            out = {"role": role, "content": content}
            if tool_calls:
                out["tool_calls"] = tool_calls
            openai_messages.append(out)
            continue

        openai_messages.append({"role": role, "content": content})

    # Convert tools
    openai_tools = None
    if tools:
        openai_tools = []
        for tool in tools:
            # Anthropic: {"name": "foo", "description": "...",
            # "input_schema": {...}}
            # OpenAI: {"type": "function", "function": {"name": "foo",
            # "description": "...", "parameters": {...}}}
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)

    return {"messages": openai_messages, "tools": openai_tools, **kwargs}


def openai_to_anthropic_response(openai_response: dict) -> dict:
    """
    Convert OpenAI Chat Completions response to Anthropic Messages API format.

    OpenAI format:
    {
      "choices": [{
        "message": {
          "content": "...",
          "tool_calls": [{"id": "...", "function": {"name": "...",
          "arguments": "..."}}]
        },
        "finish_reason": "stop"
      }],
      "usage": {"prompt_tokens": 10, "completion_tokens": 20,
      "total_tokens": 30}
    }

    Anthropic format:
    {
      "id": "msg_...",
      "type": "message",
      "role": "assistant",
      "content": [
        {"type": "text", "text": "..."},
        {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
      ],
      "stop_reason": "end_turn",
      "usage": {"input_tokens": 10, "output_tokens": 20}
    }
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})

    content_blocks = []

    # Convert text content
    text_content = message.get("content", "")
    if text_content:
        content_blocks.append({"type": "text", "text": text_content})

    # Convert tool calls
    tool_calls = message.get("tool_calls", [])
    for call in tool_calls:
        func = call.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = (
                json.loads(args_str) if isinstance(args_str, str) else args_str
            )
        except json.JSONDecodeError:
            args = {}

        content_blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": func.get("name", ""),
                "input": args,
            },
        )

    # Convert finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    # Convert usage
    usage = openai_response.get("usage", {})
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    return {
        "id": openai_response.get("id", "msg_0"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": anthropic_usage,
    }


def anthropic_to_openai_response(anthropic_response: dict) -> dict:
    """
    Convert an Anthropic Messages API response back to an OpenAI-style
    message dict: {"content": str, "tool_calls": [...]}.

    Reverse of openai_to_anthropic_response's message part — tool_use
    blocks become tool_calls entries with JSON-string arguments.
    """
    content = ""
    tool_calls = []
    for block in anthropic_response.get("content", []):
        if block.get("type") == "text":
            content += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(
                            block.get("input", {}),
                            ensure_ascii=False,
                        ),
                    },
                },
            )
    return {"content": content, "tool_calls": tool_calls}
