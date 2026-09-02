# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from dashscope.acli.providers.base import LLMChunk, LLMResponse, ToolCall

try:
    import anthropic
except ImportError:
    anthropic = None


def _convert_content_block(block: dict) -> dict:
    """Translate OpenAI-style content blocks to Anthropic format.

    Handles ``image_url`` blocks carrying a ``data:`` URL by extracting the
    MIME type and base64 payload and emitting Anthropic's ``image`` source
    schema. Other block types pass through unchanged.
    """
    if not isinstance(block, dict):
        return block
    btype = block.get("type")
    if btype == "image_url":
        url = (block.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            # data:<mime>;base64,<data>
            try:
                header, b64 = url.split(",", 1)
                mime = header.split(":", 1)[1].split(";")[0]
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                }
            except (ValueError, IndexError):
                # Malformed data URL — fall through and pass the block as-is
                pass
    return block


class AnthropicProvider:
    # Requests cap output at this many tokens; surfaced to the TUI status
    # line for a live completion percentage.
    default_max_tokens = 4096

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        if anthropic is None:
            raise ImportError(
                "The anthropic package is not installed. "
                "Run: pip install 'acli[anthropic]'",
            )
        self.model = model
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    def _convert_messages(
        self,
        messages: list[dict],
    ) -> tuple[str, list[dict]]:
        system = ""
        converted = []
        tool_results: list[dict] = []

        def _flush_tool_results():
            # Consecutive tool-role messages (parallel tool calls) merge
            # into ONE user message so user/assistant alternation holds.
            if tool_results:
                converted.append(
                    {"role": "user", "content": list(tool_results)},
                )
                tool_results.clear()

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "tool":
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_use_id", ""),
                        "content": msg["content"],
                    },
                )
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                _flush_tool_results()
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": args,
                        },
                    )
                converted.append({"role": "assistant", "content": content})
            else:
                _flush_tool_results()
                content = msg["content"]
                if isinstance(content, list):
                    content = [_convert_content_block(b) for b in content]
                converted.append({"role": msg["role"], "content": content})
        _flush_tool_results()
        return system, converted

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,  # pylint: disable=unused-argument
    ) -> LLMResponse:
        system, converted = self._convert_messages(messages)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.default_max_tokens,
            "messages": converted,
        }
        if system:
            kwargs["system"] = system
        claude_tools = self._convert_tools(tools)
        if claude_tools:
            kwargs["tools"] = claude_tools
        # Anthropic API does not support OpenAI-style response_format;
        # ignore it.

        try:
            response = await self.client.messages.create(**kwargs)
        except httpx.TimeoutException as e:
            raise RuntimeError(
                "API request timeout; check your network or retry later",
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Unable to connect to the API server; check your network",
            ) from e
        except anthropic.NotFoundError as e:
            raise RuntimeError(
                "API endpoint not found (404): base_url and protocol may "
                "not match; run /provider to check the config",
            ) from e
        except anthropic.APIError as e:
            raise RuntimeError(f"API request failed: {e}") from e

        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    ),
                )

        usage = None
        resp_usage = getattr(response, "usage", None)
        if resp_usage is not None:
            input_tokens = getattr(resp_usage, "input_tokens", 0) or 0
            output_tokens = getattr(resp_usage, "output_tokens", 0) or 0
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cached_tokens": getattr(
                    resp_usage,
                    "cache_read_input_tokens",
                    0,
                )
                or 0,
            }

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,  # pylint: disable=unused-argument
    ) -> AsyncIterator[LLMChunk]:
        system, converted = self._convert_messages(messages)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.default_max_tokens,
            "messages": converted,
        }
        if system:
            kwargs["system"] = system
        claude_tools = self._convert_tools(tools)
        if claude_tools:
            kwargs["tools"] = claude_tools
        # Anthropic API does not support OpenAI-style response_format;
        # ignore it.

        current_tool: dict | None = None
        tool_input_json = ""
        received_content = False
        stream_usage: dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
        }

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool = {
                                "id": block.id,
                                "name": block.name,
                            }
                            tool_input_json = ""
                        elif block.type == "text" and getattr(
                            block,
                            "text",
                            "",
                        ):
                            received_content = True
                            # Some OpenAI-compatible backends emit the
                            # full text
                            # on content_block_start instead of as deltas.
                            yield LLMChunk(delta_content=block.text)
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        # Anthropic SDK uses "text_delta"; some
                        # OpenAI-compatible backends (e.g. ideatalk) emit
                        # "text" or put text directly on the delta.
                        # Accept any object that carries text.
                        if getattr(delta, "type", "") == "text_delta":
                            received_content = True
                            yield LLMChunk(delta_content=delta.text)
                        elif getattr(delta, "type", "") == "text":
                            received_content = True
                            yield LLMChunk(
                                delta_content=getattr(delta, "text", ""),
                            )
                        elif hasattr(delta, "text") and delta.text:
                            received_content = True
                            yield LLMChunk(delta_content=delta.text)
                        elif getattr(delta, "type", "") == "input_json_delta":
                            tool_input_json += delta.partial_json
                    elif event.type == "content_block_stop":
                        if current_tool:
                            received_content = True
                            try:
                                args = (
                                    json.loads(tool_input_json)
                                    if tool_input_json
                                    else {}
                                )
                            except json.JSONDecodeError as e:
                                # LLM produced malformed / truncated JSON.
                                # Surface it as an empty-args tool call so
                                # the agent loop can report the error
                                # instead of crashing.
                                args = {}
                                yield LLMChunk(
                                    delta_content=(
                                        f"[Failed to parse tool args: {e}]"
                                    ),
                                )
                            # pylint: disable=unsubscriptable-object
                            yield LLMChunk(
                                tool_calls=[
                                    ToolCall(
                                        id=current_tool["id"],
                                        name=current_tool["name"],
                                        arguments=args,
                                    ),
                                ],
                            )
                            # pylint: enable=unsubscriptable-object
                            current_tool = None
                            tool_input_json = ""
                    elif event.type == "message_start":
                        usage = getattr(
                            getattr(event, "message", None),
                            "usage",
                            None,
                        )
                        if usage is not None:
                            stream_usage["input_tokens"] = (
                                getattr(usage, "input_tokens", 0) or 0
                            )
                            stream_usage["cached_tokens"] = (
                                getattr(usage, "cache_read_input_tokens", 0)
                                or 0
                            )
                    elif event.type == "message_delta":
                        usage = getattr(event, "usage", None)
                        if usage is not None:
                            stream_usage["output_tokens"] = (
                                getattr(usage, "output_tokens", 0) or 0
                            )
                    elif event.type == "message_stop":
                        yield LLMChunk(
                            finish_reason="end_turn",
                            usage={
                                **stream_usage,
                                "total_tokens": (
                                    stream_usage["input_tokens"]
                                    + stream_usage["output_tokens"]
                                ),
                            },
                        )

                # Detect silent failures: API returned 200 but no content.
                # Some backends (e.g. ideatalk rate-limit) send error in body
                # without raising, producing zero stream events.
                if not received_content:
                    stop = None
                    try:
                        final = await stream.get_final_message()
                        stop = getattr(final, "stop_reason", None)
                    except Exception:
                        pass
                    if stop and stop != "end_turn":
                        raise RuntimeError(
                            f"API stream ended abnormally: stop_reason={stop}",
                        )
                    raise RuntimeError(
                        "API stream returned no content; possibly rate "
                        "limit, model unavailable, or service disruption",
                    )
        except httpx.TimeoutException as e:
            raise RuntimeError(
                "API request timeout; check your network or retry later",
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Unable to connect to the API server; check your network",
            ) from e
        except anthropic.NotFoundError as e:
            raise RuntimeError(
                "API endpoint not found (404): base_url and protocol may "
                "not match; run /provider to check the config",
            ) from e
        except anthropic.APIError as e:
            raise RuntimeError(f"API request failed: {e}") from e
