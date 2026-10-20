# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Anthropic protocol adapter (design doc §3.1).

Translates the canonical chat models to ``POST /v1/messages`` requests so
``Client(protocol="anthropic", base_url=...)`` can talk to Anthropic
or Anthropic-compatible endpoints. System messages move into the
``system`` field, tool calls map to ``tool_use``/``tool_result``
blocks; thinking blocks (enabled by passing the native ``thinking``
object through ``extra_parameters``) surface as ``reasoning_content``
in streaming and non-streaming responses alike, with the block
signature preserved as ``reasoning_signature`` and echoed back on
assistant turns so strict endpoints accept the replayed thinking
blocks. Canonical ``tool_choice`` values translate onto the wire's
``tool_choice`` object: ``"auto"``/``"none"`` keep their type,
``"required"`` becomes ``{"type": "any"}``, function dicts become
``{"type": "tool", "name": ...}``, and Anthropic-shaped dicts pass
through unchanged. Canonical parameters with no mapped Anthropic
equivalent (``_UNSUPPORTED_CANONICAL``) raise ``BadRequestError``
rather than passing through silently — including when smuggled in via
``extra_parameters``; other ``extra_parameters`` still pass through
unless ``strict=True``.

Streaming tool use accumulates per-stream state; every live stream
gets its own parser via ``make_chunk_parser()``, so concurrent
streams on one client stay isolated.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dashscope.v2.errors import BadRequestError, DashScopeError, ErrorCode
from dashscope.v2.protocols.base import ChatProtocol, ChatRequest
from dashscope.v2.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
)

__all__ = [
    "AnthropicProtocol",
]

# Anthropic requires max_tokens; used when the caller omits it.
DEFAULT_MAX_TOKENS = 4096

# Canonical parameters this adapter does not map onto the Anthropic
# Messages wire (``tool_choice`` does map — see _translate_tool_choice;
# ``enable_thinking``/``thinking_budget`` reach the wire's ``thinking``
# object only as a pass-through via ``extra_parameters``). Rejected
# whether passed directly or smuggled through ``extra_parameters``,
# because silently dropping them is invisible to the caller (the
# request "works" but the parameter never takes effect).
_UNSUPPORTED_CANONICAL = frozenset(
    {
        "seed",
        "repetition_penalty",
        "presence_penalty",
        "enable_thinking",
        "response_format",
        "n",
        "logprobs",
        "top_logprobs",
        "thinking_budget",
        "enable_search",
        "search_options",
        "parallel_tool_calls",
    },
)

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


# The explicit context-cache marker is an Anthropic block-level field
# (prompt caching), so it maps to the same part-level sibling here.
def _copy_cache_control(
    part: Dict[str, Any],
    out: Dict[str, Any],
) -> None:
    if part.get("cache_control") is not None:
        out["cache_control"] = part["cache_control"]


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "") for p in content if p.get("type") == "text"
    )


def _translate_part(part: Dict[str, Any]) -> Dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        text: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
        _copy_cache_control(part, text)
        return text
    if part_type == "image":
        # Anthropic has no resolution-control equivalent; dropping the
        # hints would silently change token cost, so fail fast.
        if (
            part.get("min_pixels") is not None
            or part.get("max_pixels") is not None
        ):
            raise ValueError(
                "image pixel hints (min_pixels/max_pixels) are supported "
                "only with protocol='native'",
            )
        image: Dict[str, Any] = {
            "type": "image",
            "source": {"type": "url", "url": part["url"]},
        }
        _copy_cache_control(part, image)
        return image
    raise ValueError(
        f"content part {part_type!r} has no Anthropic equivalent; "
        "use protocol='native' for audio/video parts",
    )


def _tool_use_blocks(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks = []
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or ""
        try:
            tool_input = json.loads(arguments) if arguments else {}
        except (TypeError, ValueError):
            tool_input = {"_raw": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id"),
                "name": function.get("name"),
                "input": tool_input,
            },
        )
    return blocks


def _tool_result_block(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": message.get("tool_call_id"),
        "content": _text_of(message.get("content")),
    }


def _translate_message(message: Dict[str, Any]) -> Dict[str, Any]:
    role = message["role"]
    content = message.get("content")
    out: Dict[str, Any] = {"role": role}
    if role == "assistant":
        blocks: List[Dict[str, Any]] = []
        thinking = _thinking_block(message)
        if thinking is not None:
            blocks.append(thinking)
        if message.get("tool_calls") or blocks:
            if isinstance(content, list):
                # Parts lists keep their non-text parts; collapsing
                # them to text here would silently drop images.
                blocks.extend(_translate_part(p) for p in content)
            elif _text_of(content):
                blocks.append({"type": "text", "text": _text_of(content)})
            if message.get("tool_calls"):
                blocks.extend(_tool_use_blocks(message["tool_calls"]))
            out["content"] = blocks
            return out
    if isinstance(content, list):
        content = [_translate_part(p) for p in content]
    out["content"] = content if content is not None else ""
    return out


def _thinking_block(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Strict Anthropic-compatible endpoints reject thinking blocks
    # without a valid signature, and require signed ones replayed
    # verbatim on later turns; unsigned reasoning stays dropped.
    thinking = message.get("reasoning_content")
    signature = message.get("reasoning_signature")
    if thinking and signature:
        return {
            "type": "thinking",
            "thinking": thinking,
            "signature": signature,
        }
    return None


def _translate_tools(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    translated = []
    for tool in tools:
        function = tool.get("function") or tool
        item: Dict[str, Any] = {
            "name": function.get("name"),
            "input_schema": function.get("parameters")
            or {"type": "object", "properties": {}},
        }
        if function.get("description"):
            item["description"] = function["description"]
        translated.append(item)
    return translated


def _translate_tool_choice(
    tool_choice: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    # Canonical strings and the OpenAI function shape map onto the
    # Anthropic object form the wire documents (auto/any/none/tool);
    # Anthropic-shaped dicts — including extra keys such as
    # ``disable_parallel_tool_use`` — pass through unchanged. Anything
    # else raises instead of forwarding: the gateway ignores unknown
    # shapes, so a forwarded value would never take effect.
    if tool_choice in ("auto", "none"):
        return {"type": tool_choice}
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            name = (tool_choice.get("function") or {}).get("name")
            if not name:
                raise ValueError(
                    "tool_choice function dict requires function.name",
                )
            return {"type": "tool", "name": name}
        if choice_type in ("auto", "any", "none", "tool"):
            return dict(tool_choice)
    raise ValueError(
        f"tool_choice {tool_choice!r} has no Anthropic translation; use "
        "'auto'/'required'/'none', {'type': 'function', 'function': "
        "{'name': ...}}, or an Anthropic-shaped dict",
    )


def _split_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    system_blocks: List[Dict[str, Any]] = []
    wire_messages: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []

    def _flush_tool_results() -> None:
        # Consecutive tool-role messages (parallel tool calls) merge
        # into ONE user message so user/assistant alternation holds.
        if tool_results:
            wire_messages.append(
                {"role": "user", "content": list(tool_results)},
            )
            tool_results.clear()

    for message in messages:
        role = message["role"]
        if role == "system":
            content = message.get("content")
            if isinstance(content, list):
                # Text parts become individual blocks so a cache_control
                # marker survives; non-text system content keeps the
                # historical text-only extraction.
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_blocks.append(_translate_part(part))
                    elif isinstance(part, str):
                        system_blocks.append({"type": "text", "text": part})
                    else:
                        system_blocks.append({"type": "text", "text": ""})
            else:
                system_blocks.append(
                    {"type": "text", "text": _text_of(content)},
                )
        elif role == "tool":
            tool_results.append(_tool_result_block(message))
        else:
            translated = _translate_message(message)
            if role == "user" and tool_results and translated.get("content"):
                # Join the pending tool results rather than emitting a
                # second consecutive user message: tool_result blocks may
                # share a user turn with text as long as they come first,
                # and alternation has to hold.
                content = translated["content"]
                blocks = (
                    content
                    if isinstance(content, list)
                    else [{"type": "text", "text": content}]
                )
                wire_messages.append(
                    {"role": "user", "content": [*tool_results, *blocks]},
                )
                tool_results.clear()
            else:
                _flush_tool_results()
                wire_messages.append(translated)
    _flush_tool_results()
    return wire_messages, system_blocks


def _reject_unsupported(candidates: Tuple[Tuple[str, Any], ...]) -> None:
    # Unmapped canonical parameters raise instead of passing through:
    # silently dropping them is invisible to the caller (the gateway
    # ignores unknown keys), so the request "works" but the parameter
    # never takes effect.
    names = sorted(key for key, value in candidates if value is not None)
    if not names:
        return
    hint = ""
    if "enable_thinking" in names or "thinking_budget" in names:
        # The wire does have a thinking equivalent (thinking.type /
        # thinking.budget_tokens); the canonical pair stays unmapped as a
        # product-scoping decision, so point at the pass-through instead
        # of implying a wire limitation.
        hint = (
            "; for thinking control pass the native 'thinking' object "
            "via extra_parameters"
        )
    raise BadRequestError(
        f"parameter(s) {names} are not supported by protocol='anthropic' "
        f"(not mapped to an Anthropic Messages API field{hint}); drop "
        "them or switch to protocol='native'/'openai'",
        code=ErrorCode.INVALID_PARAMETER.value,
        status_code=400,
    )


def _apply_extra_parameters(
    body: Dict[str, Any],
    strict: bool,
    extra_parameters: Optional[Dict[str, Any]],
) -> None:
    if not extra_parameters:
        return
    if strict:
        raise DashScopeError(
            "unknown Anthropic parameters in strict mode: "
            f"{sorted(extra_parameters)}; drop them or use strict=False",
        )
    body.update(extra_parameters)


def _canonical_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Usage bag with the derived ``total_tokens`` filled in.

    Anthropic already spells the canonical counters ``input_tokens`` /
    ``output_tokens``, so only the total is computed here; the rest of the
    bag (``cache_creation_input_tokens``, ``cache_read_input_tokens``,
    ``service_tier``) reaches ``Usage`` as an extra. That matches the
    streaming parser, which accumulates the whole bag across events —
    reducing it to three numbers would silently drop billing detail the
    caller cannot recover.
    """
    canonical = dict(usage)
    canonical["total_tokens"] = usage.get("input_tokens", 0) + usage.get(
        "output_tokens",
        0,
    )
    return canonical


class _StreamParser:
    """Per-stream parsing state for Anthropic SSE events.

    Tool-use blocks stream as incremental JSON fragments and usage
    accumulates across events, so every live stream needs its own
    parser instance; sharing one would mix chunks across streams.
    """

    def __init__(self) -> None:
        self._tools: Dict[int, Dict[str, Any]] = {}
        self._usage: Dict[str, int] = {}

    def parse(self, payload: Dict[str, Any]) -> Optional[ChatCompletionChunk]:
        handlers = {
            "message_start": self._message_start,
            "content_block_start": self._block_start,
            "content_block_delta": self._block_delta,
            "message_delta": self._message_delta,
        }
        handler = handlers.get(payload.get("type"))
        if handler is None:
            # ping / content_block_stop / message_stop / unknown events
            # carry nothing user-facing; skip them. message_stop still
            # resets per-message tool state.
            if payload.get("type") == "message_stop":
                self._tools = {}
            return None
        parsed = handler(payload)
        if parsed is None:
            return None
        request_id, model, choices, usage = parsed
        if usage is not None:
            usage["total_tokens"] = usage.get(
                "input_tokens",
                0,
            ) + usage.get("output_tokens", 0)
        return ChatCompletionChunk(
            request_id=request_id,
            model=model,
            choices=choices,
            usage=usage,
        )

    def _message_start(self, payload: Dict[str, Any]):
        message = payload.get("message") or {}
        self._tools = {}
        self._usage = {}
        usage: Optional[Dict[str, Any]] = None
        start_usage = message.get("usage") or {}
        if start_usage:
            self._usage.update(start_usage)
            usage = dict(self._usage)
        return (
            message.get("id") or payload.get("id"),
            message.get("model") or payload.get("model"),
            [],
            usage,
        )

    def _block_start(self, payload: Dict[str, Any]):
        block = payload.get("content_block") or {}
        if block.get("type") != "tool_use":
            # Text/audio block starts carry no content yet; skip.
            return None
        choices: List[Dict[str, Any]] = []
        self._tools[payload.get("index", 0)] = {
            "id": block.get("id"),
            "name": block.get("name"),
            "arguments": "",
        }
        choices.append(
            {
                "index": 0,
                "delta": {
                    "tool_calls": self._cumulative_tool_calls(),
                },
            },
        )
        return payload.get("id"), payload.get("model"), choices, None

    def _block_delta(self, payload: Dict[str, Any]):
        delta_payload = payload.get("delta") or {}
        delta_type = delta_payload.get("type")
        delta: Dict[str, Any] = {}
        if delta_type == "text_delta":
            delta["content"] = delta_payload.get("text", "")
        elif delta_type == "thinking_delta":
            delta["reasoning_content"] = delta_payload.get("thinking", "")
        elif delta_type == "signature_delta":
            delta["reasoning_signature"] = delta_payload.get("signature", "")
        elif delta_type == "input_json_delta":
            tool = self._tools.get(payload.get("index", 0))
            if tool is not None:
                tool["arguments"] += delta_payload.get("partial_json", "")
                delta["tool_calls"] = self._cumulative_tool_calls()
        if not delta:
            # Unknown delta types (e.g. citations_delta) carry nothing.
            return None
        choices = [{"index": 0, "delta": delta}]
        return payload.get("id"), payload.get("model"), choices, None

    def _message_delta(self, payload: Dict[str, Any]):
        choices: List[Dict[str, Any]] = []
        usage: Optional[Dict[str, Any]] = None
        stop_reason = (payload.get("delta") or {}).get("stop_reason")
        end_usage = payload.get("usage") or {}
        if end_usage:
            self._usage.update(end_usage)
            usage = dict(self._usage)
        if stop_reason is not None:
            choices.append(
                {
                    "index": 0,
                    "finish_reason": _STOP_REASON_MAP.get(
                        stop_reason,
                        stop_reason,
                    ),
                    "delta": {},
                },
            )
        if not choices and usage is None:
            return None
        return payload.get("id"), payload.get("model"), choices, usage

    def _cumulative_tool_calls(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": tool["id"],
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "arguments": tool["arguments"],
                },
            }
            for _, tool in sorted(self._tools.items())
        ]


class AnthropicProtocol(ChatProtocol):
    """Adapter for ``POST /v1/messages`` (Anthropic Messages API)."""

    name = "anthropic"
    PATH = "/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, *, strict: bool = False) -> None:
        super().__init__(strict=strict)
        self._default_parser = _StreamParser()

    def make_chunk_parser(
        self,
    ) -> Callable[[Dict[str, Any]], Optional[ChatCompletionChunk]]:
        """Return a fresh parser so concurrent streams stay isolated."""
        return _StreamParser().parse

    def build_chat_request(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Union[str, List[str], None] = None,
        seed: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        tool_choice: Union[str, Dict[str, Any], None] = None,
        n: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        enable_search: Optional[bool] = None,
        search_options: Optional[Dict[str, Any]] = None,
        parallel_tool_calls: Optional[bool] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> ChatRequest:
        _reject_unsupported(
            (
                ("seed", seed),
                ("repetition_penalty", repetition_penalty),
                ("presence_penalty", presence_penalty),
                ("enable_thinking", enable_thinking),
                ("response_format", response_format),
                ("n", n),
                ("logprobs", logprobs),
                ("top_logprobs", top_logprobs),
                ("thinking_budget", thinking_budget),
                ("enable_search", enable_search),
                ("search_options", search_options),
                ("parallel_tool_calls", parallel_tool_calls),
            ),
        )
        wire_messages, system_blocks = _split_messages(messages)
        body: Dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        }
        if system_blocks:
            if any("cache_control" in b for b in system_blocks):
                # A cache_control marker needs the block-list form of the
                # Anthropic ``system`` field; without one keep the joined
                # string so the wire stays byte-identical.
                body["system"] = system_blocks
            else:
                body["system"] = "\n\n".join(b["text"] for b in system_blocks)
        if stream:
            body["stream"] = True
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if tools:
            body["tools"] = _translate_tools(tools)
        # Emitted even without tools so the value is never silently
        # dropped; a tool_choice without tools gets a clear server-side
        # 400 instead.
        if tool_choice is not None:
            body["tool_choice"] = _translate_tool_choice(tool_choice)
        # The hatch below is applied verbatim (body.update), so a
        # canonical or OpenAI-shaped tool_choice smuggled through
        # extra_parameters must go through the same translator first;
        # otherwise a bare "required" reaches the wire and the gateway
        # silently ignores it. Native shapes pass through the translator
        # unchanged, so this converges both entry points and keeps the
        # hatch-wins precedence. Copy first: never mutate the caller's
        # dict.
        if (
            extra_parameters
            and extra_parameters.get("tool_choice") is not None
        ):
            extra_parameters = dict(extra_parameters)
            extra_parameters["tool_choice"] = _translate_tool_choice(
                extra_parameters["tool_choice"],
            )
        # Reject unsupported canonical names even when passed via
        # extra_parameters, so they can never slip through silently.
        _reject_unsupported(
            tuple(
                (key, value)
                for key, value in (extra_parameters or {}).items()
                if key in _UNSUPPORTED_CANONICAL
            ),
        )
        _apply_extra_parameters(body, self._strict, extra_parameters)
        return ChatRequest(
            path=path or self.PATH,
            body=body,
            headers={"anthropic-version": self.API_VERSION},
        )

    def parse_completion(self, payload: Dict[str, Any]) -> ChatCompletion:
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        signature_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in payload.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                # Parity with the streaming thinking_delta path. The
                # signature is kept so the turn can be echoed back
                # to the API; redacted_thinking carries no readable
                # text and stays dropped (verbatim replay of it is
                # unsupported by the flat canonical model).
                thinking_parts.append(block.get("thinking", ""))
                if block.get("signature"):
                    signature_parts.append(block["signature"])
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(
                                block.get("input") or {},
                                ensure_ascii=False,
                            ),
                        },
                    },
                )
        message: Dict[str, Any] = {
            "role": payload.get("role") or "assistant",
            "content": "".join(text_parts) or None,
        }
        reasoning_content = "".join(thinking_parts)
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        # Flat canonical model: a single thinking block round-trips;
        # interleaved-thinking responses (several signed blocks) are
        # reduced like their text and cannot be replayed faithfully.
        reasoning_signature = "".join(signature_parts)
        if reasoning_signature:
            message["reasoning_signature"] = reasoning_signature
        if tool_calls:
            message["tool_calls"] = tool_calls
        stop_reason = payload.get("stop_reason")
        usage = payload.get("usage") or {}
        return ChatCompletion(
            request_id=payload.get("id"),
            model=payload.get("model"),
            choices=[
                {
                    "index": 0,
                    "finish_reason": _STOP_REASON_MAP.get(
                        stop_reason,
                        stop_reason,
                    ),
                    "message": message,
                },
            ],
            usage=_canonical_usage(usage) if usage else None,
        )

    def parse_chunk(
        self,
        payload: Dict[str, Any],
    ) -> Optional[ChatCompletionChunk]:
        # Sequential convenience path for direct calls; resources take
        # a per-stream parser from make_chunk_parser() instead.
        return self._default_parser.parse(payload)
