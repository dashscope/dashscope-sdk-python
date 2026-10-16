# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""OpenAI-compatible protocol adapter (design doc §3.1).

Translates the canonical chat models to ``/chat/completions`` requests
so ``Client(protocol="openai", base_url=...)`` can talk to any
OpenAI-compatible endpoint (Bailian compatible-mode, third-party
gateways). Unknown parameters pass through to the body unless
``strict=True``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dashscope.v2.errors import DashScopeError
from dashscope.v2.protocols.base import ChatProtocol, ChatRequest
from dashscope.v2.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
)

__all__ = [
    "OpenAIProtocol",
]


_IMAGE_HINT_KEYS = ("min_pixels", "max_pixels")
_VIDEO_HINT_KEYS = ("fps", "min_pixels", "max_pixels", "total_pixels")
# Explicit context-cache marker rides as a part-level sibling on the
# compatible-mode wire (documented example: {"type": "text", "text": ...,
# "cache_control": {"type": "ephemeral"}}).
_CACHE_CONTROL_KEYS = ("cache_control",)


def _copy_cache_control(
    part: Dict[str, Any],
    out: Dict[str, Any],
) -> None:
    for key in _CACHE_CONTROL_KEYS:
        if part.get(key) is not None:
            out[key] = part[key]


def _image_part(part: Dict[str, Any]) -> Dict[str, Any]:
    image_url: Dict[str, Any] = {"url": part["url"]}
    if part.get("detail"):
        image_url["detail"] = part["detail"]
    out: Dict[str, Any] = {"type": "image_url", "image_url": image_url}
    # Qwen-VL/Omni resize hints ride as part-level siblings of the
    # content item on the compatible-mode wire (documented example:
    # {"type": "image_url", "image_url": {...}, "min_pixels": N}).
    for key in _IMAGE_HINT_KEYS:
        if part.get(key) is not None:
            out[key] = part[key]
    _copy_cache_control(part, out)
    return out


def _audio_part(part: Dict[str, Any]) -> Dict[str, Any]:
    # Qwen-Omni input_audio: ``data`` carries a public URL or inline
    # base64 and ``format`` names the codec; both live inside the
    # input_audio object on the compatible-mode wire.
    input_audio: Dict[str, Any] = {}
    if part.get("data") is not None:
        input_audio["data"] = part["data"]
    elif part.get("url") is not None:
        input_audio["data"] = part["url"]
    if not input_audio:
        raise ValueError("audio part requires 'url' or 'data'")
    if part.get("format") is not None:
        input_audio["format"] = part["format"]
    out: Dict[str, Any] = {"type": "input_audio", "input_audio": input_audio}
    _copy_cache_control(part, out)
    return out


def _video_part(part: Dict[str, Any]) -> Dict[str, Any]:
    # A single file maps to video_url and a frame list to video; the
    # fps/pixel hints are part-level siblings on the compatible-mode
    # wire ({"type": "video_url", "video_url": {...}, "fps": 2}).
    if part.get("frames"):
        out: Dict[str, Any] = {"type": "video", "video": part["frames"]}
    else:
        out = {"type": "video_url", "video_url": {"url": part["url"]}}
    for key in _VIDEO_HINT_KEYS:
        if part.get(key) is not None:
            out[key] = part[key]
    _copy_cache_control(part, out)
    return out


def _translate_part(part: Dict[str, Any]) -> Dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        out: Dict[str, Any] = {"type": "text", "text": part.get("text", "")}
        _copy_cache_control(part, out)
        return out
    if part_type == "image":
        return _image_part(part)
    if part_type == "audio":
        return _audio_part(part)
    if part_type == "video":
        return _video_part(part)
    raise ValueError(
        f"content part {part_type!r} has no OpenAI equivalent",
    )


def _translate_message(message: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": message["role"]}
    content = message.get("content")
    if isinstance(content, list):
        content = [_translate_part(p) for p in content]
    out["content"] = content
    for key in ("name", "tool_call_id"):
        if message.get(key) is not None:
            out[key] = message[key]
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    # Reasoning models require prior turns' reasoning_content
    # echoed back; only assistant turns carry it.
    if message["role"] == "assistant" and message.get("reasoning_content"):
        out["reasoning_content"] = message["reasoning_content"]
    return out


# Renamed into the canonical counters, so the same number is not also
# readable as an extra under its wire spelling.
_USAGE_RENAMED = ("prompt_tokens", "completion_tokens")


def _translate_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    usage = usage or {}
    # Only the canonical counters are translated; every other field the
    # gateway reports (``prompt_tokens_details.cached_tokens``,
    # ``completion_tokens_details.reasoning_tokens``, ...) reaches ``Usage``
    # as an extra. ``Usage`` is ``extra="allow"`` and the native adapter
    # already hands its usage bag over untouched, so reducing the bag here
    # would silently drop billing detail the caller cannot recover.
    translated = {
        key: value for key, value in usage.items() if key not in _USAGE_RENAMED
    }
    translated["input_tokens"] = usage.get("prompt_tokens", 0)
    translated["output_tokens"] = usage.get("completion_tokens", 0)
    translated.setdefault(
        "total_tokens",
        translated["input_tokens"] + translated["output_tokens"],
    )
    return translated


class OpenAIProtocol(ChatProtocol):
    """Adapter for ``POST /chat/completions`` (OpenAI-compatible)."""

    name = "openai"
    PATH = "/chat/completions"

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
        body: Dict[str, Any] = {
            "model": model,
            "messages": [_translate_message(m) for m in messages],
        }
        if stream:
            body["stream"] = True
            # OpenAI-compatible servers omit usage from streams unless
            # asked; extra_parameters can still override this.
            body["stream_options"] = {"include_usage": True}
        for key, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("max_tokens", max_tokens),
            ("stop", stop),
            ("seed", seed),
            ("presence_penalty", presence_penalty),
            ("response_format", response_format),
            ("tools", tools),
            ("tool_choice", tool_choice),
            ("n", n),
            ("logprobs", logprobs),
            ("top_logprobs", top_logprobs),
            ("parallel_tool_calls", parallel_tool_calls),
            # DashScope extensions Bailian compatible-mode understands;
            # named SDK params, so strict must not reject them. The search
            # runs server-side, but the compatible wire returns no search
            # sources, so search_info stays None on this protocol (the
            # native protocol is where it is parsed).
            ("top_k", top_k),
            ("thinking_budget", thinking_budget),
            ("enable_search", enable_search),
            ("search_options", search_options),
        ):
            if value is not None:
                body[key] = value
        unknown: Dict[str, Any] = {}
        if repetition_penalty is not None:
            unknown["repetition_penalty"] = repetition_penalty
        if enable_thinking is not None:
            unknown["enable_thinking"] = enable_thinking
        if extra_parameters:
            unknown.update(extra_parameters)
        if unknown:
            if self._strict:
                raise DashScopeError(
                    "unknown OpenAI parameters in strict mode: "
                    f"{sorted(unknown)}; drop them or use strict=False",
                )
            body.update(unknown)
        return ChatRequest(path=path or self.PATH, body=body)

    def parse_completion(self, payload: Dict[str, Any]) -> ChatCompletion:
        return ChatCompletion(
            request_id=payload.get("request_id") or payload.get("id"),
            model=payload.get("model"),
            choices=payload.get("choices") or [],
            usage=(
                _translate_usage(payload["usage"])
                if payload.get("usage")
                else None
            ),
        )

    def parse_chunk(self, payload: Dict[str, Any]) -> ChatCompletionChunk:
        choices = []
        for raw in payload.get("choices") or []:
            raw = dict(raw)
            raw.setdefault("delta", {})
            choices.append(raw)
        return ChatCompletionChunk(
            request_id=payload.get("request_id") or payload.get("id"),
            model=payload.get("model"),
            choices=choices,
            usage=(
                _translate_usage(payload["usage"])
                if payload.get("usage")
                else None
            ),
        )
