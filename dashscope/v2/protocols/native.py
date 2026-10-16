# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native DashScope wire protocol adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dashscope.v2.protocols.base import ChatProtocol, ChatRequest
from dashscope.v2.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
    parse_chunk,
    parse_completion,
)

__all__ = [
    "NativeProtocol",
]


# Media control hints travel as part-level siblings on the native wire
# (Qwen-VL/Omni); the native protocol is the only one that carries them.
_IMAGE_HINT_KEYS = ("min_pixels", "max_pixels")
_VIDEO_HINT_KEYS = ("fps", "min_pixels", "max_pixels", "total_pixels")
# Explicit context-cache marker: also a part-level sibling (the cached
# prefix ends at the marked content block), on text and multimodal
# parts alike.
_CACHE_CONTROL_KEYS = ("cache_control",)


def _copy_hints(part: Dict[str, Any], out: Dict[str, Any], keys) -> None:
    for key in keys:
        if part.get(key) is not None:
            out[key] = part[key]


def _translate_part(part: Dict[str, Any]) -> Dict[str, Any]:
    part_type = part.get("type")
    if part_type == "text":
        text: Dict[str, Any] = {"text": part.get("text", "")}
        _copy_hints(part, text, _CACHE_CONTROL_KEYS)
        return text
    if part_type == "image":
        image: Dict[str, Any] = {"image": part["url"]}
        _copy_hints(part, image, _IMAGE_HINT_KEYS)
        _copy_hints(part, image, _CACHE_CONTROL_KEYS)
        return image
    if part_type == "audio":
        if part.get("url") is not None:
            audio: Dict[str, Any] = {"audio": part["url"]}
            _copy_hints(part, audio, _CACHE_CONTROL_KEYS)
            return audio
        # Inline base64 audio travels as a data: URI on the native wire.
        data = part.get("data")
        if data is None:
            raise ValueError("audio part requires either 'url' or 'data'")
        audio_format = part.get("format")
        if audio_format is None:
            raise ValueError(
                "audio part 'format' is required with protocol='native' "
                "when 'data' carries inline base64 audio",
            )
        inline: Dict[str, Any] = {
            "audio": f"data:audio/{audio_format};base64,{data}",
        }
        _copy_hints(part, inline, _CACHE_CONTROL_KEYS)
        return inline
    if part_type == "video":
        if part.get("frames"):
            video: Dict[str, Any] = {"video": part["frames"]}
        else:
            video = {"video": part.get("url")}
        _copy_hints(part, video, _VIDEO_HINT_KEYS)
        _copy_hints(part, video, _CACHE_CONTROL_KEYS)
        return video
    return part


def _translate_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    translated = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message = dict(message)
            message["content"] = [_translate_part(p) for p in content]
        translated.append(message)
    return translated


class NativeProtocol(ChatProtocol):
    """DashScope native protocol: /services/... + input/parameters body."""

    name = "native"

    PATH = "/services/aigc/text-generation/generation"
    MULTIMODAL_PATH = "/services/aigc/multimodal-generation/generation"
    SSE_HEADERS = {"X-DashScope-SSE": "enable"}

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
        parameters: Dict[str, Any] = {"result_format": "message"}
        if stream:
            parameters["incremental_output"] = True
        for key, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("max_tokens", max_tokens),
            ("stop", stop),
            ("seed", seed),
            ("repetition_penalty", repetition_penalty),
            ("presence_penalty", presence_penalty),
            ("enable_thinking", enable_thinking),
            ("response_format", response_format),
            ("top_k", top_k),
            ("tool_choice", tool_choice),
            ("n", n),
            ("logprobs", logprobs),
            ("top_logprobs", top_logprobs),
            ("thinking_budget", thinking_budget),
            ("enable_search", enable_search),
            ("search_options", search_options),
            ("parallel_tool_calls", parallel_tool_calls),
            ("tools", tools),
        ):
            if value is not None:
                parameters[key] = value
        if extra_parameters:
            parameters.update(extra_parameters)
        body = {
            "model": model,
            "input": {"messages": _translate_messages(messages)},
            "parameters": parameters,
        }
        headers = dict(self.SSE_HEADERS) if stream else {}
        return ChatRequest(path=path or self.PATH, body=body, headers=headers)

    def parse_completion(
        self,
        payload: Dict[str, Any],
    ) -> ChatCompletion:
        return parse_completion(payload)

    def parse_chunk(
        self,
        payload: Dict[str, Any],
    ) -> ChatCompletionChunk:
        return parse_chunk(payload)
