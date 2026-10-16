# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Canonical chat message model for the dashscope 2.0 SDK.

All resources and protocol adapters speak this model; translation to
native/openai/anthropic wire formats happens in the protocol layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "AudioPart",
    "ImagePart",
    "Message",
    "MessageInput",
    "TextPart",
    "ToolCall",
    "ToolCallFunction",
    "VideoPart",
    "extract_text",
    "has_media_parts",
    "normalize_messages",
]


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str
    # Explicit context-cache marker (Bailian 上下文缓存): a part-level
    # sibling such as {"type": "ephemeral"} on the native and
    # OpenAI-compatible wires; the cached prefix ends at this part.
    # 1.x passed content blocks through unvalidated, so the marker must
    # stay accepted on every part type.
    cache_control: Optional[Dict[str, Any]] = None


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    url: str
    detail: Optional[str] = None
    # Resolution hints are part-level siblings on both the native and
    # OpenAI-compatible wires (Qwen-VL/Omni). An adapter with no
    # equivalent (Anthropic) fails fast rather than dropping them: a
    # dropped cap silently changes token cost and model behaviour.
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    cache_control: Optional[Dict[str, Any]] = None


class AudioPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["audio"] = "audio"
    # ``url`` and inline base64 ``data`` are the two possible sources;
    # at least one is required. ``format`` describes ``data`` (e.g.
    # "mp3", "wav") and lets the native adapter build the data: URI.
    url: Optional[str] = None
    data: Optional[str] = None
    format: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _require_source(self) -> "AudioPart":
        if self.url is None and self.data is None:
            raise ValueError("audio part requires either url or data")
        return self


class VideoPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["video"] = "video"
    url: Optional[str] = None
    frames: Optional[List[str]] = None
    # Part-level sampling/resolution hints on the native and
    # OpenAI-compatible wires (Qwen-VL/Omni); same fail-fast rule as the
    # image hints above for adapters without an equivalent.
    fps: Optional[float] = None
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    total_pixels: Optional[int] = None
    cache_control: Optional[Dict[str, Any]] = None


ContentPart = Union[TextPart, ImagePart, AudioPart, VideoPart]


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    # name is absent on non-initial OpenAI stream fragments.
    name: Optional[str] = None
    arguments: str = ""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    # id is absent on non-initial OpenAI stream fragments; index
    # addresses the fragment's slot in the tool_calls array.
    id: Optional[str] = None
    type: str = "function"
    function: ToolCallFunction = ToolCallFunction()
    index: Optional[int] = None


class Message(BaseModel):
    """One chat message; ``content`` is a string, parts list, or None
    (assistant tool-call messages)."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, List[ContentPart], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None
    # Opaque vendor signature for ``reasoning_content`` (Anthropic
    # thinking blocks); required to echo the turn back verbatim.
    reasoning_signature: Optional[str] = None


MessageInput = Union[Message, Dict[str, Any], str]


def _normalize_part(  # pylint: disable=too-many-return-statements,too-many-branches
    part: Any,
) -> ContentPart:
    if isinstance(part, (TextPart, ImagePart, AudioPart, VideoPart)):
        return part
    if isinstance(part, str):
        return TextPart(text=part)
    if isinstance(part, dict):
        data = dict(part)
        part_type = data.pop("type", None)
        if part_type == "text" or "text" in data:
            return TextPart(text=data.pop("text", ""), **data)
        if (
            part_type in ("image", "image_url")
            or "image" in data
            or "image_url" in data
        ):
            url = data.pop("image", None)
            if url is None:
                url = data.pop("image_url", None)
            if isinstance(url, dict):
                # OpenAI-wire nesting: promote detail/pixel hints to
                # part level; explicit part-level keys win on conflict.
                nested, url = url, url.get("url")
                for key in ("detail", "min_pixels", "max_pixels"):
                    if key in nested:
                        data.setdefault(key, nested[key])
            if url is None and "url" in data:
                # Flat canonical dump {"type": "image", "url": ...}.
                return ImagePart(**data)
            return ImagePart(url=url, **data)
        if (
            part_type in ("audio", "input_audio")
            or "audio" in data
            or "input_audio" in data
        ):
            audio = data.pop("audio", None)
            if audio is None:
                audio = data.pop("input_audio", None)
            if isinstance(audio, dict):
                # input_audio wire shape carries data/format (or url).
                return AudioPart(**{**audio, **data})
            if audio is None:
                # Flat canonical dump {"type": "audio", ...}.
                return AudioPart(**data)
            return AudioPart(url=audio, **data)
        if (
            part_type in ("video", "video_url")
            or "video" in data
            or "video_url" in data
        ):
            video = data.pop("video", None)
            if video is None:
                video = data.pop("video_url", None)
            if isinstance(video, dict):
                return VideoPart(**{**video, **data})
            if isinstance(video, list):
                return VideoPart(frames=video, **data)
            if video is None and ("url" in data or "frames" in data):
                # Flat canonical dump {"type": "video", ...}.
                return VideoPart(**data)
            return VideoPart(url=video, **data)
    raise ValueError(f"unsupported content part: {part!r}")


def _normalize_message(message: MessageInput) -> Message:
    if isinstance(message, Message):
        return message
    if isinstance(message, str):
        return Message(role="user", content=message)
    if not isinstance(message, dict):
        raise ValueError(f"unsupported message: {message!r}")
    data = dict(message)
    content = data.get("content")
    if isinstance(content, list):
        data["content"] = [_normalize_part(p) for p in content]
    return Message(**data)


def normalize_messages(
    messages: Union[MessageInput, List[MessageInput]],
) -> List[Message]:
    if isinstance(messages, (str, dict, Message)):
        messages = [messages]
    if not messages:
        raise ValueError("messages must not be empty")
    return [_normalize_message(m) for m in messages]


def extract_text(message: Message) -> str:
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if isinstance(p, TextPart))


def has_media_parts(messages: List[Message]) -> bool:
    """True when any message carries image/audio/video content."""
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            continue
        if any(not isinstance(part, TextPart) for part in content):
            return True
    return False
