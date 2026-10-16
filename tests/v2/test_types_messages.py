# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the canonical message model (M1 task 1.5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dashscope.v2.types import (
    AudioPart,
    ImagePart,
    Message,
    TextPart,
    VideoPart,
    extract_text,
    normalize_messages,
)
from dashscope.v2.types.messages import has_media_parts


def test_plain_string_message():
    (msg,) = normalize_messages("Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert extract_text(msg) == "Hello"


def test_dict_messages():
    msgs = normalize_messages(
        [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "reasoning_content": "r",
            },
        ],
    )
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2].reasoning_content == "r"


def test_message_model_passthrough():
    original = Message(role="user", content="x")
    (msg,) = normalize_messages([original])
    assert msg is original


def test_native_multimodal_parts():
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"text": "describe"},
                    {"image": "https://example.com/a.png"},
                    {"audio": "https://example.com/a.wav"},
                    {
                        "video": [
                            "https://example.com/f1.jpg",
                            "https://example.com/f2.jpg",
                        ],
                    },
                ],
            },
        ],
    )
    parts = msg.content
    assert isinstance(parts[0], TextPart) and parts[0].text == "describe"
    assert isinstance(parts[1], ImagePart) and parts[1].url.endswith("a.png")
    assert isinstance(parts[2], AudioPart)
    assert isinstance(parts[3], VideoPart) and len(parts[3].frames) == 2
    assert extract_text(msg) == "describe"


def test_openai_style_parts():
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is here?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/b.png",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
    )
    parts = msg.content
    assert isinstance(parts[0], TextPart)
    assert isinstance(parts[1], ImagePart)
    assert parts[1].url.endswith("b.png")
    assert parts[1].detail == "low"


def test_video_url_form():
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [{"video": "https://example.com/v.mp4"}],
            },
        ],
    )
    part = msg.content[0]
    assert isinstance(part, VideoPart)
    assert part.url.endswith("v.mp4")
    assert part.frames is None


def test_input_audio_data_format():
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "QUJD", "format": "mp3"},
                    },
                ],
            },
        ],
    )
    part = msg.content[0]
    assert isinstance(part, AudioPart)
    assert part.url is None
    assert part.data == "QUJD"
    assert part.format == "mp3"


def test_audio_part_requires_a_source():
    with pytest.raises(ValidationError, match="url or data"):
        AudioPart()
    # The flat canonical dump round-trips through normalization.
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": "https://example.com/a.wav"},
                ],
            },
        ],
    )
    assert msg.content[0].url.endswith("a.wav")


def test_image_part_pixel_hints():
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/a.png",
                            "min_pixels": 64,
                            "max_pixels": 1024,
                        },
                    },
                ],
            },
        ],
    )
    part = msg.content[0]
    assert isinstance(part, ImagePart)
    assert part.min_pixels == 64
    assert part.max_pixels == 1024
    # An explicit part-level key wins over the nested placement.
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/a.png",
                            "min_pixels": 64,
                        },
                        "min_pixels": 128,
                    },
                ],
            },
        ],
    )
    assert msg.content[0].min_pixels == 128


def test_cache_control_marker_accepted():
    # 1.x passed content blocks through unvalidated, so a context-cache
    # marker must survive normalization on every part type instead of
    # raising ValidationError (a ValueError subclass) pre-flight.
    (msg,) = normalize_messages(
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "<long prompt>",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
    )
    part = msg.content[0]
    assert isinstance(part, TextPart)
    assert part.cache_control == {"type": "ephemeral"}

    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://example.com/a.png",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "audio",
                        "url": "https://example.com/a.wav",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "video",
                        "url": "https://example.com/v.mp4",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
    )
    for part in msg.content:
        assert part.cache_control == {"type": "ephemeral"}


def test_video_url_part_with_hints():
    (msg,) = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/v.mp4"},
                        "fps": 2,
                        "min_pixels": 64,
                        "max_pixels": 2048,
                        "total_pixels": 4096,
                    },
                ],
            },
        ],
    )
    part = msg.content[0]
    assert isinstance(part, VideoPart)
    assert part.url.endswith("v.mp4")
    assert part.fps == 2
    assert part.min_pixels == 64
    assert part.max_pixels == 2048
    assert part.total_pixels == 4096


def test_tool_messages():
    (assistant, tool) = normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "hz"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
    )
    assert assistant.content is None
    assert assistant.tool_calls[0].function.name == "get_weather"
    assert tool.tool_call_id == "call_1"


def test_extra_fields_preserved():
    (msg,) = normalize_messages(
        [{"role": "user", "content": "x", "custom_field": 1}],
    )
    assert msg.model_extra.get("custom_field") == 1


def test_bare_dict_message():
    (msg,) = normalize_messages({"role": "user", "content": "hi"})
    assert msg.role == "user"
    assert msg.content == "hi"


def test_invalid_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_messages([])
    with pytest.raises(ValueError, match="unsupported content part"):
        normalize_messages([{"role": "user", "content": [{"unknown": 1}]}])
    with pytest.raises(ValidationError, match="url or data"):
        normalize_messages([{"role": "user", "content": [{"type": "audio"}]}])
    with pytest.raises(ValueError, match="unsupported message"):
        normalize_messages([123])
    with pytest.raises(ValidationError):
        Message(role="user", content=[{"type": "text"}])


def test_roundtrip_serialization():
    msgs = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"text": "hi"},
                    {"image": "https://example.com/a.png"},
                ],
            },
        ],
    )
    dumped = msgs[0].model_dump(exclude_none=True)
    assert dumped["content"] == [
        {"type": "text", "text": "hi"},
        {"type": "image", "url": "https://example.com/a.png"},
    ]


def test_has_media_parts():
    assert not has_media_parts(normalize_messages("hi"))
    assert not has_media_parts(
        normalize_messages([{"role": "user", "content": [{"text": "hi"}]}]),
    )
    for content in (
        [{"image": "https://example.com/a.png"}],
        [{"audio": "https://example.com/a.wav"}],
        [{"video": "https://example.com/a.mp4"}],
        [{"text": "describe"}, {"image": "https://example.com/a.png"}],
    ):
        assert has_media_parts(
            normalize_messages([{"role": "user", "content": content}]),
        )
