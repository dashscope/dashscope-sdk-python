# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Protocol adapters (native / openai / anthropic).

See design doc §3 for the translation contract.
"""

from typing import Any, Dict, Optional

from dashscope.v2._config import VALID_PROTOCOLS
from dashscope.v2.protocols.anthropic import AnthropicProtocol
from dashscope.v2.protocols.base import ChatProtocol, ChatRequest
from dashscope.v2.protocols.native import NativeProtocol
from dashscope.v2.protocols.openai import OpenAIProtocol

__all__ = [
    "AnthropicProtocol",
    "ChatProtocol",
    "ChatRequest",
    "NativeProtocol",
    "OpenAIProtocol",
    "get_protocol",
]

_REGISTRY = {
    NativeProtocol.name: NativeProtocol,
    OpenAIProtocol.name: OpenAIProtocol,
    AnthropicProtocol.name: AnthropicProtocol,
}


def get_protocol(
    name: str,
    protocol_options: Optional[Dict[str, Any]] = None,
) -> ChatProtocol:
    """Resolve a protocol adapter by name, passing adapter options."""
    protocol_cls = _REGISTRY.get(name)
    if protocol_cls is None:
        raise ValueError(
            f"unknown protocol {name!r}, expected one of {VALID_PROTOCOLS}",
        )
    try:
        return protocol_cls(  # type: ignore[abstract]
            **dict(protocol_options or {}),
        )
    except TypeError as exc:
        raise ValueError(
            f"invalid protocol_options for protocol {name!r}: {exc}",
        ) from exc
