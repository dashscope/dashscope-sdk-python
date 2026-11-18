# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed models for the OpenAI-compatible Conversations API (C2).

A conversation is server-side context for the Responses API: its items
are injected as history automatically, and each response's input/output
is appended back to it. The wire is a flat OpenAI ``conversation`` object
plus ``message`` items, modelled here independently of chat.

Every model is permissive (``extra="allow"``, optional defaulted fields)
and parses through :func:`~dashscope.types._parse.parse_model`, so a
drifted field degrades rather than sinks an otherwise healthy reply.
``created_at`` is Unix **milliseconds** (unlike the response object's
seconds). The service keeps the latest 100 items within 7 days.

These types are exported under ``dashscope.types`` (not the top-level
``dashscope`` namespace, which already carries the 1.x ``Conversation``).

Contract:
https://help.aliyun.com/zh/model-studio/openai-compatible-conversations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

from dashscope.types._parse import parse_model
from dashscope.types.common import ResponseBase, response_envelope

if TYPE_CHECKING:
    import httpx

__all__ = [
    "Conversation",
    "ConversationContentPart",
    "ConversationDeleted",
    "ConversationItem",
    "ConversationItemDeleted",
    "parse_conversation",
    "parse_conversation_deleted",
    "parse_conversation_item",
    "parse_conversation_item_deleted",
]


class ConversationContentPart(BaseModel):
    """A content block: ``input_text`` (user) or ``output_text`` (model)."""

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = None
    text: Optional[str] = None


class Conversation(ResponseBase):
    """A conversation (create / get / update). ``object`` is
    ``conversation``; ``metadata`` is at most 16 string key/value pairs."""

    id: Optional[str] = None
    object: Optional[str] = "conversation"
    created_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationDeleted(ResponseBase):
    """Result of ``conversations.delete``. Deleting a conversation does
    not delete its items. ``object`` is ``conversation.deleted``."""

    id: Optional[str] = None
    deleted: bool = False
    object: Optional[str] = "conversation.deleted"


class ConversationItem(ResponseBase):
    """One ``message`` item. ``content`` is a plain string or a list of
    content blocks; ``status`` is ``in_progress`` / ``completed`` /
    ``incomplete``. Returned top-level by ``items.get`` and nested in
    ``items.list`` / ``items.create``."""

    id: Optional[str] = None
    type: Optional[str] = "message"
    role: Optional[str] = None
    status: Optional[str] = None
    content: Union[str, List[ConversationContentPart]] = []

    @property
    def text(self) -> str:
        """Concatenated text of the content (string or block list)."""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content if part.text)


class ConversationItemDeleted(ResponseBase):
    """Result of ``items.delete``. ``object`` is
    ``conversation.item.deleted``."""

    id: Optional[str] = None
    deleted: bool = False
    object: Optional[str] = "conversation.item.deleted"


def parse_conversation(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> Conversation:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        Conversation,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def parse_conversation_deleted(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> ConversationDeleted:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        ConversationDeleted,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def parse_conversation_item(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> ConversationItem:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        ConversationItem,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def parse_conversation_item_deleted(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> ConversationItemDeleted:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        ConversationItemDeleted,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )