# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.conversations -- OpenAI-compatible Conversations API (C2).

Server-side conversation context for the Responses API: a conversation's
items are injected as history automatically, and each response's input
and output are appended back to it. Pass ``conversation=conv.id`` to
``client.responses.create`` (mutually exclusive with
``previous_response_id``); the SDK stays a stateless wrapper -- no
client-side history or memory.

The resource covers the full documented surface: conversation
``create`` / ``get`` / ``update`` / ``delete`` and ``items.create`` /
``items.list`` / ``items.get`` / ``items.delete``.

Like ``client.responses`` / ``client.batches``, every call derives its own
base URL via ``compatible_mode_base_url`` (``{gateway}/compatible-mode/v1``)
so the chat ``protocol`` setting cannot move it, and tags ``sdk_module``
``conversations`` for the ``x-dashscope-sdk-client`` header.

Limits (mirrored as local pre-validation): at most 20 items per
create/add; metadata at most 16 string pairs (key <= 64, value <= 512
chars); the service keeps the latest 100 items within 7 days.

Contract:
https://help.aliyun.com/zh/model-studio/openai-compatible-conversations
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from pydantic import BaseModel

from dashscope._config import Config, compatible_mode_base_url
from dashscope.errors import response_error
from dashscope.resources.batches import _CursorPage
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import response_envelope, with_headers
from dashscope.types.conversations import (
    Conversation,
    ConversationDeleted,
    ConversationItem,
    ConversationItemDeleted,
    parse_conversation,
    parse_conversation_deleted,
    parse_conversation_item,
    parse_conversation_item_deleted,
)

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_CONVERSATIONS_PATH = "/conversations"

MODULE = "conversations"

# items.list caps limit at 100 (default 20); order defaults to desc.
_MAX_LIMIT = 100
# create conversation / create items accept at most 20 items per call.
_MAX_ITEMS = 20
_MAX_METADATA_PAIRS = 16
_MAX_METADATA_KEY_LEN = 64
_MAX_METADATA_VALUE_LEN = 512

ItemsInput = Sequence[Union[Mapping[str, Any], BaseModel]]
MetadataInput = Mapping[str, str]


def _endpoint_base(config: Config) -> str:
    # The compatible-mode Conversations API sits under /compatible-mode/v1
    # on every gateway; the chat protocol setting must not move it.
    return compatible_mode_base_url(
        config.workspace,
        config.region,
        config.configured_base_url,
    )


def _conversation_id(value: Union[Conversation, str]) -> str:
    cid = value.id if isinstance(value, Conversation) else value
    if not cid or not str(cid).strip():
        raise ValueError("conversation_id must be a non-empty string")
    return str(cid)


def _item_id(value: Union[ConversationItem, str]) -> str:
    iid = value.id if isinstance(value, ConversationItem) else value
    if not iid or not str(iid).strip():
        raise ValueError("item_id must be a non-empty string")
    return str(iid)


def _validate_metadata(
    metadata: Optional[MetadataInput],
) -> Optional[Dict[str, str]]:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping of string keys/values")
    if len(metadata) > _MAX_METADATA_PAIRS:
        raise ValueError(
            f"metadata supports at most {_MAX_METADATA_PAIRS} key/value "
            f"pairs (got {len(metadata)})"
        )
    result: Dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or len(key) > _MAX_METADATA_KEY_LEN:
            raise ValueError(
                f"metadata keys must be strings of at most "
                f"{_MAX_METADATA_KEY_LEN} characters"
            )
        if not isinstance(value, str) or len(value) > _MAX_METADATA_VALUE_LEN:
            raise ValueError(
                f"metadata values must be strings of at most "
                f"{_MAX_METADATA_VALUE_LEN} characters"
            )
        result[key] = value
    return result


def _normalize_items(value: ItemsInput) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, BaseModel):
            items.append(item.model_dump(exclude_none=True))
        elif isinstance(item, Mapping):
            items.append(dict(item))
        else:
            raise ValueError("each item must be a mapping or a pydantic model")
    if len(items) > _MAX_ITEMS:
        raise ValueError(
            f"items supports at most {_MAX_ITEMS} entries per call "
            f"(got {len(items)})"
        )
    return items


def _build_create_body(
    *,
    metadata: Optional[MetadataInput],
    items: Optional[ItemsInput],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    normalized_metadata = _validate_metadata(metadata)
    if normalized_metadata is not None:
        body["metadata"] = normalized_metadata
    if items is not None:
        normalized = _normalize_items(items)
        if normalized:
            body["items"] = normalized
    return body


def _build_items_body(items: ItemsInput) -> Dict[str, Any]:
    if items is None:
        raise ValueError("items is required")
    normalized = _normalize_items(items)
    if not normalized:
        raise ValueError("items must not be empty")
    return {"items": normalized}


def _items_params(
    *,
    after: Optional[str],
    limit: Optional[int],
    order: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if after is not None:
        params["after"] = after
    if limit is not None:
        if not 1 <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
        params["limit"] = limit
    if order is not None:
        if order not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")
        params["order"] = order
    return params


def _checked(client, resp, model_cls, parser):
    """Raise on failure, or fill the error envelope when not raising.

    Compatible-mode failures nest ``code`` / ``message`` under ``error``
    (OpenAI shape), which the flat success parser cannot see; routing the
    reply through ``response_error`` keeps ``raise_on_error=False``
    reporting the failure via ``is_error`` / ``code`` / ``message``.
    """
    http_resp = resp.http_response
    error = response_error(http_resp)
    if error is None:
        return with_headers(
            parser(
                resp.json(),
                strict=client.config.strict_validation,
                response=http_resp,
            ),
            http_resp,
        )
    if client.raise_on_error:
        raise error
    return with_headers(
        model_cls(
            request_id=error.request_id,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            is_error=True,
        ),
        http_resp,
    )


class ConversationItemList(_CursorPage):
    """One cursor page of conversation items (``items.list`` /
    ``items.create``).

    The wire carries ``first_id`` / ``last_id`` (item ids, ``msg_xxx``);
    the next-page cursor prefers ``last_id``. ``object`` is ``list`` on
    ``items.list`` (absent on ``items.create``, defaulted here).
    """

    data: List[ConversationItem] = []
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    object: Optional[str] = "list"

    @property
    def _next_cursor(self) -> Optional[str]:
        if self.last_id:
            return self.last_id
        return self.data[-1].id if self.data else None


def _parse_conversation_item_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> ConversationItemList:
    payload = dict(body) if isinstance(body, dict) else {}
    data = payload.get("data")
    if not isinstance(data, list):
        data = []
    return parse_model(
        ConversationItemList,
        {
            "data": data,
            "first_id": payload.get("first_id"),
            "last_id": payload.get("last_id"),
            "object": payload.get("object", "list"),
            "has_more": bool(payload.get("has_more", False)),
            **response_envelope(payload),
        },
        strict=strict,
        response=response,
    )


class ConversationItems:
    """``client.conversations.items`` -- a conversation's message items."""

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        conversation_id: Union[Conversation, str],
        *,
        items: ItemsInput,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItemList, "APIResponse"]:
        """Add message items (``POST /conversations/{id}/items``).

        ``items`` is required, at most 20 per call; each is a ``message``
        with ``role`` (user/assistant/system/developer) and ``content``
        (string or content-block list). Returns the created items as a
        cursor page (``data`` / ``first_id`` / ``last_id`` / ``has_more``).
        """
        cid = _conversation_id(conversation_id)
        body = _build_items_body(items)
        resp = self._client.transport.request_api(
            "POST",
            f"{_CONVERSATIONS_PATH}/{cid}/items",
            json=body,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationItemList,
            _parse_conversation_item_list,
        )

    def list(  # pylint: disable=redefined-builtin
        self,
        conversation_id: Union[Conversation, str],
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItemList, "APIResponse"]:
        """List items, one cursor page (``GET /conversations/{id}/items``).

        ``after`` is an item id (``msg_xxx``) -- the previous page's
        ``last_id``; ``limit`` is in [1, 100] (default 20); ``order`` is
        ``asc`` / ``desc`` (default ``desc``). Cross pages with
        ``iter_pages()`` / ``next_page()`` on the parsed result.
        """
        cid = _conversation_id(conversation_id)
        params = _items_params(after=after, limit=limit, order=order)
        resp = self._client.transport.request_api(
            "GET",
            f"{_CONVERSATIONS_PATH}/{cid}/items",
            params=params,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        result = _checked(
            self._client,
            resp,
            ConversationItemList,
            _parse_conversation_item_list,
        )
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    cid,
                    after=cursor,
                    limit=limit,
                    order=order,
                    timeout=timeout,
                    api_key=api_key,
                    workspace=workspace,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    max_retries=max_retries,
                ),
            )
        return result

    def get(
        self,
        conversation_id: Union[Conversation, str],
        item_id: Union[ConversationItem, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItem, "APIResponse"]:
        """Retrieve one item (``GET /conversations/{id}/items/{item_id}``)."""
        cid = _conversation_id(conversation_id)
        iid = _item_id(item_id)
        resp = self._client.transport.request_api(
            "GET",
            f"{_CONVERSATIONS_PATH}/{cid}/items/{iid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationItem,
            parse_conversation_item,
        )

    def delete(
        self,
        conversation_id: Union[Conversation, str],
        item_id: Union[ConversationItem, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItemDeleted, "APIResponse"]:
        """Delete one item (``DELETE /conversations/{id}/items/{item_id}``).

        The wire returns ``{id, deleted, object: conversation.item.deleted}``.
        """
        cid = _conversation_id(conversation_id)
        iid = _item_id(item_id)
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_CONVERSATIONS_PATH}/{cid}/items/{iid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationItemDeleted,
            parse_conversation_item_deleted,
        )


class AsyncConversationItems:
    """Async twin of :class:`ConversationItems`."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        conversation_id: Union[Conversation, str],
        *,
        items: ItemsInput,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItemList, "APIResponse"]:
        """Async twin of :meth:`ConversationItems.create`."""
        cid = _conversation_id(conversation_id)
        body = _build_items_body(items)
        resp = await self._client.transport.request_api(
            "POST",
            f"{_CONVERSATIONS_PATH}/{cid}/items",
            json=body,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationItemList,
            _parse_conversation_item_list,
        )

    async def list(  # pylint: disable=redefined-builtin
        self,
        conversation_id: Union[Conversation, str],
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItemList, "APIResponse"]:
        """Async twin of :meth:`ConversationItems.list`."""
        cid = _conversation_id(conversation_id)
        params = _items_params(after=after, limit=limit, order=order)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_CONVERSATIONS_PATH}/{cid}/items",
            params=params,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        result = _checked(
            self._client,
            resp,
            ConversationItemList,
            _parse_conversation_item_list,
        )
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    cid,
                    after=cursor,
                    limit=limit,
                    order=order,
                    timeout=timeout,
                    api_key=api_key,
                    workspace=workspace,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    max_retries=max_retries,
                ),
            )
        return result

    async def get(
        self,
        conversation_id: Union[Conversation, str],
        item_id: Union[ConversationItem, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItem, "APIResponse"]:
        """Async twin of :meth:`ConversationItems.get`."""
        cid = _conversation_id(conversation_id)
        iid = _item_id(item_id)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_CONVERSATIONS_PATH}/{cid}/items/{iid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationItem,
            parse_conversation_item,
        )

    async def delete(
        self,
        conversation_id: Union[Conversation, str],
        item_id: Union[ConversationItem, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationItemDeleted, "APIResponse"]:
        """Async twin of :meth:`ConversationItems.delete`."""
        cid = _conversation_id(conversation_id)
        iid = _item_id(item_id)
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_CONVERSATIONS_PATH}/{cid}/items/{iid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationItemDeleted,
            parse_conversation_item_deleted,
        )


class Conversations:
    """``client.conversations`` -- the OpenAI-compatible Conversations
    lifecycle, with ``items`` as a sub-resource."""

    def __init__(self, client) -> None:
        self._client = client
        self.items = ConversationItems(client)

    def create(
        self,
        *,
        metadata: Optional[MetadataInput] = None,
        items: Optional[ItemsInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Conversation, "APIResponse"]:
        """Create a conversation (``POST /conversations``).

        ``metadata`` (<= 16 string pairs) and initial ``items`` (<= 20
        ``message`` entries) are both optional. The returned ``id`` seeds
        ``client.responses.create(conversation=...)`` for multi-turn
        context; ``created_at`` is Unix milliseconds.
        """
        body = _build_create_body(metadata=metadata, items=items)
        resp = self._client.transport.request_api(
            "POST",
            _CONVERSATIONS_PATH,
            json=body,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Conversation, parse_conversation)

    def get(
        self,
        conversation_id: Union[Conversation, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Conversation, "APIResponse"]:
        """Retrieve a conversation (``GET /conversations/{id}``)."""
        cid = _conversation_id(conversation_id)
        resp = self._client.transport.request_api(
            "GET",
            f"{_CONVERSATIONS_PATH}/{cid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Conversation, parse_conversation)

    def update(
        self,
        conversation_id: Union[Conversation, str],
        *,
        metadata: MetadataInput,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Conversation, "APIResponse"]:
        """Update a conversation's metadata (``POST /conversations/{id}``).

        ``metadata`` is required and **fully replaces** the existing
        metadata (<= 16 string pairs, key <= 64, value <= 512 chars).
        """
        cid = _conversation_id(conversation_id)
        normalized = _validate_metadata(metadata)
        if normalized is None:
            raise ValueError("metadata is required for update")
        resp = self._client.transport.request_api(
            "POST",
            f"{_CONVERSATIONS_PATH}/{cid}",
            json={"metadata": normalized},
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Conversation, parse_conversation)

    def delete(
        self,
        conversation_id: Union[Conversation, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationDeleted, "APIResponse"]:
        """Delete a conversation (``DELETE /conversations/{id}``).

        The conversation's items are **not** deleted. The wire returns
        ``{id, deleted, object: conversation.deleted}``.
        """
        cid = _conversation_id(conversation_id)
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_CONVERSATIONS_PATH}/{cid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationDeleted,
            parse_conversation_deleted,
        )


class AsyncConversations:
    """Async twin of :class:`Conversations`."""

    def __init__(self, client) -> None:
        self._client = client
        self.items = AsyncConversationItems(client)

    async def create(
        self,
        *,
        metadata: Optional[MetadataInput] = None,
        items: Optional[ItemsInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Conversation, "APIResponse"]:
        """Async twin of :meth:`Conversations.create`."""
        body = _build_create_body(metadata=metadata, items=items)
        resp = await self._client.transport.request_api(
            "POST",
            _CONVERSATIONS_PATH,
            json=body,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Conversation, parse_conversation)

    async def get(
        self,
        conversation_id: Union[Conversation, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Conversation, "APIResponse"]:
        """Async twin of :meth:`Conversations.get`."""
        cid = _conversation_id(conversation_id)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_CONVERSATIONS_PATH}/{cid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Conversation, parse_conversation)

    async def update(
        self,
        conversation_id: Union[Conversation, str],
        *,
        metadata: MetadataInput,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Conversation, "APIResponse"]:
        """Async twin of :meth:`Conversations.update`."""
        cid = _conversation_id(conversation_id)
        normalized = _validate_metadata(metadata)
        if normalized is None:
            raise ValueError("metadata is required for update")
        resp = await self._client.transport.request_api(
            "POST",
            f"{_CONVERSATIONS_PATH}/{cid}",
            json={"metadata": normalized},
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Conversation, parse_conversation)

    async def delete(
        self,
        conversation_id: Union[Conversation, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ConversationDeleted, "APIResponse"]:
        """Async twin of :meth:`Conversations.delete`."""
        cid = _conversation_id(conversation_id)
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_CONVERSATIONS_PATH}/{cid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ConversationDeleted,
            parse_conversation_deleted,
        )


__all__ = [
    "AsyncConversationItems",
    "AsyncConversations",
    "ConversationItemList",
    "ConversationItems",
    "Conversations",
]