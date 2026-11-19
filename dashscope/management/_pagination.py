# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Token (``nextToken``) pagination for the modelstudio list ops.

The modelstudio lists answer with ``nextToken`` / ``maxResults`` /
``totalCount`` rather than the offset ``page_no`` / ``page_size`` the
tokenplan ops use, so they cannot ride :class:`BasePage`. :class:`TokenPage`
mirrors BasePage's traversal contract exactly -- one class serves both
clients, ``__iter__`` walks the current page's items, ``iter_pages``
crosses pages -- so the two pagination styles feel identical to callers.

The fetch callback is a ``PrivateAttr`` injected by the resource after
parsing; it never appears in ``model_dump()``.
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
)

from pydantic import AliasChoices, ConfigDict, Field, PrivateAttr

from dashscope.types.common import ResponseBase
from dashscope.types.pagination import (
    _reject_async_callback,
    _reject_awaitable_page,
)

__all__ = ["TokenPage"]

T = TypeVar("T")


class TokenPage(ResponseBase, Generic[T]):
    """One page of a ``nextToken``-paginated list, with traversal helpers."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    items: List[T] = Field(default_factory=list)
    next_token: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("next_token", "nextToken"),
    )
    max_results: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("max_results", "maxResults"),
    )
    total_count: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("total_count", "totalCount"),
    )

    _fetch_next: Optional[Callable[[str], Any]] = PrivateAttr(default=None)

    def bind_pagination(
        self,
        callback: Optional[Callable[[str], Any]],
    ) -> "TokenPage[T]":
        """Attach the ``(next_token) -> page`` fetch callback; returns self."""
        self._fetch_next = callback
        return self

    def has_next_page(self) -> bool:
        """Whether a ``nextToken`` and a fetch callback are both present."""
        return bool(self.next_token) and self._fetch_next is not None

    def next_page(self) -> Optional["TokenPage[T]"]:
        """Fetch the following page, or ``None`` at the end of the list."""
        if not self.has_next_page() or self._fetch_next is None:
            return None
        _reject_async_callback(self._fetch_next)
        return _reject_awaitable_page(self._fetch_next(self.next_token))

    async def next_page_async(self) -> Optional["TokenPage[T]"]:
        """Async :meth:`next_page`; awaits the callback when it is a coroutine."""
        if not self.has_next_page() or self._fetch_next is None:
            return None
        result = self._fetch_next(self.next_token)
        if hasattr(result, "__await__"):
            result = await result
        return result

    def __iter__(self) -> Iterator[T]:
        """Iterate this page's items only; use :meth:`iter_pages` to cross pages."""
        return iter(self.items)

    def __aiter__(self) -> AsyncIterator[T]:
        """Async counterpart of :meth:`__iter__`: this page's items only."""

        async def _gen() -> AsyncIterator[T]:
            for item in self.items:
                yield item

        return _gen()

    def iter_pages(self) -> Iterator["TokenPage[T]"]:
        """Yield this page and every following one, advancing via ``next_page``."""
        if self._fetch_next is not None:
            _reject_async_callback(self._fetch_next)
        page: Optional["TokenPage[T]"] = self
        while page is not None:
            yield page
            page = page.next_page()

    async def iter_pages_async(self) -> AsyncIterator["TokenPage[T]"]:
        """Async :meth:`iter_pages`, advancing via ``next_page_async``."""
        page: Optional["TokenPage[T]"] = self
        while page is not None:
            yield page
            page = await page.next_page_async()
