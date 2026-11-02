# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offset pagination shared by every list endpoint in the dashscope 2.0 SDK.

``models.list`` / ``files.list`` / ``jobs.list`` all answer with the same
envelope -- ``page_no`` / ``page_size`` / ``total_count`` plus a list of
items -- but each used to expose it as a bare model, leaving callers to
loop ``page_no`` by hand and compare it against the total. :class:`BasePage`
folds that traversal into the response object so a page can be iterated
and advanced uniformly.

The concrete list models keep their own item field name (``models`` /
``files`` / ``tasks``) and expose it through the :attr:`BasePage.items`
property, so iteration is uniform without renaming the public fields.

One class serves both the sync and the async client: the resource injects a
matching fetch callback (a plain callable or a coroutine function) into the
private ``_fetch_next`` slot after parsing, and the sync / async traversal
methods each drive it appropriately. Because it is a ``PrivateAttr``, the
callback never appears in ``model_dump()``.
"""

from __future__ import annotations

import inspect
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

from pydantic import AliasChoices, Field, PrivateAttr

from dashscope.types.common import ResponseBase

__all__ = ["BasePage"]

T = TypeVar("T")


class BasePage(ResponseBase, Generic[T]):
    """One page of an offset-paginated list, with traversal helpers.

    Subclasses declare their item list under a resource-specific field and
    override :attr:`items` to return it. ``page_no`` / ``page_size`` /
    ``total_count`` are inherited from here.
    """

    # Fetches another page as ``(page_no, page_size) -> page`` (a page or an
    # awaitable of one). Injected by the resource after parsing; a PrivateAttr
    # so it stays out of ``model_dump()`` and never round-trips as a field.
    _fetch_next: Optional[Callable[[int, int], Any]] = PrivateAttr(
        default=None,
    )

    page_no: int = 1
    page_size: int = 10
    # The wire reports the total as ``total``; ``total_count`` is the
    # canonical name here. Accept both so a page parsed straight from a body
    # traverses correctly instead of silently reading a zero total.
    total_count: int = Field(
        0,
        validation_alias=AliasChoices("total_count", "total"),
    )

    @property
    def items(self) -> List[T]:
        """The items on *this* page; overridden by each concrete list model."""
        return []

    def bind_pagination(
        self,
        callback: Optional[Callable[[int, int], Any]],
    ) -> "BasePage[T]":
        """Attach the page-fetch callback the traversal methods use.

        Resources call this right after parsing so ``next_page()`` /
        ``iter_pages()`` can request the following page. Returns ``self``
        for chaining. ``callback`` is stored in a ``PrivateAttr``, so
        binding it never changes ``model_dump()``.
        """
        self._fetch_next = callback
        return self

    def has_next_page(self) -> bool:
        """Whether ``total_count`` implies a page after this one.

        Offset arithmetic only: it trusts the server's ``total_count`` and
        does not consult ``_fetch_next``, so it stays meaningful on a page
        built without a callback (e.g. parsed straight from a body).
        """
        return self.page_no * self.page_size < self.total_count

    def next_page(self) -> Optional["BasePage[T]"]:
        """Fetch the following page, or ``None`` at the end of the list.

        Returns ``None`` when there is no next page or when no fetch callback
        was injected (a standalone page cannot advance itself).
        """
        if not self.has_next_page() or self._fetch_next is None:
            return None
        return self._fetch_next(self.page_no + 1, self.page_size)

    async def next_page_async(self) -> Optional["BasePage[T]"]:
        """Async :meth:`next_page`; awaits the callback when it is a coroutine."""
        if not self.has_next_page() or self._fetch_next is None:
            return None
        result = self._fetch_next(self.page_no + 1, self.page_size)
        if inspect.isawaitable(result):
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

    def iter_pages(self) -> Iterator["BasePage[T]"]:
        """Yield this page and every following one, advancing via :meth:`next_page`."""
        page: Optional["BasePage[T]"] = self
        while page is not None:
            yield page
            page = page.next_page()

    async def iter_pages_async(self) -> AsyncIterator["BasePage[T]"]:
        """Async :meth:`iter_pages`, advancing via :meth:`next_page_async`."""
        page: Optional["BasePage[T]"] = self
        while page is not None:
            yield page
            page = await page.next_page_async()
