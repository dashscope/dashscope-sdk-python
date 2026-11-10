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

_ASYNC_TRAVERSAL_HINT = (
    "this page was fetched through AsyncClient, so advancing it is a "
    "coroutine; use next_page_async() or iter_pages_async() instead of the "
    "synchronous traversal"
)


def _reject_async_callback(callback: Callable[[int, int], Any]) -> None:
    """Best-effort early refusal of a sync traversal over an async page.

    Fires when the resource bound the coroutine function itself. Resources
    that bind a closure around it (``lambda no, size: self.list(...)``) are
    not coroutine functions and are caught later, by
    :func:`_reject_awaitable_page`.
    """
    if inspect.iscoroutinefunction(callback):
        raise TypeError(_ASYNC_TRAVERSAL_HINT)


def _reject_awaitable_page(result: Any) -> Any:
    """Refuse a page that came back un-awaited, and say which method to use.

    One class serves both clients, so nothing static stops a caller from
    pairing an async-fetched page with the synchronous traversal. Handing
    back the coroutine instead would let it read as a page and fail two
    frames later with ``AttributeError``, naming neither the mistake nor the
    fix; the coroutine is closed first so it does not also leak a
    ``RuntimeWarning`` about never having been awaited.
    """
    if not inspect.isawaitable(result):
        return result
    close = getattr(result, "close", None)
    if callable(close):
        close()
    raise TypeError(_ASYNC_TRAVERSAL_HINT)


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
    # The page this page was *requested* as, when the resource bound one.
    # Traversal advances from here rather than from the parsed ``page_no`` /
    # ``page_size``; see :meth:`bind_pagination`.
    _requested_no: Optional[int] = PrivateAttr(default=None)
    _requested_size: Optional[int] = PrivateAttr(default=None)

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
        requested_page_no: Optional[int] = None,
        requested_page_size: Optional[int] = None,
    ) -> "BasePage[T]":
        """Attach the page-fetch callback the traversal methods use.

        Resources call this right after parsing so ``next_page()`` /
        ``iter_pages()`` can request the following page. Returns ``self``
        for chaining. ``callback`` is stored in a ``PrivateAttr``, so
        binding it never changes ``model_dump()``.

        ``requested_page_no`` / ``requested_page_size`` are the offsets the
        resource actually sent, and traversal advances from them rather than
        from the parsed :attr:`page_no` / :attr:`page_size`. A gateway that
        omits ``page_no`` leaves every page reading the field's default of 1,
        so advancing from the echo would request page 2 forever; one that
        omits ``page_size`` would silently change the window and skip or
        repeat items. The resource is the only party that reliably knows what
        it asked for.
        """
        self._fetch_next = callback
        self._requested_no = requested_page_no
        self._requested_size = requested_page_size
        return self

    @property
    def _cursor_no(self) -> int:
        """The page number to advance from: the request's, else the echo."""
        if self._requested_no is not None:
            return self._requested_no
        return self.page_no

    @property
    def _cursor_size(self) -> int:
        """The page size to advance with: the request's, else the echo."""
        if self._requested_size is not None:
            return self._requested_size
        return self.page_size

    def has_next_page(self) -> bool:
        """Whether ``total_count`` implies a page after this one.

        Offset arithmetic only: it trusts the server's ``total_count`` and
        does not consult ``_fetch_next``, so it stays meaningful on a page
        built without a callback (e.g. parsed straight from a body).
        """
        return self._cursor_no * self._cursor_size < self.total_count

    def next_page(self) -> Optional["BasePage[T]"]:
        """Fetch the following page, or ``None`` at the end of the list.

        Returns ``None`` when there is no next page or when no fetch callback
        was injected (a standalone page cannot advance itself). Raises
        ``TypeError`` on a page fetched through ``AsyncClient`` -- use
        :meth:`next_page_async` there.
        """
        if not self.has_next_page() or self._fetch_next is None:
            return None
        _reject_async_callback(self._fetch_next)
        return _reject_awaitable_page(
            self._fetch_next(self._cursor_no + 1, self._cursor_size),
        )

    async def next_page_async(self) -> Optional["BasePage[T]"]:
        """Async :meth:`next_page`; awaits the callback when it is a coroutine."""
        if not self.has_next_page() or self._fetch_next is None:
            return None
        result = self._fetch_next(self._cursor_no + 1, self._cursor_size)
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
        """Yield this page and every following one, advancing via :meth:`next_page`.

        Raises ``TypeError`` on a page fetched through ``AsyncClient``; use
        :meth:`iter_pages_async` there. When the resource bound the coroutine
        function directly the refusal happens before anything is yielded;
        when it bound a closure around one, the first page is yielded and the
        advance that follows raises.
        """
        if self._fetch_next is not None:
            _reject_async_callback(self._fetch_next)
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
