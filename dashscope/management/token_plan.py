# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.token_plan`` -- TokenPlan account, members, seats, subscription.

Read-only (G5). The tokenplan ops use the PascalCase envelope and offset
pagination (``PageNo`` / ``PageNum`` / ``PageSize`` / ``Total``), so their
lists ride :class:`~dashscope.types.pagination.BasePage` rather than the
modelstudio ``TokenPage``. Two page shapes occur: ``members`` returns the
item array directly under ``Data`` with pagination at the root, while
``seats`` / ``shared_packages`` nest both under ``Data.Items`` /
``Data.{PageNo,PageSize,Total}``.

Write ops (invite links, key create/rotate, member add/update/remove,
seat assign/revoke, org update) are high-risk and deferred to G6 -- not
implemented here.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from dashscope.management._errors import decode_body
from dashscope.management._parse import parse_base_page, parse_one
from dashscope.management._resource import (
    AsyncResource,
    SyncResource,
    options_kwargs,
)
from dashscope.management.types import (
    OrgMemberPage,
    SeatPage,
    SeatStats,
    SharedPackagePage,
    SubscriptionStats,
    TokenPlanAccount,
)
from dashscope.transport._options import request_options

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

__all__ = [
    "AsyncTokenPlan",
    "TokenPlan",
]


def _root_array_page(body: dict) -> tuple:
    items = body.get("Data")
    items = items if isinstance(items, list) else []
    return (
        items,
        body.get("PageNo", 1),
        body.get("PageSize", 10),
        body.get("Total", len(items)),
    )


def _nested_page(body: dict) -> tuple:
    data = body.get("Data")
    data = data if isinstance(data, dict) else {}
    items = data.get("Items")
    items = items if isinstance(items, list) else []
    return (
        items,
        data.get("PageNo", 1),
        data.get("PageSize", 10),
        data.get("Total", len(items)),
    )


class _Account(SyncResource):
    @overload
    def get(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPlanAccount:
        ...

    @overload
    def get(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def get(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPlanAccount, "APIResponse"]:
        """The TokenPlan account detail and its org memberships."""
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "GetTokenPlanAccountDetail",
            {},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            TokenPlanAccount,
            resp,
            data_field="Data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class _Members(SyncResource):
    @overload
    def list(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        has_seat: Optional[bool] = None,
        page_num: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> OrgMemberPage:
        ...

    @overload
    def list(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        has_seat: Optional[bool] = None,
        page_num: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def list(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        has_seat: Optional[bool] = None,
        page_num: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[OrgMemberPage, "APIResponse"]:
        """List organization members (``MaskedApiKey`` is already masked)."""
        filters = {
            "name": name,
            "status": status,
            "has_seat": has_seat,
            "page_num": page_num,
            "page_size": page_size,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "ListOrganizationMembers",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        items, no, size, total = _root_array_page(
            decode_body(resp.http_response),
        )

        def fetch(next_no: int, next_size: int) -> OrgMemberPage:
            return self.list(  # type: ignore[call-overload]
                **{**filters, "page_num": next_no, "page_size": next_size},
                **opts,
            )

        return parse_base_page(
            OrgMemberPage,
            resp,
            items_field="members",
            items=items,
            page_no=no,
            page_size=size,
            total_count=total,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
            requested_no=page_num,
            requested_size=page_size,
        )


class _Seats(SyncResource):
    @overload
    def list(
        self,
        *,
        status_list: Optional[Any] = None,
        seat_id: Optional[str] = None,
        seat_type: Optional[str] = None,
        query_assigned: Optional[bool] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SeatPage:
        ...

    @overload
    def list(
        self,
        *,
        status_list: Optional[Any] = None,
        seat_id: Optional[str] = None,
        seat_type: Optional[str] = None,
        query_assigned: Optional[bool] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def list(
        self,
        *,
        status_list: Optional[Any] = None,
        seat_id: Optional[str] = None,
        seat_type: Optional[str] = None,
        query_assigned: Optional[bool] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SeatPage, "APIResponse"]:
        """List subscription seat details, paginated by ``PageNo``."""
        filters = {
            "status_list": status_list,
            "seat_id": seat_id,
            "seat_type": seat_type,
            "query_assigned": query_assigned,
            "page_no": page_no,
            "page_size": page_size,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "GetSubscriptionSeatDetails",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        items, no, size, total = _nested_page(
            decode_body(resp.http_response),
        )

        def fetch(next_no: int, next_size: int) -> SeatPage:
            return self.list(
                **{**filters, "page_no": next_no, "page_size": next_size},
                **opts,
            )

        return parse_base_page(
            SeatPage,
            resp,
            items_field="seats",
            items=items,
            page_no=no,
            page_size=size,
            total_count=total,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
            requested_no=page_no,
            requested_size=page_size,
        )

    @overload
    def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SeatStats:
        ...

    @overload
    def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SeatStats, "APIResponse"]:
        """Organization member seat statistics (a bare root DTO)."""
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "GetOrganizationMemberSeatStats",
            {},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            SeatStats,
            resp,
            data_field=None,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class _Subscription(SyncResource):
    @overload
    def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SubscriptionStats:
        ...

    @overload
    def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SubscriptionStats, "APIResponse"]:
        """Subscription seat totals and remaining credits, by seat type."""
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "GetSubscriptionStats",
            {},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            SubscriptionStats,
            resp,
            data_field="Data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class _SharedPackages(SyncResource):
    @overload
    def list(
        self,
        *,
        status_list: Optional[Any] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SharedPackagePage:
        ...

    @overload
    def list(
        self,
        *,
        status_list: Optional[Any] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def list(
        self,
        *,
        status_list: Optional[Any] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SharedPackagePage, "APIResponse"]:
        """List subscription shared packages, paginated by ``PageNo``."""
        filters = {
            "status_list": status_list,
            "page_no": page_no,
            "page_size": page_size,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "ListSubscriptionSharedPackages",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        items, no, size, total = _nested_page(
            decode_body(resp.http_response),
        )

        def fetch(next_no: int, next_size: int) -> SharedPackagePage:
            return self.list(
                **{**filters, "page_no": next_no, "page_size": next_size},
                **opts,
            )

        return parse_base_page(
            SharedPackagePage,
            resp,
            items_field="packages",
            items=items,
            page_no=no,
            page_size=size,
            total_count=total,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
            requested_no=page_no,
            requested_size=page_size,
        )


class TokenPlan(SyncResource):
    """TokenPlan read surface; sub-resources group the ops by area."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.account = _Account(client)
        self.members = _Members(client)
        self.seats = _Seats(client)
        self.subscription = _Subscription(client)
        self.shared_packages = _SharedPackages(client)


class _AsyncAccount(AsyncResource):
    @overload
    async def get(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPlanAccount:
        ...

    @overload
    async def get(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def get(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPlanAccount, "APIResponse"]:
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "GetTokenPlanAccountDetail",
            {},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            TokenPlanAccount,
            resp,
            data_field="Data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class _AsyncMembers(AsyncResource):
    @overload
    async def list(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        has_seat: Optional[bool] = None,
        page_num: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> OrgMemberPage:
        ...

    @overload
    async def list(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        has_seat: Optional[bool] = None,
        page_num: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def list(
        self,
        *,
        name: Optional[str] = None,
        status: Optional[str] = None,
        has_seat: Optional[bool] = None,
        page_num: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[OrgMemberPage, "APIResponse"]:
        filters = {
            "name": name,
            "status": status,
            "has_seat": has_seat,
            "page_num": page_num,
            "page_size": page_size,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "ListOrganizationMembers",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        items, no, size, total = _root_array_page(
            decode_body(resp.http_response),
        )

        async def fetch(next_no: int, next_size: int) -> OrgMemberPage:
            return await self.list(  # type: ignore[call-overload]
                **{**filters, "page_num": next_no, "page_size": next_size},
                **opts,
            )

        return parse_base_page(
            OrgMemberPage,
            resp,
            items_field="members",
            items=items,
            page_no=no,
            page_size=size,
            total_count=total,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
            requested_no=page_num,
            requested_size=page_size,
        )


class _AsyncSeats(AsyncResource):
    @overload
    async def list(
        self,
        *,
        status_list: Optional[Any] = None,
        seat_id: Optional[str] = None,
        seat_type: Optional[str] = None,
        query_assigned: Optional[bool] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SeatPage:
        ...

    @overload
    async def list(
        self,
        *,
        status_list: Optional[Any] = None,
        seat_id: Optional[str] = None,
        seat_type: Optional[str] = None,
        query_assigned: Optional[bool] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def list(
        self,
        *,
        status_list: Optional[Any] = None,
        seat_id: Optional[str] = None,
        seat_type: Optional[str] = None,
        query_assigned: Optional[bool] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SeatPage, "APIResponse"]:
        filters = {
            "status_list": status_list,
            "seat_id": seat_id,
            "seat_type": seat_type,
            "query_assigned": query_assigned,
            "page_no": page_no,
            "page_size": page_size,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "GetSubscriptionSeatDetails",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        items, no, size, total = _nested_page(
            decode_body(resp.http_response),
        )

        async def fetch(next_no: int, next_size: int) -> SeatPage:
            return await self.list(
                **{**filters, "page_no": next_no, "page_size": next_size},
                **opts,
            )

        return parse_base_page(
            SeatPage,
            resp,
            items_field="seats",
            items=items,
            page_no=no,
            page_size=size,
            total_count=total,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
            requested_no=page_no,
            requested_size=page_size,
        )

    @overload
    async def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SeatStats:
        ...

    @overload
    async def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SeatStats, "APIResponse"]:
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "GetOrganizationMemberSeatStats",
            {},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            SeatStats,
            resp,
            data_field=None,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class _AsyncSubscription(AsyncResource):
    @overload
    async def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SubscriptionStats:
        ...

    @overload
    async def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def stats(
        self,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SubscriptionStats, "APIResponse"]:
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "GetSubscriptionStats",
            {},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            SubscriptionStats,
            resp,
            data_field="Data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class _AsyncSharedPackages(AsyncResource):
    @overload
    async def list(
        self,
        *,
        status_list: Optional[Any] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> SharedPackagePage:
        ...

    @overload
    async def list(
        self,
        *,
        status_list: Optional[Any] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def list(
        self,
        *,
        status_list: Optional[Any] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[SharedPackagePage, "APIResponse"]:
        filters = {
            "status_list": status_list,
            "page_no": page_no,
            "page_size": page_size,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "ListSubscriptionSharedPackages",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        items, no, size, total = _nested_page(
            decode_body(resp.http_response),
        )

        async def fetch(next_no: int, next_size: int) -> SharedPackagePage:
            return await self.list(
                **{**filters, "page_no": next_no, "page_size": next_size},
                **opts,
            )

        return parse_base_page(
            SharedPackagePage,
            resp,
            items_field="packages",
            items=items,
            page_no=no,
            page_size=size,
            total_count=total,
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
            requested_no=page_no,
            requested_size=page_size,
        )


class AsyncTokenPlan(AsyncResource):
    """Async TokenPlan read surface; mirrors :class:`TokenPlan`."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.account = _AsyncAccount(client)
        self.members = _AsyncMembers(client)
        self.seats = _AsyncSeats(client)
        self.subscription = _AsyncSubscription(client)
        self.shared_packages = _AsyncSharedPackages(client)
