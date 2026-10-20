# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Transparent v2 error-alignment proxy for the mounted AgentStudio sub-SDK.

Design doc M3 task 3.12. ``client.agentstudio`` lazily mounts the 1.x
AgentStudio sub-SDK, which keeps its own exception family rooted at
``dashscope.common.error.DashScopeException``. That family shares no
lineage with ``dashscope.v2.errors.DashScopeError``, so a user writing
``except dashscope.v2.errors.DashScopeError`` would miss every failure
the mounted sub-SDK raises.

This module closes that gap at the *mount boundary only*: it wraps the
constructed AgentStudio client in a transparent proxy that catches
``AgentStudioError`` and re-raises the equivalent v2 error. No 1.x code
is touched, and the standalone ``dashscope.agentstudio.Client`` keeps its
native error contract -- only the v2 mount is translated.

Two-branch translation (design doc §7):

* HTTP-layer errors (``APIStatusError`` and its subclasses) carry a real
  ``status_code``; they belong to the status map, not ``SDK_ERRORS``, so
  they go through :func:`~dashscope.v2.errors.error_from_status`.
* Local / stream errors that have no HTTP response behind them
  (``APIConnectionError`` / ``APITimeoutError`` / ``StreamError`` /
  ``StreamClosedError``) go through
  :func:`~dashscope.v2.errors.error_from_sdk_exception` with family
  ``sdk.agentstudio``. The registry carries exactly these four and, by
  design, no ``sdk.agentstudio.Error`` family fallback, so any other
  non-HTTP class degrades to a v2 ``APIConnectionError`` rather than let
  the two-tier lookup raise ``KeyError`` and mask the original failure.

The proxy is transparent (``__class__`` spoofing plus attribute
forwarding) so ``isinstance(client.agentstudio, AgentStudioClient)``,
``.transport`` access and the context-manager protocol keep working.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Iterator
from typing import Any, Dict, Optional

import httpx

from dashscope.agentstudio import exceptions as _as_exceptions
from dashscope.agentstudio.pagination import AsyncCursorPage, CursorPage
from dashscope.v2 import errors as _v2_errors

_AGENTSTUDIO_FAMILY = "sdk.agentstudio"

# The only AgentStudio classes registered under ``sdk.agentstudio.*`` in
# the error catalog. Everything else non-HTTP has no entry and no family
# fallback, so it must not reach ``error_from_sdk_exception``.
_SDK_ERROR_CLASSES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "StreamError",
        "StreamClosedError",
    },
)

# Resource classes (and the typed event streams) live under this package;
# attribute access recurses into them so nested method calls stay guarded.
_RESOURCE_PACKAGE = "dashscope.agentstudio.resources"

# AgentStudio raises a few HTTP-family errors locally, without a response,
# so they carry no ``status_code`` (the missing-``api_key`` guard in
# ``transport.build_headers`` is the live example). Recover the status the
# class stands for so ``error_from_status`` still picks the right v2 class
# instead of degrading to a 500. Values mirror agentstudio's own
# ``_STATUS_TO_DEFAULT``; only the class's canonical status is kept.
_STATUS_BY_CLASS: Dict[type, int] = {
    _as_exceptions.InvalidRequestError: 400,
    _as_exceptions.AuthenticationError: 401,
    _as_exceptions.PermissionDeniedError: 403,
    _as_exceptions.NotFoundError: 404,
    _as_exceptions.ConflictError: 409,
    _as_exceptions.RateLimitError: 429,
    _as_exceptions.OverloadedError: 503,
    _as_exceptions.InternalServerError: 500,
}


def _effective_status(exc: _as_exceptions.AgentStudioError) -> int:
    """The HTTP status to map on, recovering it from the class if absent."""

    status = getattr(exc, "status_code", None)
    if status is not None:
        return status
    for cls in type(exc).__mro__:
        if cls in _STATUS_BY_CLASS:
            return _STATUS_BY_CLASS[cls]
    return 500


def translate_agentstudio_error(
    exc: _as_exceptions.AgentStudioError,
) -> _v2_errors.DashScopeError:
    """Map a 1.x AgentStudio exception into the v2 error family."""

    status = getattr(exc, "status_code", None)
    if isinstance(exc, _as_exceptions.APIStatusError) or status is not None:
        # V1 already resolved compatibility fields and header precedence.
        request_id = getattr(exc, "request_id", None)
        headers = {"x-request-id": request_id} if request_id else None
        error = _v2_errors.error_from_status(
            _effective_status(exc),
            message=getattr(exc, "message", None) or str(exc),
            headers=headers,
        )
        error.code = exc.code
        error.body = exc.raw
        return error

    if type(exc).__name__ in _SDK_ERROR_CLASSES:
        return _v2_errors.error_from_sdk_exception(
            exc,
            family=_AGENTSTUDIO_FAMILY,
        )

    # Unregistered non-HTTP class (in practice only the bare base class):
    # surface it as a connection error instead of a masking KeyError.
    return _v2_errors.APIConnectionError(
        getattr(exc, "message", None) or str(exc),
        request_id=getattr(exc, "request_id", None),
    )


def _is_resource(obj: Any) -> bool:
    return type(obj).__module__.startswith(_RESOURCE_PACKAGE)


def _is_sync_stream(obj: Any) -> bool:
    # CursorPage is iterable but not a context manager, so requiring both
    # keeps pagination results out of the stream proxy.
    return hasattr(obj, "__enter__") and hasattr(obj, "__iter__")


def _is_async_stream(obj: Any) -> bool:
    return hasattr(obj, "__aenter__") and hasattr(obj, "__aiter__")


def _wrap_result(result: Any) -> Any:
    """Guard lazy stream and pagination calls without changing model data."""

    if isinstance(result, AsyncCursorPage):
        return _AsyncPageProxy(result)
    if isinstance(result, CursorPage):
        return _SyncPageProxy(result)
    if _is_async_stream(result):
        return _AsyncStreamProxy(result)
    if _is_sync_stream(result):
        return _SyncStreamProxy(result)
    return result


_STREAM_ERRORS = (
    _as_exceptions.AgentStudioError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def _translate_stream_error(exc: Exception) -> _v2_errors.DashScopeError:
    if isinstance(exc, _as_exceptions.AgentStudioError):
        return translate_agentstudio_error(exc)
    cls = (
        _as_exceptions.APITimeoutError
        if isinstance(exc, httpx.TimeoutException)
        else _as_exceptions.APIConnectionError
    )
    return translate_agentstudio_error(cls(str(exc)))


def _iterate_sync(target: Any) -> Any:
    try:
        yield from target
    except _STREAM_ERRORS as exc:
        raise _translate_stream_error(exc) from exc


async def _iterate_async(target: Any) -> Any:
    iterator = None
    try:
        iterator = target.__aiter__()
        async for item in iterator:
            yield item
    except _STREAM_ERRORS as exc:
        raise _translate_stream_error(exc) from exc
    finally:
        if hasattr(iterator, "aclose"):
            await iterator.aclose()


def _guard_sync(func: Any) -> Any:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = func(*args, **kwargs)
        except _as_exceptions.AgentStudioError as exc:
            raise translate_agentstudio_error(exc) from exc
        return _wrap_result(result)

    return wrapper


def _guard_async(func: Any) -> Any:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = await func(*args, **kwargs)
        except _as_exceptions.AgentStudioError as exc:
            raise translate_agentstudio_error(exc) from exc
        return _wrap_result(result)

    return wrapper


class _SyncStreamProxy:
    """Guards a sync event stream whose errors surface on enter/iterate."""

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # noqa: D105 - transparency helper
        return type(object.__getattribute__(self, "_target"))

    def __getattr__(self, name: str) -> Any:
        value = getattr(object.__getattribute__(self, "_target"), name)
        if isinstance(value, Iterator):
            return _iterate_sync(value)
        if callable(value):
            return _guard_sync(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    def __enter__(self) -> "_SyncStreamProxy":
        target = object.__getattribute__(self, "_target")
        try:
            target.__enter__()
        except _as_exceptions.AgentStudioError as exc:
            raise translate_agentstudio_error(exc) from exc
        return self

    def __exit__(self, *exc_info: Any) -> Optional[bool]:
        target = object.__getattribute__(self, "_target")
        return target.__exit__(*exc_info)

    def __iter__(self) -> Any:
        return _iterate_sync(object.__getattribute__(self, "_target"))


class _AsyncStreamProxy:
    """Guards an async event stream whose errors surface on enter/iterate."""

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # noqa: D105 - transparency helper
        return type(object.__getattribute__(self, "_target"))

    def __getattr__(self, name: str) -> Any:
        value = getattr(object.__getattribute__(self, "_target"), name)
        if isinstance(value, AsyncIterator):
            return _iterate_async(value)
        if callable(value):
            return (
                _guard_async(value)
                if asyncio.iscoroutinefunction(value)
                else _guard_sync(value)
            )
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    async def __aenter__(self) -> "_AsyncStreamProxy":
        target = object.__getattribute__(self, "_target")
        try:
            await target.__aenter__()
        except _as_exceptions.AgentStudioError as exc:
            raise translate_agentstudio_error(exc) from exc
        return self

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]:
        target = object.__getattribute__(self, "_target")
        return await target.__aexit__(*exc_info)

    def __aiter__(self) -> Any:
        return _iterate_async(object.__getattribute__(self, "_target"))


class AgentStudioProxy:
    """Transparent recursive proxy over an AgentStudio client or resource.

    Attribute access returns a guarded callable for methods, a nested
    proxy for sub-resources, and the raw value for everything else (so
    ``.transport`` and returned model objects pass through untouched).
    """

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_cache", {})

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # noqa: D105 - transparency helper
        return type(object.__getattribute__(self, "_target"))

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        cache = object.__getattribute__(self, "_cache")
        value = getattr(target, name)
        if _is_resource(value) and not callable(value):
            cached = cache.get(name)
            if cached is None or cached[0] is not value:
                cached = cache[name] = (value, AgentStudioProxy(value))
            return cached[1]
        cache.pop(name, None)
        if callable(value):
            return (
                _guard_async(value)
                if asyncio.iscoroutinefunction(value)
                else _guard_sync(value)
            )
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)
        object.__getattribute__(self, "_cache").pop(name, None)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return repr(object.__getattribute__(self, "_target"))

    def __enter__(self) -> "AgentStudioProxy":
        object.__getattribute__(self, "_target").__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> Optional[bool]:
        target = object.__getattribute__(self, "_target")
        return target.__exit__(*exc_info)

    async def __aenter__(self) -> "AgentStudioProxy":
        await object.__getattribute__(self, "_target").__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> Optional[bool]:
        target = object.__getattribute__(self, "_target")
        return await target.__aexit__(*exc_info)


class _SyncPageProxy(AgentStudioProxy):
    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_target"))

    def __iter__(self) -> Any:
        try:
            yield from object.__getattribute__(self, "_target")
        except _as_exceptions.AgentStudioError as exc:
            raise translate_agentstudio_error(exc) from exc


class _AsyncPageProxy(AgentStudioProxy):
    def __len__(self) -> int:
        return len(object.__getattribute__(self, "_target"))

    async def __aiter__(self) -> Any:
        try:
            async for item in object.__getattribute__(self, "_target"):
                yield item
        except _as_exceptions.AgentStudioError as exc:
            raise translate_agentstudio_error(exc) from exc


def wrap_agentstudio(client: Any) -> AgentStudioProxy:
    """Wrap a constructed AgentStudio client in the v2 translation proxy."""

    return AgentStudioProxy(client)
