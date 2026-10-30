# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-request customization channel for the v2 transports.

Every resource method offers the same three escape hatches --
``extra_headers`` / ``extra_query`` / ``extra_body`` -- and hands them to
the transport as one :class:`RequestOptions`. They exist so a caller can
attach something this SDK does not model yet (a gateway experiment flag,
a parameter a newer backend added) without waiting for a release, and so
``max_retries`` can be narrowed for one call -- a long-running job
submission that must not be replayed, without deriving a second client
through ``with_options`` just to change that one number.

``None`` means "no opinion" everywhere here, so a call that sets no
option behaves exactly as it did before this channel existed. Header
precedence is documented on :func:`dashscope.transport.http._request_headers`,
which is where the layers actually meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import httpx

__all__ = [
    "RequestOptions",
    "deep_merge",
    "merge_query",
    "request_options",
    "resolve_max_retries",
    "resolve_timeout",
]


@dataclass(frozen=True)
class RequestOptions:
    """Per-request overrides, every field optional.

    Frozen because one instance is naturally built once and reused across
    calls; a mutable option object would let a later request inherit an
    earlier one's edits.

    ``timeout`` is the *fallback* for the transport's own ``timeout=``
    argument, which wins when both are given: resources already forward a
    per-method ``timeout``, so two channels setting one value with no
    rule between them would be ambiguous.
    """

    extra_headers: Optional[Mapping[str, str]] = None
    extra_query: Optional[Mapping[str, Any]] = None
    extra_body: Optional[Mapping[str, Any]] = None
    timeout: Optional[Union[float, httpx.Timeout]] = None
    max_retries: Optional[int] = None


def request_options(
    extra_headers: Optional[Mapping[str, str]] = None,
    extra_query: Optional[Mapping[str, Any]] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
) -> Optional[RequestOptions]:
    """Options for a resource call's three extras, or ``None``.

    Resources call this rather than building :class:`RequestOptions`
    themselves, so an all-defaults call stays ``None`` and the transport
    keeps its no-options path.
    """
    if not (extra_headers or extra_query or extra_body):
        return None
    return RequestOptions(
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
    )


def deep_merge(
    base: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge ``extra`` over ``base``: nested dicts merge, all else replaces.

    Neither input is mutated. A key holding a dict on both sides merges
    recursively, so adding one nested field does not drop its siblings;
    every other type replaces, including lists -- concatenating them
    would silently change what the caller asked to overwrite.
    """
    merged = dict(base)
    for key, value in extra.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def merge_query(
    params: Optional[Mapping[str, Any]],
    options: Optional[RequestOptions],
) -> Optional[Mapping[str, Any]]:
    """``params`` with ``options.extra_query`` merged on top.

    Shallow, unlike the body merge: a query value is a scalar or a
    repeated key, never a structure, so there is nothing to recurse into.
    """
    if options is None or not options.extra_query:
        return params
    return {**(params or {}), **options.extra_query}


def resolve_timeout(
    timeout: Optional[Union[float, httpx.Timeout]],
    options: Optional[RequestOptions],
) -> Optional[Union[float, httpx.Timeout]]:
    """The call's timeout, falling back to ``options.timeout``."""
    if timeout is not None or options is None:
        return timeout
    return options.timeout


def resolve_max_retries(
    default: int,
    options: Optional[RequestOptions],
) -> int:
    """Retry budget for one request: the override, else the client default.

    Clamped at zero like the client-level setting, so a negative override
    means "do not retry" rather than a negative attempt count.
    """
    if options is None or options.max_retries is None:
        return default
    return max(options.max_retries, 0)
