# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Explicit cross-Region model discovery (A5, scope-review §22.7).

A :class:`RegionRouter` aggregates several per-Region Clients that the
caller supplies explicitly -- the SDK never guesses another Region's
credential or workspace (§22.4). It answers "which Regions serve this
model" by probing each Client's own catalog; it does **not** route or
replay inference requests, and there is **no automatic cross-Region
fallback** (§22.8): a Region that errors is reported as an error entry,
never silently skipped or retried elsewhere.

**Disabled by default.** Cross-Region discovery may move data across
borders, so constructing a router requires an explicit
``allow_cross_region=True`` opt-in, and enabling it warns that
**cross-border data compliance is the user's sole responsibility**.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Mapping, Optional

from pydantic import BaseModel, Field

from dashscope._config import _is_safe_region_label
from dashscope.client import AsyncClient, Client
from dashscope.errors import DashScopeError
from dashscope.resources.models import Model

__all__ = [
    "AsyncRegionRouter",
    "ModelAvailability",
    "RegionModelInfo",
    "RegionRouter",
]

_CROSS_BORDER_NOTICE = (
    "RegionRouter performs cross-region model discovery, which may involve "
    "cross-border data transfer. Cross-border data compliance is the user's "
    "sole responsibility: you must ensure compliance with all applicable "
    "data-export, residency and privacy laws before enabling it."
)

_OPT_IN_REQUIRED = (
    "cross-region discovery is disabled by default. Pass "
    "allow_cross_region=True to opt in. " + _CROSS_BORDER_NOTICE
)


def _gate_cross_region(allow_cross_region: bool) -> None:
    if not allow_cross_region:
        raise ValueError(_OPT_IN_REQUIRED)
    warnings.warn(_CROSS_BORDER_NOTICE, UserWarning, stacklevel=3)


class RegionModelInfo(BaseModel):
    """One Region's answer for a model.

    ``available`` is True only when that Region's catalog lists the model.
    ``model`` carries the catalog entry when available. ``error`` is set
    when the Region query itself failed (auth/transport/HTTP), which is
    distinct from "the model is not in this Region's catalog"; the Region
    label is always present so a failure is attributable (§22.8).
    """

    region: str
    available: bool = False
    model: Optional[Model] = None
    error: Optional[str] = None


class ModelAvailability(BaseModel):
    """Aggregated cross-Region availability for one model."""

    model: str
    regions: Dict[str, RegionModelInfo] = Field(default_factory=dict)

    @property
    def available_in(self) -> List[str]:
        """The Region labels whose catalog serves the model."""
        return [
            region
            for region, info in self.regions.items()
            if info.available
        ]


def _validate_clients(clients, expected_type):
    if not clients:
        raise ValueError(
            "RegionRouter requires at least one per-region Client",
        )
    validated = {}
    for region, client in clients.items():
        if not _is_safe_region_label(region):
            raise ValueError(
                f"invalid region label {region!r}: expected a lowercase DNS "
                f"label (letters, digits, hyphens), e.g. 'cn-beijing'",
            )
        if not isinstance(client, expected_type):
            raise TypeError(
                f"clients[{region!r}] must be a {expected_type.__name__}",
            )
        if client.config.region != region:
            raise ValueError(
                f"clients[{region!r}] is configured for region "
                f"{client.config.region!r}; the key must match the client's "
                f"own region (region/workspace/key are one config unit)",
            )
        validated[region] = client
    return validated


def _error_text(exc: DashScopeError) -> str:
    return f"{exc.code}: {exc.message}"


class _RouterModels:
    def __init__(self, clients: Mapping[str, Client]) -> None:
        self._clients = clients

    def available_regions(
        self,
        model: str,
        *,
        timeout: Optional[float] = None,
        refresh: bool = False,
    ) -> ModelAvailability:
        """Probe every supplied Region's catalog for ``model``.

        Each Region is queried through its own Client (its own credential,
        workspace and endpoint); ``timeout`` / ``refresh`` forward to the
        catalog read. A Region that fails is recorded with its error rather
        than aborting the discovery or falling back to another Region.
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty model id")
        regions = {}
        for region, client in self._clients.items():
            regions[region] = _probe_region(
                client,
                model,
                region,
                timeout=timeout,
                refresh=refresh,
            )
        return ModelAvailability(model=model, regions=regions)


def _probe_region(client, model, region, *, timeout, refresh):
    try:
        available = client.models.is_available(
            model,
            timeout=timeout,
            refresh=refresh,
        )
    except DashScopeError as exc:
        return RegionModelInfo(
            region=region,
            available=False,
            error=_error_text(exc),
        )
    if not available:
        return RegionModelInfo(region=region, available=False)
    try:
        # is_available already made (or refreshed) the shared catalog read,
        # so get reuses it from the cache rather than fetching again.
        entry = client.models.get(model, timeout=timeout)
    except DashScopeError as exc:
        return RegionModelInfo(
            region=region,
            available=True,
            error=_error_text(exc),
        )
    return RegionModelInfo(region=region, available=True, model=entry)


class RegionRouter:
    """Aggregate explicit per-Region Clients for cross-Region discovery.

    >>> router = RegionRouter(
    ...     clients={"cn-beijing": bj, "ap-southeast-1": sg},
    ...     allow_cross_region=True,
    ... )
    >>> router.models.available_regions("qwen-plus").available_in
    ['cn-beijing', 'ap-southeast-1']

    Keys must be safe DNS region labels matching each Client's own
    ``config.region``; values must be :class:`~dashscope.Client` instances.
    Discovery only -- no inference routing, no automatic fallback.

    Disabled by default: ``allow_cross_region`` must be explicitly True,
    which warns that cross-border data compliance is the user's sole
    responsibility.
    """

    def __init__(
        self,
        clients: Mapping[str, Client],
        *,
        allow_cross_region: bool = False,
    ) -> None:
        _gate_cross_region(allow_cross_region)
        self._clients = _validate_clients(clients, Client)
        self.models = _RouterModels(self._clients)


class _AsyncRouterModels:
    def __init__(self, clients: Mapping[str, AsyncClient]) -> None:
        self._clients = clients

    async def available_regions(
        self,
        model: str,
        *,
        timeout: Optional[float] = None,
        refresh: bool = False,
    ) -> ModelAvailability:
        """Async :meth:`_RouterModels.available_regions`."""
        if not model or not model.strip():
            raise ValueError("model must be a non-empty model id")
        regions = {}
        for region, client in self._clients.items():
            regions[region] = await _probe_region_async(
                client,
                model,
                region,
                timeout=timeout,
                refresh=refresh,
            )
        return ModelAvailability(model=model, regions=regions)


async def _probe_region_async(client, model, region, *, timeout, refresh):
    try:
        available = await client.models.is_available(
            model,
            timeout=timeout,
            refresh=refresh,
        )
    except DashScopeError as exc:
        return RegionModelInfo(
            region=region,
            available=False,
            error=_error_text(exc),
        )
    if not available:
        return RegionModelInfo(region=region, available=False)
    try:
        # is_available already made (or refreshed) the shared catalog read,
        # so get reuses it from the cache rather than fetching again.
        entry = await client.models.get(model, timeout=timeout)
    except DashScopeError as exc:
        return RegionModelInfo(
            region=region,
            available=True,
            error=_error_text(exc),
        )
    return RegionModelInfo(region=region, available=True, model=entry)


class AsyncRegionRouter:
    """Async :class:`RegionRouter`; per-Region ``AsyncClient`` instances.

    Disabled by default: ``allow_cross_region`` must be explicitly True,
    which warns that cross-border data compliance is the user's sole
    responsibility.
    """

    def __init__(
        self,
        clients: Mapping[str, AsyncClient],
        *,
        allow_cross_region: bool = False,
    ) -> None:
        _gate_cross_region(allow_cross_region)
        self._clients = _validate_clients(clients, AsyncClient)
        self.models = _AsyncRouterModels(self._clients)