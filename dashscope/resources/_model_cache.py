# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Short-TTL in-memory cache for the read-only model catalog (A4).

The catalog (``models.list`` / ``get`` / ``is_available``) is slow-changing
reference data, so repeated identical reads within a short window can be
served from memory instead of the gateway. The cache is per-Models-instance
and keyed by Region, workspace, credential identity and the exact query, so
an entry never crosses tenants or Regions. Only successful responses are
stored. Errors, ``raw=True`` reads and externally managed HTTP clients
bypass it; ``refresh=True`` forces a fresh fetch. Nothing is persisted to
disk and the cache key holds only a credential fingerprint.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

MODEL_CATALOG_CACHE_TTL = 60.0

_MAX_ENTRIES = 256

__all__ = [
    "MODEL_CATALOG_CACHE_TTL",
    "catalog_key",
    "credential_fingerprint",
    "normalize_params",
]


def credential_fingerprint(api_key: Optional[str]) -> str:
    """A short, non-reversible identity for the effective credential.

    Two calls with the same key share a cache entry; different keys (different
    tenants) do not. The plaintext key is never stored -- only this digest.
    """
    return hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:16]


def normalize_params(params: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """A hashable, order-independent form of the catalog query params.

    Keys are sorted and list values become tuples so the same filter set
    hashes identically regardless of dict ordering or list identity.
    """
    items = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, (list, tuple)):
            value = tuple(value)
        items.append((key, value))
    return tuple(items)


def catalog_key(
    config: Any,
    params: Mapping[str, Any],
    *,
    api_key: Optional[str],
    workspace: Optional[str],
) -> Tuple[Any, ...]:
    """Build the isolation key for one catalog read.

    ``api_key`` / ``workspace`` are the per-call overrides; ``None`` falls
    back to the client config, mirroring what the transport actually sends.
    """
    eff_workspace = workspace if workspace is not None else config.workspace
    eff_api_key = api_key if api_key is not None else config.api_key
    return (
        config.region,
        eff_workspace,
        credential_fingerprint(eff_api_key),
        normalize_params(params),
    )


class _CatalogCache:
    """Per-instance TTL cache of successful catalog responses.

    Bounded to ``_MAX_ENTRIES``; the oldest entry is evicted when full.
    Thread-safe for the sync client. The async client shares one instance on
    a single event loop and never awaits inside the lock, so the same short
    critical section is safe there too.
    """

    def __init__(self, ttl: float = MODEL_CATALOG_CACHE_TTL) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        self._entries: Dict[Any, Tuple[float, "APIResponse"]] = {}

    def get(self, key: Any) -> Optional["APIResponse"]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, resp = entry
            if time.monotonic() - stored_at >= self._ttl:
                del self._entries[key]
                return None
            return resp

    def set(self, key: Any, resp: "APIResponse") -> None:
        with self._lock:
            if key not in self._entries and len(self._entries) >= _MAX_ENTRIES:
                oldest = min(self._entries, key=lambda k: self._entries[k][0])
                del self._entries[oldest]
            self._entries[key] = (time.monotonic(), resp)
