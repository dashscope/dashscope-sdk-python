# -*- coding: utf-8 -*-
"""Identifier and timestamp helpers used across local storage providers."""

from __future__ import annotations

import hashlib
import os
import platform
import uuid
from datetime import datetime, timezone


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def short_uuid(length: int = 12) -> str:
    """Return a short hex UUID of the requested length."""
    return uuid.uuid4().hex[:length]


def stable_memory_user_id(length: int = 8) -> str:
    """Return a stable, opaque user identifier for local memory storage.

    Derived from machine hostname and OS user so the same human on the same
    machine always gets the same memory file, without exposing the actual
    username/hostname in the ID.
    """
    host = platform.node() or "unknown"
    user = (
        str(os.getuid())
        if hasattr(os, "getuid")
        else os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    )
    digest = hashlib.sha256(f"{host}:{user}:acli-memory".encode()).hexdigest()
    return f"acli-{digest[:length]}"
