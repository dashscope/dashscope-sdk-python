# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Logging setup for the dashscope 2.0 SDK.

Enable with ``DASHSCOPE_LOG=debug|info|warning|error``. Requests,
response status, request_id and retry decisions are logged; sensitive
headers (Authorization etc.) and bearer tokens are redacted.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Mapping, Optional

LOG_ENV = "DASHSCOPE_LOG"
ROOT_NAME = "dashscope.v2"

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

SENSITIVE_HEADER_KEYS = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "cookie",
        "proxy-authorization",
        "x-dashscope-encryptionkey",
    },
)

_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s'\"]+")
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")

REDACTED = "[REDACTED]"


def redact_headers(
    headers: Optional[Mapping[str, Any]],
) -> Optional[dict]:
    if headers is None:
        return None
    return {
        k: REDACTED if k.lower() in SENSITIVE_HEADER_KEYS else v
        for k, v in headers.items()
    }


def redact_text(text: str) -> str:
    text = _BEARER_RE.sub(r"\1" + REDACTED, text)
    return _API_KEY_RE.sub(REDACTED, text)


class SensitiveHeadersFilter(logging.Filter):
    """Redacts sensitive headers and tokens from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        headers = getattr(record, "headers", None)
        if headers is not None:
            record.headers = redact_headers(headers)
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(
                redact_text(a) if isinstance(a, str) else a
                for a in record.args
            )
        elif isinstance(record.args, dict):
            record.args = {
                k: redact_text(v) if isinstance(v, str) else v
                for k, v in record.args.items()
            }
        return True


_SENSITIVE_FILTER = SensitiveHeadersFilter()

_configured = False


def _configure(level: int) -> None:
    logger = logging.getLogger(ROOT_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(_SENSITIVE_FILTER)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
            ),
        )
        logger.addHandler(handler)


def set_log_level(level_name: str) -> None:
    """Apply a log level to the v2 logger immediately."""
    level = _LEVELS.get(str(level_name).strip().lower())
    if level is None:
        raise ValueError(
            f"invalid log level {level_name!r}, "
            f"expected one of {sorted(_LEVELS)}",
        )
    _configure(level)


def reset_log_level() -> None:
    """Restore the v2 logger level to the env-configured level or WARNING."""
    level_name = os.environ.get(LOG_ENV, "").strip().lower()
    level = _LEVELS.get(level_name, logging.WARNING)
    logging.getLogger(ROOT_NAME).setLevel(level)


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    level_name = os.environ.get(LOG_ENV, "").strip().lower()
    level = _LEVELS.get(level_name)
    if level is None:
        return
    _configure(level)


def get_logger(name: str) -> logging.Logger:
    _ensure_configured()
    logger = logging.getLogger(f"{ROOT_NAME}.{name}")
    # Logger-level filters don't see records propagated from children, so
    # the filter must sit on each emitting logger (and on the handler).
    if _SENSITIVE_FILTER not in logger.filters:
        logger.addFilter(_SENSITIVE_FILTER)
    return logger
