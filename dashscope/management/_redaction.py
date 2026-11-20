# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Secret redaction for the management surface.

Read ops here are not secret-free: ``ListApiKeys`` / ``GetApiKey`` return
the plaintext ``apiKeyValue``. The typed models hold it in a pydantic
``SecretStr`` (masked in ``repr`` / ``model_dump``), but two paths bypass
that field type -- the raw error body attached to an exception, and log
records formatted from a response dict. This module scrubs both so a
key value never reaches a log line, a traceback, or ``err.body``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from dashscope._logging import get_logger

__all__ = [
    "REDACTED",
    "SecretRedactionFilter",
    "get_management_logger",
    "mask_secret",
    "redact_body",
]

REDACTED = "***"

# Lowercased wire keys whose values are credentials. Matched
# case-insensitively so both the modelstudio (``apiKeyValue``) and the
# aliyun gateway (``AccessKeySecret``) spellings are caught.
_SECRET_KEYS = frozenset(
    {
        "apikeyvalue",
        "accesskeysecret",
        "accesskeyid",
        "securitytoken",
        "bearertoken",
        "password",
        "secret",
    },
)


def mask_secret(secret: str) -> str:
    """A log-safe rendering of a secret: ``***`` plus its last 4 chars.

    The tail is enough to tell two keys apart in a support ticket without
    exposing the credential. Anything shorter than 8 chars is fully
    masked -- a short secret has no safe suffix to show.
    """
    if not secret or len(secret) < 8:
        return REDACTED
    return f"{REDACTED}{secret[-4:]}"


def redact_body(body: Any) -> Any:
    """A deep copy of ``body`` with credential-key values replaced.

    Non-dict/list values pass through untouched; the original is never
    mutated, so the caller's parsed model keeps its real ``SecretStr``
    while the error/log copy is safe to print.
    """
    if isinstance(body, dict):
        return {
            key: (
                REDACTED
                if str(key).lower() in _SECRET_KEYS
                else redact_body(value)
            )
            for key, value in body.items()
        }
    if isinstance(body, list):
        return [redact_body(item) for item in body]
    return body


class SecretRedactionFilter(logging.Filter):
    """Scrub credential-looking tokens from a management log record.

    Key-oriented redaction (``redact_body``) cannot catch a secret that
    was interpolated into a free-text message, so this filter also masks
    ``sk-`` prefixed API key values wherever they appear in the formatted
    record. Attached to every management logger via
    ``get_management_logger``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "sk-" in message:
            record.msg = _mask_sk_tokens(message)
            record.args = ()
        return True


# Word-bounded so keys adjacent to '=' or embedded in JSON are caught
# without touching lookalikes such as "task-123". {5,} keeps the old
# space-split threshold (token length > 7).
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{5,}")


def _mask_sk_tokens(message: str) -> str:
    return _SK_RE.sub(lambda m: mask_secret(m.group()), message)


_SECRET_FILTER = SecretRedactionFilter()


def get_management_logger(name: str) -> logging.Logger:
    """A core SDK logger with the management secret filter attached.

    Logger-level filters only run on the emitting logger (not on records
    propagated from children), so every management module must fetch its
    logger through here rather than calling ``get_logger`` directly.
    """
    logger = get_logger(name)
    if _SECRET_FILTER not in logger.filters:
        logger.addFilter(_SECRET_FILTER)
    return logger
