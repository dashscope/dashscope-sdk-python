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
from typing import Any, Iterator

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


def _secret_values(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS:
                if isinstance(item, str) and item:
                    yield item
            else:
                yield from _secret_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _secret_values(item)


def redact_body(body: Any) -> Any:
    """Copy and scrub credential fields and their echoes in free text.

    Also masks recognizable API keys when there is no dedicated credential
    field. The original payload remains intact for successful typed reads.
    """
    values = set(_secret_values(body))
    pattern = (
        re.compile(
            "|".join(map(re.escape, sorted(values, key=len, reverse=True))),
        )
        if values
        else None
    )

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: REDACTED
                if str(key).lower() in _SECRET_KEYS
                else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, str):
            if pattern is not None:
                value = pattern.sub(REDACTED, value)
            return _mask_sk_tokens(value)
        return value

    return redact(body)


class SecretRedactionFilter(logging.Filter):
    """Scrub credential-looking tokens from a management log record.

    Log records may contain text that never passed through ``redact_body``,
    so this filter also masks ``sk-`` prefixed API key values wherever
    they appear in the formatted record. Attached to every management logger via
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
