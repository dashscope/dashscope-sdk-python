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
from typing import Any

__all__ = [
    "REDACTED",
    "SecretRedactionFilter",
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
    record. Attached to the management logger only.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "sk-" in message:
            record.msg = _mask_sk_tokens(message)
            record.args = ()
        return True


def _mask_sk_tokens(message: str) -> str:
    out = []
    for token in message.split(" "):
        if token.startswith("sk-") and len(token) > 7:
            out.append(mask_secret(token))
        else:
            out.append(token)
    return " ".join(out)
