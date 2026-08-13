# -*- coding: utf-8 -*-
"""Sensitive data sanitization — redact API keys, tokens, and secrets
before persisting to disk.

Security improvements:
- Expanded pattern coverage (Alibaba Cloud RAM, WeChat, DingTalk, JWT, etc.)
- Performance: compiled regex list built once at module import, not per-call
"""

from __future__ import annotations

import re
from typing import Any

_REDACT = "[REDACTED]"

# Patterns are compiled once. `has_group` indicates whether group(1) is a
# prefix to keep (e.g. "Bearer ") vs. the whole match being the secret.
_PATTERNS: list[tuple[re.Pattern, bool]] = [
    # Bearer tokens
    (re.compile(r"(?i)(bearer\s+)\S+"), True),
    # Generic key=value / key: value assignments (>= 8 char values)
    (
        re.compile(
            r"(?i)((?:api[_-]?key|api[_-]?secret|password|passwd|token|"
            r"secret|access[_-]?key|secret[_-]?key|auth[_-]?token|"
            r"credentials?|private[_-]?key|encryption[_-]?key|"
            r"client[_-]?secret|app[_-]?secret|session[_-]?token|"
            r"refresh[_-]?token|signing[_-]?key|"
            r"hmac[_-]?key)\s*[=:]\s*[\"']?)"
            r'[^\s"\']{8,}',
        ),
        True,
    ),
    # Anthropic
    (re.compile(r"sk-ant-[A-Za-z0-9\-]{6,}"), False),
    # OpenAI
    (re.compile(r"sk-proj-[A-Za-z0-9\-]{6,}"), False),
    (re.compile(r"sk-[A-Za-z0-9\-]{6,}"), False),
    # Alibaba Cloud AccessKey ID
    (re.compile(r"LTAI[A-Za-z0-9]{12,20}"), False),
    # AWS Access Key / Session Token prefix
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), False),
    # GitHub PATs / OAuth
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), False),
    # GitLab PAT
    (re.compile(r"glpat-[A-Za-z0-9\-]{20,}"), False),
    # Slack tokens
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), False),
    # WeChat AppSecret / AccessToken
    (
        re.compile(
            r"(?:appsecret|access_token)[\"']?\s*[:=]\s*[\"']?"
            r"[A-Za-z0-9]{20,}",
        ),
        False,
    ),
    # DingTalk access_token
    (re.compile(r"access_token=[A-Za-z0-9]{30,}"), False),
    # JWT (header.payload.signature)
    (
        re.compile(r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
        False,
    ),
    # PEM private keys (RSA, EC, etc.) — match the BEGIN line + first
    # content line
    (
        re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[^\n]+\n[^\n]+"),
        False,
    ),
    # Azure Storage Account Key
    (re.compile(r"AccountKey=[A-Za-z0-9+/=]{40,}"), False),
    # Google Cloud service account key (base64-ish block after "private_key")
    (re.compile(r'"private_key"\s*:\s*"[^"]{50,}"'), False),
]


def is_secret_field(name: str) -> bool:
    """Check if a field name indicates it holds a secret value."""
    n = name.lower()
    return "key" in n or "secret" in n or "token" in n or "password" in n


def sanitize_text(text: str) -> str:
    """Redact secrets from a plain string."""
    if not isinstance(text, str):
        return text
    for pat, has_group in _PATTERNS:
        if has_group:
            text = pat.sub(lambda m: m.group(1) + _REDACT, text)
        else:
            text = pat.sub(_REDACT, text)
    return text


def sanitize(obj: Any) -> Any:
    """Recursively redact secrets from str, list, or dict."""
    if isinstance(obj, str):
        return sanitize_text(obj)
    if isinstance(obj, list):
        return [sanitize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    return obj
