# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""ACS3-HMAC-SHA256 (V3, ROA) request signing for the management surface.

Delegates the cryptographic canonicalization to the official
``alibabacloud_openapi_util`` client so the signature matches what the
aliyun gateway recomputes. Two invariants make that hold:

* the signed header set is exactly the sent header set -- the SDK
  identity headers (``x-dashscope-sdk-client: python-sdk/<ver>/management2``
  and friends) are folded in *before* signing, not appended after;
* the canonical query string is also the wire query string, so a
  ``nextToken`` carrying ``=``/``+``/``/`` is encoded identically on both
  sides instead of drifting between httpx's default quoting and the
  signer's ``quote(safe='~')``.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

from alibabacloud_openapi_util.client import Client as _OpenApiUtil
from alibabacloud_openapi_util.client import (
    get_canonical_query_string as _canonical_query_string,
)

from dashscope.management._credentials import ResolvedCredential

__all__ = ["API_VERSION", "SIGN_TYPE", "SignedRequest", "sign_request"]

SIGN_TYPE = "ACS3-HMAC-SHA256"
API_VERSION = "2026-02-10"


@dataclass(frozen=True)
class SignedRequest:
    """A fully signed request ready for httpx to send verbatim."""

    method: str
    url: str
    headers: Dict[str, str]
    content: Optional[bytes]


def sign_request(
    *,
    method: str,
    host: str,
    path: str,
    action: str,
    query: Optional[Mapping[str, Any]],
    body: Optional[bytes],
    credential: ResolvedCredential,
    user_agent: str,
    sdk_headers: Mapping[str, str],
    extra_headers: Optional[Mapping[str, str]] = None,
) -> SignedRequest:
    """Sign one management request and return its wire form.

    ``extra_headers`` are merged in *before* signing (caller wins over the
    defaults), so the signed header set stays equal to the sent set.
    """
    payload = body or b""
    payload_hex = hashlib.sha256(payload).hexdigest()
    flat_query = _OpenApiUtil.query(_normalize_query(query))
    canonical_query = _canonical_query_string(flat_query)
    encoded_path = _OpenApiUtil.get_encode_path(path)

    headers = {
        "host": host,
        "x-acs-version": API_VERSION,
        "x-acs-action": action,
        "x-acs-date": _utc_timestamp(),
        "x-acs-signature-nonce": uuid.uuid4().hex,
        "x-acs-content-sha256": payload_hex,
        "accept": "application/json",
        "user-agent": user_agent,
    }
    if body is not None:
        headers["content-type"] = "application/json"
    if credential.security_token:
        headers["x-acs-security-token"] = credential.security_token
        headers["x-acs-accesskey-id"] = credential.access_key_id
    if credential.provider_name:
        headers["x-acs-credentials-provider"] = credential.provider_name
    headers.update(sdk_headers)
    if extra_headers:
        headers.update({k.lower(): v for k, v in extra_headers.items()})

    shim = SimpleNamespace(
        method=method.upper(),
        pathname=encoded_path,
        query=flat_query,
        headers=headers,
    )
    headers["authorization"] = _OpenApiUtil.get_authorization(
        shim,
        SIGN_TYPE,
        payload_hex,
        credential.access_key_id,
        credential.access_key_secret,
    )
    url = f"https://{host}{encoded_path}"
    if canonical_query:
        url = f"{url}?{canonical_query}"
    return SignedRequest(method.upper(), url, headers, body)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_query(
    query: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Lowercase booleans so ``Client.query`` does not emit ``True``.

    ``Client.query`` stringifies scalars with ``str()``, which renders a
    Python bool as ``True``/``False``; the gateway expects ``true``/
    ``false``. JSON-style arrays arrive pre-serialized as strings and
    flat-style arrays as lists, both left for ``Client.query`` to flatten.
    """
    if not query:
        return {}
    return {key: _normalize_value(value) for key, value in query.items()}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value
