# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""V3 ACS3-HMAC-SHA256 signing tests for the management surface.

The oracle here recomputes the signature the way the aliyun gateway does
-- from the *wire* URL and headers, with hashlib/hmac directly, independent
of openapi_util -- so a drift between what we sign and what we send fails
loudly. A second check compares against openapi_util.get_authorization.
"""

from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlsplit

import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from alibabacloud_openapi_util.client import (  # noqa: E402
    Client as OpenApiUtil,
)
from alibabacloud_openapi_util.client import (  # noqa: E402
    get_canonical_query_string,
)

from dashscope.management import _signer  # noqa: E402
from dashscope.management._credentials import ResolvedCredential  # noqa: E402
from dashscope.management._signer import sign_request  # noqa: E402

AK = "ak-test"
SK = "sk-test"
HOST = "modelstudio.cn-beijing.aliyuncs.com"
SDK_HEADERS = {
    "x-dashscope-sdk-client": "python-sdk/2.0.0/management2",
    "x-dashscope-sdk-session-id": "sess-test",
    "x-dashscope-client-request-id": "crid-test",
}


def _credential(token=None, provider=None):
    return ResolvedCredential(
        access_key_id=AK,
        access_key_secret=SK,
        security_token=token,
        provider_name=provider,
    )


def _sign(**kw):
    params = {
        "method": "GET",
        "host": HOST,
        "path": "/modelstudio/workspaces",
        "action": "ListWorkspaces",
        "query": None,
        "body": None,
        "credential": _credential(),
        "user_agent": "dashscope-sdk-python/2.0.0 management",
        "sdk_headers": SDK_HEADERS,
    }
    params.update(kw)
    return sign_request(**params)


def _gateway_recompute(signed, ak=AK, sk=SK):
    """Independently rebuild the Authorization from the wire form."""
    parts = urlsplit(signed.url)
    canonical_uri = parts.path
    canonical_query = parts.query
    headers = {
        k.lower(): v
        for k, v in signed.headers.items()
        if k.lower() != "authorization"
    }
    keys = sorted(headers)
    canonical_headers = "".join(
        f"{k}:{str(headers[k]).strip()}\n" for k in keys
    )
    signed_headers = ";".join(keys)
    payload = headers["x-acs-content-sha256"]
    canonical_request = (
        f"{signed.method}\n{canonical_uri}\n{canonical_query}\n"
        f"{canonical_headers}\n{signed_headers}\n{payload}"
    )
    hashed = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"ACS3-HMAC-SHA256\n{hashed}"
    signature = hmac.new(
        sk.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return (
        f"ACS3-HMAC-SHA256 Credential={ak},"
        f"SignedHeaders={signed_headers},Signature={signature}"
    )


def _signed_header_set(signed):
    auth = signed.headers["authorization"]
    fragment = auth.split("SignedHeaders=", 1)[1].split(",", 1)[0]
    return set(fragment.split(";"))


def test_signature_matches_gateway_oracle(monkeypatch):
    monkeypatch.setattr(
        _signer,
        "_utc_timestamp",
        lambda: "2026-02-10T00:00:00Z",
    )
    signed = _sign(query={"maxResults": 10, "workspaceName": "demo"})
    assert signed.headers["authorization"] == _gateway_recompute(signed)


def test_signature_matches_openapi_util_oracle():
    signed = _sign(query={"maxResults": 10})
    parts = urlsplit(signed.url)
    headers = {
        k: v for k, v in signed.headers.items() if k.lower() != "authorization"
    }
    shim = type(
        "Shim",
        (),
        {
            "method": signed.method,
            "pathname": parts.path,
            "query": dict(
                pair.split("=", 1) for pair in parts.query.split("&")
            ),
            "headers": headers,
        },
    )()
    expected = OpenApiUtil.get_authorization(
        shim,
        "ACS3-HMAC-SHA256",
        headers["x-acs-content-sha256"],
        AK,
        SK,
    )
    assert signed.headers["authorization"] == expected


def test_signed_headers_equal_sent_headers():
    signed = _sign(query={"maxResults": 5})
    sent = {k.lower() for k in signed.headers if k.lower() != "authorization"}
    assert _signed_header_set(signed) == sent


def test_sdk_headers_are_signed():
    signed = _sign()
    signed_set = _signed_header_set(signed)
    assert "x-dashscope-sdk-client" in signed_set
    assert "x-dashscope-sdk-session-id" in signed_set
    assert signed.headers["x-dashscope-sdk-client"].endswith("management2")


def test_special_char_query_wire_matches_canonical():
    token = "abc=+/def~ghi jkl"
    signed = _sign(query={"nextToken": token, "maxResults": 10})
    parts = urlsplit(signed.url)
    flat = OpenApiUtil.query({"nextToken": token, "maxResults": "10"})
    canonical = get_canonical_query_string(flat)
    assert parts.query == canonical
    assert signed.headers["authorization"] == _gateway_recompute(signed)


def test_path_is_encoded():
    signed = _sign(path="/modelstudio/apikeys/123", action="GetApiKey")
    assert urlsplit(signed.url).path == "/modelstudio/apikeys/123"
    encoded = _sign(path="/modelstudio/a b/c~d")
    assert urlsplit(encoded.url).path == OpenApiUtil.get_encode_path(
        "/modelstudio/a b/c~d",
    )


def test_bool_query_lowercased_on_wire():
    signed = _sign(
        action="GetBillingOverview",
        path="/modelstudio/billing/overview",
        query={"zeroFilter": True, "includeHistoryPeriod": False},
    )
    query = urlsplit(signed.url).query
    assert "zeroFilter=true" in query
    assert "includeHistoryPeriod=false" in query
    assert "True" not in query and "False" not in query


def test_sts_token_headers_present_and_signed():
    signed = _sign(credential=_credential(token="sts-token-test"))
    assert signed.headers["x-acs-security-token"] == "sts-token-test"
    assert signed.headers["x-acs-accesskey-id"] == AK
    signed_set = _signed_header_set(signed)
    assert "x-acs-security-token" in signed_set
    assert signed.headers["authorization"] == _gateway_recompute(signed)


def test_credentials_provider_header_when_named():
    signed = _sign(credential=_credential(provider="ecs_ram_role"))
    assert signed.headers["x-acs-credentials-provider"] == "ecs_ram_role"
    assert "x-acs-credentials-provider" in _signed_header_set(signed)


def test_empty_body_payload_hash():
    signed = _sign()
    assert (
        signed.headers["x-acs-content-sha256"]
        == hashlib.sha256(
            b"",
        ).hexdigest()
    )
    assert signed.content is None
    assert "content-type" not in signed.headers


def test_json_body_payload_hash_and_content_type():
    body = b'{"workspaceName":"demo"}'
    signed = _sign(method="POST", body=body)
    assert (
        signed.headers["x-acs-content-sha256"]
        == hashlib.sha256(
            body,
        ).hexdigest()
    )
    assert signed.headers["content-type"] == "application/json"
    assert signed.content == body
    assert signed.headers["authorization"] == _gateway_recompute(signed)


def test_extra_headers_merged_before_signing():
    signed = _sign(extra_headers={"X-Custom": "v"})
    assert signed.headers["x-custom"] == "v"
    assert "x-custom" in _signed_header_set(signed)
    assert signed.headers["authorization"] == _gateway_recompute(signed)


def test_authorization_format():
    signed = _sign()
    auth = signed.headers["authorization"]
    assert auth.startswith(f"ACS3-HMAC-SHA256 Credential={AK},")
    signature = auth.split("Signature=", 1)[1]
    assert len(signature) == 64
    int(signature, 16)  # valid hex


def test_acs_version_and_action_headers():
    signed = _sign(action="ListWorkspaces")
    assert signed.headers["x-acs-version"] == "2026-02-10"
    assert signed.headers["x-acs-action"] == "ListWorkspaces"
    assert signed.headers["host"] == HOST
    assert signed.headers["x-acs-date"]
    assert signed.headers["x-acs-signature-nonce"]
