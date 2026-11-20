# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""RAM error crosswalk + success-gate tests for the management surface."""

from __future__ import annotations

import httpx
import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope.errors import (  # noqa: E402
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
)
from dashscope.management._errors import (  # noqa: E402
    check_management_response,
    management_error,
    read_envelope,
)
from dashscope.management._redaction import REDACTED  # noqa: E402


def _response(body, status=200, headers=None):
    return httpx.Response(status, json=body, headers=headers or {})


# --- success gate ---------------------------------------------------------


def test_http200_code200_success_true_is_not_error():
    body = {"code": "200", "success": True, "requestId": "r1"}
    env = read_envelope(body, 200)
    assert env.success is True
    assert check_management_response(_response(body)) is None


def test_tokenplan_success_true_is_not_error():
    body = {"Success": True, "Code": "200", "RequestId": "r2"}
    assert read_envelope(body, 200).success is True


def test_success_false_is_error_even_at_http200():
    body = {"code": "200", "success": False, "message": "boom"}
    env = read_envelope(body, 200)
    assert env.success is False


def test_no_flag_success_code_at_http200_is_success():
    assert read_envelope({"code": "OK"}, 200).success is True
    assert read_envelope({}, 200).success is True


def test_no_flag_non_success_code_at_http200_is_error():
    assert read_envelope({"code": "500"}, 200).success is False


def test_non_2xx_without_flag_is_error():
    assert read_envelope({}, 500).success is False


def test_request_id_normalized_both_casings():
    assert read_envelope({"requestId": "a"}, 200).request_id == "a"
    assert read_envelope({"RequestId": "b"}, 200).request_id == "b"


def test_non_dict_body_degrades_to_empty():
    assert read_envelope(["not", "a", "dict"], 200).success is True
    assert read_envelope(None, 500).success is False


# --- crosswalk by code ----------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("Throttling.User", RateLimitError),
        ("Throttling", RateLimitError),
        ("InvalidAccessKeyId", AuthenticationError),
        ("InvalidAccessKeyId.NotFound", AuthenticationError),
        ("SignatureDoesNotMatch", AuthenticationError),
        ("InvalidSecurityToken.Expired", AuthenticationError),
        ("Forbidden.RAM", PermissionDeniedError),
        ("NoPermission", PermissionDeniedError),
        ("WorkspaceNotExist", NotFoundError),
        ("ResourceNotFound", NotFoundError),
        ("InvalidParameter.WorkspaceId", BadRequestError),
    ],
)
def test_crosswalk_by_code(code, expected):
    env = read_envelope({"code": code, "success": False}, 400)
    assert isinstance(management_error(env), expected)


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, AuthenticationError),
        (403, PermissionDeniedError),
        (404, NotFoundError),
        (429, RateLimitError),
        (500, InternalServerError),
        (503, ServiceUnavailableError),
        (400, BadRequestError),
    ],
)
def test_crosswalk_by_status(status, expected):
    env = read_envelope({}, status)
    assert isinstance(management_error(env), expected)


def test_code_takes_precedence_over_status():
    env = read_envelope({"Code": "Throttling.User"}, 500)
    assert isinstance(management_error(env), RateLimitError)


def test_non_string_code_falls_back_to_status():
    # A numeric code must not crash the crosswalk with AttributeError;
    # _class_by_code coerces like _is_success does.
    env = read_envelope(
        {"success": False, "code": 500, "message": "boom"},
        500,
    )
    assert isinstance(management_error(env), InternalServerError)


def test_gateway_error_without_success_flag():
    body = {
        "Code": "InvalidAccessKeyId",
        "Message": "ak not found",
        "RequestId": "gw-1",
        "Recommend": "https://example",
        "HostId": "modelstudio.cn-beijing.aliyuncs.com",
    }
    err = check_management_response(_response(body, 404), raise_on_error=False)
    assert isinstance(err, AuthenticationError)
    assert err.request_id == "gw-1"
    assert err.code == "InvalidAccessKeyId"


# --- raise vs return ------------------------------------------------------


def test_check_raises_by_default():
    body = {"success": False, "code": "Forbidden", "message": "no"}
    with pytest.raises(PermissionDeniedError):
        check_management_response(_response(body, 403))


def test_check_returns_error_when_not_raising():
    body = {"success": False, "code": "Forbidden", "message": "no"}
    err = check_management_response(
        _response(body, 403),
        raise_on_error=False,
    )
    assert isinstance(err, PermissionDeniedError)
    assert err.message == "no"


# --- redaction ------------------------------------------------------------


def test_error_body_redacts_secret_keys():
    body = {
        "success": False,
        "code": "InvalidParameter",
        "apiKeyValue": "sk-PLAINTEXTSECRET123",
        "nested": {"AccessKeySecret": "super-secret-value"},
    }
    env = read_envelope(body, 400)
    err = management_error(env)
    assert err.body["apiKeyValue"] == REDACTED
    assert err.body["nested"]["AccessKeySecret"] == REDACTED
    assert "sk-PLAINTEXTSECRET123" not in str(err.body)
    assert "super-secret-value" not in str(err.body)


def test_redaction_does_not_mutate_original():
    body = {"apiKeyValue": "sk-PLAINTEXTSECRET123", "success": False}
    env = read_envelope(body, 400)
    management_error(env)
    assert body["apiKeyValue"] == "sk-PLAINTEXTSECRET123"


def test_error_message_and_status_threaded():
    env = read_envelope(
        {"success": False, "code": "Throttling", "message": "slow down"},
        429,
    )
    err = management_error(env, headers={"x-request-id": "h1"})
    assert err.status_code == 429
    assert err.message == "slow down"
    assert err.code == "Throttling"
