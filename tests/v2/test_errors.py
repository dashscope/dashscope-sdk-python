# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the dashscope 2.0 error model.

Two things are pinned here. First, the runtime behaviour of the
mapping functions (which class a given status / wire code produces).
Second, the coupling to the generated catalog in
``dashscope/v2/_error_registry.py``: exception class names, their
``status_code``, and the wire-code crosswalk must all be derivable
from it, so an upstream regeneration shows up as a test failure
instead of silent drift.
"""

from __future__ import annotations

import httpx
import pytest

from dashscope.v2 import _error_registry as registry
from dashscope.v2 import errors
from dashscope.v2.errors import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ContentFilterError,
    DashScopeError,
    ErrorCode,
    GatewayTimeoutError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    SdkError,
    ServiceUnavailableError,
    check_response,
    error_from_business_code,
    error_from_sdk_exception,
)
from dashscope.v2.errors import (
    _CODE_MAP,
    _REFINED_CLASS,
    _SDK_ERRORS,
    _STATUS_MAP,
    _WIRE_TO_PUBLIC,
)
from dashscope.v2.types.common import JobStatus, ResponseBase


def _resp(status, body=None, headers=None):
    return httpx.Response(
        status,
        json=body,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.test.local/x"),
    )


def test_status_mapping_and_attributes():
    resp = _resp(
        401,
        {
            "code": "InvalidApiKey",
            "message": "Invalid API-key provided.",
            "request_id": "req-1",
        },
    )
    with pytest.raises(AuthenticationError) as ei:
        check_response(resp)
    err = ei.value
    assert err.code == "InvalidApiKey"
    assert err.status_code == 401
    assert err.request_id == "req-1"
    assert err.message == "Invalid API-key provided."
    assert err.body["code"] == "InvalidApiKey"
    assert "AuthenticationError" in str(err)


def test_openai_error_envelope_parsed():
    resp = _resp(
        401,
        {
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            },
        },
    )
    with pytest.raises(AuthenticationError) as ei:
        check_response(resp)
    err = ei.value
    assert err.message == "Incorrect API key provided."
    assert err.code == "invalid_api_key"
    assert err.status_code == 401


def test_anthropic_error_envelope_parsed():
    resp = _resp(
        429,
        {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Number of requests exceeded.",
            },
        },
    )
    with pytest.raises(RateLimitError) as ei:
        check_response(resp)
    assert ei.value.message == "Number of requests exceeded."
    assert ei.value.code == "rate_limit_error"


def test_native_fields_take_precedence_over_envelope():
    resp = _resp(
        400,
        {
            "code": "InvalidParameter",
            "message": "native says no",
            "error": {"message": "nested", "code": "nested_code"},
        },
    )
    with pytest.raises(BadRequestError) as ei:
        check_response(resp)
    assert ei.value.message == "native says no"
    assert ei.value.code == "InvalidParameter"


def test_non_dict_error_envelope_ignored():
    resp = _resp(400, {"message": "plain", "error": "oops"})
    with pytest.raises(BadRequestError) as ei:
        check_response(resp)
    assert ei.value.message == "plain"


def test_status_only_mapping():
    cases = [
        (400, {"message": "bad"}, BadRequestError),
        (401, None, AuthenticationError),
        (403, {"code": "AccessDenied"}, PermissionDeniedError),
        (404, None, NotFoundError),
        (413, None, RequestTooLargeError),
        (429, {"code": "Throttling.RateQuota"}, RateLimitError),
        (500, {"code": "InternalError"}, InternalServerError),
        (503, None, ServiceUnavailableError),
        (504, None, GatewayTimeoutError),
    ]
    for status, body, expected in cases:
        with pytest.raises(expected) as ei:
            check_response(_resp(status, body))
        assert type(ei.value) is expected, status
        assert ei.value.status_code == status


def test_uncatalogued_status_falls_back_to_the_base_class():
    # The catalog defines 400/401/403/404/413/429/500/503/504. Anything
    # else -- including 5xx the catalog does not list -- stays the bare
    # base class rather than being guessed into InternalServerError.
    for status in (418, 409, 502, 505):
        with pytest.raises(APIStatusError) as ei:
            check_response(_resp(status, {"message": "nope"}))
        assert type(ei.value) is APIStatusError, status
        assert ei.value.status_code == status


def test_content_filter_from_status():
    resp = _resp(400, {"code": "DataInspectionFailed", "message": "blocked"})
    with pytest.raises(ContentFilterError) as ei:
        check_response(resp)
    assert isinstance(ei.value, BadRequestError)
    assert ei.value.status_code == 400


def test_business_error_on_http_200():
    resp = _resp(
        200,
        {
            "code": "DataInspectionFailed",
            "message": "output blocked",
            "request_id": "r2",
        },
    )
    with pytest.raises(ContentFilterError) as ei:
        check_response(resp)
    assert ei.value.status_code == 200
    assert ei.value.request_id == "r2"
    # Same class on both layers, so one `except ContentFilterError`
    # covers moderation blocks however the gateway chose to report them.
    assert isinstance(ei.value, BadRequestError)


def test_unknown_business_code_on_http_200():
    resp = _resp(200, {"code": "Some.Future.Code", "message": "boom"})
    with pytest.raises(DashScopeError) as ei:
        check_response(resp)
    assert type(ei.value) is DashScopeError
    assert ei.value.code == "Some.Future.Code"


def test_ok_response_passes():
    check_response(_resp(200, {"output": {"text": "hi"}, "request_id": "r3"}))


def test_raise_on_error_false_swallows():
    check_response(
        _resp(429, {"code": "Throttling", "message": "slow down"}),
        raise_on_error=False,
    )
    check_response(
        _resp(200, {"code": "Throttling", "message": "slow down"}),
        raise_on_error=False,
    )


def test_response_base_flags_business_code():
    # Non-raising mode leaves the parsed model as the only channel, so it
    # has to report the failure itself: same rule response_error applies.
    assert ResponseBase(code="Throttling", message="slow down").is_error
    assert not ResponseBase(request_id="r1").is_error
    # An explicitly flagged envelope stays flagged without a code.
    assert ResponseBase(is_error=True).is_error


def test_response_base_subclasses_inherit_the_flag():
    assert JobStatus(status="FAILED", code="Job.Failed").is_error
    assert not JobStatus(status="SUCCEEDED").is_error


def test_non_json_body():
    resp = httpx.Response(
        502,
        content=b"Bad Gateway",
        headers={"content-type": "text/plain"},
        request=httpx.Request("GET", "https://api.test.local/x"),
    )
    with pytest.raises(APIStatusError) as ei:
        check_response(resp)
    assert ei.value.body is None
    assert ei.value.status_code == 502


def test_sse_error_body_parsed():
    # A streaming request rejected before any output arrives as a 4xx
    # whose body is the error frame itself; the frame's payload must
    # drive the mapping exactly like the JSON body of a non-streaming
    # failure.
    body = (
        "event:error\n"
        'data:{"code":"InvalidParameter",'
        '"message":"Partial mode is not supported when '
        'enable_thinking is true","request_id":"sse-req-1"}'
    )
    resp = httpx.Response(
        400,
        content=body.encode("utf-8"),
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", "https://api.test.local/x"),
    )
    with pytest.raises(BadRequestError) as ei:
        check_response(resp)
    err = ei.value
    assert err.code == "InvalidParameter"
    assert err.message == (
        "Partial mode is not supported when enable_thinking is true"
    )
    assert err.status_code == 400
    assert err.request_id == "sse-req-1"
    assert err.body["code"] == "InvalidParameter"


def test_sse_error_body_wire_code_picks_the_class():
    # Wire-code priority must survive the SSE framing: the status alone
    # would say BadRequestError, the frame says content moderation. The
    # content-type header is deliberately absent -- frames are detected
    # by shape, like resources/_sse.py.
    body = (
        "event:error\n"
        'data:{"code":"DataInspectionFailed","message":"blocked",'
        '"request_id":"sse-req-2"}'
    )
    resp = httpx.Response(
        400,
        content=body.encode("utf-8"),
        request=httpx.Request("POST", "https://api.test.local/x"),
    )
    with pytest.raises(ContentFilterError) as ei:
        check_response(resp)
    assert isinstance(ei.value, BadRequestError)
    assert ei.value.code == "DataInspectionFailed"
    assert ei.value.request_id == "sse-req-2"


def test_sse_body_without_decodable_frame_keeps_raw_text():
    # No JSON in the frame: the raw-text fallback stands and the class
    # comes from the status alone.
    resp = httpx.Response(
        400,
        content=b"event:error\ndata:oops",
        request=httpx.Request("POST", "https://api.test.local/x"),
    )
    with pytest.raises(BadRequestError) as ei:
        check_response(resp)
    assert ei.value.code is None
    assert ei.value.body is None
    assert ei.value.message == "event:error\ndata:oops"


def test_error_code_enum():
    assert ErrorCode.INVALID_API_KEY.value == "InvalidApiKey"
    assert ErrorCode.DATA_INSPECTION_FAILED.value == "DataInspectionFailed"
    assert ErrorCode("Throttling") is ErrorCode.THROTTLING


def test_request_id_from_header_fallback():
    resp = _resp(
        404,
        {"message": "not found"},
        headers={"x-request-id": "hdr-req-1"},
    )
    with pytest.raises(NotFoundError) as ei:
        check_response(resp)
    assert ei.value.request_id == "hdr-req-1"


def test_every_catalog_error_code_names_a_real_class():
    for entry in registry.PUBLIC_ERRORS:
        cls = getattr(errors, entry.error_code, None)
        assert isinstance(cls, type), entry.error_code
        assert issubclass(cls, APIStatusError), entry.error_code
        assert cls.status_code == entry.status_code, entry.error_code


def test_catalog_classes_are_flat_siblings():
    classes = {
        entry.error_code: getattr(errors, entry.error_code)
        for entry in registry.PUBLIC_ERRORS
    }
    assert len(classes) == 9
    for name, cls in classes.items():
        assert cls.__bases__ == (APIStatusError,), name
        for other_name, other in classes.items():
            if other is not cls:
                assert not issubclass(cls, other), (name, other_name)


def test_status_map_is_derived_from_the_catalog():
    catalogued = {e.status_code for e in registry.PUBLIC_ERRORS}
    assert set(_STATUS_MAP) == catalogued
    for entry in registry.PUBLIC_ERRORS:
        expected = getattr(errors, entry.error_code)
        assert _STATUS_MAP[entry.status_code] is expected, entry.key


def test_catalog_keys_and_wire_codes_are_separate_namespaces():
    # Why the crosswalk has to exist at all: the catalog's `key` is an
    # internal canonical name, never a value the gateway sends.
    keys = {entry.key for entry in registry.PUBLIC_ERRORS}
    assert keys.isdisjoint({member.value for member in ErrorCode})


def test_crosswalk_covers_every_declared_wire_code():
    assert set(_WIRE_TO_PUBLIC) == {member.value for member in ErrorCode}
    assert set(_CODE_MAP) == set(_WIRE_TO_PUBLIC)


def test_crosswalk_values_are_catalog_entries():
    catalogued = {id(entry) for entry in registry.PUBLIC_ERRORS}
    for wire, entry in _WIRE_TO_PUBLIC.items():
        assert id(entry) in catalogued, wire


def test_wire_code_never_leaves_the_catalogued_class():
    # A wire code may resolve to a more specific V2 class, but only a
    # subclass of the one its catalog entry names -- never a sibling.
    for wire, entry in _WIRE_TO_PUBLIC.items():
        catalogued = getattr(errors, entry.error_code)
        assert issubclass(_CODE_MAP[wire], catalogued), wire


def test_unrefined_wire_codes_map_to_exactly_the_catalogued_class():
    for wire, entry in _WIRE_TO_PUBLIC.items():
        if wire in _REFINED_CLASS:
            continue
        catalogued = getattr(errors, entry.error_code)
        assert _CODE_MAP[wire] is catalogued, wire


def test_content_filter_is_the_only_refinement():
    assert _REFINED_CLASS == {"DataInspectionFailed": ContentFilterError}
    assert issubclass(ContentFilterError, BadRequestError)


def test_previously_unmapped_wire_codes_now_resolve():
    # Both were declared in ErrorCode but missing from the old
    # hand-written code map, so InvalidParameter fell back to whatever
    # the HTTP status implied and Model.NotExist never became a 404.
    assert _CODE_MAP["InvalidParameter"] is BadRequestError
    assert _CODE_MAP["Model.NotExist"] is NotFoundError


def test_service_unavailable_and_gateway_timeout_are_distinguishable():
    assert _CODE_MAP["ServiceUnavailable"] is ServiceUnavailableError
    assert _CODE_MAP["GatewayTimeout"] is GatewayTimeoutError
    assert not issubclass(ServiceUnavailableError, GatewayTimeoutError)
    assert not issubclass(GatewayTimeoutError, ServiceUnavailableError)


def test_business_layer_resolves_wire_codes_without_a_status():
    # HTTP 200 with a business code has no status to fall back on, so
    # the crosswalk is the only thing that can pick a class there.
    err = error_from_business_code(
        message="no such model",
        code="Model.NotExist",
    )
    assert type(err) is NotFoundError
    assert err.status_code == 200
    assert err.code == "Model.NotExist"


# -- catalog text as the last-resort message ---------------------------


def _bare(status, headers=None):
    """A response with no body at all, so ``response.text`` is empty."""
    return httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.test.local/x"),
    )


def test_bodyless_failure_gets_the_catalog_text():
    with pytest.raises(ServiceUnavailableError) as ei:
        check_response(_bare(503))
    assert ei.value.message == (
        "The service is temporarily unavailable. Please try again later."
    )


def test_gateway_message_wins_over_catalog_text():
    resp = _resp(503, {"message": "upstream drained"})
    with pytest.raises(ServiceUnavailableError) as ei:
        check_response(resp)
    assert ei.value.message == "upstream drained"


def test_placeholders_survive_the_catalog_fallback():
    # Nothing in the SDK knows the resource name, and dropping the
    # placeholder would hide the field the catalog declares.
    with pytest.raises(NotFoundError) as ei:
        check_response(_bare(404))
    assert "{resource}" in ei.value.message


def test_ambiguous_status_gets_no_catalog_text():
    # 400 / 401 / 429 each carry several entries with different text, so
    # a status alone cannot pick one without inventing a reason.
    for status in (400, 401, 429):
        with pytest.raises(APIStatusError) as ei:
            check_response(_bare(status))
        assert ei.value.message == "", status
        assert ei.value.status_code == status


def test_wire_code_identifies_an_entry_on_an_ambiguous_status():
    err = errors.error_from_status(
        400,
        message="",
        body={"code": "InvalidParameter"},
    )
    assert type(err) is BadRequestError
    assert err.message == (
        "The request is invalid. Please check the request and try again."
    )


def test_uncatalogued_status_gets_no_catalog_text():
    with pytest.raises(APIStatusError) as ei:
        check_response(_bare(502))
    assert type(ei.value) is APIStatusError
    assert ei.value.message == ""


def test_business_layer_fills_text_from_the_code_alone():
    err = error_from_business_code(message="", code="Throttling")
    assert type(err) is RateLimitError
    assert err.message == (
        "Rate limit exceeded. Please slow down your requests."
    )
    assert err.status_code == 200


# -- SDK-side errors (catalog SDK_ERRORS, no HTTP response) ------------


class OSSUploadError(Exception):
    """Stand-in matching a catalogued 1.x class name."""


class NotInCatalog(Exception):
    """Stand-in with no catalog entry under any family."""


def test_sdk_error_has_no_http_semantics():
    assert issubclass(SdkError, DashScopeError)
    assert not issubclass(SdkError, APIStatusError)
    assert SdkError("boom").status_code is None


def test_sdk_error_translation_resolves_the_catalog_by_class_name():
    err = error_from_sdk_exception(
        OSSUploadError("bucket denied"),
        family="sdk.agentic_rl",
    )
    entry = _SDK_ERRORS["sdk.agentic_rl.OSSUploadError"]
    assert err.code == entry.name
    assert err.solution == entry.solution
    assert err.allow_retry == entry.allow_retry


def test_sdk_error_translation_keeps_the_original_message():
    # The catalog message is generic ("File upload operation to OSS
    # failed."); what the subsystem actually said is more useful.
    err = error_from_sdk_exception(
        OSSUploadError("bucket denied"),
        family="sdk.agentic_rl",
    )
    entry = _SDK_ERRORS["sdk.agentic_rl.OSSUploadError"]
    assert err.message == "bucket denied"
    assert err.message != entry.message


def test_sdk_error_translation_falls_back_to_the_family_entry():
    err = error_from_sdk_exception(
        NotInCatalog("odd"),
        family="sdk.agentic_rl",
    )
    assert err.code == "sdk.agentic_rl.Error"
    assert err.message == "odd"


def test_sdk_error_translation_needs_a_family_fallback():
    # sdk.agentstudio.* has no .Error entry, so the two-tier lookup
    # cannot serve it -- which is why that mount needs its own scheme.
    with pytest.raises(KeyError) as ei:
        error_from_sdk_exception(
            NotInCatalog("odd"),
            family="sdk.agentstudio",
        )
    assert "sdk.agentstudio.Error" in str(ei.value)
