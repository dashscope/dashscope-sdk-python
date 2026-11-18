# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for PTU throughput reservation (milestone D3).

Covers ``client.deployments`` PTU create/scale/renew/overflow plus the
``capacity_instances`` and ``capacity_operations`` subresources: wire
bodies, billing safeguards (max_retries=0 default, x-acs-req-uuid
idempotency), the HTTP-200 operation_status=FAILED check, the
operation_request_id collision, camelCase instance pagination, and
sync/async symmetry. Model names are public/arbitrary, never internal.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import DashScopeError
from dashscope.resources.deployments import (
    CapacityInstance,
    CapacityInstanceList,
    CapacityOperation,
    Deployment,
    PrePaidInfo,
    PtuCapacity,
)

BASE_URL = "https://api.test.local/api/v1"

_PTU_CAPACITY = {"input_tpm": 100, "output_tpm": 50}
_PRE_PAID = {"pricing_cycle": "Day", "duration": 1, "auto_renewal": False}


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def _deployment_output(**extra):
    output = {
        "deployed_model": "pm-abc123",
        "model_name": "qwen-plus",
        "plan": "ptu",
        "status": "RUNNING",
        "ptu_service_tier": "ptu_fast",
        "ptu_capacity": _PTU_CAPACITY,
        "charge_type": "pre_paid",
        "overflow_strategy": "disable",
        "operation_id": "op-1",
        "instance_id": "ci-1",
    }
    output.update(extra)
    return {"request_id": "req-dep", "output": output}


def _operation_output(status="PROCESSING", **extra):
    output = {
        "operation_id": "op-1",
        "request_id": "req-original-write",
        "operation_type": "CREATE",
        "operation_status": status,
        "model_service_id": "pm-abc123",
        "instance_id": "ci-1",
    }
    output.update(extra)
    return {"request_id": "req-poll", "output": output}


def _instance_output(**extra):
    output = {
        "model_service_id": "pm-abc123",
        "instance_id": "ci-1",
        "charge_type": "pre_paid",
        "status": "RUNNING",
        "deleted": False,
        "effective_capacity": _PTU_CAPACITY,
        "can_scale": True,
        "can_renew": True,
        "can_delete": False,
    }
    output.update(extra)
    return {"request_id": "req-inst", "output": output}


# --- PTU create ----------------------------------------------------------


def test_ptu_create_wire_body_and_no_idempotency():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        return httpx.Response(200, json=_deployment_output())

    client = _make_client(handler)
    created = client.deployments.create(
        model="qwen-plus",
        plan="ptu",
        charge_type="pre_paid",
        ptu_capacity=_PTU_CAPACITY,
        pre_paid_info=_PRE_PAID,
        service_tier="ptu_fast",
        name="my-ptu",
        suffix="alpha",
    )
    body = captured["body"]
    assert body["model_name"] == "qwen-plus"
    assert body["plan"] == "ptu"
    assert body["charge_type"] == "pre_paid"
    assert body["ptu_capacity"] == _PTU_CAPACITY
    assert body["pre_paid_info"] == _PRE_PAID
    assert body["service_tier"] == "ptu_fast"
    assert body["name"] == "my-ptu"
    assert body["suffix"] == "alpha"
    assert "capacity" not in body
    # First ModelCode creation must NOT carry an instance idempotency key.
    assert "x-acs-req-uuid" not in captured["headers"]
    assert "x-dashscope-requestid" not in captured["headers"]
    assert isinstance(created, Deployment)
    assert created.plan == "ptu"
    assert created.ptu_capacity.input_tpm == 100
    assert created.ptu_capacity.output_tpm == 50
    assert created.overflow_strategy == "disable"
    assert created.operation_id == "op-1"
    assert created.instance_id == "ci-1"
    client.close()


def test_ptu_create_accepts_typed_models():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_deployment_output())

    client = _make_client(handler)
    client.deployments.create(
        model="qwen-plus",
        plan="ptu",
        charge_type="post_paid",
        ptu_capacity=PtuCapacity(input_tpm=100, output_tpm=50),
        pre_paid_info=PrePaidInfo(pricing_cycle="Day", duration=1),
    )
    assert captured["body"]["ptu_capacity"] == _PTU_CAPACITY
    assert captured["body"]["pre_paid_info"]["duration"] == 1
    client.close()


def test_ptu_create_defaults_to_no_retry():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, json={"message": "busy"})

    client = _make_client(handler)
    with pytest.raises(DashScopeError):
        client.deployments.create(
            model="qwen-plus",
            plan="ptu",
            charge_type="post_paid",
            ptu_capacity=_PTU_CAPACITY,
        )
    assert len(attempts) == 1
    client.close()


def test_ptu_create_explicit_max_retries_retries(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(200, json=_deployment_output())

    from dashscope.transport import http as transport_http

    monkeypatch.setattr(transport_http, "_sleep", lambda seconds: None)
    client = _make_client(handler)
    created = client.deployments.create(
        model="qwen-plus",
        plan="ptu",
        charge_type="post_paid",
        ptu_capacity=_PTU_CAPACITY,
        max_retries=2,
    )
    assert created.deployed_model == "pm-abc123"
    assert len(attempts) == 3
    client.close()


def test_mu_create_keeps_default_retry(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, json={"message": "busy"})

    from dashscope.transport import http as transport_http

    monkeypatch.setattr(transport_http, "_sleep", lambda seconds: None)
    client = _make_client(handler)
    with pytest.raises(DashScopeError):
        client.deployments.create(model="qwen-plus", capacity=2, plan="mu")
    # The purchase safeguard is PTU-specific; MU create still retries.
    assert len(attempts) > 1
    client.close()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"plan": "ptu", "ptu_capacity": _PTU_CAPACITY}, "charge_type"),
        ({"plan": "ptu", "charge_type": "post_paid"}, "ptu_capacity"),
        (
            {
                "plan": "ptu",
                "charge_type": "pre_paid",
                "ptu_capacity": _PTU_CAPACITY,
            },
            "pre_paid_info",
        ),
        (
            {
                "plan": "ptu",
                "charge_type": "post_paid",
                "ptu_capacity": {"input_tpm": 100},
            },
            "output_tpm",
        ),
        (
            {
                "plan": "ptu",
                "charge_type": "post_paid",
                "ptu_capacity": {"input_tpm": -1, "output_tpm": 5},
            },
            "non-negative",
        ),
    ],
)
def test_ptu_create_validation(kwargs, match):
    client = _make_client(
        lambda r: httpx.Response(200, json=_deployment_output()),
    )
    with pytest.raises(ValueError, match=match):
        client.deployments.create(model="qwen-plus", **kwargs)
    client.close()


# --- PTU scale -----------------------------------------------------------


def test_ptu_scale_body_idempotency_and_no_retry():
    captured = {}
    attempts = []

    def handler(request):
        attempts.append(1)
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        captured["path"] = request.url.path
        return httpx.Response(200, json=_deployment_output())

    client = _make_client(handler)
    client.deployments.scale(
        "pm-abc123",
        ptu_capacity={"input_tpm": 200, "output_tpm": 100},
        instance_id="ci-1",
        order_type="UPGRADE",
    )
    assert captured["path"] == "/api/v1/deployments/pm-abc123/scale"
    assert captured["body"] == {
        "deployed_model": "pm-abc123",
        "ptu_capacity": {"input_tpm": 200, "output_tpm": 100},
        "instance_id": "ci-1",
        "order_type": "UPGRADE",
    }
    key = captured["headers"]["x-acs-req-uuid"]
    assert key
    assert captured["headers"]["x-dashscope-requestid"] == key
    assert len(attempts) == 1
    client.close()


def test_mu_scale_has_no_idempotency_key():
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_deployment_output())

    client = _make_client(handler)
    client.deployments.scale("pm-abc123", capacity=4)
    assert captured["body"] == {"deployed_model": "pm-abc123", "capacity": 4}
    assert "x-acs-req-uuid" not in captured["headers"]
    client.close()


def test_scale_requires_capacity_or_ptu():
    client = _make_client(
        lambda r: httpx.Response(200, json=_deployment_output()),
    )
    with pytest.raises(ValueError, match="capacity or ptu_capacity"):
        client.deployments.scale("pm-abc123")
    with pytest.raises(ValueError, match="order_type"):
        client.deployments.scale(
            "pm-abc123",
            capacity=2,
            order_type="SIDEWAYS",
        )
    client.close()


def test_scale_reuses_caller_idempotency_key():
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        return httpx.Response(200, json=_deployment_output())

    client = _make_client(handler)
    client.deployments.scale(
        "pm-abc123",
        ptu_capacity=_PTU_CAPACITY,
        extra_headers={"x-acs-req-uuid": "my-key"},
    )
    assert captured["headers"]["x-acs-req-uuid"] == "my-key"
    assert captured["headers"]["x-dashscope-requestid"] == "my-key"
    client.close()


def _assert_scale_reuses_variant_key(variant):
    captured = {}

    def handler(request):
        captured["headers"] = request.headers
        captured["raw"] = request.headers.raw
        return httpx.Response(200, json=_deployment_output())

    client = _make_client(handler)
    client.deployments.scale(
        "pm-abc123",
        ptu_capacity=_PTU_CAPACITY,
        extra_headers={variant: "my-key"},
    )
    # One key on the wire, not a caller/SDK pair with conflicting values
    # (httpx joins duplicate header lines with ", ").
    assert captured["headers"]["x-acs-req-uuid"] == "my-key"
    assert captured["headers"]["x-dashscope-requestid"] == "my-key"
    names = [name.lower() for name, _ in captured["raw"]]
    assert names.count(b"x-acs-req-uuid") == 1
    assert names.count(b"x-dashscope-requestid") == 1
    client.close()


def test_scale_reuses_case_variant_idempotency_key():
    for variant in ("X-Acs-Req-Uuid", "x-dashscope-requestid"):
        _assert_scale_reuses_variant_key(variant)


# --- renew / overflow ----------------------------------------------------


def test_renew_body_and_safeguards():
    captured = {}
    attempts = []

    def handler(request):
        attempts.append(1)
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        captured["path"] = request.url.path
        return httpx.Response(200, json=_deployment_output())

    client = _make_client(handler)
    client.deployments.renew(
        "pm-abc123",
        pre_paid_info=_PRE_PAID,
        instance_id="ci-1",
        is_change=True,
        ptu_capacity=_PTU_CAPACITY,
    )
    assert captured["path"] == "/api/v1/deployments/pm-abc123/renew"
    assert captured["body"] == {
        "pre_paid_info": _PRE_PAID,
        "instance_id": "ci-1",
        "is_change": True,
        "ptu_capacity": _PTU_CAPACITY,
    }
    assert captured["headers"]["x-acs-req-uuid"]
    assert len(attempts) == 1
    client.close()


def test_renew_capacity_change_requires_is_change():
    client = _make_client(
        lambda r: httpx.Response(200, json=_deployment_output()),
    )
    with pytest.raises(ValueError, match="is_change"):
        client.deployments.renew(
            "pm-abc123",
            pre_paid_info=_PRE_PAID,
            ptu_capacity=_PTU_CAPACITY,
        )
    client.close()


def test_update_overflow_strategy():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        return httpx.Response(
            200,
            json=_deployment_output(overflow_strategy="enable"),
        )

    client = _make_client(handler)
    updated = client.deployments.update_overflow_strategy(
        "pm-abc123",
        overflow_strategy="enable",
    )
    assert captured["path"] == (
        "/api/v1/deployments/pm-abc123/update-overflowstrategy"
    )
    assert captured["body"] == {"overflow_strategy": "enable"}
    # Config toggle, not a purchase: no idempotency key forced.
    assert "x-acs-req-uuid" not in captured["headers"]
    assert updated.overflow_strategy == "enable"
    with pytest.raises(ValueError, match="overflow_strategy"):
        client.deployments.update_overflow_strategy(
            "pm-abc123",
            overflow_strategy="ENABLE",
        )
    client.close()


# --- capacity instances --------------------------------------------------


def test_instance_create_body_idempotency_and_operation():
    captured = {}
    attempts = []

    def handler(request):
        attempts.append(1)
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        captured["path"] = request.url.path
        return httpx.Response(200, json=_operation_output("SUCCEEDED"))

    client = _make_client(handler)
    op = client.deployments.capacity_instances.create(
        "pm-abc123",
        billing_method="PRE_PAY",
        ptu_capacity=_PTU_CAPACITY,
        pre_paid_info=_PRE_PAID,
    )
    assert captured["path"] == (
        "/api/v1/deployments/pm-abc123/capacity-instances"
    )
    assert captured["body"] == {
        "billing_method": "PRE_PAY",
        "ptu_capacity": _PTU_CAPACITY,
        "pre_paid_info": _PRE_PAID,
    }
    assert captured["headers"]["x-acs-req-uuid"]
    assert len(attempts) == 1
    assert isinstance(op, CapacityOperation)
    assert op.succeeded
    assert op.operation_id == "op-1"
    # Inner write request id preserved separately from the poll envelope.
    assert op.request_id == "req-poll"
    assert op.operation_request_id == "req-original-write"
    client.close()


def test_instance_create_postpaid_no_prepaid_ok():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_operation_output("SUCCEEDED"))

    client = _make_client(handler)
    client.deployments.capacity_instances.create(
        "pm-abc123",
        billing_method="POST_PAY",
        ptu_capacity=_PTU_CAPACITY,
    )
    assert "pre_paid_info" not in captured["body"]
    client.close()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"billing_method": "pre_pay", "ptu_capacity": _PTU_CAPACITY},
            "billing_method",
        ),
        (
            {"billing_method": "PRE_PAY", "ptu_capacity": _PTU_CAPACITY},
            "pre_paid_info",
        ),
        (
            {"billing_method": "POST_PAY", "ptu_capacity": {"input_tpm": 1}},
            "output_tpm",
        ),
    ],
)
def test_instance_create_validation(kwargs, match):
    client = _make_client(
        lambda r: httpx.Response(200, json=_operation_output()),
    )
    with pytest.raises(ValueError, match=match):
        client.deployments.capacity_instances.create("pm-abc123", **kwargs)
    client.close()


def test_instance_list_camelcase_pagination():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "request_id": "req-list",
                "output": {
                    "records": [_instance_output()["output"]],
                    "items": 7,
                    "page": 2,
                    "itemsPerPage": 20,
                    "pageCount": 1,
                },
            },
        )

    client = _make_client(handler)
    page = client.deployments.capacity_instances.list(
        "pm-abc123",
        page_no=2,
        page_size=20,
        statuses="RUNNING,STOPPED",
        charge_types="pre_paid,post_paid",
    )
    assert captured["path"] == (
        "/api/v1/deployments/pm-abc123/capacity-instances"
    )
    assert captured["params"] == {
        "page_no": "2",
        "page_size": "20",
        "include_deleted": "true",
        "statuses": "RUNNING,STOPPED",
        "charge_types": "pre_paid,post_paid",
    }
    assert isinstance(page, CapacityInstanceList)
    assert page.total_count == 7
    assert page.page == 2
    assert page.items_per_page == 20
    assert page.page_count == 1
    assert isinstance(page.records[0], CapacityInstance)
    assert page.records[0].instance_id == "ci-1"
    assert page.records[0].can_scale is True
    client.close()


def test_instance_list_snake_case_pagination():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-list",
                "output": {
                    "records": [_instance_output()["output"]],
                    "total_count": 7,
                    "page_no": 1,
                    "page_size": 1,
                    "page_count": 7,
                },
            },
        )

    client = _make_client(handler)
    page = client.deployments.capacity_instances.list(
        "pm-abc123",
        page_no=1,
        page_size=1,
    )
    # The snake_case totals must survive: falling back to len(records)/20
    # would report one full page and stop pagination early.
    assert page.total_count == 7
    assert page.page == 1
    assert page.items_per_page == 1
    assert page.page_count == 7
    client.close()


def test_instance_get():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return httpx.Response(200, json=_instance_output(can_delete=True))

    client = _make_client(handler)
    inst = client.deployments.capacity_instances.get("pm-abc123", "ci-1")
    assert captured["path"] == (
        "/api/v1/deployments/pm-abc123/capacity-instances/ci-1"
    )
    assert inst.model_service_id == "pm-abc123"
    assert inst.effective_capacity.input_tpm == 100
    assert inst.can_delete is True
    client.close()


def test_instance_scale_renew_delete_paths():
    seen = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body, request.headers))
        return httpx.Response(200, json=_operation_output("SUCCEEDED"))

    client = _make_client(handler)
    ci = client.deployments.capacity_instances
    ci.scale(
        "pm-abc123",
        "ci-1",
        ptu_capacity=_PTU_CAPACITY,
        order_type="DOWNGRADE",
    )
    ci.renew("pm-abc123", "ci-1", pre_paid_info=_PRE_PAID)
    ci.delete("pm-abc123", "ci-1", reason="scaling down")
    assert seen[0][0] == "PUT"
    assert seen[0][1] == (
        "/api/v1/deployments/pm-abc123/capacity-instances/ci-1/scale"
    )
    assert seen[0][2] == {
        "ptu_capacity": _PTU_CAPACITY,
        "order_type": "DOWNGRADE",
    }
    assert seen[1][1] == (
        "/api/v1/deployments/pm-abc123/capacity-instances/ci-1/renew"
    )
    assert seen[1][2] == {"pre_paid_info": _PRE_PAID}
    assert seen[2][0] == "DELETE"
    assert seen[2][1] == (
        "/api/v1/deployments/pm-abc123/capacity-instances/ci-1"
    )
    # Delete carries no JSON body; reason rides as a query parameter.
    assert seen[2][2] is None
    for _, _, _, headers in seen:
        assert headers["x-acs-req-uuid"]
    client.close()


def test_instance_delete_reason_query():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=_operation_output("SUCCEEDED"))

    client = _make_client(handler)
    client.deployments.capacity_instances.delete(
        "pm-abc123",
        "ci-1",
        reason="bye",
    )
    assert captured["params"] == {"reason": "bye"}
    client.close()


# --- operation failure / polling -----------------------------------------


def test_operation_failed_raises_catalog_error():
    def handler(request):
        return httpx.Response(
            200,
            json=_operation_output(
                "FAILED",
                error_code="CAPACITY_INSTANCE_REQUIRED",
                error_message="instance id required",
            ),
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError) as exc:
        client.deployments.capacity_instances.create(
            "pm-abc123",
            billing_method="POST_PAY",
            ptu_capacity=_PTU_CAPACITY,
        )
    assert exc.value.code == "CAPACITY_INSTANCE_REQUIRED"
    client.close()


def test_operation_failed_compat_returns_envelope():
    def handler(request):
        return httpx.Response(
            200,
            json=_operation_output(
                "FAILED",
                error_code="PREPAID_UNSUBSCRIBE_REQUIRED",
                error_message="unsubscribe first",
            ),
        )

    client = _make_client(handler, raise_on_error=False)
    op = client.deployments.capacity_instances.delete("pm-abc123", "ci-1")
    assert op.failed
    assert op.operation_status == "FAILED"
    assert op.error_code == "PREPAID_UNSUBSCRIBE_REQUIRED"
    client.close()


def test_operations_get_path_and_request_id_collision():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return httpx.Response(200, json=_operation_output("SUCCEEDED"))

    client = _make_client(handler)
    op = client.deployments.capacity_operations.get("pm-abc123", "op-1")
    assert captured["path"] == (
        "/api/v1/deployments/pm-abc123/capacity-operations/op-1"
    )
    assert op.request_id == "req-poll"
    assert op.operation_request_id == "req-original-write"
    assert op.finished
    client.close()


def test_operations_get_404_maps_error():
    def handler(request):
        return httpx.Response(
            404,
            json={"code": "CAPACITY_OPERATION_NOT_FOUND", "message": "no op"},
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError) as exc:
        client.deployments.capacity_operations.get("pm-abc123", "999")
    assert exc.value.status_code == 404
    client.close()


def test_wait_polls_until_terminal():
    states = iter(["PROCESSING", "PROCESSING", "SUCCEEDED"])
    polls = []

    def handler(request):
        polls.append(1)
        return httpx.Response(200, json=_operation_output(next(states)))

    client = _make_client(handler)
    op = client.deployments.capacity_operations.wait(
        "pm-abc123",
        "op-1",
        _sleep=lambda seconds: None,
    )
    assert op.succeeded
    assert len(polls) == 3
    client.close()


def test_wait_failed_raises():
    states = iter(["PROCESSING", "FAILED"])

    def handler(request):
        return httpx.Response(
            200,
            json=_operation_output(
                next(states),
                error_code="InternalError",
                error_message="boom",
            ),
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError) as exc:
        client.deployments.capacity_operations.wait(
            "pm-abc123",
            "op-1",
            _sleep=lambda seconds: None,
        )
    assert exc.value.code == "InternalError"
    client.close()


def test_wait_timeout():
    def handler(request):
        return httpx.Response(200, json=_operation_output("PROCESSING"))

    client = _make_client(handler)
    with pytest.raises(DashScopeError, match="timed out"):
        client.deployments.capacity_operations.wait(
            "pm-abc123",
            "op-1",
            timeout=0,
            _sleep=lambda seconds: None,
        )
    client.close()


def test_wait_compat_returns_failed():
    def handler(request):
        return httpx.Response(
            200,
            json=_operation_output("FAILED", error_code="InternalError"),
        )

    client = _make_client(handler, raise_on_error=False)
    op = client.deployments.capacity_operations.wait(
        "pm-abc123",
        "op-1",
        _sleep=lambda seconds: None,
    )
    assert op.failed
    client.close()


# --- async symmetry ------------------------------------------------------


async def test_async_ptu_create_and_safeguard():
    captured = {}
    attempts = []

    def handler(request):
        attempts.append(1)
        captured["body"] = json.loads(request.content)
        captured["headers"] = request.headers
        return httpx.Response(200, json=_deployment_output())

    client = _make_async_client(handler)
    created = await client.deployments.create(
        model="qwen-plus",
        plan="ptu",
        charge_type="post_paid",
        ptu_capacity=_PTU_CAPACITY,
    )
    assert created.plan == "ptu"
    assert captured["body"]["ptu_capacity"] == _PTU_CAPACITY
    assert "x-acs-req-uuid" not in captured["headers"]
    assert len(attempts) == 1
    await client.aclose()


async def test_async_instance_create_operation_and_failure():
    def handler(request):
        return httpx.Response(
            200,
            json=_operation_output(
                "FAILED",
                error_code="POSTPAID_INSTANCE_ALREADY_EXISTS",
                error_message="exists",
            ),
        )

    client = _make_async_client(handler)
    with pytest.raises(DashScopeError) as exc:
        await client.deployments.capacity_instances.create(
            "pm-abc123",
            billing_method="POST_PAY",
            ptu_capacity=_PTU_CAPACITY,
        )
    assert exc.value.code == "POSTPAID_INSTANCE_ALREADY_EXISTS"
    await client.aclose()


async def test_async_instance_list_camelcase():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-list",
                "output": {
                    "records": [_instance_output()["output"]],
                    "items": 3,
                    "page": 1,
                    "itemsPerPage": 20,
                    "pageCount": 1,
                },
            },
        )

    client = _make_async_client(handler)
    page = await client.deployments.capacity_instances.list("pm-abc123")
    assert page.total_count == 3
    assert page.records[0].instance_id == "ci-1"
    await client.aclose()


async def test_async_renew_and_overflow():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json=_deployment_output())

    client = _make_async_client(handler)
    await client.deployments.renew("pm-abc123", pre_paid_info=_PRE_PAID)
    await client.deployments.update_overflow_strategy(
        "pm-abc123",
        overflow_strategy="disable",
    )
    assert seen == [
        ("PUT", "/api/v1/deployments/pm-abc123/renew"),
        ("PUT", "/api/v1/deployments/pm-abc123/update-overflowstrategy"),
    ]
    await client.aclose()


async def test_async_wait_polls_until_terminal():
    states = iter(["PROCESSING", "SUCCEEDED"])
    polls = []

    def handler(request):
        polls.append(1)
        return httpx.Response(200, json=_operation_output(next(states)))

    async def nosleep(seconds):
        return None

    client = _make_async_client(handler)
    op = await client.deployments.capacity_operations.wait(
        "pm-abc123",
        "op-1",
        _sleep=nosleep,
    )
    assert op.succeeded
    assert len(polls) == 2
    await client.aclose()
