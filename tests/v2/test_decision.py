# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.decision (TypeSafe System One decision model).

The System One endpoint is pinned to the compatible-mode path on every
gateway, independent of the client's chat ``protocol`` -- the routing
tests below pin that for native / openai / anthropic alike.
"""

# Mock handlers intentionally ignore the request argument; pytest
# fixture injection looks like unused/redefined names to pylint.
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import BadRequestError, RateLimitError
from dashscope.v2.resources.decision import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)
from dashscope.v2.transport._ident import sdk_client_value

BASE_URL = "https://api.test.local/api/v1"
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_SYSTEMONE = "/compatible-mode/v1/systemone"

_STATE = {
    "ticket_id": "T-1001",
    "content": "订单支付后超过 24 小时仍未到账，用户无法继续使用核心服务。",
}

# The response example from the decision-model API doc.
_DECISION_BODY = {
    "model": "decision-model-preview",
    "request_id": "7b986c65-b223-9341-b5f0-b988e27ecaac",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.88,
            "probabilities": {"billing": 0.94, "technical": 0.06},
        },
        "escalate": {"type": "noul", "noul": 0.99},
        "severity": {
            "type": "score",
            "score": 2.25,
            "confidence": 0.91,
            "legend": {
                "0": "轻微问题，不影响功能",
                "1": "部分功能受影响，但存在替代方案",
                "2": "核心功能不可用，没有替代方案",
                "3": "造成严重业务或安全影响",
            },
            "probabilities": {"0": 0.0, "1": 0.01, "2": 0.73, "3": 0.26},
        },
    },
    "usage": {"input_tokens": 125},
    "latency_ms": 52.9,
}


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


@pytest.fixture
def clean_env(monkeypatch):
    from dashscope.v2._config import settings

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_REGION",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


def _questions():
    return {
        "department": ChoiceQuestion(
            instructions="应该由哪个团队处理？",
            criteria={
                "billing": "支付、退款和账单问题",
                "technical": "产品故障和集成问题",
            },
        ),
        "escalate": NoulQuestion(instructions="是否需要立即通知值班人员？"),
        "severity": ScoreQuestion(
            instructions="这个问题有多严重？",
            criteria=[
                "轻微问题，不影响功能",
                "部分功能受影响，但存在替代方案",
                "核心功能不可用，没有替代方案",
                "造成严重业务或安全影响",
            ],
        ),
    }


def test_create_builds_wire_body():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_DECISION_BODY)

    client = _make_client(handler)
    client.decision.create(
        model="decision-model-preview",
        state=_STATE,
        questions=_questions(),
    )
    # Explicit base_url stays the service root (same rule as chat).
    assert captured["path"] == "/api/v1/systemone"
    body = captured["body"]
    assert body["model"] == "decision-model-preview"
    assert body["state"] == _STATE
    department = body["questions"]["department"]
    assert department["type"] == "choice"
    assert department["criteria"]["billing"] == "支付、退款和账单问题"
    escalate = body["questions"]["escalate"]
    # An unset optional (noul criteria) stays off the wire.
    assert escalate == {
        "type": "noul",
        "instructions": "是否需要立即通知值班人员？",
    }
    severity = body["questions"]["severity"]
    assert severity["type"] == "score"
    assert len(severity["criteria"]) == 4
    client.close()


def test_create_accepts_string_state_and_raw_dict_questions():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_DECISION_BODY)

    client = _make_client(handler)
    client.decision.create(
        model="decision-model-preview",
        state="这段评论是否在索要退款？",
        questions={
            "refund": {
                "type": "noul",
                "instructions": "是否在索要退款？",
                # Unknown fields pass through for forward compatibility.
                "future_field": 1,
            },
        },
    )
    assert captured["body"]["state"] == "这段评论是否在索要退款？"
    assert captured["body"]["questions"]["refund"] == {
        "type": "noul",
        "instructions": "是否在索要退款？",
        "future_field": 1,
    }
    client.close()


def test_response_parsing():
    def handler(request):
        return httpx.Response(200, json=_DECISION_BODY)

    client = _make_client(handler)
    resp = client.decision.create(
        model="decision-model-preview",
        state=_STATE,
        questions=_questions(),
    )
    assert resp.request_id == "7b986c65-b223-9341-b5f0-b988e27ecaac"
    assert resp.model == "decision-model-preview"
    assert resp.is_error is False
    department = resp.answers["department"]
    assert department.type == "choice"
    assert department.choice == "billing"
    assert department.confidence == 0.88
    assert department.probabilities == {"billing": 0.94, "technical": 0.06}
    assert resp.answers["escalate"].noul == 0.99
    severity = resp.answers["severity"]
    assert severity.score == 2.25
    assert severity.legend["2"] == "核心功能不可用，没有替代方案"
    assert severity.probabilities["3"] == 0.26
    assert resp.usage.input_tokens == 125
    assert resp.latency_ms == 52.9
    client.close()


def test_response_headers_attached():
    def handler(request):
        return httpx.Response(
            200,
            json=_DECISION_BODY,
            headers={"x-request-id": "rid-1"},
        )

    client = _make_client(handler)
    resp = client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert resp.headers["x-request-id"] == "rid-1"
    client.close()


def test_create_requires_questions():
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="questions must not be empty"):
        client.decision.create(
            model="decision-model-preview",
            state="s",
            questions={},
        )
    client.close()


def test_question_model_validation():
    # choice needs at least one option, score at least two levels.
    with pytest.raises(ValidationError):
        ChoiceQuestion()
    with pytest.raises(ValidationError):
        ChoiceQuestion(criteria={})
    with pytest.raises(ValidationError):
        ScoreQuestion(criteria=["only one level"])
    # A bare noul question is valid (criteria is optional).
    assert NoulQuestion().type == "noul"


@pytest.mark.parametrize("protocol", ["native", "openai", "anthropic"])
def test_routing_default_gateway_ignores_chat_protocol(clean_env, protocol):
    urls = []
    http_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: (
                urls.append(str(request.url)),
                httpx.Response(200, json=_DECISION_BODY),
            )[1],
        ),
        base_url=_WS_HOST + "/api/v1",
    )
    client = Client(
        api_key="sk-test",
        workspace="ws-123",
        protocol=protocol,
        http_client=http_client,
    )
    client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert urls == [_WS_HOST + _SYSTEMONE], protocol
    client.close()


@pytest.mark.parametrize(
    "region",
    ["cn-beijing", "ap-southeast-1"],
)
def test_routing_documented_region_gateways(clean_env, region):
    """Both endpoints the API doc lists (Beijing / Singapore) resolve."""
    urls = []
    ws_host = f"https://ws-123.{region}.maas.aliyuncs.com"

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_DECISION_BODY)

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=ws_host + "/api/v1",
    )
    client = Client(
        api_key="sk-test",
        workspace="ws-123",
        region=region,
        http_client=http_client,
    )
    client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert urls == [ws_host + _SYSTEMONE]
    client.close()


def test_routing_explicit_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_DECISION_BODY)

    client = _make_client(handler)
    client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert urls == [BASE_URL + "/systemone"]
    client.close()


def test_routing_legacy_fallback(clean_env, tmp_path):
    legacy = "https://dashscope.aliyuncs.com"
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_DECISION_BODY)

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=legacy + "/api/v1",
    )
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        client = Client(
            api_key="sk-test",
            http_client=http_client,
            config_path=tmp_path / "absent.toml",
        )
    client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert urls == [legacy + _SYSTEMONE]
    client.close()


def test_http_error_maps_openai_style_body():
    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "InvalidParameter",
                    "message": "questions.q.criteria is required",
                },
            },
            headers={"x-request-id": "rid-err"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as exc_info:
        client.decision.create(
            model="decision-model-preview",
            state="s",
            questions={"q": {"type": "choice"}},
        )
    err = exc_info.value
    assert err.code == "InvalidParameter"
    assert err.status_code == 400
    assert err.request_id == "rid-err"
    client.close()


def test_http_429_maps_rate_limit():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    client = _make_client(handler)
    with pytest.raises(RateLimitError):
        client.decision.create(
            model="decision-model-preview",
            state="s",
            questions={"q": {"type": "noul"}},
        )
    client.close()


def test_http_error_envelope_when_not_raising():
    """raise_on_error=False must not mark a non-2xx reply as success.

    The compatible-mode endpoint nests the envelope under ``error``,
    so the success parser sees none of it; the response must still
    carry is_error/code/message/request_id/status_code.
    """

    def handler(request):
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": "Model.NotExist",
                    "message": "model decision-model-typo does not exist",
                },
            },
            headers={"x-request-id": "rid-404"},
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.decision.create(
        model="decision-model-typo",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert resp.is_error is True
    assert resp.status_code == 404
    assert resp.code == "Model.NotExist"
    assert resp.message == "model decision-model-typo does not exist"
    assert resp.request_id == "rid-404"
    assert resp.answers == {}
    assert resp.headers["x-request-id"] == "rid-404"
    client.close()


def test_business_error_layer_raise_and_report():
    body = {
        "code": "InvalidParameter",
        "message": "state too long",
        "request_id": "rid-biz",
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as exc_info:
        client.decision.create(
            model="decision-model-preview",
            state="s",
            questions={"q": {"type": "noul"}},
        )
    assert exc_info.value.status_code == 200
    client.close()

    # raise_on_error=False fills the envelope instead of raising.
    client = _make_client(handler, raise_on_error=False)
    resp = client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert resp.is_error is True
    assert resp.code == "InvalidParameter"
    assert resp.status_code == 200
    assert resp.answers == {}
    client.close()


def test_sdk_module_header():
    seen = {}

    def handler(request):
        seen["client"] = request.headers.get("x-dashscope-sdk-client")
        return httpx.Response(200, json=_DECISION_BODY)

    client = _make_client(handler)
    client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert seen["client"] == sdk_client_value("decision")
    assert seen["client"].endswith("/decision2")
    client.close()


async def test_async_create():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return httpx.Response(200, json=_DECISION_BODY)

    client = _make_async_client(handler)
    resp = await client.decision.create(
        model="decision-model-preview",
        state=_STATE,
        questions=_questions(),
    )
    assert captured["path"] == "/api/v1/systemone"
    assert resp.answers["department"].choice == "billing"
    assert resp.answers["escalate"].noul == 0.99
    await client.aclose()


async def test_async_http_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": "Model.NotExist",
                    "message": "model decision-model-typo does not exist",
                },
            },
            headers={"x-request-id": "rid-404"},
        )

    client = _make_async_client(handler, raise_on_error=False)
    resp = await client.decision.create(
        model="decision-model-typo",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert resp.is_error is True
    assert resp.status_code == 404
    assert resp.code == "Model.NotExist"
    assert resp.message == "model decision-model-typo does not exist"
    assert resp.request_id == "rid-404"
    assert resp.answers == {}
    await client.aclose()


async def test_async_routing_default_gateway(clean_env):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_DECISION_BODY)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = AsyncClient(
        api_key="sk-test",
        workspace="ws-123",
        http_client=http_client,
    )
    await client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert urls == [_WS_HOST + _SYSTEMONE]
    await client.aclose()
