# -*- coding: utf-8 -*-
from typing import Any

# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import json

import pytest

import dashscope
from dashscope import AioDecisionModel, DecisionModel
from dashscope.common.error import (
    InputRequired,
    InvalidParameter,
    ModelRequired,
)
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer

RESPONSE_BODY = {
    "model": "decision-model-preview",
    "request_id": "7b986c65-b223-9341-b5f0-b988e27ecaac",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.88,
            "probabilities": {
                "billing": 0.94,
                "technical": 0.06,
            },
        },
        "escalate": {
            "type": "noul",
            "noul": 0.99,
        },
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
            "probabilities": {
                "0": 0,
                "1": 0.01,
                "2": 0.73,
                "3": 0.26,
            },
        },
    },
    "usage": {
        "input_tokens": 125,
    },
    "latency_ms": 52.9,
}

STATE = {
    "ticket_id": "T-1001",
    "content": "订单支付后超过 24 小时仍未到账，用户无法继续使用核心服务，要求立即处理。",
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "department": {
        "type": "choice",
        "instructions": "应该由哪个团队处理？",
        "criteria": {
            "billing": "支付、退款和账单问题",
            "technical": "产品故障和集成问题",
        },
    },
    "escalate": {
        "type": "noul",
        "instructions": "是否需要立即通知值班人员？",
    },
    "severity": {
        "type": "score",
        "instructions": "这个问题有多严重？",
        "criteria": [
            "轻微问题，不影响功能",
            "部分功能受影响，但存在替代方案",
            "核心功能不可用，没有替代方案",
            "造成严重业务或安全影响",
        ],
    },
}


class TestDecisionModel(MockServerBase):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.origin_base_compatible_api_url = dashscope.base_compatible_api_url
        dashscope.base_compatible_api_url = (
            "http://localhost:8089/compatible-mode/v1"
        )

    @classmethod
    def teardown_class(cls):
        super().teardown_class()
        dashscope.base_compatible_api_url = cls.origin_base_compatible_api_url

    def test_call(self, mock_server: MockServer):
        mock_server.responses.put(json.dumps(RESPONSE_BODY))
        response = DecisionModel.call(
            model=DecisionModel.Models.decision_model_preview,
            state=STATE,
            questions=QUESTIONS,
        )
        req = mock_server.requests.get(block=True)
        assert req["path"] == "/compatible-mode/v1/systemone"
        assert req["body"] == {
            "model": "decision-model-preview",
            "state": STATE,
            "questions": QUESTIONS,
        }
        assert response.status_code == 200
        assert response.request_id == RESPONSE_BODY["request_id"]
        assert response.model == "decision-model-preview"
        assert response.usage is not None
        assert response.usage.input_tokens == 125
        assert response.latency_ms == 52.9

        assert response.answers is not None
        department = response.answers["department"]
        assert department.type == "choice"
        assert department.choice == "billing"
        assert department.confidence == 0.88
        assert department.probabilities == {
            "billing": 0.94,
            "technical": 0.06,
        }

        escalate = response.answers["escalate"]
        assert escalate.type == "noul"
        assert escalate.noul == 0.99

        severity = response.answers["severity"]
        assert severity.type == "score"
        assert severity.score == 2.25
        assert severity.confidence == 0.91
        assert severity.legend is not None
        assert severity.legend["2"] == "核心功能不可用，没有替代方案"
        assert severity.probabilities is not None
        assert severity.probabilities["2"] == 0.73

    def test_call_with_text_state(self, mock_server: MockServer):
        mock_server.responses.put(json.dumps(RESPONSE_BODY))
        state = "订单支付后超过 24 小时仍未到账"
        response = DecisionModel.call(
            model="decision-model-preview",
            state=state,
            questions=QUESTIONS,
        )
        req = mock_server.requests.get(block=True)
        assert req["body"]["state"] == state
        assert response.status_code == 200

    def test_call_error_response(self, mock_server: MockServer):
        error_body = {
            "status_code": 400,
            "code": "InvalidParameter",
            "message": "questions is invalid",
            "request_id": "4cc9f235-6fe0-9d62-9b83-2e1f5e06f6b0",
        }
        mock_server.responses.put(json.dumps(error_body))
        response = DecisionModel.call(
            model="decision-model-preview",
            state=STATE,
            questions=QUESTIONS,
        )
        mock_server.requests.get(block=True)
        assert response.status_code == 400
        assert response.code == "InvalidParameter"
        assert response.message == "questions is invalid"
        assert response.answers is None

    def test_aio_call(self, mock_server: MockServer):
        mock_server.responses.put(json.dumps(RESPONSE_BODY))

        response = asyncio.run(
            AioDecisionModel.call(
                model=AioDecisionModel.Models.decision_model_preview,
                state=STATE,
                questions=QUESTIONS,
            ),
        )

        req = mock_server.requests.get(block=True)
        assert req["path"] == "/compatible-mode/v1/systemone"
        assert req["body"] == {
            "model": "decision-model-preview",
            "state": STATE,
            "questions": QUESTIONS,
        }
        assert response.status_code == 200
        assert response.request_id == RESPONSE_BODY["request_id"]
        assert response.usage is not None
        assert response.usage.input_tokens == 125
        assert response.answers is not None
        assert response.answers["department"].choice == "billing"
        assert response.answers["escalate"].noul == 0.99
        assert response.answers["severity"].score == 2.25

    def test_stream_not_supported(self):
        with pytest.raises(InvalidParameter):
            DecisionModel.call(
                model="decision-model-preview",
                state=STATE,
                questions=QUESTIONS,
                stream=True,
            )
        with pytest.raises(InvalidParameter):
            asyncio.run(
                AioDecisionModel.call(
                    model="decision-model-preview",
                    state=STATE,
                    questions=QUESTIONS,
                    stream=True,
                ),
            )

    def test_model_required(self):
        with pytest.raises(ModelRequired):
            DecisionModel.call(
                model="",
                state=STATE,
                questions=QUESTIONS,
            )

    def test_state_required(self):
        with pytest.raises(InputRequired):
            DecisionModel.call(
                model="decision-model-preview",
                state=None,
                questions=QUESTIONS,
            )

    def test_questions_required(self):
        with pytest.raises(InputRequired):
            DecisionModel.call(
                model="decision-model-preview",
                state=STATE,
                questions=None,
            )
        with pytest.raises(InputRequired):
            DecisionModel.call(
                model="decision-model-preview",
                state=STATE,
                questions={},
            )
