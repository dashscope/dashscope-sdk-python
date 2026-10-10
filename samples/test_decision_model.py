# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import os

from dashscope import AioDecisionModel, DecisionModel

STATE = {
    "ticket_id": "T-1001",
    "content": "订单支付后超过 24 小时仍未到账，用户无法继续使用核心服务，要求立即处理。",
}

QUESTIONS = {
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


def test_decision_model():
    response = DecisionModel.call(
        model=os.getenv("MODEL_NAME", DecisionModel.Models.decision_model_preview),
        state=STATE,
        questions=QUESTIONS,
    )

    if response.status_code == 200:
        for question_id, answer in response.answers.items():
            print(f"{question_id}: {answer}")
    else:
        print(f"request_id: {response.request_id}")
        print(f"code: {response.code}")
        print(f"message: {response.message}")


async def test_aio_decision_model():
    response = await AioDecisionModel.call(
        model=os.getenv("MODEL_NAME", AioDecisionModel.Models.decision_model_preview),
        state=STATE,
        questions=QUESTIONS,
    )
    print(f"response:\n{response}")


if __name__ == "__main__":
    test_decision_model()
    asyncio.run(test_aio_decision_model())
