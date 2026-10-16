# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 decision-model (System One) example.

One ``client.decision.create`` call answers a set of typed questions
about a business ``state`` in a single forward pass -- multi-choice
(``choice``), yes/no (``noul``) and ordinal scale (``score``) -- each
answer carrying its probability distribution. The model generates no
text, so latency and cost do not grow with output length. Typical
uses: ticket routing, content moderation, agent routing.

The endpoint sits under the compatible-mode path on every gateway, so
it does not move with the client's chat ``protocol`` setting.
"""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "decision-model-preview")

# The business state under decision: free text, a dialogue, or a
# structured object like this support ticket.
STATE = {
    "ticket_id": "T-1001",
    "content": "订单支付后超过 24 小时仍未到账，用户无法继续使用核心服务。",
}

if __name__ == "__main__":
    client = dashscope.Client()

    print("----- decision request -----")
    # Question values may also be the typed models ChoiceQuestion /
    # NoulQuestion / ScoreQuestion from dashscope.v2.resources.decision;
    # raw dicts below show the exact wire shape (and pass future fields
    # through untouched).
    resp = client.decision.create(
        model=MODEL,
        state=STATE,
        questions={
            # choice: pick one of the criteria options (1-255 options;
            # give the full set and add an "other" fallback).
            "department": {
                "type": "choice",
                "instructions": "应该由哪个团队处理？",
                "criteria": {
                    "billing": "支付、退款和账单问题",
                    "technical": "产品故障和集成问题",
                    "other": "不属于以上两类的问题",
                },
            },
            # noul: yes/no judgement; the answer is P(yes), 0=no, 1=yes.
            "escalate": {
                "type": "noul",
                "instructions": "是否需要立即通知值班人员？",
            },
            # score: ordinal scale described low to high (2-255 levels;
            # 3-7 clearly distinguishable levels work best).
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
        },
    )

    # ``answers`` keys mirror the request's question ids.
    department = resp.answers["department"]
    print(
        f"department={department.choice}, "
        f"confidence={department.confidence}, "
        f"probabilities={department.probabilities}",
    )

    escalate = resp.answers["escalate"]
    print(f"escalate P(yes)={escalate.noul}")

    severity = resp.answers["severity"]
    # score is the probability-weighted level index and may fall
    # between two levels; legend maps an index back to its text.
    print(
        f"severity={severity.score}, "
        f"confidence={severity.confidence}, "
        f"legend={severity.legend}",
    )

    print(
        f"input_tokens={resp.usage.input_tokens}, "
        f"latency_ms={resp.latency_ms}",
    )


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Ticket triage: 04_real_world_scenarios/ticket_triage_harness.py
#   (threshold policy + fail-closed degradation around the call)
# -------------------------------------------------------------
