# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 real-world scenario: decision-model ticket triage.

This is not "how to call ``client.decision.create``". It is the
production harness a ticket system needs *around* that call:

1. **One decision round per ticket** -- the decision model
   (``decision-model-preview``) answers a batch of typed questions
   about the ticket in a single forward pass: which team owns it
   (``choice``), whether to page the on-call engineer (``noul``) and
   how severe it is (``score``). No text is generated, so latency and
   cost stay flat per ticket.
2. **A local threshold policy** -- the model returns probability
   distributions, never "what to do". Harness-owned cut lines (a
   confidence floor, an escalation threshold, severity bands) turn
   distributions into routing actions, so the policy is auditable and
   tunable without touching the model.
3. **Fail-closed degradation** -- low routing confidence sends the
   ticket to the HUMAN queue; a failed call -- an exception, or an
   ``is_error`` envelope, which can ride HTTP 200 -- degrades the
   ticket to the same queue with the error attached. No ticket is ever
   auto-routed on a guess.
4. **Per-ticket isolation + a token ledger** -- one ticket's backend
   failure never stops the batch, and ``usage.input_tokens`` (the only
   quantity the decision model bills) is booked per ticket.

It runs **fully offline and deterministically by default**: the mock
seam is the one-method decision backend, so ``MockDecisionBackend``
serves scripted answers and injects a fault on demand while the
policy, degradation and ledger logic under test stays identical. Set
``TRIAGE_HARNESS_LIVE=1`` to run the exact same harness against the
real gateway with ``DASHSCOPE_API_KEY``.
"""

import os

import dashscope
from dashscope.v2 import errors

# ---- configuration (override via env) ---- #
DECISION_MODEL = os.environ.get("DASHSCOPE_MODEL", "decision-model-preview")
LIVE = os.environ.get("TRIAGE_HARNESS_LIVE") == "1"

# Policy thresholds: business decisions owned by the harness, not by
# the model. Tune them from observed routing quality.
CONFIDENCE_FLOOR = 0.6  # below this, routing goes to a human
ESCALATE_THRESHOLD = 0.8  # P(yes) required to page the on-call engineer
P1_SCORE = 2.5  # probability-weighted severity index cut lines
P2_SCORE = 1.5
DECISION_TIMEOUT = 10.0  # per-call timeout in seconds (live mode)

TRIAGED, ESCALATED, HUMAN_REVIEW, DEGRADED = (
    "triaged",
    "escalated",
    "human_review",
    "degraded",
)

# One decision round asks every question at once; ``answers`` come back
# keyed by these same question ids. Raw dicts show the wire shape; the
# typed ChoiceQuestion / NoulQuestion / ScoreQuestion models in
# dashscope.v2.resources.decision add local validation if wanted.
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "应该由哪个团队处理这张工单？",
        "criteria": {
            "billing": "支付、退款和账单问题",
            "technical": "产品故障、报错和集成问题",
            "logistics": "发货、运输和签收问题",
            "other": "不属于以上任何一类的问题",
        },
    },
    "escalate": {
        "type": "noul",
        "instructions": "是否紧急到需要立即通知值班工程师？",
    },
    "severity": {
        "type": "score",
        "instructions": "这张工单有多严重？",
        "criteria": [
            "轻微问题，不影响功能使用",
            "部分功能受影响，但存在替代方案",
            "核心功能不可用，没有替代方案",
            "造成严重业务资损或安全影响",
        ],
    },
}

# A stand-in ticket intake queue; load from your ticket system in prod.
TICKETS = [
    {
        "ticket_id": "T-2001",
        "content": "订单支付后超过 24 小时仍未到账，用户无法继续使用核心服务。",
    },
    {
        "ticket_id": "T-2002",
        "content": "快递显示已签收但本人没有收到包裹，需要核实。",
    },
    {
        "ticket_id": "T-2003",
        "content": "你们这个系统到底怎么回事，给个说法。",
    },
    {
        "ticket_id": "T-2004",
        "content": "线上服务全面报错，所有用户无法登录，疑似严重故障。",
    },
    {
        "ticket_id": "T-2005",
        "content": "发票抬头开错了，需要重新开具。",
    },
]


# ===================== verdict & policy (local, deterministic) =========== #
class TicketVerdict:
    """Structured per-ticket outcome: what to do, why, and what it cost."""

    def __init__(
        self,
        ticket_id,
        state,
        department=None,
        priority=None,
        escalate=False,
        confidence=None,
        reason="",
        input_tokens=0,
        latency_ms=None,
        request_id=None,
        error_code=None,
    ):
        self.ticket_id = ticket_id
        self.state = state
        self.department = department
        self.priority = priority
        self.escalate = escalate
        self.confidence = confidence
        self.reason = reason
        self.input_tokens = input_tokens
        self.latency_ms = latency_ms
        self.request_id = request_id
        self.error_code = error_code


class TriagePolicy:
    """Turn answer distributions into a routing action.

    Every threshold here is a local business decision: the model
    supplies distributions, the policy owns the consequences.
    """

    def __init__(
        self,
        confidence_floor=CONFIDENCE_FLOOR,
        escalate_threshold=ESCALATE_THRESHOLD,
    ):
        self._floor = confidence_floor
        self._escalate = escalate_threshold

    @staticmethod
    def _priority(score):
        # ``score`` is the probability-WEIGHTED level index (0-based,
        # and it may fall between two levels), not an integer class.
        if score is None:
            return "P3"
        if score >= P1_SCORE:
            return "P1"
        if score >= P2_SCORE:
            return "P2"
        return "P3"

    def decide(self, ticket_id, response):
        answers = response.answers
        department = answers.get("department")
        escalate = answers.get("escalate")
        severity = answers.get("severity")
        dept_name = getattr(department, "choice", None)
        dept_conf = getattr(department, "confidence", None)
        p_yes = getattr(escalate, "noul", None) or 0.0
        usage = getattr(response, "usage", None)
        base = {
            "department": dept_name,
            "priority": self._priority(getattr(severity, "score", None)),
            "escalate": p_yes >= self._escalate,
            "confidence": dept_conf,
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "latency_ms": getattr(response, "latency_ms", None),
            "request_id": getattr(response, "request_id", None),
        }
        # Fail closed #1: uncertain routing is worse than no routing.
        if dept_conf is None or dept_conf < self._floor:
            return TicketVerdict(
                ticket_id,
                HUMAN_REVIEW,
                reason=f"路由置信度 {dept_conf} 低于阈值 {self._floor}",
                **base,
            )
        # Fail closed #2: a high P(yes) pages a human by design.
        if p_yes >= self._escalate:
            return TicketVerdict(
                ticket_id,
                ESCALATED,
                reason=f"P(需值班介入)={p_yes:.2f} 达到阈值 {self._escalate}",
                **base,
            )
        return TicketVerdict(
            ticket_id,
            TRIAGED,
            reason="置信度与阈值检查均通过",
            **base,
        )


# ===================== decision backends (the mock seam) ================= #
class DashScopeDecisionBackend:
    """Live backend: one ``client.decision.create`` round per ticket.

    Only constructed in LIVE mode, so the offline demo never touches
    the network. ``decision.create`` is non-streaming, so the SDK's own
    ``max_retries`` absorbs transient 429/5xx before this ever raises.
    Note the endpoint lives under the compatible-mode path on every
    gateway: the client's chat ``protocol`` setting does not move it.
    """

    def __init__(self, client):
        self._client = client

    def decide(self, *, model, state, questions, timeout):
        return self._client.decision.create(
            model=model,
            state=state,
            questions=questions,
            timeout=timeout,
        )


class MockAnswer:
    """Attribute-compatible stand-in for the SDK's DecisionAnswer."""

    def __init__(self):
        self.type = None
        self.choice = None
        self.noul = None
        self.score = None
        self.probabilities = None
        self.confidence = None
        self.legend = None


class MockDecisionResponse:
    """Attribute-compatible stand-in for client.decision's response."""

    def __init__(self, answers, input_tokens, latency_ms, request_id):
        self.answers = answers
        self.usage = type("Usage", (), {"input_tokens": input_tokens})()
        self.latency_ms = latency_ms
        self.request_id = request_id
        self.is_error = False
        self.code = None
        self.message = None
        self.status_code = 200


def _spread(winner, confidence, options):
    """Probabilities for a choice answer: winner + equal remainder."""
    others = [name for name in options if name != winner]
    share = round((1.0 - confidence) / len(others), 4) if others else 0.0
    probs = {name: share for name in others}
    probs[winner] = confidence
    return probs


def _mock_answers(spec, questions):
    dept_name, dept_conf = spec["department"]
    department = MockAnswer()
    department.type = "choice"
    department.choice = dept_name
    department.confidence = dept_conf
    department.probabilities = _spread(
        dept_name,
        dept_conf,
        questions["department"]["criteria"],
    )

    p_yes = spec["escalate"]
    escalate = MockAnswer()
    escalate.type = "noul"
    escalate.noul = p_yes
    escalate.probabilities = {"no": round(1.0 - p_yes, 4), "yes": p_yes}

    score_value, score_conf = spec["severity"]
    levels = questions["severity"]["criteria"]
    severity = MockAnswer()
    severity.type = "score"
    severity.score = score_value
    severity.confidence = score_conf
    severity.legend = {str(i): text for i, text in enumerate(levels)}
    center = int(round(score_value))
    severity.probabilities = {
        str(i): round(score_conf if i == center else 0.05, 4)
        for i in range(len(levels))
    }
    return {
        "department": department,
        "escalate": escalate,
        "severity": severity,
    }


# Scripted per-ticket outcomes: department (name, confidence), escalate
# P(yes), severity (weighted index, confidence). T-2003 is deliberately
# vague so its low confidence exercises the human-review path.
MOCK_SCRIPT = {
    "T-2001": {
        "department": ("billing", 0.93),
        "escalate": 0.06,
        "severity": (2.3, 0.81),
    },
    "T-2002": {
        "department": ("logistics", 0.88),
        "escalate": 0.12,
        "severity": (1.4, 0.77),
    },
    "T-2003": {
        "department": ("other", 0.34),
        "escalate": 0.08,
        "severity": (0.9, 0.41),
    },
    "T-2004": {
        "department": ("technical", 0.97),
        "escalate": 0.96,
        "severity": (2.9, 0.93),
    },
}

# Tickets whose decision call "fails" in the offline demo.
FAIL_TICKETS = {"T-2005"}


class MockDecisionBackend:
    """Offline stand-in for the decision resource.

    Serves scripted, attribute-compatible answers keyed by ticket id
    and raises a typed SDK error for tickets in ``fail_ids``, so the
    demo exercises the low-confidence and the degradation path without
    a network. Swap for :class:`DashScopeDecisionBackend` to go live.
    """

    def __init__(self, script, fail_ids=()):
        self._script = script
        self._fail_ids = set(fail_ids)

    def decide(self, *, model, state, questions, timeout):
        ticket_id = state.get("ticket_id", "?")
        if ticket_id in self._fail_ids:
            raise errors.APIConnectionError(
                f"mocked decision backend outage for {ticket_id}",
            )
        answers = _mock_answers(self._script[ticket_id], questions)
        return MockDecisionResponse(
            answers,
            input_tokens=40 + len(state.get("content", "")),
            latency_ms=min(37.5, timeout * 1000),
            request_id=f"mock-{model}-{ticket_id}",
        )


# ===================== the harness ===================== #
class TicketTriageHarness:
    """Batch triage driver: per-ticket isolation + a token ledger.

    A backend failure becomes a DEGRADED verdict for that ticket alone;
    the batch never stops. The decision model generates no text, so
    ``usage.input_tokens`` is the whole bill -- the ledger is the cost
    model, exact rather than estimated.
    """

    def __init__(
        self,
        backend,
        policy,
        model=DECISION_MODEL,
        timeout=DECISION_TIMEOUT,
    ):
        self._backend = backend
        self._policy = policy
        self._model = model
        self._timeout = timeout
        self.total_input_tokens = 0
        self.triaged_ok = 0

    def triage(self, ticket):
        """Triage one ticket; never raises for backend failures."""
        ticket_id = ticket["ticket_id"]
        try:
            response = self._backend.decide(
                model=self._model,
                state=ticket,
                questions=QUESTIONS,
                timeout=self._timeout,
            )
        except errors.DashScopeError as exc:
            return self._degrade(ticket_id, exc=exc)
        # Belt and suspenders: DashScope can report a failure with HTTP
        # 200, so under raise_on_error=False the failure arrives as an
        # envelope, not an exception. Never trust an is_error body.
        if getattr(response, "is_error", False):
            return self._degrade(ticket_id, envelope=response)
        verdict = self._policy.decide(ticket_id, response)
        self.total_input_tokens += verdict.input_tokens
        self.triaged_ok += 1
        return verdict

    def run(self, tickets):
        verdicts = []
        for ticket in tickets:
            verdict = self.triage(ticket)
            self._report(ticket, verdict)
            verdicts.append(verdict)
        return verdicts

    @staticmethod
    def _degrade(ticket_id, exc=None, envelope=None):
        if envelope is not None:
            code = getattr(envelope, "code", None)
            note = getattr(envelope, "message", None)
        else:
            code = getattr(exc, "code", None) or type(exc).__name__
            note = str(exc)
        return TicketVerdict(
            ticket_id,
            DEGRADED,
            reason=f"决策后端不可用（{note}），已转人工队列，不凭空猜测路由",
            error_code=code,
        )

    @staticmethod
    def _report(ticket, verdict):
        print(f"\n  工单 {ticket['ticket_id']}: {ticket['content'][:22]}...")
        print(
            f"    状态={verdict.state}, 团队={verdict.department}, "
            f"优先级={verdict.priority}, 值班介入={verdict.escalate}",
        )
        print(
            f"    置信度={verdict.confidence}, "
            f"input_tokens={verdict.input_tokens}, "
            f"request_id={verdict.request_id}",
        )
        print(f"    判定依据: {verdict.reason}")
        if verdict.error_code:
            print(f"    error_code: {verdict.error_code}")


# ===================== wiring & demo ===================== #
def build_harness():
    """Build the harness on a live or scripted decision backend."""
    if LIVE:
        client = dashscope.Client(max_retries=2)
        backend = DashScopeDecisionBackend(client)
    else:
        backend = MockDecisionBackend(MOCK_SCRIPT, fail_ids=FAIL_TICKETS)
    return TicketTriageHarness(backend, TriagePolicy())


def main():
    harness = build_harness()
    mode = "LIVE 真实网关" if LIVE else "OFFLINE 脚本化 Mock 后端"
    print(f"模式: {mode}，模型: {DECISION_MODEL}")
    print(
        f"策略阈值: 置信度下限={CONFIDENCE_FLOOR}, "
        f"值班介入 P(yes)>={ESCALATE_THRESHOLD}, "
        f"严重度 P1/P2 分数线={P1_SCORE}/{P2_SCORE}",
    )
    if LIVE:
        print("(LIVE 模式：全部工单走真实决策模型，无故障注入)")
    verdicts = harness.run(TICKETS)
    counts = {}
    for verdict in verdicts:
        counts[verdict.state] = counts.get(verdict.state, 0) + 1
    print("\n===== 批次汇总 =====")
    print(f"  工单总数: {len(verdicts)}, 状态分布: {counts}")
    print(
        f"  自动路由成功 {harness.triaged_ok} 单，"
        f"计费 input_tokens 合计 {harness.total_input_tokens}",
    )
    print("  说明: 决策模型只按输入 token 计费（不生成文本）；")
    print("  转人工/降级的工单不计入自动路由量。")


if __name__ == "__main__":
    main()
