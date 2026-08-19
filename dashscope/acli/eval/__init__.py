# -*- coding: utf-8 -*-
"""Evaluation framework for agent quality and provider comparison.

Library form (no slash command): import and drive it from scripts or tests.

Provides:
  - ``EvalCase``: a single test case (input + expected behaviour)
  - ``EvalRunner``: runs cases against an Agent and scores results
  - ``ProviderComparator``: A/B comparison of two providers on the same cases
"""
# pylint: disable=too-many-branches

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation test case.

    ``input`` is the user prompt. ``expected_keywords`` are terms that should
    appear in a successful response. ``expected_tools`` are tools that should
    be called. ``must_not_contain`` flags failure indicators.
    """

    name: str
    input: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    max_turns: int = 10


@dataclass
class EvalResult:
    """Result of running one EvalCase."""

    case_name: str
    passed: bool
    score: float  # 0..1
    response: str = ""
    tools_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "passed": self.passed,
            "score": self.score,
            "response": self.response[:500],
            "tools_used": self.tools_used,
            "errors": self.errors,
            "duration": self.duration,
        }


class EvalRunner:
    """Runs EvalCases against an Agent and scores results.

    Scoring:
      - 0.4: keyword overlap (expected_keywords in response)
      - 0.3: tool usage (expected_tools were called)
      - 0.2: no forbidden content (must_not_contain absent)
      - 0.1: no errors
    """

    def __init__(self, agent_factory: Any = None):
        """``agent_factory(config) -> Agent`` creates a fresh agent per
        case."""
        self._agent_factory = agent_factory

    async def run_case(self, case: EvalCase, agent: Any = None) -> EvalResult:
        """Run a single evaluation case."""
        import time

        start = time.time()
        ag = agent or (self._agent_factory() if self._agent_factory else None)
        if ag is None:
            return EvalResult(
                case_name=case.name,
                passed=False,
                score=0.0,
                errors=["No agent available"],
            )

        response_text = ""
        tools_used: list[str] = []
        errors: list[str] = []

        # Honor the per-case turn limit (previously written but never applied).
        ag.max_turns = case.max_turns

        try:
            async for chunk in ag.run_stream(case.input):
                response_text += chunk
            # Collect tools used from the last assistant message with
            # tool_calls
            for msg in reversed(ag.messages):
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        if "function" in tc:
                            tools_used.append(tc["function"]["name"])
                    break
        except Exception as e:
            errors.append(str(e))

        duration = time.time() - start

        # Score
        score = 0.0
        # Keyword overlap (0.4)
        if case.expected_keywords:
            response_lower = response_text.lower()
            found = sum(
                1
                for kw in case.expected_keywords
                if kw.lower() in response_lower
            )
            score += 0.4 * (found / len(case.expected_keywords))
        else:
            score += 0.4  # no keywords to check = full marks

        # Tool usage (0.3)
        if case.expected_tools:
            used_set = set(tools_used)
            expected_set = set(case.expected_tools)
            overlap = len(used_set & expected_set)
            score += 0.3 * (overlap / len(expected_set))
        else:
            score += 0.3

        # No forbidden content (0.2)
        if case.must_not_contain:
            response_lower = response_text.lower()
            if any(
                term.lower() in response_lower
                for term in case.must_not_contain
            ):
                errors.append("Response contains forbidden content")
            else:
                score += 0.2
        else:
            score += 0.2

        # No errors (0.1)
        if not errors:
            score += 0.1

        passed = score >= 0.7 and not errors

        return EvalResult(
            case_name=case.name,
            passed=passed,
            score=min(score, 1.0),
            response=response_text,
            tools_used=tools_used,
            errors=errors,
            duration=duration,
        )

    async def run_suite(
        self,
        cases: list[EvalCase],
        agent: Any = None,
    ) -> list[EvalResult]:
        """Run multiple cases. Returns results in order."""
        results = []
        for case in cases:
            result = await self.run_case(case, agent)
            results.append(result)
        return results


class ProviderComparator:
    """A/B comparison of two providers on the same eval suite.

    Runs the same cases against two agent configurations (e.g. tongyi vs
    anthropic) and produces a comparison report.
    """

    def __init__(self, runner: EvalRunner | None = None):
        self._runner = runner or EvalRunner()

    async def compare(
        self,
        cases: list[EvalCase],
        agent_a: Any,
        agent_b: Any,
        label_a: str = "A",
        label_b: str = "B",
    ) -> dict[str, Any]:
        """Run cases against both agents and return a comparison report."""
        results_a = await self._runner.run_suite(cases, agent_a)
        results_b = await self._runner.run_suite(cases, agent_b)

        avg_a = sum(r.score for r in results_a) / max(len(results_a), 1)
        avg_b = sum(r.score for r in results_b) / max(len(results_b), 1)

        return {
            "label_a": label_a,
            "label_b": label_b,
            "avg_score_a": avg_a,
            "avg_score_b": avg_b,
            "pass_rate_a": sum(1 for r in results_a if r.passed)
            / max(len(results_a), 1),
            "pass_rate_b": sum(1 for r in results_b if r.passed)
            / max(len(results_b), 1),
            "avg_duration_a": sum(r.duration for r in results_a)
            / max(len(results_a), 1),
            "avg_duration_b": sum(r.duration for r in results_b)
            / max(len(results_b), 1),
            "winner": label_a
            if avg_a > avg_b
            else label_b
            if avg_b > avg_a
            else "tie",
            "results_a": [r.to_dict() for r in results_a],
            "results_b": [r.to_dict() for r in results_b],
        }


def load_eval_suite(path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSON file.

    Format:
    ``[{"name": "...", "input": "...", "expected_keywords": [...], ...}]``
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for item in data:
        cases.append(
            EvalCase(
                name=item["name"],
                input=item["input"],
                expected_keywords=item.get("expected_keywords", []),
                expected_tools=item.get("expected_tools", []),
                must_not_contain=item.get("must_not_contain", []),
                max_turns=item.get("max_turns", 10),
            ),
        )
    return cases
