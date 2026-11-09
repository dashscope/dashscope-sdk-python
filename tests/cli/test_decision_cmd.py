# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope decision`` -- one round of typed questions."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.decision import DecisionResponse

_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "invoices", "technical": "bugs"},
    },
    "escalate": {"type": "noul", "instructions": "Page on-call?"},
}


def _response(**overrides: Any) -> DecisionResponse:
    data: Dict[str, Any] = {
        "model": "decision-model-preview",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.91,
                "probabilities": {"billing": 0.91, "technical": 0.09},
            },
            "escalate": {"type": "noul", "noul": 0.2},
            "severity": {"type": "score", "score": 1.43, "confidence": 0.7},
        },
        "usage": {"input_tokens": 30},
        "request_id": "req-1",
    }
    data.update(overrides)
    return DecisionResponse(**data)


class _Decision:
    """Stand-in for ``client.decision``."""

    def __init__(self, result: Any = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._result = result if result is not None else _response()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def decision(make_client):
    def install(result: Any = None) -> _Decision:
        resource = _Decision(result)
        make_client(decision=resource)
        return resource

    return install


class TestCreate:
    def test_one_row_per_answer(self, invoke, decision):
        resource = decision()
        result = invoke(
            [
                "decision",
                "create",
                "--state",
                "cannot open the invoice pdf",
                "--questions",
                json.dumps(_QUESTIONS),
            ],
        )
        assert result.exit_code == 0
        for word in ("department", "billing", "escalate", "severity"):
            assert word in result.stdout
        assert resource.calls[0]["model"] == "decision-model-preview"
        assert resource.calls[0]["state"] == "cannot open the invoice pdf"
        assert resource.calls[0]["questions"] == _QUESTIONS

    def test_a_json_state_is_parsed(self, invoke, decision):
        resource = decision()
        invoke(
            [
                "decision",
                "create",
                "--state",
                '{"ticket": 7, "tags": ["billing"]}',
                "--questions",
                json.dumps(_QUESTIONS),
            ],
        )
        assert resource.calls[0]["state"] == {"ticket": 7, "tags": ["billing"]}

    def test_a_json_array_state_is_parsed(self, invoke, decision):
        resource = decision()
        invoke(
            [
                "decision",
                "create",
                "--state",
                '[{"role": "user", "text": "hi"}]',
                "--questions",
                json.dumps(_QUESTIONS),
            ],
        )
        assert resource.calls[0]["state"] == [
            {"role": "user", "text": "hi"},
        ]

    def test_reads_the_state_from_a_file(self, invoke, decision, tmp_path):
        payload = tmp_path / "state.txt"
        payload.write_text("a long ticket body\n", encoding="utf-8")
        resource = decision()
        invoke(
            [
                "decision",
                "create",
                "--state",
                f"@{payload}",
                "--questions",
                json.dumps(_QUESTIONS),
            ],
        )
        assert resource.calls[0]["state"] == "a long ticket body\n"

    def test_reads_the_questions_from_stdin(self, invoke, decision):
        resource = decision()
        result = invoke(
            [
                "decision",
                "create",
                "--state",
                "hi",
                "--questions",
                "-",
            ],
            input=json.dumps(_QUESTIONS),
        )
        assert result.exit_code == 0
        assert resource.calls[0]["questions"] == _QUESTIONS

    def test_the_model_can_be_overridden(self, invoke, decision):
        resource = decision()
        invoke(
            [
                "decision",
                "create",
                "-m",
                "decision-model-2",
                "--state",
                "hi",
                "--questions",
                json.dumps(_QUESTIONS),
            ],
        )
        assert resource.calls[0]["model"] == "decision-model-2"

    def test_a_malformed_json_state_is_a_usage_error(self, invoke, decision):
        resource = decision()
        result = invoke(
            [
                "decision",
                "create",
                "--state",
                "{not json",
                "--questions",
                json.dumps(_QUESTIONS),
            ],
        )
        assert result.exit_code == 2
        assert "--state" in result.stderr
        assert resource.calls == []

    def test_the_state_is_required(self, invoke, decision):
        resource = decision()
        result = invoke(
            ["decision", "create", "--questions", json.dumps(_QUESTIONS)],
        )
        assert result.exit_code == 2
        assert "--state is required" in result.stderr
        assert resource.calls == []

    def test_the_questions_are_required(self, invoke, decision):
        decision()
        assert invoke(["decision", "create", "--state", "hi"]).exit_code == 2

    def test_an_empty_question_set_is_a_usage_error(self, invoke, decision):
        resource = decision()
        result = invoke(
            ["decision", "create", "--state", "hi", "--questions", "{}"],
        )
        assert result.exit_code == 2
        assert "must not be empty" in result.stderr
        assert resource.calls == []

    def test_json_output_is_the_whole_response(self, invoke, decision):
        decision()
        result = invoke(
            [
                "decision",
                "create",
                "--state",
                "hi",
                "--questions",
                json.dumps(_QUESTIONS),
            ],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["answers"]["department"]["choice"] == "billing"
        assert doc["request_id"] == "req-1"

    def test_an_answerless_round_still_renders(self, invoke, decision):
        """A success body may omit the answers table entirely."""
        decision(_response(answers={}))
        result = invoke(
            [
                "decision",
                "create",
                "--state",
                "hi",
                "--questions",
                json.dumps(_QUESTIONS),
            ],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        assert json.loads(result.stdout)["answers"] == {}

    def test_the_usage_note_stays_off_stdout(self, invoke, decision):
        decision()
        result = invoke(
            [
                "decision",
                "create",
                "--state",
                "hi",
                "--questions",
                json.dumps(_QUESTIONS),
            ],
        )
        assert "usage:" in result.stderr
        assert "usage" not in result.stdout


class TestGroupShape:
    def test_bare_decision_shows_help(self, app, runner):
        assert "Usage" in runner.invoke(app, ["decision"]).stdout

    def test_help_carries_a_worked_question_set(self, app, runner):
        """`--questions` is the one flag whose shape is not guessable."""
        text = runner.invoke(app, ["decision", "--help"]).stdout
        assert "criteria" in text
