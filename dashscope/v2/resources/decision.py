# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.decision resource (TypeSafe System One decision model).

The decision model (``decision-model-preview``) answers typed questions
about a business ``state`` in one forward pass -- multi-choice
(``choice``), yes/no (``noul``) and ordinal scale (``score``) -- and
returns each answer with its probability distribution, plus a confidence
for ``choice`` / ``score``. It generates no text, so latency and cost do
not grow with output length. Typical uses: ticket routing, content
moderation, agent routing and result verification.

Wire: ``POST {gateway}/compatible-mode/v1/systemone`` (no streaming, no
function calling, no web search). The endpoint lives under the
compatible-mode path on every gateway, independent of the client's chat
``protocol`` setting, so this resource derives its own base URL via
``compatible_mode_base_url`` instead of using ``Config.base_url``
(native) or ``Config.protocol_base_url`` (chat-protocol dependent).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from dashscope.v2._config import Config, compatible_mode_base_url
from dashscope.v2.errors import response_error
from dashscope.v2.types.common import ResponseBase, Usage, with_headers

_PATH = "/systemone"
# Functional-area tag for x-dashscope-sdk-client. The decision model is
# new in v2 (no 1.x API), so "decision" extends the 1.x vocabulary and
# rides the same V2_MODULE_MARKER mechanism (emitted as "decision2").
MODULE = "decision"

__all__ = [
    "AsyncDecision",
    "ChoiceQuestion",
    "Decision",
    "DecisionAnswer",
    "DecisionResponse",
    "NoulQuestion",
    "QuestionInput",
    "ScoreQuestion",
]


class ChoiceQuestion(BaseModel):
    """A pick-one-of-N question (``type="choice"``).

    ``criteria`` maps each option name to its description (1-255
    options; give the full set and add an ``other`` fallback).
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"] = "choice"
    instructions: Optional[str] = None
    criteria: Dict[str, str] = Field(min_length=1, max_length=255)


class NoulQuestion(BaseModel):
    """A yes/no judgement (``type="noul"``); the answer is P(yes).

    ``criteria`` may describe the two outcomes as
    ``{"true": ..., "false": ...}`` and is optional.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"] = "noul"
    instructions: Optional[str] = None
    criteria: Optional[Dict[str, str]] = None


class ScoreQuestion(BaseModel):
    """An ordinal-scale rating (``type="score"``).

    ``criteria`` describes the levels from low to high (2-255 levels;
    3-7 clearly distinguishable levels work best). The answer is the
    probability-weighted level index and may fall between two levels.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["score"] = "score"
    instructions: Optional[str] = None
    criteria: List[str] = Field(min_length=2, max_length=255)


# A question is given either as one of the typed models above or as a
# raw wire dict (``{"type": "choice", "criteria": {...}, ...}``); raw
# dicts pass through untouched, which keeps future question fields
# usable before the SDK models learn them.
QuestionInput = Union[
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    Dict[str, Any],
]

# ``state`` is the business state under decision: free text, a dialogue,
# or a structured object (serialized before the model sees it).
StateInput = Union[str, Dict[str, Any], List[Any]]


class DecisionAnswer(BaseModel):
    """The answer to one question; the filled field follows ``type``.

    ``choice`` carries the selected option name, ``noul`` P(yes) with
    0=no and 1=yes, and ``score`` the probability-weighted level index
    (e.g. ``1.43``). ``probabilities`` gives every option's / level's
    probability (they sum to 1); ``confidence`` is returned for
    ``choice`` / ``score`` only, and ``legend`` maps a ``score`` level
    index back to its description.
    """

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = None
    choice: Optional[str] = None
    noul: Optional[float] = None
    score: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    legend: Optional[Dict[str, str]] = None


class DecisionResponse(ResponseBase):
    """One System One decision round; ``answers`` keys mirror the
    request's question ids."""

    model: Optional[str] = None
    answers: Dict[str, DecisionAnswer] = {}
    usage: Optional[Usage] = None
    latency_ms: Optional[float] = None


def _parse(body: Dict[str, Any]) -> DecisionResponse:
    payload = dict(body) if isinstance(body, dict) else {}
    # Tolerate a success body that omits the answers table.
    payload["answers"] = payload.get("answers") or {}
    return DecisionResponse(**payload)


def _dump_question(question: QuestionInput) -> Dict[str, Any]:
    if isinstance(question, BaseModel):
        # exclude_none keeps an unset optional (noul criteria, missing
        # instructions) off the wire instead of sending explicit nulls.
        return question.model_dump(exclude_none=True)
    return dict(question)


def _build_body(
    *,
    model: str,
    state: StateInput,
    questions: Dict[str, QuestionInput],
) -> Dict[str, Any]:
    if not questions:
        raise ValueError("questions must not be empty")
    return {
        "model": model,
        "state": state,
        "questions": {
            key: _dump_question(question)
            for key, question in questions.items()
        },
    }


def _endpoint_base(config: Config) -> str:
    # The System One endpoint sits under the compatible-mode path on
    # every gateway; the chat protocol setting must not move it.
    return compatible_mode_base_url(
        config.workspace,
        config.region,
        config.configured_base_url,
    )


def _checked_response(client, response) -> DecisionResponse:
    """Raise on failure, or fill the error envelope when not raising.

    A failed call (non-2xx, or HTTP 200 with a business ``code``)
    carries its ``code`` / ``message`` in the error envelope -- nested
    under ``error`` on this compatible-mode endpoint -- which the
    success parser cannot see. Routing the reply through
    ``response_error`` keeps ``raise_on_error=False`` reporting the
    failure via ``is_error`` / ``code`` / ``message`` / ``request_id``
    / ``status_code`` instead of surfacing it as an empty success.
    """
    error = response_error(response)
    if error is None:
        return with_headers(_parse(response.json()), response)
    if client.raise_on_error:
        raise error
    return with_headers(
        DecisionResponse(
            request_id=error.request_id,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            is_error=True,
        ),
        response,
    )


class Decision:
    """Synchronous decision-model calls (``client.decision``)."""

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        state: StateInput,
        questions: Dict[str, QuestionInput],
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DecisionResponse:
        """Run one decision round.

        Usage::

            resp = client.decision.create(
                model="decision-model-preview",
                state={"ticket_id": "T-1001", "content": "..."},
                questions={
                    "department": ChoiceQuestion(
                        instructions="Which team should handle this?",
                        criteria={"billing": "...", "technical": "..."},
                    ),
                    "escalate": NoulQuestion(
                        instructions="Page the on-call engineer?",
                    ),
                },
            )
            print(resp.answers["department"].choice)
        """
        body = _build_body(model=model, state=state, questions=questions)
        response = self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            base_url=_endpoint_base(self._client.config),
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        return _checked_response(self._client, response)


class AsyncDecision:
    """Asynchronous decision-model calls (``client.decision``)."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        state: StateInput,
        questions: Dict[str, QuestionInput],
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DecisionResponse:
        body = _build_body(model=model, state=state, questions=questions)
        response = await self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            base_url=_endpoint_base(self._client.config),
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        return _checked_response(self._client, response)
