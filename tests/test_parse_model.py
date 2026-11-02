# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for fault-tolerant response parsing (T2)."""

# pytest fixture injection looks like unused/redefined names to pylint.
# pylint: disable=redefined-outer-name,unused-argument

from __future__ import annotations

import logging
from typing import Optional

import httpx
import pytest
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError, model_validator

from dashscope.errors import APIResponseValidationError, DashScopeError
from dashscope.resources.models import Model
from dashscope.types._parse import parse_model
from dashscope.types.common import Usage
from dashscope.types.completions import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from dashscope.types.messages import TextPart, ToolCall


def _resp(status, body=None, headers=None):
    return httpx.Response(
        status,
        json=body,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.test.local/x"),
    )


def _completion(**overrides):
    body = {
        "request_id": "req-1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
            },
        ],
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_a_wellformed_body_parses_exactly_as_plain_validation():
    body = _completion(usage={"input_tokens": 3, "output_tokens": 4})
    assert parse_model(ChatCompletion, body) == (
        ChatCompletion.model_validate(body)
    )


def test_a_wellformed_body_logs_no_drift(caplog):
    # The degraded path has to stay invisible when nothing drifted, or
    # the debug record stops meaning anything.
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        parse_model(ChatCompletion, _completion())
    drift = [r for r in caplog.records if "response drift" in r.getMessage()]
    assert not drift


# --------------------------------------------------------------------------
# Degradation: the crash this exists to prevent
# --------------------------------------------------------------------------


def test_field_type_drift_degrades_instead_of_raising():
    # ``usage`` arrives as a string: before this, a bare ChatCompletion()
    # let a raw pydantic ValidationError escape to the caller.
    body = _completion(usage="unlimited")
    parsed = parse_model(ChatCompletion, body)
    assert parsed.usage is None
    # Everything healthy still parsed.
    assert parsed.request_id == "req-1"
    assert parsed.output_text == "hi"


def test_several_drifted_fields_are_all_repaired_in_one_pass():
    body = _completion(usage="unlimited", search_info="none", model=7)
    body["choices"][0]["index"] = "zero"
    parsed = parse_model(ChatCompletion, body)
    assert parsed.usage is None
    assert parsed.search_info is None
    assert parsed.model is None
    assert parsed.choices[0].index == 0
    assert parsed.output_text == "hi"


def test_a_degraded_parse_keeps_nested_models_as_instances():
    # ``model_construct`` would leave ``usage`` a plain dict, so
    # ``parsed.usage.input_tokens`` raises AttributeError on an object
    # whose healthy twin has a Usage there.
    parsed = parse_model(
        ChatCompletion,
        _completion(usage={"input_tokens": "many", "output_tokens": 5}),
    )
    assert isinstance(parsed.usage, Usage)
    assert parsed.usage.output_tokens == 5
    assert parsed.usage.input_tokens == 0


def test_a_missing_required_nested_field_becomes_a_zeroed_instance():
    # Choice.message is required; a server that drops it must not turn
    # every downstream ``.message.role`` read into an AttributeError.
    parsed = parse_model(Choice, {"index": 0})
    assert isinstance(parsed.message, ChatCompletionMessage)
    assert parsed.message.role == "assistant"


def test_the_business_error_flag_survives_a_degraded_parse():
    # ``is_error`` comes from a mode="after" validator. Skipping it would
    # let a degraded error response masquerade as a success, which is
    # worse than the crash this path replaces.
    parsed = parse_model(
        ChatCompletion,
        {
            "request_id": "req-1",
            "code": "InvalidParameter",
            "message": "bad",
            "usage": "unlimited",
        },
    )
    assert parsed.is_error is True
    assert parsed.code == "InvalidParameter"


def test_list_elements_are_repaired_in_place_keeping_every_index():
    # Callers merge tool calls by position, so a dropped element would
    # silently misalign the ones after it.
    good = {"id": "1", "type": "function", "function": {"name": "f"}}
    parsed = parse_model(
        ChatCompletion,
        _completion(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [good, 5, good],
                    },
                },
            ],
        ),
    )
    calls = parsed.choices[0].message.tool_calls
    assert len(calls) == 3
    assert calls[0].function.name == "f"
    assert calls[2].function.name == "f"
    # The repaired slot is a real model at its original index, not a
    # hole the two healthy calls collapsed into.
    assert isinstance(calls[1], ToolCall)
    assert calls[1].function.name is None


def test_two_drifted_siblings_keep_the_healthy_third():
    # Repairing ``usage.input_tokens`` yields a Usage instance; the second
    # error on the same root must not then mistake that instance for
    # unrepairable input and throw the whole bag -- healthy
    # ``total_tokens`` included -- away.
    parsed = parse_model(
        ChatCompletion,
        _completion(
            usage={
                "input_tokens": "many",
                "output_tokens": "lots",
                "total_tokens": 9,
            },
        ),
    )
    assert isinstance(parsed.usage, Usage)
    assert parsed.usage.total_tokens == 9
    assert (parsed.usage.input_tokens, parsed.usage.output_tokens) == (0, 0)


def test_two_drifted_elements_keep_their_healthy_neighbours():
    parsed = parse_model(
        ChatCompletion,
        _completion(
            choices=[
                {
                    "index": "zero",
                    "message": {"role": "assistant", "content": "a"},
                },
                {"index": 1, "message": {"role": "assistant", "content": "b"}},
                {"index": 2, "message": {"role": "assistant", "content": 3}},
            ],
        ),
    )
    assert [c.message.content for c in parsed.choices] == ["a", "b", None]
    assert [c.index for c in parsed.choices] == [0, 1, 2]


def test_a_nested_missing_field_keeps_its_healthy_siblings():
    # The gap is at ``choices[1].message``. Zeroing the whole ``choices``
    # root would discard the element pydantic had already accepted.
    parsed = parse_model(
        ChatCompletion,
        _completion(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "keep"},
                },
                {"index": 1},
            ],
        ),
    )
    assert len(parsed.choices) == 2
    assert parsed.choices[0].message.content == "keep"
    assert parsed.choices[1].index == 1
    assert parsed.choices[1].message.role == "assistant"


def test_a_non_mapping_body_degrades_instead_of_raising():
    # ``dict(data)`` on a string raised ValueError and on a number
    # TypeError, and it did so in strict mode too -- ahead of the
    # validation error strict mode promises.
    for body in ("a string", 42, None):
        parsed = parse_model(ChatCompletion, body)
        assert isinstance(parsed, ChatCompletion)
        assert parsed.request_id is None
        assert parsed.choices == []


def test_a_non_mapping_body_reports_a_validation_error_in_strict_mode():
    with pytest.raises(APIResponseValidationError) as caught:
        parse_model(
            ChatCompletion,
            "a string",
            strict=True,
            response=_resp(200),
        )
    assert caught.value.body == "a string"
    assert caught.value.response.status_code == 200


def test_extra_fields_survive_a_degraded_parse():
    parsed = parse_model(
        ChatCompletion,
        _completion(usage="unlimited", provider_new_field={"x": 1}),
    )
    assert parsed.model_extra["provider_new_field"] == {"x": 1}
    assert parsed.provider_new_field == {"x": 1}


def test_drift_on_an_aliased_field_is_repaired():
    # ValidationError reports the alias it read (``open_source``), not the
    # field name, so resolving the location needs pydantic's aliasing
    # reversed. Both spellings of the alias are covered.
    assert parse_model(Model, {"open_source": "not-a-bool"}).open_source is (
        None
    )
    assert parse_model(Model, {"opensource": "not-a-bool"}).open_source is (
        None
    )


# --------------------------------------------------------------------------
# Stage 3: cross-field invariants that re-validation cannot satisfy
# --------------------------------------------------------------------------


class _CrossField(BaseModel):
    model_config = ConfigDict(extra="allow")

    low: int = 0
    high: int = 0
    flagged: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "_CrossField":
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        self.flagged = True
        return self


class _BeforeAlwaysFails(BaseModel):
    """Forces stage 3: no validation pass can ever succeed."""

    model_config = ConfigDict(extra="allow")

    code: Optional[str] = None
    is_error: bool = False

    @model_validator(mode="before")
    @classmethod
    def _refuse(cls, value):
        raise ValueError("before validators are never re-run")

    @model_validator(mode="after")
    def _flag(self) -> "_BeforeAlwaysFails":
        if self.code:
            self.is_error = True
        return self


def test_a_cross_field_invariant_falls_back_to_construct():
    # Each field is individually valid, so repairing cannot help; the
    # body is still worth keeping rather than losing.
    parsed = parse_model(_CrossField, {"low": 5, "high": 1, "extra": "kept"})
    assert parsed.low == 5
    assert parsed.high == 1
    assert parsed.model_extra == {"extra": "kept"}


def test_an_after_validator_that_still_fails_does_not_raise():
    # The invariant is genuinely violated by this body, so re-running it
    # fails again; a degraded parse logs that instead of raising.
    assert parse_model(_CrossField, {"low": 5, "high": 1}).flagged is False


def test_the_construct_fallback_reruns_after_validators():
    # Proof the fallback restores the state ``model_construct`` skips:
    # the before-validator is deliberately not re-run (it transforms raw
    # input), while the after-validator's effect is present.
    parsed = parse_model(_BeforeAlwaysFails, {"code": "InvalidParameter"})
    assert parsed.code == "InvalidParameter"
    assert parsed.is_error is True


# --------------------------------------------------------------------------
# strict mode
# --------------------------------------------------------------------------


def test_strict_raises_api_response_validation_error_carrying_the_body():
    body = _completion(usage="unlimited")
    response = _resp(200, body, {"x-request-id": "header-id"})
    with pytest.raises(APIResponseValidationError) as caught:
        parse_model(ChatCompletion, body, strict=True, response=response)
    err = caught.value
    assert err.body == body
    assert isinstance(err.cause, ValidationError)
    assert err.response is response
    # request_id comes from the body first, the header second.
    assert err.request_id == "req-1"


def test_a_strict_failure_is_catchable_as_a_dashscope_error():
    # The whole point of raising the catalogued class instead of letting
    # pydantic escape: one except clause covers it.
    with pytest.raises(DashScopeError):
        parse_model(
            ChatCompletion,
            _completion(usage="unlimited"),
            strict=True,
        )


def test_strict_mode_logs_before_raising(caplog):
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        with pytest.raises(APIResponseValidationError):
            parse_model(
                ChatCompletion,
                _completion(usage="unlimited"),
                strict=True,
            )
    messages = [r.getMessage() for r in caplog.records]
    assert any("strict validation" in m for m in messages)


# --------------------------------------------------------------------------
# Logging contract
# --------------------------------------------------------------------------


def test_drift_is_logged_at_debug_level(caplog):
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        parse_model(ChatCompletion, _completion(usage="unlimited"))
    records = [r for r in caplog.records if "response drift" in r.getMessage()]
    assert records
    assert all(r.levelno == logging.DEBUG for r in records)
    messages = " ".join(r.getMessage() for r in records)
    assert "ChatCompletion.usage" in messages
    # The type is what identifies the drift; the value is user content.
    assert "(got str)" in messages


def test_drift_logs_never_carry_the_raw_value(caplog):
    # A response body holds user prompts and model output, and the
    # redaction filter is key-oriented, so it would not catch these.
    secret = "the-user-sent-this-prompt"
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        parse_model(ChatCompletion, _completion(usage=secret))
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert secret not in messages


# --------------------------------------------------------------------------
# Scope boundary: input models keep failing fast
# --------------------------------------------------------------------------


def test_a_user_supplied_part_type_still_raises():
    # TextPart is an input model. Fault tolerance is for the server
    # drifting, not for absorbing a caller's typo silently.
    with pytest.raises(ValidationError):
        TextPart(type="texst", text="hello")
