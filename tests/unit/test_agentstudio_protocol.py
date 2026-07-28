# -*- coding: utf-8 -*-
"""Protocol-alignment tests: error / snake_case / data flatten.

These three contracts were agreed on with the backend team:

1. Error envelope uses ``error.code`` / ``error.message`` (the
   documented shape). Legacy ``error_code`` / ``error_message`` is
   still tolerated for compatibility.
2. Wire format is snake_case throughout — both request bodies emitted
   by the SDK and response bodies returned by the server. The only
   defensive translation is ``requestId`` → ``request_id`` because
   some legacy gateways still emit camelCase for that single field.
3. Responses come back as flat bare JSON; the SDK exposes the resource
   at the top level via :func:`unwrap`.
"""

import json

import pytest

from dashscope.agentstudio import exceptions
from dashscope.agentstudio.transport import is_error_payload, unwrap
from dashscope.agentstudio.types import (
    user_custom_tool_result,
    user_define_outcome,
    user_interrupt,
    user_message,
    user_tool_confirmation,
)

# ---------------------------------------------------------------------------
# 1. error.{code, message}
# ---------------------------------------------------------------------------


def test_error_uses_nested_code_and_message():
    body = {
        "type": "error",
        "error": {"code": "invalid_request_error", "message": "bad arg"},
        "request_id": "req_001",
    }
    err = exceptions.from_response(status_code=400, body=body)
    assert isinstance(err, exceptions.InvalidRequestError)
    assert err.code == "invalid_request_error"
    assert err.message == "bad arg"
    assert err.request_id == "req_001"


def test_error_legacy_underscored_fields_still_parsed():
    body = {
        "type": "error",
        "error": {"error_code": "rate_limit_error", "error_message": "slow"},
    }
    err = exceptions.from_response(status_code=429, body=body)
    assert isinstance(err, exceptions.RateLimitError)
    assert err.code == "rate_limit_error"


def test_legacy_permission_code_is_normalized():
    """Pre-standard ``permission_denied_error`` normalizes to the unified
    ``permission_error`` (SDK normalization wins over the raw server code)."""
    body = {"type": "error", "error": {"code": "permission_denied_error"}}
    err = exceptions.from_response(status_code=403, body=body)
    assert isinstance(err, exceptions.PermissionDeniedError)
    assert err.code == "permission_error"


def test_404_default_message_has_no_unresolved_placeholder():
    """The registry 404 message is templated; the default must not leak the
    raw ``{resource}`` placeholder when the server omits a message."""
    err = exceptions.from_response(status_code=404, body=None)
    assert isinstance(err, exceptions.NotFoundError)
    assert "{" not in err.message
    assert err.message == "The requested resource was not found."


def test_504_code_and_message_are_consistent():
    """504 code and default message come from the same registry row, so
    they no longer disagree (previously ``api_error`` + a timeout message)."""
    err = exceptions.from_response(status_code=504, body=None)
    assert err.code == "timeout_error"
    assert "timed out" in err.message


def test_default_messages_sourced_from_registry():
    """Statuses the registry models carry a placeholder-free default."""
    for status in (400, 401, 403, 404, 429, 500, 503, 504):
        err = exceptions.from_response(status_code=status, body=None)
        assert "{" not in err.message
        assert not err.message.startswith("HTTP ")


def test_is_error_payload_detects_both_shapes():
    assert is_error_payload(
        {"type": "error", "error": {"code": "x", "message": "y"}},
    )
    assert is_error_payload(
        {"type": "error", "error": {"error_code": "x", "error_message": "y"}},
    )


# ---------------------------------------------------------------------------
# 2. snake_case wire format
# ---------------------------------------------------------------------------


def test_client_event_keys_are_snake_case():
    """Every key the SDK emits in user.* events must be snake_case."""
    samples = [
        user_message(
            "hi",
            session_thread_id="th_1",
            metadata={"k": "v"},
        ),
        user_tool_confirmation(
            tool_use_id="t_1",
            result="allow",
        ),
        user_tool_confirmation(
            tool_use_id="t_1",
            result="deny",
            deny_message="nope",
        ),
        user_custom_tool_result(
            custom_tool_use_id="ctu_1",
            content="ok",
            is_error=False,
        ),
        user_define_outcome(
            description="desc",
            rubric="r",
            max_iterations=3,
        ),
    ]

    def _walk_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "_" in k or k.islower(), f"non-snake_case key: {k}"
                # No camel-case (i.e. no internal uppercase letters)
                assert not any(c.isupper() for c in k), f"camelCase leak: {k}"
                _walk_keys(v)
        elif isinstance(obj, list):
            for it in obj:
                _walk_keys(it)

    for evt in samples:
        # Re-serialize through json to make sure no funny key types slipped in
        _walk_keys(json.loads(json.dumps(evt)))


def test_unwrap_keeps_snake_case_payloads_unchanged():
    payload = {
        "id": "agt_1",
        "name": "demo",
        "system_prompt": "you are helpful",
        "created_at": "2026-06-16T10:00:00Z",
        "request_id": "req_1",
    }
    data, rid = unwrap(payload)
    assert data == payload
    assert rid == "req_1"


def test_unwrap_only_camelcase_translation_is_request_id():
    """Other camelCase keys are left as-is; SDK trusts the backend
    to send snake_case per the agreed wire contract.
    """
    payload = {"id": "agt_1", "fooBar": 1, "requestId": "req_x"}
    data, rid = unwrap(payload)
    assert data["fooBar"] == 1  # not auto-converted
    assert "request_id" in data and "requestId" not in data
    assert rid == "req_x"


# ---------------------------------------------------------------------------
# 3. flat response
# ---------------------------------------------------------------------------


def test_flat_response_unchanged():
    payload = {"id": "agt_1", "request_id": "req_1"}
    data, rid = unwrap(payload)
    assert data == {"id": "agt_1", "request_id": "req_1"}
    assert rid == "req_1"


# ---------------------------------------------------------------------------
# Tests from test_agentstudio_client_events.py
# ---------------------------------------------------------------------------


def test_user_message_string_wraps_to_text_block():
    evt = user_message("hello")
    assert evt["type"] == "message"
    assert evt["role"] == "user"
    assert evt["content"] == [{"type": "text", "text": "hello"}]


def test_user_message_list_passthrough():
    blocks = [{"type": "image", "source": "..."}]
    evt = user_message(blocks, session_thread_id="th_1", metadata={"k": "v"})
    assert evt["type"] == "message"
    assert evt["role"] == "user"
    assert evt["content"] == blocks
    assert evt["session_thread_id"] == "th_1"
    assert evt["metadata"] == {"k": "v"}


def test_user_interrupt():
    evt = user_interrupt()
    assert evt == {"role": "user", "type": "interrupt"}


def test_user_tool_confirmation_validates_result():
    with pytest.raises(ValueError):
        user_tool_confirmation(tool_use_id="t_1", result="MAYBE")
    deny = user_tool_confirmation(
        tool_use_id="t_1",
        result="deny",
        deny_message="nope",
    )
    assert deny["role"] == "user"
    assert deny["type"] == "tool_confirmation"
    data_block = deny["content"][0]
    assert data_block["type"] == "data"
    assert data_block["data"]["call_id"] == "t_1"
    assert data_block["data"]["result"] == "deny"
    assert data_block["data"]["deny_message"] == "nope"


def test_user_custom_tool_result_string_to_text():
    evt = user_custom_tool_result(custom_tool_use_id="ctu_1", content="42")
    assert evt["role"] == "tool"
    assert evt["type"] == "function_call_output"
    data_block = evt["content"][0]
    assert data_block["type"] == "data"
    assert data_block["data"]["call_id"] == "ctu_1"
    assert data_block["data"]["output"] == "42"
    assert evt["is_error"] is False


def test_user_custom_tool_result_dict_to_data_block():
    evt = user_custom_tool_result(
        custom_tool_use_id="ctu_1",
        content={"x": 1},
        is_error=True,
    )
    assert evt["role"] == "tool"
    assert evt["type"] == "function_call_output"
    data_block = evt["content"][0]
    assert data_block["type"] == "data"
    assert data_block["data"]["call_id"] == "ctu_1"
    assert data_block["data"]["output"] == {"x": 1}
    assert evt["is_error"] is True


def test_user_define_outcome():
    evt = user_define_outcome(
        description="task A",
        rubric="must be JSON",
        max_iterations=3,
    )
    assert evt["role"] == "user"
    assert evt["type"] == "define_outcome"
    data_block = evt["content"][0]
    assert data_block["type"] == "data"
    assert data_block["data"]["description"] == "task A"
    assert data_block["data"]["rubric"] == "must be JSON"
    assert data_block["data"]["max_iterations"] == 3


# ---------------------------------------------------------------------------
# Tests for Agent/Session type field mapping
# ---------------------------------------------------------------------------


def test_agent_system_field_and_property():
    """Server returns ``system`` field; SDK exposes it both as
    ``agent.system`` and the convenience ``agent.system_prompt`` property.
    """
    from dashscope.agentstudio.types import Agent

    agent = Agent(id="agt_1", system="You are helpful.", name="Test")
    assert agent.system == "You are helpful."
    assert agent.system_prompt == "You are helpful."


def test_session_stats_and_usage_fields():
    """Session now parses ``stats`` and ``usage`` from server response."""
    from dashscope.agentstudio.types import Session

    stats = {"active_seconds": 42, "duration_seconds": 5}
    usage = {"input_tokens": 100, "output_tokens": 200}
    s = Session(id="sesn_1", status="idle", stats=stats, usage=usage)
    assert s.stats.active_seconds == 42
    assert s.stats.duration_seconds == 5
    assert s.usage.input_tokens == 100
    assert s.usage.output_tokens == 200


def test_from_response_spring_default():
    """Spring Boot default error page is coerced to the right error type."""
    body = {
        "timestamp": "...",
        "status": 404,
        "error": "Not Found",
        "path": "/api/v1/agentstudio/agents",
    }
    err = exceptions.from_response(status_code=404, body=body)
    assert isinstance(err, exceptions.NotFoundError)


# ---------------------------------------------------------------------------
# 4. agents.update version contract (no auto-retrieve)
# ---------------------------------------------------------------------------


class _RecordingTransport:
    """Minimal transport that records requests and returns a canned agent."""

    def __init__(self):
        self.calls = []

    def request(self, method, path, **kwargs):
        self.calls.append({"method": method, "path": path, **kwargs})
        from dashscope.agentstudio.transport import APIResponse

        return APIResponse(
            data={"id": "agent_1", "version": 3, "name": "demo"},
            request_id="req_1",
        )


def _client_with_recording_transport():
    from dashscope.agentstudio import Client

    c = Client(api_key="test-key", base_url="http://test")
    c.transport = _RecordingTransport()
    return c


def test_agents_update_requires_version_kwarg():
    """version is required — omitting it is a TypeError."""
    client = _client_with_recording_transport()
    with pytest.raises(TypeError):
        # pylint: disable=missing-kwoa
        client.agents.update(  # type: ignore[call-arg]
            "agent_1",
            name="new-name",
        )


def test_agents_update_with_version_sends_body():
    """update() sends POST /agents/{id} with version."""
    client = _client_with_recording_transport()
    client.agents.update(
        "agent_1",
        version=3,
        name="new-name",
    )
    assert len(client.transport.calls) == 1
    call = client.transport.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/agents/agent_1"
    body = call["json"]
    assert body["version"] == 3
    assert body["name"] == "new-name"


def test_agents_update_does_not_auto_retrieve():
    """SDK must NOT call retrieve() internally."""
    client = _client_with_recording_transport()
    client.agents.update(
        "agent_1",
        version=3,
        name="new-name",
    )
    assert len(client.transport.calls) == 1
    assert client.transport.calls[0]["method"] == "POST"
