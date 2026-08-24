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


def test_thread_event_exposes_thread_id():
    """thread_* events carry a top-level thread_id; it must be a real field."""
    from dashscope.agentstudio.types import Message

    ev = Message(
        object="message",
        type="thread_status",
        id="sevt_1",
        thread_id="sthr_01M0SG9KZ4TMW0QPEKHHM43QRK",
        content=[
            {
                "type": "data",
                "data": {"agent_name": "worker", "thread_status": "running"},
            },
        ],
    )
    assert ev.thread_id == "sthr_01M0SG9KZ4TMW0QPEKHHM43QRK"
    # not swallowed into extra
    assert "thread_id" not in ev.extra
    # thread_status value stays readable from the data block
    assert ev.content[0].data["thread_status"] == "running"


def test_agent_create_body_includes_multiagent():
    """multiagent roster is forwarded verbatim on create."""
    from dashscope.agentstudio.types.params import AgentCreateParams

    body = AgentCreateParams(
        name="coordinator",
        model="qwen-max",
        multiagent={
            "type": "coordinator",
            "agents": [
                {"type": "self"},
                {"type": "agent", "id": "agent_worker", "version": 1},
            ],
        },
    ).to_dict()
    assert body["multiagent"]["type"] == "coordinator"
    assert body["multiagent"]["agents"][0] == {"type": "self"}
    assert body["multiagent"]["agents"][1]["id"] == "agent_worker"
    assert body["multiagent"]["agents"][1]["version"] == 1

    # omitted -> not emitted
    plain = AgentCreateParams(name="plain", model="qwen-max").to_dict()
    assert "multiagent" not in plain


def test_agent_update_body_includes_multiagent():
    """multiagent is forwarded on update alongside the required version."""
    from dashscope.agentstudio.types.params import AgentUpdateParams

    body = AgentUpdateParams(
        name="coordinator",
        version=2,
        multiagent={"type": "coordinator", "agents": [{"type": "self"}]},
    ).to_dict()
    assert body["version"] == 2
    assert body["multiagent"]["agents"] == [{"type": "self"}]

    # An empty list must not be dropped — it clears the roster server-side.
    cleared = AgentUpdateParams(
        name="coordinator",
        version=3,
        multiagent={"type": "coordinator", "agents": []},
    ).to_dict()
    assert cleared["multiagent"] == {"type": "coordinator", "agents": []}


def test_agent_model_hydrates_multiagent():
    """Agent response hydrates the multiagent dict into typed models."""
    from dashscope.agentstudio.types import (
        Agent,
        MultiAgentConfig,
        MultiAgentRosterEntry,
    )

    agent = Agent(
        id="agent_1",
        version=1,
        multiagent={
            "type": "coordinator",
            "agents": [{"type": "self"}, {"type": "agent", "id": "agent_2"}],
        },
    )
    assert isinstance(agent.multiagent, MultiAgentConfig)
    assert agent.multiagent.type == "coordinator"
    assert isinstance(agent.multiagent.agents[0], MultiAgentRosterEntry)
    assert agent.multiagent.agents[0].type == "self"
    assert agent.multiagent.agents[1].id == "agent_2"
