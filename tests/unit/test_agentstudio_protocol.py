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
    user_tool_approval_response,
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
        user_tool_approval_response(
            batch_id="response_xxx:9f2c",
            call_id="t_1",
            result="allow",
        ),
        user_tool_approval_response(
            batch_id="response_xxx:9f2c",
            call_id="t_1",
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


def test_user_tool_approval_response_validates_result():
    with pytest.raises(ValueError):
        user_tool_approval_response(
            batch_id="b1",
            call_id="t_1",
            result="MAYBE",
        )
    deny = user_tool_approval_response(
        batch_id="b1",
        call_id="t_1",
        result="deny",
        deny_message="nope",
    )
    assert deny["role"] == "user"
    assert deny["type"] == "tool_approval_response"
    data_block = deny["content"][0]
    assert data_block["type"] == "data"
    assert data_block["data"]["batch_id"] == "b1"
    assert data_block["data"]["call_id"] == "t_1"
    assert data_block["data"]["result"] == "deny"
    assert data_block["data"]["deny_message"] == "nope"
    # allow carries no deny_message
    allow = user_tool_approval_response(
        batch_id="b1",
        call_id="t_1",
        result="allow",
    )
    assert "deny_message" not in allow["content"][0]["data"]


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

    # A MultiAgentConfig read off a response must be accepted back as-is,
    # so read-modify-write round trips work.
    from dashscope.agentstudio.types import Agent

    hydrated = Agent(
        id="agent_1",
        multiagent={"type": "coordinator", "agents": [{"type": "self"}]},
    ).multiagent
    round_tripped = AgentUpdateParams(
        name="coordinator",
        version=4,
        multiagent=hydrated,
    ).to_dict()
    assert round_tripped["multiagent"] == {
        "type": "coordinator",
        "agents": [{"type": "self"}],
    }


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


# ---------------------------------------------------------------------------
# Tool approval
# ---------------------------------------------------------------------------


def _approval_request_msg():
    from dashscope.agentstudio.types import parse_message

    return parse_message(
        {
            "object": "message",
            "status": "completed",
            "id": "msg_approval_xxx",
            "role": "assistant",
            "type": "tool_approval_request",
            "content": [
                {
                    "type": "data",
                    "data": {
                        "batch_id": "response_xxx:9f2c",
                        "call_id": "call_xxx",
                        "name": "bash",
                        "arguments": '{"command": "ls -la"}',
                        "tool_type": "builtin",
                    },
                },
            ],
        },
    )


def test_tool_approval_request_accessor():
    from dashscope.agentstudio.types import parse_message

    m = _approval_request_msg()
    ap = m.tool_approval_request
    assert ap["batch_id"] == "response_xxx:9f2c"
    assert ap["call_id"] == "call_xxx"
    assert (
        ap["arguments"] == '{"command": "ls -la"}'
    )  # JSON string, not object
    assert ap["tool_type"] == "builtin"
    # None for non-approval events
    assert parse_message({"type": "tool_call"}).tool_approval_request is None


def test_stop_reason_status_and_enum():
    from dashscope.agentstudio.types import StopReason
    from dashscope.agentstudio.constants import (
        SSEEventType,
        SessionStatus,
        StopReasonType,
    )

    sr = StopReason(
        type="requires_action",
        pending_batch_id="b1",
        pending_call_ids=["c1", "c2"],
    )
    d = sr.to_dict()
    assert d["pending_call_ids"] == ["c1", "c2"]
    assert SessionStatus.RESCHEDULED == "rescheduled"
    assert SessionStatus.DELETED == "deleted"
    members = {m.value for m in SSEEventType}
    assert len(members) == 23
    assert "tool_confirmation" not in members
    assert (
        "tool_approval_request" in members
        and "tool_approval_response" in members
    )
    assert StopReasonType.REQUIRES_ACTION == "requires_action"


def test_permission_policy_validation():
    from dashscope.agentstudio.types import PermissionPolicy

    assert PermissionPolicy(type="always_ask").type == "always_ask"
    assert PermissionPolicy().type == "always_allow"  # default
    with pytest.raises(ValueError):
        PermissionPolicy(type="always_x")
    # pylint: disable=too-many-function-args
    with pytest.raises(TypeError):
        PermissionPolicy("always_ask")
    # pylint: enable=too-many-function-args


def test_approval_errors_map_by_http_status():
    """Approval error codes are not enumerated as separate exception types;
    they map by HTTP status, with the reason in .code/.message."""
    from dashscope.agentstudio.exceptions import (
        InternalServerError,
        InvalidRequestError,
        OverloadedError,
        from_response,
    )

    cases = {
        ("bma_invalid_event", 400): InvalidRequestError,
        ("invalid_tool_approval", 400): InvalidRequestError,
        ("pending_tool_approval_unresolved", 400): InvalidRequestError,
        ("malformed_model_tool_call", 500): InternalServerError,
        ("tool_approval_service_unavailable", 503): OverloadedError,
    }
    for (code, status), cls in cases.items():
        err = from_response(
            status_code=status,
            body={"type": "error", "error": {"code": code, "message": "why"}},
            headers={},
        )
        assert isinstance(err, cls), (code, type(err))
        assert err.code == code and err.message == "why"


def test_event_list_types_repeated_query():
    """Multi-type filter serializes as repeated types[]= keys, not a single
    comma-joined string (which the server parses as one bogus value)."""
    import httpx
    from dashscope.agentstudio.types.params import SessionEventListParams

    params = SessionEventListParams(
        types=["message", "tool_call", "tool_approval_request"],
        limit=50,
    ).to_dict()
    q = httpx.Request("GET", "https://x/", params=params).url.query.decode()
    assert "types%5B%5D=message" in q
    assert "types%5B%5D=tool_call" in q
    assert "types%5B%5D=tool_approval_request" in q
    assert "," not in q


# ---------------------------------------------------------------------------
# Event deltas
# ---------------------------------------------------------------------------


def _delta_payloads():
    return [
        {"type": "event_start", "event": {"id": "sevt_a", "type": "message"}},
        {
            "type": "event_delta",
            "event_id": "sevt_a",
            "delta": {
                "type": "content_delta",
                "index": 0,
                "content": {"type": "text", "text": "事件增量"},
            },
        },
        {
            "type": "event_delta",
            "event_id": "sevt_a",
            "delta": {
                "type": "content_delta",
                "index": 0,
                "content": {"type": "text", "text": "流"},
            },
        },
        {
            "object": "message",
            "status": "completed",
            "id": "sevt_a",
            "role": "assistant",
            "type": "message",
            "content": [{"type": "text", "text": "事件增量流"}],
        },
    ]


def test_delta_frames_and_text_deltas():
    from dashscope.agentstudio.resources.session_events import (
        _TypedEventStream,
    )
    from dashscope.agentstudio.types import parse_message

    msgs = [parse_message(p) for p in _delta_payloads()]
    assert msgs[0].event_start == {"id": "sevt_a", "type": "message"}
    assert msgs[1].event_delta["event_id"] == "sevt_a"
    assert msgs[1].delta_text == "事件增量"
    assert msgs[3].delta_text is None  # terminal message is not a delta frame

    class _Fake:
        def __init__(self, items):
            self._items = items

        def __iter__(self):
            return iter(self._items)

        def close(self):
            pass

    ts = _TypedEventStream.__new__(_TypedEventStream)
    ts._stream = _Fake(_delta_payloads())  # pylint: disable=protected-access
    assert list(ts.text_deltas) == ["事件增量", "流"]


# ---------------------------------------------------------------------------
# Session resources & threads
# ---------------------------------------------------------------------------


def test_session_resource_model_and_add_params():
    from dashscope.agentstudio import SessionResource
    from dashscope.agentstudio.types.params import SessionResourceAddParams

    r = SessionResource(
        id="sesrsc_1",
        type="file",
        file_id="file_copy",  # session-scoped copy, != source id
        mount_path="/mnt/session/uploads/data.csv",
        created_at="t",
        updated_at="t",
        request_id="r",
    )
    assert r.file_id == "file_copy"
    body = SessionResourceAddParams(
        type="file",
        file_id="file_src",
        mount_path="/uploads/data.csv",
    ).to_dict()
    assert body == {
        "type": "file",
        "file_id": "file_src",
        "mount_path": "/uploads/data.csv",
    }


def test_session_thread_model_fields():
    from dashscope.agentstudio.types import SessionThread

    th = SessionThread(
        id="sthr_01",
        type="session_thread",
        session_id="sesn_01",
        parent_thread_id="sthr_primary",
        agent={"id": "agent_01", "version": 1},
        status="idle",
        created_at="t",
        updated_at="t",
    )
    d = th.to_dict()
    assert d["type"] == "session_thread"
    assert d["agent"] == {"id": "agent_01", "version": 1}
    assert "title" not in d  # legacy field removed


def test_message_error_and_pending_signal():
    from dashscope.agentstudio.types import parse_message

    ev = parse_message(
        {
            "type": "error",
            "error": {
                "code": "tool_approval_service_unavailable",
                "message": "down",
            },
            "metadata": {
                "pending_tool_approvals": {
                    "batch_id": "b1",
                    "call_ids": ["c1"],
                },
            },
        },
    )
    assert ev.error["code"] == "tool_approval_service_unavailable"
    assert ev.pending_tool_approvals == {"batch_id": "b1", "call_ids": ["c1"]}
    # non-error events: both None
    plain = parse_message(
        {
            "type": "message",
            "content": [{"type": "text", "text": "hi"}],
        },
    )
    assert plain.error is None and plain.pending_tool_approvals is None
    # error without the suspend signal: barrier is not up
    err_only = parse_message(
        {
            "type": "error",
            "error": {"code": "malformed_model_tool_call"},
        },
    )
    assert err_only.pending_tool_approvals is None


# ---------------------------------------------------------------------------
# P2: Message data accessors & multiagent
# ---------------------------------------------------------------------------


def test_message_data_accessors():
    from dashscope.agentstudio.types import parse_message

    # model_request_end
    mre = parse_message(
        {
            "object": "message",
            "status": "completed",
            "id": "m1",
            "type": "model_request_end",
            "content": [
                {
                    "type": "data",
                    "data": {
                        "model_request_start_id": "m0",
                        "is_error": False,
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 80,
                        "speed": "standard",
                    },
                },
            ],
        },
    )
    assert mre.model_request_end["output_tokens"] == 50
    assert mre.data["model_request_start_id"] == "m0"
    assert mre.outcome_evaluation is None  # wrong type

    # outcome_evaluation
    oe = parse_message(
        {
            "object": "message",
            "status": "completed",
            "id": "m2",
            "type": "outcome_evaluation",
            "content": [
                {
                    "type": "data",
                    "data": {
                        "outcome_id": "out_x",
                        "iteration": 1,
                        "phase": "end",
                        "result": "pass",
                        "explanation": "ok",
                    },
                },
            ],
        },
    )
    assert oe.outcome_evaluation["phase"] == "end"
    assert oe.outcome_evaluation["result"] == "pass"

    # thread_status / thread_created / session_updated share the data shape
    ts = parse_message(
        {
            "type": "thread_status",
            "content": [
                {
                    "type": "data",
                    "data": {
                        "session_thread_id": "sthr_1",
                        "agent_name": "researcher",
                        "thread_status": "idle",
                        "stop_reason": {"type": "end_turn"},
                    },
                },
            ],
        },
    )
    assert ts.thread_status["thread_status"] == "idle"

    tc = parse_message(
        {
            "type": "thread_created",
            "content": [
                {
                    "type": "data",
                    "data": {
                        "session_thread_id": "sthr_1",
                        "agent_name": "researcher",
                    },
                },
            ],
        },
    )
    assert tc.thread_created["agent_name"] == "researcher"

    su = parse_message(
        {
            "type": "session_updated",
            "content": [{"type": "data", "data": {"title": "新标题"}}],
        },
    )
    assert su.session_updated["title"] == "新标题"
    # non-matching types return None
    assert ts.session_updated is None


def test_thread_message_routing_accessor():
    from dashscope.agentstudio.types import parse_message

    sent = parse_message(
        {
            "type": "thread_message_sent",
            "role": "assistant",
            "content": [{"type": "text", "text": "查一下"}],
            "metadata": {
                "to_session_thread_id": "sthr_r",
                "to_agent_name": "researcher",
            },
        },
    )
    assert sent.thread_message_routing == {
        "to_session_thread_id": "sthr_r",
        "to_agent_name": "researcher",
    }
    recv = parse_message(
        {
            "type": "thread_message_received",
            "role": "assistant",
            "content": [{"type": "text", "text": "查到了"}],
            "metadata": {
                "from_session_thread_id": "sthr_r",
                "from_agent_name": "researcher",
            },
        },
    )
    assert recv.thread_message_routing["from_agent_name"] == "researcher"
    # non-thread-message event -> None
    assert parse_message({"type": "message"}).thread_message_routing is None


def test_multiagent_config_normalization_and_roster_fields():
    """The SDK normalizes multiagent config (type default, self cleanup) but
    does not reject server data — the 1-20 / at-most-one-self limits are the
    server's to enforce, so parsing a valid agent never fails."""
    from dashscope.agentstudio.types import (
        MultiAgentConfig,
        MultiAgentRosterEntry,
    )

    cfg = MultiAgentConfig(
        agents=[
            {"type": "self"},
            {"type": "agent", "id": "agent_2", "version": 3},
        ],
    )
    assert cfg.type == "coordinator"  # defaulted
    assert all(isinstance(a, MultiAgentRosterEntry) for a in cfg.agents)
    assert cfg.agents[0].type == "self"
    # name/description are enriched response fields (present on retrieval)
    entry = MultiAgentRosterEntry(
        type="agent",
        id="a1",
        name="worker",
        description="reads files",
    )
    assert entry.name == "worker" and entry.description == "reads files"
    # empty list clears the roster
    assert MultiAgentConfig(agents=[]).agents == []
    # None agents normalizes to []
    assert MultiAgentConfig(agents=None).agents == []


def test_text_stream_stops_on_session_status_idle():
    """Regression: text_stream/text_deltas must not crash on session_status events.
    RESCHEDULING was deleted from constants but stop-set tuples still referenced it,
    causing AttributeError on every stream's normal end path."""
    from dashscope.agentstudio.resources.session_events import (
        _TypedEventStream,
    )

    raw_events = [
        {
            "object": "message",
            "status": "completed",
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
        },
        {
            "object": "message",
            "status": "completed",
            "id": "m2",
            "type": "session_status",
            "content": [
                {
                    "type": "data",
                    "data": {
                        "session_status": "idle",
                        "stop_reason": {"type": "end_turn"},
                    },
                },
            ],
        },
    ]

    ts = _TypedEventStream.from_raw_events(raw_events)
    assert list(ts.text_stream) == ["hello"]

    ts2 = _TypedEventStream.from_raw_events(raw_events)
    assert not list(ts2.text_deltas)
