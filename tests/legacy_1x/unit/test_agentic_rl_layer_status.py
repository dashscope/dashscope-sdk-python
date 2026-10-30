# -*- coding: utf-8 -*-
"""Layer readiness must gate instance loading instead of being swallowed.

``get_layer`` used to catch its own ``FunctionLayerError`` and return
``"SUCCESS"``, so an unbuilt layer surfaced later as an unrelated
``Empty instance ID`` load failure. These cover the terminal states, the
stalled build, and the two error paths in ``load``.
"""

import pytest

from dashscope.finetune.reinforcement import (
    FunctionComponentModel,
    FunctionComponentRuntime,
    FunctionLayerError,
    RolloutFunctionComponent,
)
from dashscope.finetune.reinforcement.common import model

LAYER_CODE = "layer_c5669b3b602d4ae9b99bc8f576e4fdce"
ENTITY_ID = "ro-85a1dc67-cd85-49a9-855b-b2bb575d80ed"

# The payload the FaaS gateway returns when a layer is missing in a region:
# HTTP 200 with the failure described in the body and no ``output.status``.
LAYER_NOT_FOUND_BODY = {
    "request_id": "55bd69f2-3e3b-9cc0-989c-07b867aa9f17",
    "message": "FaaS layer not found for region cn-wulanchabu",
    "code": "BadRequest",
    "status": {"code": 200, "message": "OK"},
    "output": {},
}

# A layer that is already built, followed by a successful instance load.
READY_LAYER_THEN_INSTANCE = [
    {"output": {"status": "SUCCESS"}},
    {
        "output": {
            "instanceId": "in-1",
            "trigger_url": "https://example.com/invoke",
            "trigger_token": "token",
        },
    },
]


def _stub_responses(monkeypatch, responses):
    """Serve ``responses`` in order, repeating the last one, and count calls."""
    calls = {"count": 0, "payloads": []}

    async def fake_client_fc(*args, **kwargs):
        index = calls["count"]
        calls["count"] += 1
        calls["payloads"].append(args[2] if len(args) > 2 else kwargs)
        response = responses[min(index, len(responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(model, "client_fc", fake_client_fc)
    return calls


def _component(with_runtime=True):
    component = RolloutFunctionComponent(name="rollout-1")
    component.entity_id = ENTITY_ID
    component.fcmodel = FunctionComponentModel()
    if with_runtime:
        component.runtime = FunctionComponentRuntime(layer_code=LAYER_CODE)
    return component


async def test_get_layer_returns_ready_status(monkeypatch):
    calls = _stub_responses(monkeypatch, [{"output": {"status": "SUCCESS"}}])

    assert await FunctionComponentModel().get_layer(LAYER_CODE) == "SUCCESS"
    assert calls["count"] == 1


async def test_get_layer_waits_out_a_slow_build(monkeypatch):
    calls = _stub_responses(
        monkeypatch,
        [
            {"output": {"status": "RUNNING"}},
            {"output": {"status": "RUNNING"}},
            {"output": {"status": "SUCCESS"}},
        ],
    )
    monkeypatch.setattr(model, "FC_LAYER_STATUS_INTERVAL", 0)

    assert await FunctionComponentModel().get_layer(LAYER_CODE) == "SUCCESS"
    assert calls["count"] == 3


async def test_get_layer_raises_on_failed_build(monkeypatch):
    _stub_responses(
        monkeypatch,
        [{"output": {"status": "FAILED"}, "message": "pip install exploded"}],
    )

    with pytest.raises(FunctionLayerError) as excinfo:
        await FunctionComponentModel().get_layer(LAYER_CODE)

    message = str(excinfo.value)
    assert "pip install exploded" in message
    assert LAYER_CODE in message
    # The class prefix names the subsystem, not the create stage, so a build
    # failure must not read as "create failed: ... reported FAILED".
    assert "Function layer failed: build reported FAILED" in message
    assert excinfo.value.layer_code == LAYER_CODE
    assert excinfo.value.layer_name is None


async def test_get_layer_raises_on_region_rejection(monkeypatch):
    calls = _stub_responses(monkeypatch, [LAYER_NOT_FOUND_BODY])

    with pytest.raises(FunctionLayerError) as excinfo:
        await FunctionComponentModel().get_layer(LAYER_CODE)

    assert "cn-wulanchabu" in str(excinfo.value)
    assert calls["count"] == 1


async def test_get_layer_raises_when_build_stalls(monkeypatch):
    calls = _stub_responses(monkeypatch, [{"output": {"status": "RUNNING"}}])
    monkeypatch.setattr(model, "FC_LAYER_STATUS_TIMEOUT", 0.2)
    monkeypatch.setattr(model, "FC_LAYER_STATUS_INTERVAL", 0.05)

    with pytest.raises(FunctionLayerError) as excinfo:
        await FunctionComponentModel().get_layer(LAYER_CODE)

    message = str(excinfo.value)
    assert "timed out" in message
    assert "RUNNING" in message
    assert "FC_LAYER_STATUS_TIMEOUT" in message
    assert calls["count"] > 3


async def test_get_layer_raises_on_transport_error(monkeypatch):
    _stub_responses(monkeypatch, [RuntimeError("connection reset")])

    with pytest.raises(FunctionLayerError) as excinfo:
        await FunctionComponentModel().get_layer(LAYER_CODE)

    assert "connection reset" in str(excinfo.value)


async def test_load_reports_layer_failure_as_such(monkeypatch):
    _stub_responses(monkeypatch, [{"output": {"status": "FAILED"}}])
    monkeypatch.setattr(model, "FC_LAYER_USED", True)

    result = await _component().load(entity_id=ENTITY_ID)

    assert not result.status.success
    assert result.status.name == "FunctionLayerError"
    assert result.status.code == 526
    assert LAYER_CODE in result.status.message
    assert "Empty instance ID" not in result.status.message


async def test_load_without_runtime_explains_missing_layer_code(monkeypatch):
    _stub_responses(monkeypatch, [{"output": {"status": "SUCCESS"}}])
    monkeypatch.setattr(model, "FC_LAYER_USED", True)

    result = await _component(with_runtime=False).load(entity_id=ENTITY_ID)

    assert not result.status.success
    assert "layer_code is required" in result.status.message
    assert "NoneType" not in result.status.message


async def test_load_accepts_layer_code_from_the_runtime_argument(monkeypatch):
    """The 2022 message tells callers to pass a runtime carrying layer_code,
    so ``load`` must honour that argument when ``self.runtime`` is absent."""
    _stub_responses(monkeypatch, READY_LAYER_THEN_INSTANCE)
    monkeypatch.setattr(model, "FC_LAYER_USED", True)

    result = await _component(with_runtime=False).load(
        entity_id=ENTITY_ID,
        runtime=FunctionComponentRuntime(layer_code=LAYER_CODE),
    )

    assert result.status.success
    assert result.output["instance_id"] == "in-1"


async def test_load_waits_on_the_runtime_argument_layer(monkeypatch):
    """The argument already overrides ``self.runtime`` in the request body, so
    it must also decide which layer build is waited on."""
    calls = _stub_responses(monkeypatch, READY_LAYER_THEN_INSTANCE)
    monkeypatch.setattr(model, "FC_LAYER_USED", True)

    override_code = "layer_0f3d9c1a77b24e08a5c6d4b1e9f07a23"
    result = await _component().load(
        entity_id=ENTITY_ID,
        runtime=FunctionComponentRuntime(layer_code=override_code),
    )

    assert result.status.success
    assert calls["payloads"][0] == {"layer_code": override_code}
    assert calls["payloads"][1]["layer_code"] == override_code


async def test_load_reports_server_rejection_with_context(monkeypatch):
    _stub_responses(
        monkeypatch,
        [{"output": {"status": "SUCCESS"}}, LAYER_NOT_FOUND_BODY],
    )
    monkeypatch.setattr(model, "FC_LAYER_USED", True)

    result = await _component().load(entity_id=ENTITY_ID)

    assert not result.status.success
    assert result.status.name == "FunctionLoadError"
    assert ENTITY_ID in result.status.message
    assert LAYER_CODE in result.status.message
    assert "cn-wulanchabu" in result.status.message


async def test_load_succeeds_once_layer_is_ready(monkeypatch):
    _stub_responses(monkeypatch, READY_LAYER_THEN_INSTANCE)
    monkeypatch.setattr(model, "FC_LAYER_USED", True)

    result = await _component().load(entity_id=ENTITY_ID)

    assert result.status.success
    assert result.output["instance_id"] == "in-1"
