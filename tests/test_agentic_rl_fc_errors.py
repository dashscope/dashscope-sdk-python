# -*- coding: utf-8 -*-
"""Preserve FaaS business failures even when the transport reports success."""

from unittest.mock import AsyncMock

import pytest

from dashscope.applications.finetune.reinforcement import (
    FunctionComponentModel,
    FunctionComponentRuntime,
    FunctionLayerError,
    FunctionLoadError,
    RolloutFunctionComponent,
    TuningModel,
)
from dashscope.applications.finetune.reinforcement.common import model


LAYER_CODE = "layer_9328e5982668484d9d5269a62868f6f8"
ENTITY_ID = "ro-48aa9fc3-30ec-45c7-bf4f-920f9dcb7d16"
REGION_REJECTION = {
    "request_id": "577f2378-85a2-95ae-902d-444b2dac0045",
    "code": "BadRequest",
    "message": (
        "FaaS layer not found for region null "
        "(globalLayerCode: global_layer_0f1c73df65d64214a3a0c506eff3bf4c)"
    ),
    "status": {"code": 200, "message": "OK"},
}
INSTANCE_OUTPUT = {
    "instanceId": "in-1",
    "trigger_url": "https://example.com/invoke",
    "trigger_token": "sensitive-trigger-token",
}


@pytest.fixture
def component(monkeypatch):
    monkeypatch.setattr(model, "FC_LAYER_USED", True)
    return RolloutFunctionComponent(
        name="rollout-1",
        entity_id=ENTITY_ID,
        fcmodel=FunctionComponentModel(),
        runtime=FunctionComponentRuntime(layer_code=LAYER_CODE),
    )


@pytest.mark.parametrize("output", [None, {}, {"layer_code": LAYER_CODE}])
async def test_create_layer_rejects_business_failure(monkeypatch, output):
    response = {**REGION_REJECTION, "output": output}
    monkeypatch.setattr(model, "client_fc", AsyncMock(return_value=response))

    with pytest.raises(FunctionLayerError) as excinfo:
        await FunctionComponentModel().create_layer(requirements_file="")

    error = excinfo.value
    assert "create rejected by server" in str(error)
    assert "region null" in str(error)
    assert error.request_id == response["request_id"]
    assert error.body == response
    assert error.code == "sdk.agentic_rl.FunctionLayerError"
    assert error.allow_retry is False


@pytest.mark.parametrize("output", [None, {}, {"layer_code": ""}])
async def test_create_layer_requires_nonempty_layer_code(monkeypatch, output):
    response = {"status": {"code": 200}, "output": output}
    monkeypatch.setattr(model, "client_fc", AsyncMock(return_value=response))

    with pytest.raises(FunctionLayerError, match="Empty layer code"):
        await FunctionComponentModel().create_layer(requirements_file="")


@pytest.mark.parametrize("output", [None, {}, {"status": "SUCCESS"}])
async def test_get_layer_rejects_business_failure(monkeypatch, output):
    response = {**REGION_REJECTION, "output": output}
    client = AsyncMock(return_value=response)
    monkeypatch.setattr(model, "client_fc", client)

    with pytest.raises(FunctionLayerError) as excinfo:
        await FunctionComponentModel().get_layer(LAYER_CODE)

    error = excinfo.value
    assert "status query rejected by server" in str(error)
    assert error.request_id == response["request_id"]
    assert error.body == response
    assert error.layer_code == LAYER_CODE
    assert error.allow_retry is False
    assert client.await_count == 1


@pytest.mark.parametrize("output", [None, {}, INSTANCE_OUTPUT])
async def test_load_preserves_business_failure(monkeypatch, component, output):
    response = {**REGION_REJECTION, "output": output}
    client = AsyncMock(
        side_effect=[{"output": {"status": "SUCCESS"}}, response],
    )
    monkeypatch.setattr(model, "client_fc", client)

    result = await component.load()

    assert not result.status.success
    assert result.status.name == "FunctionLoadError"
    assert result.status.code == 522
    assert "load rejected by server" in result.status.message
    assert "Empty instance ID" not in result.status.message
    assert "region null" in result.status.message
    assert ENTITY_ID in result.status.message
    assert LAYER_CODE in result.status.message
    assert result.request_id == response["request_id"]
    assert result.body == response
    assert INSTANCE_OUTPUT["trigger_token"] not in result.status.message
    assert INSTANCE_OUTPUT["trigger_token"] not in repr(result)
    assert component.instance_id is None
    assert client.await_count == 2


@pytest.mark.parametrize("output", [None, {}])
async def test_load_missing_instance_is_still_reported(
    monkeypatch,
    component,
    output,
):
    response = {"request_id": "missing-instance", "output": output}
    monkeypatch.setattr(
        model,
        "client_fc",
        AsyncMock(side_effect=[{"output": {"status": "SUCCESS"}}, response]),
    )

    result = await component.load()

    assert not result.status.success
    assert "Empty instance ID" in result.status.message
    assert "NoneType" not in result.status.message
    assert result.request_id == response["request_id"]
    assert result.body == response


async def test_get_layer_missing_status_remains_retryable(monkeypatch):
    response = {"request_id": "missing-status", "output": None}
    monkeypatch.setattr(model, "client_fc", AsyncMock(return_value=response))

    with pytest.raises(FunctionLayerError) as excinfo:
        await FunctionComponentModel().get_layer(LAYER_CODE)

    assert "no status" in str(excinfo.value)
    assert excinfo.value.allow_retry is True
    assert excinfo.value.request_id == response["request_id"]
    assert excinfo.value.body == response


async def test_registration_preserves_layer_creation_failure(
    monkeypatch,
    component,
    tmp_path,
):
    component.fcmodel.zipdir = str(tmp_path)
    (tmp_path / "requirements.txt").write_text("")
    monkeypatch.setattr(FunctionComponentModel, "to_oss", AsyncMock())
    client = AsyncMock(return_value=REGION_REJECTION)
    monkeypatch.setattr(model, "client_fc", client)

    result = await component.register(
        oss_id="oss-id",
        oss_url="https://example.com/package",
    )

    assert not result.status.success
    assert result.status.code == 524
    assert result.request_id == REGION_REJECTION["request_id"]
    assert result.body == REGION_REJECTION
    assert "create rejected by server" in result.status.message
    assert client.await_count == 1


async def test_load_preserves_layer_query_rejection(monkeypatch, component):
    client = AsyncMock(return_value=REGION_REJECTION)
    monkeypatch.setattr(model, "client_fc", client)

    result = await component.load()

    assert not result.status.success
    assert result.status.name == "FunctionLayerError"
    assert result.status.code == 526
    assert result.request_id == REGION_REJECTION["request_id"]
    assert result.body == REGION_REJECTION
    assert client.await_count == 1


@pytest.mark.parametrize(
    "failure",
    [{"status": {"code": 400}}, {"success": False}],
)
async def test_load_rejects_other_failure_envelopes(
    monkeypatch,
    component,
    failure,
):
    client = AsyncMock(
        side_effect=[
            {"output": {"status": "SUCCESS"}},
            {**failure, "output": INSTANCE_OUTPUT},
        ],
    )
    monkeypatch.setattr(model, "client_fc", client)

    result = await component.load()

    assert not result.status.success
    assert "load rejected by server" in result.status.message
    assert component.instance_id is None


@pytest.mark.parametrize("code", [None, "", 0, 200])
async def test_successful_fc_responses_remain_compatible(
    monkeypatch,
    component,
    code,
):
    monkeypatch.setattr(
        model,
        "client_fc",
        AsyncMock(
            side_effect=[
                {
                    "code": code,
                    "status": {"code": 200},
                    "output": {"layer_code": LAYER_CODE},
                },
                {"code": code, "output": {"status": "SUCCESS"}},
                {"code": code, "output": INSTANCE_OUTPUT},
            ],
        ),
    )

    layer_code = await component.fcmodel.create_layer(requirements_file="")
    result = await component.load()

    assert layer_code == LAYER_CODE
    assert result.status.success
    assert result.output["instance_id"] == "in-1"


async def test_eager_registration_preserves_load_failure(
    monkeypatch,
    component,
):
    client = AsyncMock(
        side_effect=[{"output": {"status": "SUCCESS"}}, REGION_REJECTION],
    )
    monkeypatch.setattr(model, "client_fc", client)
    tuning = TuningModel(functions=[component])

    with pytest.raises(FunctionLoadError) as excinfo:
        await tuning.register_functions(lazy_load=False)

    error = excinfo.value
    assert error.request_id == REGION_REJECTION["request_id"]
    assert error.body == REGION_REJECTION
    assert error.code == "sdk.agentic_rl.FunctionLoadError"
    assert error.entity_id == ENTITY_ID
    assert "region null" in str(error)
    assert client.await_count == 2
