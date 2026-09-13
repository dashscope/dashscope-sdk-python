# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Regression tests for the two P0 defects fixed on dev/errors.

P0-1: ``AioHttpRequest._handle_response`` yielded the result of the
      ``async def`` ``_handle_aiohttp_failed_response`` without awaiting
      it, so every ``Aio*`` API handed callers a bare coroutine instead
      of a ``DashScopeAPIResponse`` on any non-2xx status.

P0-2: six FAILED-response sites in ``reinforcement/common/model.py`` read
      ``SdkErrorDef.external``, a field deleted together with
      ``ClientErrorDef``. Every real function-component deployment
      failure therefore raised ``AttributeError`` instead of returning a
      structured ``ResponseFC``.
"""
import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dashscope.api_entities.aiohttp_request import AioHttpRequest
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.common.error_registry import (
    INTERNAL_ERROR,
    SDK_AGENTIC_RL_BASE_CONNECTION_ERROR,
    SDK_AGENTIC_RL_FUNCTION_LAYER_ERROR,
    SDK_AGENTIC_RL_FUNCTION_LOAD_ERROR,
    SDK_AGENTIC_RL_INSTANCE_QUERY_ERROR,
    SDK_AGENTIC_RL_INSTANCE_WARMUP_ERROR,
    SDK_AGENTIC_RL_REGISTRATION_ERROR,
)
from dashscope.finetune.reinforcement.common.errors import FunctionLayerError
from dashscope.finetune.reinforcement.common.model import (
    AgenticRLFunctionComponent,
)
from dashscope.finetune.reinforcement.common.model_types import (
    FunctionType,
    StatusType,
)


class _MockErrorResponse:
    """Minimal stand-in for an aiohttp.ClientResponse carrying an error."""

    def __init__(self, status, content_type, payload=None, text=""):
        self.status = status
        self.content_type = content_type
        self.headers = {"Content-Type": content_type}
        self._payload = payload
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


def _make_request(stream=False):
    return AioHttpRequest(
        url="https://dashscope.aliyuncs.com/api/v1/x",
        api_key="sk-test",
        http_method="POST",
        stream=stream,
    )


async def _collect(request, response):
    return [rsp async for rsp in request._handle_response(response)]


class TestAioHttpErrorResponseIsAwaited:
    """P0-1: non-2xx responses must yield a real response object."""

    def test_json_error_yields_response_not_coroutine(self):
        response = _MockErrorResponse(
            status=400,
            content_type="application/json",
            payload={
                "code": "InvalidParameter",
                "message": "Model not exist.",
                "request_id": "req-1",
            },
        )
        results = asyncio.run(_collect(_make_request(), response))

        assert len(results) == 1
        rsp = results[0]
        assert isinstance(rsp, DashScopeAPIResponse)
        assert not asyncio.iscoroutine(rsp)
        assert rsp.status_code == 400
        assert rsp.code == "InvalidParameter"
        assert rsp.message == "Model not exist."
        assert rsp.request_id == "req-1"

    def test_json_error_without_code_falls_back(self):
        response = _MockErrorResponse(
            status=500,
            content_type="application/json",
            payload={"request_id": "req-2"},
        )
        rsp = asyncio.run(_collect(_make_request(), response))[0]

        assert isinstance(rsp, DashScopeAPIResponse)
        assert rsp.code == INTERNAL_ERROR.error_code
        assert rsp.message == INTERNAL_ERROR.format_msg()

    def test_non_json_error_yields_response(self):
        response = _MockErrorResponse(
            status=502,
            content_type="text/html",
            text="<html>Bad Gateway</html>",
        )
        rsp = asyncio.run(_collect(_make_request(), response))[0]

        assert isinstance(rsp, DashScopeAPIResponse)
        assert rsp.status_code == 502
        assert rsp.code == INTERNAL_ERROR.error_code
        assert rsp.message == "<html>Bad Gateway</html>"

    def test_no_unawaited_coroutine_warning(self):
        response = _MockErrorResponse(
            status=429,
            content_type="application/json",
            payload={"code": "Throttling", "message": "slow down"},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asyncio.run(_collect(_make_request(), response))

        assert not [
            w for w in caught if "never awaited" in str(w.message)
        ], "error response coroutine was not awaited"

    def test_registry_has_no_external_field(self):
        """Guard the assumption behind the P0-2 fix."""
        for err_def in (
            SDK_AGENTIC_RL_REGISTRATION_ERROR,
            SDK_AGENTIC_RL_FUNCTION_LAYER_ERROR,
            SDK_AGENTIC_RL_BASE_CONNECTION_ERROR,
            SDK_AGENTIC_RL_FUNCTION_LOAD_ERROR,
            SDK_AGENTIC_RL_INSTANCE_WARMUP_ERROR,
            SDK_AGENTIC_RL_INSTANCE_QUERY_ERROR,
        ):
            assert not hasattr(err_def, "external")


def _make_component(**kwargs):
    return AgenticRLFunctionComponent(
        type=FunctionType.ROLLOUT,
        name="test_fn",
        **kwargs,
    )


def _assert_failed(rsp, expected_name):
    """A FAILED ResponseFC carrying the real diagnostic, not AttributeError."""
    assert rsp.status.task == StatusType.FAILED
    assert rsp.status.name == expected_name
    assert rsp.status.code == INTERNAL_ERROR.status_code
    assert "{" not in rsp.status.message, "unfilled placeholder in message"
    assert rsp.status.message


class TestFunctionComponentFailurePaths:
    """P0-2: every deployment-failure path returns a structured response."""

    def test_query_with_empty_instance_id(self):
        rsp = asyncio.run(AgenticRLFunctionComponent.query(""))

        _assert_failed(rsp, SDK_AGENTIC_RL_INSTANCE_QUERY_ERROR.name)
        assert "No instance ID available" in rsp.status.message
        assert rsp.output == {"instance_id": ""}

    def test_load_without_entity_id(self):
        rsp = asyncio.run(_make_component().load())

        _assert_failed(rsp, SDK_AGENTIC_RL_FUNCTION_LOAD_ERROR.name)
        assert "No valid registration ID provided" in rsp.status.message

    def test_register_generic_failure(self):
        component = _make_component()
        component.fcmodel = MagicMock()
        component.fcmodel.generate_id.side_effect = RuntimeError("oss down")

        rsp = asyncio.run(component.register())

        _assert_failed(rsp, SDK_AGENTIC_RL_REGISTRATION_ERROR.name)
        assert "oss down" in rsp.status.message

    def test_register_function_layer_failure(self):
        component = _make_component()
        component.fcmodel = MagicMock()
        component.fcmodel.generate_id.side_effect = FunctionLayerError(
            "layer build failed",
        )

        rsp = asyncio.run(component.register())

        _assert_failed(rsp, SDK_AGENTIC_RL_FUNCTION_LAYER_ERROR.name)
        assert "layer build failed" in rsp.status.message

    def test_register_fc_call_failure(self):
        component = _make_component()
        fcmodel = MagicMock()
        fcmodel.classpath = ""
        fcmodel.get_oss = AsyncMock()
        fcmodel.to_oss = AsyncMock()
        fcmodel.oss_id = "oss-id-1"
        fcmodel.filepath = "/pkg/rollout.py"
        fcmodel.classname = "RolloutFn"
        fcmodel.oss_signed_url = "https://oss/x.zip"
        component.fcmodel = fcmodel

        with patch(
            "dashscope.finetune.reinforcement.common.model.client_fc",
            new=AsyncMock(side_effect=ConnectionError("fc unreachable")),
        ), patch(
            "dashscope.finetune.reinforcement.common.model.FC_LAYER_USED",
            False,
        ):
            rsp = asyncio.run(component.register())

        _assert_failed(rsp, SDK_AGENTIC_RL_BASE_CONNECTION_ERROR.name)
        assert "fc unreachable" in rsp.status.message

    def test_load_warmup_failure(self):
        component = _make_component()
        component.entity_id = "entity-1"
        component.instance_id = "inst-1"
        component.instance_url = "ftp://not-http"
        component.instance_token = "tok"

        with patch(
            "dashscope.finetune.reinforcement.common.model.client_fc",
            new=AsyncMock(
                return_value={
                    "output": {
                        "instanceId": "inst-1",
                        "trigger_url": "ftp://not-http",
                        "trigger_token": "tok",
                    },
                },
            ),
        ), patch(
            "dashscope.finetune.reinforcement.common.model.FC_LAYER_USED",
            False,
        ):
            rsp = asyncio.run(component.load(warmup=True))

        _assert_failed(rsp, SDK_AGENTIC_RL_INSTANCE_WARMUP_ERROR.name)
        assert "Invalid instance URL format" in rsp.status.message
        assert rsp.output == {"instance_id": "inst-1"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
