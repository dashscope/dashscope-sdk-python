# -*- coding: utf-8 -*-
"""verify_function must not flatten service failures into a 400.

Both of its handlers wrapped every exception in ``ValidationError``, so a failed
instance query, a missing trigger URL or a request timeout all reached
``AgenticRL.test_functions`` as "invalid input" and were reported to the user as
a 400. The reason the deployed function itself reported was discarded the same
way, replaced by a generic "Function output validation failed".
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dashscope.common.error import DashScopeException, InvalidParameter
from dashscope.finetune.agentic_rl import AgenticRL
from dashscope.finetune.reinforcement import FunctionType
from dashscope.finetune.reinforcement.common.errors import (
    InstanceQueryError,
    OutputError,
    ValidationError,
)
from dashscope.finetune.reinforcement.common.model import (
    AgenticRLFunctionComponent,
    RolloutInput,
)
from dashscope.finetune.reinforcement.common.model_types import StatusType

_MODEL = "dashscope.finetune.reinforcement.common.model"


def _query_result(task, output=None):
    result = MagicMock()
    result.status.task = task
    result.output = output if output is not None else {}
    return result


def _instance_output(url="https://fc.example.com", token="tok"):
    return {"output": {"trigger_url": url, "trigger_token": token}}


class TestVerificationFailuresKeepTheirType:
    """The handler around instance lookup and the test request."""

    @pytest.mark.asyncio
    async def test_failed_instance_query_is_not_a_400(self):
        with patch.object(
            AgenticRLFunctionComponent,
            "query",
            new=AsyncMock(
                return_value=_query_result(StatusType.FAILED),
            ),
        ):
            with pytest.raises(InstanceQueryError):
                await AgenticRLFunctionComponent.verify_function(
                    MagicMock(spec=RolloutInput),
                    "inst-1",
                )

    @pytest.mark.asyncio
    async def test_missing_trigger_url_is_not_a_400(self):
        with patch.object(
            AgenticRLFunctionComponent,
            "query",
            new=AsyncMock(
                return_value=_query_result(
                    StatusType.SUCCEEDED,
                    _instance_output(url=""),
                ),
            ),
        ):
            with pytest.raises(OutputError):
                await AgenticRLFunctionComponent.verify_function(
                    MagicMock(spec=RolloutInput),
                    "inst-1",
                )

    @pytest.mark.asyncio
    async def test_request_timeout_stays_a_timeout(self):
        """A timeout flattened into ValidationError could never reach the 504
        branch in test_functions."""
        with patch.object(
            AgenticRLFunctionComponent,
            "query",
            new=AsyncMock(
                return_value=_query_result(
                    StatusType.SUCCEEDED,
                    _instance_output(),
                ),
            ),
        ):
            with patch(
                f"{_MODEL}.client_fc",
                new=AsyncMock(
                    side_effect=TimeoutError("Request timeout (600s)"),
                ),
            ):
                with pytest.raises(TimeoutError) as exc_info:
                    await AgenticRLFunctionComponent.verify_function(
                        MagicMock(spec=RolloutInput),
                        "inst-1",
                    )

        assert "600s" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unexpected_failure_is_still_wrapped(self):
        """The wrap is only removed for failures that classify themselves."""
        with patch.object(
            AgenticRLFunctionComponent,
            "query",
            new=AsyncMock(side_effect=RuntimeError("bug")),
        ):
            with pytest.raises(ValidationError) as exc_info:
                await AgenticRLFunctionComponent.verify_function(
                    MagicMock(spec=RolloutInput),
                    "inst-1",
                )

        assert "Function verification failed" in exc_info.value.message


class TestFunctionReportedReasonSurvives:
    """The handler around response validation."""

    @pytest.mark.asyncio
    async def test_deployed_function_reason_is_kept(self):
        response = {
            "status": {"code": 500, "message": "model not deployed"},
        }
        with patch.object(
            AgenticRLFunctionComponent,
            "query",
            new=AsyncMock(
                return_value=_query_result(
                    StatusType.SUCCEEDED,
                    _instance_output(),
                ),
            ),
        ):
            with patch(
                f"{_MODEL}.client_fc",
                new=AsyncMock(return_value=response),
            ):
                with pytest.raises(ValidationError) as exc_info:
                    await AgenticRLFunctionComponent.verify_function(
                        MagicMock(spec=RolloutInput),
                        "inst-1",
                    )

        assert "model not deployed" in exc_info.value.message


class TestTestFunctionsReportsServiceFailuresAs5xx:
    """End to end: what the user actually sees."""

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @pytest.mark.parametrize(
        "underlying, expected_status, expected_code, expected_type",
        [
            (
                InstanceQueryError("Status query failed"),
                500,
                "InternalServerError",
                DashScopeException,
            ),
            (
                OutputError("No instance url/token provided"),
                500,
                "InternalServerError",
                DashScopeException,
            ),
            (
                TimeoutError("Request timeout (600s)"),
                504,
                "GatewayTimeoutError",
                DashScopeException,
            ),
            (
                ValidationError("model not deployed"),
                400,
                "BadRequestError",
                InvalidParameter,
            ),
        ],
    )
    async def test_classification(
        self,
        _mock_set_api_key,
        underlying,
        expected_status,
        expected_code,
        expected_type,
    ):
        with patch(
            "dashscope.finetune.agentic_rl.RolloutInput.model_validate",
            return_value=MagicMock(),
        ):
            with patch(
                "dashscope.finetune.agentic_rl."
                "AgenticRLFunctionComponent.verify_function",
                new=AsyncMock(side_effect=underlying),
            ):
                with pytest.raises(expected_type) as exc_info:
                    await AgenticRL.test_functions(
                        instance_id="inst-1",
                        functype=FunctionType.ROLLOUT,
                        input_data={},
                    )

        assert exc_info.value.status_code == expected_status
        assert exc_info.value.error_code == expected_code
        assert exc_info.value.__cause__ is underlying

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_timeout_logs_the_timeout_code(
        self,
        mock_logger,
        _mock_set_api_key,
    ):
        """The FunctionTestTimeout registry entry was unreachable before."""
        with patch(
            "dashscope.finetune.agentic_rl.RolloutInput.model_validate",
            return_value=MagicMock(),
        ):
            with patch(
                "dashscope.finetune.agentic_rl."
                "AgenticRLFunctionComponent.verify_function",
                new=AsyncMock(
                    side_effect=TimeoutError("Request timeout (600s)"),
                ),
            ):
                with pytest.raises(DashScopeException):
                    await AgenticRL.test_functions(
                        instance_id="inst-1",
                        functype=FunctionType.ROLLOUT,
                        input_data={},
                    )

        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.FunctionTestTimeout"
        )

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    async def test_service_failure_message_names_the_cause(
        self,
        _mock_set_api_key,
    ):
        underlying = InstanceQueryError("Status query failed")
        with patch(
            "dashscope.finetune.agentic_rl.RolloutInput.model_validate",
            return_value=MagicMock(),
        ):
            with patch(
                "dashscope.finetune.agentic_rl."
                "AgenticRLFunctionComponent.verify_function",
                new=AsyncMock(side_effect=underlying),
            ):
                with pytest.raises(DashScopeException) as exc_info:
                    await AgenticRL.test_functions(
                        instance_id="inst-1",
                        functype=FunctionType.ROLLOUT,
                        input_data={},
                    )

        assert "InstanceQueryError" in str(exc_info.value)
