# -*- coding: utf-8 -*-
"""Tests for AgenticRL error handling with standard SDK exceptions.

Validates that AgenticRL methods raise correct standard SDK exceptions
(AuthenticationError, InvalidParameter, DashScopeException) with proper
status_code and error_code attributes in error scenarios.
"""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    DashScopeException,
)
from dashscope.common.error_registry import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    PERMISSION_DENIED,
    REQUEST_TIMEOUT,
)
from dashscope.finetune.agentic_rl import AgenticRL, _get_public_error
from dashscope.finetune.reinforcement import FunctionType
from dashscope.finetune.reinforcement import HTTP_REQUEST_TIMEOUT
from dashscope.finetune.reinforcement.common.errors import (
    BasePermissionError,
    ConfigurationError,
    InputError,
    ValidationError,
)


class TestInitAuthenticationError:
    """Test __init__ raises AuthenticationError for invalid API key."""

    @patch("dashscope.finetune.agentic_rl.set_api_key")
    def test_init_invalid_api_key_raises_authentication_error(
        self,
        mock_set_api_key,
    ):
        """__init__ should raise AuthenticationError when set_api_key fails."""
        mock_set_api_key.side_effect = ValueError("Invalid API key")

        with pytest.raises(AuthenticationError) as exc_info:
            AgenticRL(api_key="invalid-key")

        assert exc_info.value.status_code == 401
        assert exc_info.value.error_code == "AuthenticationError"
        assert "Incorrect API key" in str(exc_info.value)


class TestSubmitJobInvalidParameter:
    """Test submit_job raises InvalidParameter for duplicate function names."""

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    def test_submit_job_duplicate_function_names_raises_invalid_parameter(
        self,
        _mock_random_id,
        _mock_parent_init,
    ):
        """submit_job should raise InvalidParameter when
        check_function_names returns False."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = False
        agent.tuning.find_function_name_problems.return_value = (
            ["rollout_fn"],
            [],
        )
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(InvalidParameter) as exc_info:
            agent.submit_job()

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "BadRequestError"
        assert "request is invalid" in str(exc_info.value).lower()


class TestRegisterFunctionsDashScopeException:
    """Test register_functions raises DashScopeException on failure."""

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    async def test_register_functions_failure_raises_dashscope_exception(
        self,
        _mock_parent_init,
    ):
        """register_functions should raise DashScopeException
        when internal call fails."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.register_functions = AsyncMock(
            side_effect=RuntimeError("Registration failed"),
        )

        with pytest.raises(DashScopeException) as exc_info:
            await agent.register_functions()

        assert exc_info.value.status_code == 500
        assert exc_info.value.error_code == "InternalServerError"
        assert "internal error" in str(exc_info.value).lower()


class TestUploadDatasetsDashScopeException:
    """Test upload_datasets raises DashScopeException on failure."""

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    async def test_upload_datasets_failure_raises_dashscope_exception(
        self,
        _mock_parent_init,
    ):
        """upload_datasets should raise DashScopeException
        when upload fails."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.upload_datasets = AsyncMock(
            side_effect=IOError("Upload failed"),
        )

        with pytest.raises(DashScopeException) as exc_info:
            await agent.upload_datasets()

        assert exc_info.value.status_code == 500
        assert exc_info.value.error_code == "InternalServerError"


class TestSubmitJobCallFailure:
    """Test submit_job raises DashScopeException when API call fails."""

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    @patch("dashscope.finetune.agentic_rl.CreateMixin.call")
    def test_submit_job_call_failure_raises_dashscope_exception(
        self,
        mock_call,
        _mock_random_id,
        _mock_parent_init,
    ):
        """submit_job should raise DashScopeException
        when super().call() fails."""
        mock_call.side_effect = RuntimeError("API call failed")

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = True
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(DashScopeException) as exc_info:
            agent.submit_job()

        assert exc_info.value.status_code == 500
        assert exc_info.value.error_code == "InternalServerError"


class TestRunFailure:
    """Test run raises DashScopeException on failure."""

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    async def test_run_failure_raises_dashscope_exception(
        self,
        _mock_parent_init,
    ):
        """run should raise DashScopeException when any step fails."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        object.__setattr__(
            agent,
            "register_functions",
            AsyncMock(
                side_effect=RuntimeError("Step failed"),
            ),
        )

        with pytest.raises(DashScopeException) as exc_info:
            await agent.run()

        assert exc_info.value.status_code == 500
        assert exc_info.value.error_code == "InternalServerError"


class TestTestFunctionsInvalidParameter:
    """Test test_functions raises InvalidParameter for
    unsupported types and validation failures."""

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    async def test_test_functions_unsupported_type_raises_invalid_parameter(
        self,
        _mock_set_api_key,
    ):
        """test_functions should raise InvalidParameter
        for unsupported FunctionType."""
        with pytest.raises(InvalidParameter) as exc_info:
            await AgenticRL.test_functions(
                instance_id="inst-123",
                functype=MagicMock(spec=FunctionType),  # Unsupported type
                input_data={},
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "BadRequestError"

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @patch("dashscope.finetune.agentic_rl.RolloutInput.model_validate")
    async def test_test_functions_validation_failure_raises_invalid_parameter(
        self,
        mock_validate,
        _mock_set_api_key,
    ):
        """test_functions should raise InvalidParameter
        when validation fails."""
        mock_validate.side_effect = ValueError("Validation failed")

        with pytest.raises(InvalidParameter) as exc_info:
            await AgenticRL.test_functions(
                instance_id="inst-123",
                functype=FunctionType.ROLLOUT,
                input_data={"invalid": "data"},
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "BadRequestError"


class TestDashScopeExceptionPassthrough:
    """Test that DashScopeException from API is passed through unchanged."""

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    @patch("dashscope.finetune.agentic_rl.CreateMixin.call")
    def test_submit_job_passthrough_dashscope_exception(
        self,
        mock_call,
        _mock_random_id,
        _mock_parent_init,
    ):
        """submit_job should pass through DashScopeException
        from API unchanged."""
        original_exc = DashScopeException("API error")
        original_exc.status_code = 429
        original_exc.error_code = "RateLimitError"
        original_exc.request_id = "req-123"
        mock_call.side_effect = original_exc

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = True
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(DashScopeException) as exc_info:
            agent.submit_job()

        # Should be the exact same exception object
        assert exc_info.value is original_exc
        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == "RateLimitError"
        assert exc_info.value.request_id == "req-123"

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    async def test_register_functions_passthrough_dashscope_exception(
        self,
        _mock_parent_init,
    ):
        """register_functions should pass through
        DashScopeException unchanged."""
        original_exc = DashScopeException("Service error")
        original_exc.status_code = 503
        original_exc.error_code = "ServiceUnavailableError"

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.register_functions = AsyncMock(side_effect=original_exc)

        with pytest.raises(DashScopeException) as exc_info:
            await agent.register_functions()

        assert exc_info.value is original_exc
        assert exc_info.value.status_code == 503
        assert exc_info.value.error_code == "ServiceUnavailableError"

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    async def test_run_passthrough_dashscope_exception(
        self,
        _mock_parent_init,
    ):
        """run should pass through DashScopeException
        from inner calls unchanged."""
        original_exc = DashScopeException("Inner error")
        original_exc.status_code = 400
        original_exc.error_code = "BadRequestError"

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        object.__setattr__(
            agent,
            "register_functions",
            AsyncMock(side_effect=original_exc),
        )

        with pytest.raises(DashScopeException) as exc_info:
            await agent.run()

        assert exc_info.value is original_exc
        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == "BadRequestError"


class TestExceptionChainPreserved:
    """Verify that __cause__ is correctly preserved in all error scenarios."""

    @patch("dashscope.finetune.agentic_rl.set_api_key")
    def test_init_preserves_cause(self, mock_set_api_key):
        """__init__ should preserve the original exception as __cause__."""
        original = ValueError("bad key")
        mock_set_api_key.side_effect = original

        with pytest.raises(AuthenticationError) as exc_info:
            AgenticRL(api_key="invalid")

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    async def test_register_functions_preserves_cause(self, _mock_parent_init):
        """register_functions should preserve __cause__
        for non-DashScopeException."""
        original = IOError("network down")
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.register_functions = AsyncMock(side_effect=original)

        with pytest.raises(DashScopeException) as exc_info:
            await agent.register_functions()

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    async def test_upload_datasets_preserves_cause(self, _mock_parent_init):
        """upload_datasets should preserve __cause__."""
        original = OSError("disk full")
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.upload_datasets = AsyncMock(side_effect=original)

        with pytest.raises(DashScopeException) as exc_info:
            await agent.upload_datasets()

        assert exc_info.value.__cause__ is original

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    @patch("dashscope.finetune.agentic_rl.CreateMixin.call")
    def test_submit_job_call_failure_preserves_cause(
        self,
        mock_call,
        _mock_random_id,
        _mock_parent_init,
    ):
        """submit_job should preserve __cause__ when API call fails."""
        original = TimeoutError("connection timeout")
        mock_call.side_effect = original

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = True
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(DashScopeException) as exc_info:
            agent.submit_job()

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    async def test_test_functions_validation_preserves_cause(
        self,
        _mock_set_api_key,
    ):
        """test_functions should preserve __cause__ on validation failure."""
        original = TypeError("wrong type")

        with patch(
            "dashscope.finetune.agentic_rl.RolloutInput.model_validate",
            side_effect=original,
        ):
            with pytest.raises(InvalidParameter) as exc_info:
                await AgenticRL.test_functions(
                    instance_id="inst-1",
                    functype=FunctionType.ROLLOUT,
                    input_data={"bad": "data"},
                )

        assert exc_info.value.__cause__ is original


class TestInternalLogCodes:
    """Verify each failure path logs its internal registry code."""

    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @patch("dashscope.finetune.agentic_rl.logger")
    def test_init_logs_invalid_api_key(self, mock_logger, mock_set_api_key):
        """__init__ should log sdk.InvalidApiKey."""
        mock_set_api_key.side_effect = ValueError("bad")

        with pytest.raises(AuthenticationError):
            AgenticRL(api_key="x")

        mock_logger.error.assert_called_once()
        # Logger uses placeholder format: "[%s] %s | %s",
        # name, message, exception
        assert mock_logger.error.call_args[0][1] == "sdk.InvalidApiKey"

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_register_functions_logs_registration_failed(
        self,
        mock_logger,
        _mock_parent_init,
    ):
        """register_functions should log
        sdk.agentic_rl.FunctionRegistrationFailed."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.register_functions = AsyncMock(
            side_effect=RuntimeError("fail"),
        )

        with pytest.raises(DashScopeException):
            await agent.register_functions()

        mock_logger.error.assert_called_once()
        # Logger uses placeholder format: "[%s] %s | %s",
        # name, message, exception
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.FunctionRegistrationFailed"
        )

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_upload_datasets_logs_upload_failed(
        self,
        mock_logger,
        _mock_parent_init,
    ):
        """upload_datasets should log
        sdk.agentic_rl.DatasetsUploadFailed."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.upload_datasets = AsyncMock(side_effect=IOError("fail"))

        with pytest.raises(DashScopeException):
            await agent.upload_datasets()

        mock_logger.error.assert_called_once()
        # Logger uses placeholder format: "[%s] %s | %s",
        # name, message, exception
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.DatasetsUploadFailed"
        )

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    @patch("dashscope.finetune.agentic_rl.logger")
    def test_submit_job_duplicate_names_logs_duplicate_names(
        self,
        mock_logger,
        _mock_random_id,
        _mock_parent_init,
    ):
        """submit_job duplicate names should log
        sdk.agentic_rl.DuplicateFunctionNames."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = False
        agent.tuning.find_function_name_problems.return_value = (
            ["rollout_fn"],
            [],
        )
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(InvalidParameter):
            agent.submit_job()

        mock_logger.error.assert_called_once()
        # Logger uses placeholder format: "[%s] %s", name, message
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.DuplicateFunctionNames"
        )
        rendered = mock_logger.error.call_args[0][2]
        assert "rollout_fn" in rendered
        assert "check_function_names() failed" not in rendered
        assert "{" not in rendered

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    @patch("dashscope.finetune.agentic_rl.logger")
    def test_submit_job_missing_name_is_not_reported_as_duplicate(
        self,
        mock_logger,
        _mock_random_id,
        _mock_parent_init,
    ):
        """check_function_names() also fails when a component carries no
        `name` attribute; that must not be logged as a duplication."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = False
        agent.tuning.find_function_name_problems.return_value = ([], [2])
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(InvalidParameter) as exc_info:
            agent.submit_job()

        assert exc_info.value.status_code == 400
        mock_logger.error.assert_called_once()
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.ValidationError"
        )
        assert "Duplicate" not in mock_logger.error.call_args[0][2]

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    @patch("dashscope.finetune.agentic_rl.CreateMixin.call")
    @patch("dashscope.finetune.agentic_rl.logger")
    def test_submit_job_call_failure_logs_submission_failed(
        self,
        mock_logger,
        mock_call,
        _mock_random_id,
        _mock_parent_init,
    ):
        """submit_job call failure should log
        sdk.agentic_rl.JobSubmissionFailed."""
        mock_call.side_effect = RuntimeError("API down")

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = True
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(DashScopeException):
            agent.submit_job()

        mock_logger.error.assert_called_once()
        # Logger uses placeholder format: "[%s] %s | %s",
        # name, message, exception
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.JobSubmissionFailed"
        )

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_run_logs_workflow_failed(
        self,
        mock_logger,
        _mock_parent_init,
    ):
        """run should log sdk.agentic_rl.WorkflowFailed on failure."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        object.__setattr__(
            agent,
            "register_functions",
            AsyncMock(side_effect=RuntimeError("step failed")),
        )

        with pytest.raises(DashScopeException):
            await agent.run()

        mock_logger.error.assert_called_once()
        # Logger uses placeholder format: "[%s] %s | %s",
        # name, message, exception
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.WorkflowFailed"
        )

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_test_functions_unsupported_type_logs_unsupported_type(
        self,
        mock_logger,
        _mock_set_api_key,
    ):
        """test_functions unsupported type should log
        sdk.agentic_rl.UnsupportedFunctionType."""
        with pytest.raises(InvalidParameter):
            await AgenticRL.test_functions(
                instance_id="inst-1",
                functype=MagicMock(spec=FunctionType),
                input_data={},
            )

        # Check all logged messages for the error name
        error_names = [call[0][1] for call in mock_logger.error.call_args_list]
        assert "sdk.agentic_rl.UnsupportedFunctionType" in error_names
        assert "sdk.agentic_rl.FunctionTestFailed" not in error_names

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_test_functions_validation_failure_logs_test_failed(
        self,
        mock_logger,
        _mock_set_api_key,
    ):
        """test_functions validation failure should log
        sdk.agentic_rl.FunctionTestFailed."""
        with patch(
            "dashscope.finetune.agentic_rl.RolloutInput.model_validate",
            side_effect=ValueError("bad input"),
        ):
            with pytest.raises(InvalidParameter):
                await AgenticRL.test_functions(
                    instance_id="inst-1",
                    functype=FunctionType.ROLLOUT,
                    input_data={"bad": "data"},
                )

        mock_logger.error.assert_called_once()
        # Logger uses placeholder format: "[%s] %s | %s",
        # name, message, exception
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.FunctionTestFailed"
        )

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_test_functions_timeout_logs_configured_timeout(
        self,
        mock_logger,
        _mock_set_api_key,
    ):
        """The {timeout} variable must carry the configured request timeout
        rather than the literal "unknown"."""
        with patch(
            "dashscope.finetune.agentic_rl.RolloutInput.model_validate",
            return_value=MagicMock(),
        ):
            with patch(
                "dashscope.finetune.agentic_rl."
                "AgenticRLFunctionComponent.verify_function",
                side_effect=TimeoutError("too slow"),
            ):
                with pytest.raises(DashScopeException) as exc_info:
                    await AgenticRL.test_functions(
                        instance_id="inst-1",
                        functype=FunctionType.ROLLOUT,
                        input_data={},
                    )

        assert exc_info.value.status_code == 504
        rendered = mock_logger.error.call_args[0][2]
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.FunctionTestTimeout"
        )
        assert f"after {HTTP_REQUEST_TIMEOUT} seconds" in rendered
        assert "unknown seconds" not in rendered
        assert "{" not in rendered


class TestPassthroughAttributeCompleteness:
    """Verify that passthrough DashScopeException preserves ALL attributes."""

    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch(
        "dashscope.finetune.agentic_rl.generate_random_id",
        return_value="abcd1234",
    )
    @patch("dashscope.finetune.agentic_rl.CreateMixin.call")
    def test_submit_job_passthrough_preserves_all_attributes(
        self,
        mock_call,
        _mock_random_id,
        _mock_parent_init,
    ):
        """Passthrough should preserve status_code,
        error_code, request_id, and message."""
        original = DashScopeException("Rate limited by server")
        original.status_code = 429
        original.error_code = "RateLimitError"
        original.request_id = "req-abc-123"
        mock_call.side_effect = original

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.name = "test-job"
        agent.tuning.model.name = "test-model"
        agent.tuning.check_function_names.return_value = True
        agent.tuning.datasets = []
        agent.tuning.combine_ids_runtimes.return_value = []
        agent.tuning.training.hyper_parameters = None
        agent.tuning.training.resources = None
        agent.tuning.training.type = "rl"

        with pytest.raises(DashScopeException) as exc_info:
            agent.submit_job()

        assert exc_info.value is original
        assert exc_info.value.status_code == 429
        assert exc_info.value.error_code == "RateLimitError"
        assert exc_info.value.request_id == "req-abc-123"
        assert "Rate limited" in str(exc_info.value)


class TestMultipleUnderlyingExceptionTypes:
    """Verify correct behavior with various underlying exception types."""

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @pytest.mark.parametrize(
        "underlying_exc, expected_status, expected_code",
        [
            (IOError("disk error"), 500, "InternalServerError"),
            (TimeoutError("timed out"), 504, "GatewayTimeoutError"),
            (
                ConnectionError("connection refused"),
                500,
                "InternalServerError",
            ),
            (PermissionError("access denied"), 500, "InternalServerError"),
            (MemoryError("out of memory"), 500, "InternalServerError"),
        ],
    )
    async def test_register_functions_handles_various_exceptions(
        self,
        _mock_parent_init,
        underlying_exc,
        expected_status,
        expected_code,
    ):
        """register_functions should convert various
        exception types to DashScopeException."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.register_functions = AsyncMock(side_effect=underlying_exc)

        with pytest.raises(DashScopeException) as exc_info:
            await agent.register_functions()

        assert exc_info.value.status_code == expected_status
        assert exc_info.value.error_code == expected_code
        assert exc_info.value.__cause__ is underlying_exc

    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @pytest.mark.parametrize(
        "underlying_exc",
        [
            OSError("os error"),
            RuntimeError("runtime error"),
            SystemError("system error"),
        ],
    )
    async def test_upload_datasets_handles_various_exceptions(
        self,
        _mock_parent_init,
        underlying_exc,
    ):
        """upload_datasets should convert various
        exception types to DashScopeException."""
        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        agent.tuning.upload_datasets = AsyncMock(side_effect=underlying_exc)

        with pytest.raises(DashScopeException) as exc_info:
            await agent.upload_datasets()

        assert exc_info.value.status_code == 500
        assert exc_info.value.error_code == "InternalServerError"
        assert exc_info.value.__cause__ is underlying_exc


class TestPublicErrorMapping:
    """Lock the exception -> public-error mapping table.

    ``test_functions`` used to carry a second copy of this table and the two
    drifted apart; both now route through ``_get_public_error``.
    """

    @pytest.mark.parametrize(
        "underlying_exc, expected_error",
        [
            (BasePermissionError("denied"), PERMISSION_DENIED),
            (ConfigurationError("bad config"), INVALID_REQUEST),
            (ValidationError("bad payload"), INVALID_REQUEST),
            (InputError("bad field"), INVALID_REQUEST),
            (TimeoutError("too slow"), REQUEST_TIMEOUT),
            (OSError("io"), INTERNAL_ERROR),
            (RuntimeError("boom"), INTERNAL_ERROR),
            (ValueError("plain"), INTERNAL_ERROR),
            (TypeError("plain"), INTERNAL_ERROR),
        ],
    )
    def test_default_mapping(self, underlying_exc, expected_error):
        assert _get_public_error(underlying_exc) is expected_error

    @pytest.mark.parametrize(
        "underlying_exc",
        [ValueError("bad payload"), TypeError("bad payload")],
    )
    def test_input_errors_opt_in_to_400(self, underlying_exc):
        """A flow that validates a payload before sending it can classify a
        generic value/type failure as the caller's fault."""
        mapped = _get_public_error(
            underlying_exc,
            input_errors=(ValueError, TypeError),
        )
        assert mapped is INVALID_REQUEST

    def test_input_errors_do_not_override_permission(self):
        mapped = _get_public_error(
            BasePermissionError("denied"),
            input_errors=(BasePermissionError,),
        )
        assert mapped is PERMISSION_DENIED

    def test_input_errors_do_not_swallow_timeout(self):
        mapped = _get_public_error(
            TimeoutError("too slow"),
            input_errors=(ValueError, TypeError),
        )
        assert mapped is REQUEST_TIMEOUT

    @pytest.mark.asyncio
    @patch("dashscope.finetune.agentic_rl.set_api_key")
    @patch("dashscope.finetune.agentic_rl.RolloutInput.model_validate")
    @pytest.mark.parametrize(
        "underlying_exc, expected_status, expected_code",
        [
            (ValueError("bad payload"), 400, "BadRequestError"),
            (TypeError("bad payload"), 400, "BadRequestError"),
            (TimeoutError("too slow"), 504, "GatewayTimeoutError"),
            (OSError("io"), 500, "InternalServerError"),
        ],
    )
    async def test_functions_uses_the_shared_mapping(
        self,
        mock_validate,
        _mock_set_api_key,
        underlying_exc,
        expected_status,
        expected_code,
    ):
        mock_validate.side_effect = underlying_exc

        with pytest.raises(DashScopeException) as exc_info:
            await AgenticRL.test_functions(
                instance_id="inst-1",
                functype=FunctionType.ROLLOUT,
                input_data={},
            )

        assert exc_info.value.status_code == expected_status
        assert exc_info.value.error_code == expected_code
