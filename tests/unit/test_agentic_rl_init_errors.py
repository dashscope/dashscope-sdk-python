# -*- coding: utf-8 -*-
"""Tests for AgenticRL.init() error reporting and the shared error helpers.

``init()`` used to let ``IOErrorWithCode`` escape untouched, so a caller who
passed a bad config path got an exception with no ``status_code`` and no public
error code. The log-and-classify sequence was also copy-pasted into five other
methods; these tests pin the extracted helpers so the call sites can stay short.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest

from dashscope.common.error import DashScopeException, InvalidParameter
from dashscope.common.error_registry import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    PERMISSION_DENIED,
    REQUEST_TIMEOUT,
)
from dashscope.finetune import agentic_rl
from dashscope.finetune.agentic_rl import (
    AgenticRL,
    _log_internal_error,
    _public_exception,
)
from dashscope.finetune.reinforcement import FunctionType, HTTP_REQUEST_TIMEOUT
from dashscope.finetune.reinforcement.common.errors import (
    IOErrorWithCode,
    RuntimeErrorWithCode,
)

_MODULE = "dashscope.finetune.agentic_rl"


def _agent():
    """Build an AgenticRL without running ``__init__`` (which needs a key)."""
    return AgenticRL.__new__(AgenticRL)


class TestInitReportsConfigProblemsAsBadRequest:
    def test_missing_config_file_has_a_status_code(self):
        agent = _agent()

        with pytest.raises(DashScopeException) as exc_info:
            agent.init("/nonexistent/agentic-rl-config.yaml")

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == INVALID_REQUEST.error_code

    def test_missing_config_file_is_not_an_internal_error(self):
        """A path the caller got wrong must not look like a server failure."""
        agent = _agent()

        with pytest.raises(DashScopeException) as exc_info:
            agent.init("/nonexistent/agentic-rl-config.yaml")

        assert exc_info.value.status_code != INTERNAL_ERROR.status_code

    def test_missing_config_file_is_an_invalid_parameter(self):
        agent = _agent()

        with pytest.raises(InvalidParameter):
            agent.init("/nonexistent/agentic-rl-config.yaml")

    def test_message_names_the_file_that_failed(self, tmp_path):
        missing = str(tmp_path / "gone.yaml")
        agent = _agent()

        with pytest.raises(DashScopeException) as exc_info:
            agent.init(missing)

        assert missing in str(exc_info.value)

    def test_original_io_error_is_chained(self):
        agent = _agent()

        with pytest.raises(DashScopeException) as exc_info:
            agent.init("/nonexistent/agentic-rl-config.yaml")

        assert isinstance(exc_info.value.__cause__, IOErrorWithCode)

    def test_malformed_config_is_also_a_bad_request(self, tmp_path):
        """``load_from_dict`` reports unknown keys through IOErrorWithCode."""
        config = tmp_path / "bad.yaml"
        config.write_text("not_a_real_field: 1\n", encoding="utf-8")
        agent = _agent()

        with pytest.raises(DashScopeException) as exc_info:
            agent.init(str(config))

        assert exc_info.value.status_code == 400
        assert exc_info.value.error_code == INVALID_REQUEST.error_code

    def test_empty_config_path_is_a_bad_request(self):
        """``init()`` with no argument must not surface an unwrapped IOError."""
        agent = _agent()

        with pytest.raises(DashScopeException) as exc_info:
            agent.init()

        assert exc_info.value.status_code == 400

    @patch(f"{_MODULE}.logger")
    def test_failure_is_logged_as_a_configuration_error(self, mock_logger):
        agent = _agent()

        with pytest.raises(DashScopeException):
            agent.init("/nonexistent/agentic-rl-config.yaml")

        mock_logger.error.assert_called_once()
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.ConfigurationError"
        )

    @patch(f"{_MODULE}.logger")
    def test_log_message_has_no_unfilled_placeholder(self, mock_logger):
        agent = _agent()

        with pytest.raises(DashScopeException):
            agent.init("/nonexistent/agentic-rl-config.yaml")

        rendered = mock_logger.error.call_args[0][2]
        assert "{" not in rendered
        assert rendered == "Invalid system configuration detected."


class _Collect(logging.Handler):
    """Collects records straight off the SDK logger.

    ``caplog`` attaches at the root, and the SDK logger does not propagate, so
    it sees nothing here.
    """

    def __init__(self, sink):
        super().__init__(level=logging.ERROR)
        self._sink = sink

    def emit(self, record):
        self._sink.append(record)


class TestLogInternalError:
    def test_keeps_the_positional_log_shape(self):
        error_def = MagicMock()
        error_def.name = "sdk.agentic_rl.SomeFailure"
        error_def.format_message.return_value = "rendered message"
        cause = RuntimeErrorWithCode("boom")

        _log_internal_error(error_def, cause)

        args, kwargs = error_def.format_message.call_args
        assert error_def.name == "sdk.agentic_rl.SomeFailure"
        assert args == ({"inner_code": "sdk.agentic_rl.RuntimeErrorWithCode"},)
        assert kwargs == {}

    def test_inner_code_falls_back_to_unknown(self):
        mock_logger = MagicMock()
        with patch(f"{_MODULE}.logger", mock_logger):
            _log_internal_error(MagicMock(name="def"), ValueError("boom"))

        assert mock_logger.error.call_args[0][3] == "unknown"

    def test_extra_vars_reach_the_rendered_message(self):
        error_def = MagicMock()
        error_def.format_message.return_value = "timed out after 300 seconds"
        cause = RuntimeErrorWithCode("boom")

        with patch(f"{_MODULE}.logger"):
            _log_internal_error(error_def, cause, {"timeout": "300"})

        variables = error_def.format_message.call_args[0][0]
        assert variables["timeout"] == "300"
        assert "inner_code" in variables

    def test_cause_is_logged_with_a_traceback(self):
        mock_logger = MagicMock()
        cause = RuntimeErrorWithCode("boom")

        with patch(f"{_MODULE}.logger", mock_logger):
            _log_internal_error(MagicMock(), cause)

        assert mock_logger.error.call_args[0][4] is cause
        assert mock_logger.error.call_args.kwargs["exc_info"] is True

    def test_record_is_attributed_to_the_caller(self):
        """Every handler logs through this helper, so without the stacklevel
        the record's location would be the same line for all five."""
        error_def = MagicMock()
        error_def.name = "sdk.agentic_rl.SomeFailure"
        error_def.format_message.return_value = "rendered"
        records = []
        handler = _Collect(records)

        agentic_rl.logger.addHandler(handler)
        try:
            _log_internal_error(error_def, RuntimeErrorWithCode("boom"))
        finally:
            agentic_rl.logger.removeHandler(handler)

        assert records[-1].funcName == (
            "test_record_is_attributed_to_the_caller"
        )


class TestPublicException:
    def test_bad_request_keeps_the_narrower_type(self):
        exc = _public_exception(INVALID_REQUEST, ValueError("bad payload"))

        assert isinstance(exc, InvalidParameter)
        assert exc.status_code == 400
        assert exc.error_code == INVALID_REQUEST.error_code

    @pytest.mark.parametrize(
        "public_error",
        [INTERNAL_ERROR, PERMISSION_DENIED, REQUEST_TIMEOUT],
    )
    def test_other_statuses_are_plain_dashscope_exceptions(self, public_error):
        exc = _public_exception(public_error, RuntimeError("boom"))

        assert type(exc) is DashScopeException
        assert exc.status_code == public_error.status_code
        assert exc.error_code == public_error.error_code

    def test_message_keeps_the_cause_summary(self):
        exc = _public_exception(
            INTERNAL_ERROR,
            RuntimeErrorWithCode("upload exploded"),
        )

        assert "RuntimeErrorWithCode" in str(exc)
        assert "upload exploded" in str(exc)

    def test_message_prefers_the_clean_message_attribute(self):
        """``str()`` on an AgenticRLError carries the registry prefix and the
        ``(caused by ...)`` tail; the caller-facing text should not."""
        cause = IOErrorWithCode(
            "Failed to load YAML file: /tmp/x.yaml",
            path="/tmp/x.yaml",
        )

        exc = _public_exception(INVALID_REQUEST, cause)

        assert "I/O error: Failed to load YAML file" in str(exc)
        assert "General I/O operation failure" not in str(exc)


class TestFunctionTestLogMessages:
    """Guards the rendered log text after the helper extraction."""

    async def _run(self, error):
        """Drive test_functions into ``error``; the caller asserts on it."""
        with patch(f"{_MODULE}.set_api_key"), patch(
            f"{_MODULE}.RolloutInput.model_validate",
            return_value=MagicMock(
                model_dump=MagicMock(return_value={}),
            ),
        ), patch(
            f"{_MODULE}.AgenticRLFunctionComponent.verify_function",
            side_effect=error,
        ):
            await AgenticRL.test_functions(
                "inst-1",
                FunctionType.ROLLOUT,
                {"prompt": "hi"},
            )

    @pytest.mark.asyncio
    async def test_timeout_renders_the_timeout_message(self):
        error = TimeoutError("too slow")

        with patch(f"{_MODULE}.logger") as mock_logger:
            with pytest.raises(DashScopeException):
                await self._run(error)

        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.FunctionTestTimeout"
        )
        rendered = mock_logger.error.call_args[0][2]
        assert rendered == (
            f"Function test timed out after {HTTP_REQUEST_TIMEOUT} seconds."
        )

    @pytest.mark.asyncio
    async def test_timeout_maps_to_504(self):
        with pytest.raises(DashScopeException) as exc_info:
            await self._run(TimeoutError("too slow"))

        assert exc_info.value.status_code == REQUEST_TIMEOUT.status_code

    @pytest.mark.asyncio
    async def test_failure_renders_the_inner_code(self):
        error = RuntimeErrorWithCode("deploy broke")

        with patch(f"{_MODULE}.logger") as mock_logger:
            with pytest.raises(DashScopeException):
                await self._run(error)

        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.FunctionTestFailed"
        )
        rendered = mock_logger.error.call_args[0][2]
        assert rendered == (
            "Function test failed: sdk.agentic_rl.RuntimeErrorWithCode."
        )
        assert "{" not in rendered
