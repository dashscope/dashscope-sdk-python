# -*- coding: utf-8 -*-
"""Tests for dashscope/cli/common.py error handling and exit-code tiers."""
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

from dashscope.cli import embeddings
from dashscope.cli.common import (
    _exit_code_for,
    _handle_exception,
    err_console,
    print_failed_message,
    ensure_ok,
    handle_sdk_error,
    set_verbose_errors,
)
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.common.error import AuthenticationError
from dashscope.finetune.reinforcement.common.errors import OutputError


class TestPrintFailedMessage:
    """Test print_failed_message with various response scenarios."""

    def test_complete_response(self, capsys):
        """Test with all fields present."""
        rsp = DashScopeAPIResponse(
            status_code=500,
            request_id="req_123",
            code="ServerError",
            message="Internal server error",
        )
        print_failed_message(rsp)
        captured = capsys.readouterr()
        assert "req_123" in captured.err
        assert "500" in captured.err
        assert "ServerError" in captured.err
        assert "Internal server error" in captured.err

    def test_missing_request_id(self, capsys):
        """Test when request_id is missing."""
        rsp = Mock()
        rsp.status_code = 400
        rsp.code = "BadRequest"
        rsp.message = "Invalid parameter"
        # Simulate missing request_id attribute
        del rsp.request_id

        print_failed_message(rsp)
        captured = capsys.readouterr()
        # Missing request_id should not be displayed (empty fields are omitted)
        assert "request_id:" not in captured.err
        # Error code is kept in original camelCase format
        assert "BadRequest" in captured.err
        assert "400" in captured.err

    def test_empty_code_and_message(self, capsys):
        """Test when code and message are empty strings."""
        rsp = DashScopeAPIResponse(
            status_code=503,
            request_id="req_456",
            code="",
            message="",
        )
        print_failed_message(rsp)
        captured = capsys.readouterr()
        # Should not show empty code/message fields
        # Use word boundary check: "code: " with space after colon
        assert ", code: " not in captured.err
        assert ", message: " not in captured.err
        assert "req_456" in captured.err

    def test_none_attributes(self, capsys):
        """Test when attributes are None."""
        rsp = Mock()
        rsp.status_code = 502
        rsp.request_id = None
        rsp.code = None
        rsp.message = None

        print_failed_message(rsp)
        captured = capsys.readouterr()
        # None values should not be displayed
        assert ", request_id:" not in captured.err
        assert ", code: " not in captured.err
        assert ", message: " not in captured.err
        assert "502" in captured.err


class TestEnsureOk:
    """Test ensure_ok with various response scenarios."""

    def test_successful_response(self):
        """Test with successful HTTP 200 and no business error."""
        rsp = DashScopeAPIResponse(
            status_code=HTTPStatus.OK,
            request_id="req_ok",
            code="",
            message="",
            output={"result": "success"},
        )
        result = ensure_ok(rsp)
        assert result == {"result": "success"}

    def test_http_error(self, capsys):
        """Test with HTTP error status."""
        rsp = DashScopeAPIResponse(
            status_code=404,
            request_id="req_404",
            code="NotFound",
            message="Resource not found",
        )

        with pytest.raises(typer.Exit):
            ensure_ok(rsp)

        captured = capsys.readouterr()
        # Should only print once (not duplicated)
        # Error message now uses "Request failed" instead of "Failed"
        assert captured.err.count("Request failed") == 1

    def test_business_error_in_dict_output(self, capsys):
        """Test with HTTP 200 but business error in dict output."""
        rsp = DashScopeAPIResponse(
            status_code=HTTPStatus.OK,
            request_id="req_biz_err",
            code="",
            message="",
            output={"code": "InvalidParameter", "message": "Model not found"},
        )

        with pytest.raises(typer.Exit):
            ensure_ok(rsp)

        captured = capsys.readouterr()
        # Rich may wrap long lines and add extra spaces, normalize whitespace
        normalized_err = " ".join(captured.err.split())
        # Error code is kept in original camelCase format
        assert "InvalidParameter" in normalized_err
        assert "Model not found" in normalized_err

    def test_business_error_without_message(self, capsys):
        """Test business error without message field."""
        rsp = DashScopeAPIResponse(
            status_code=HTTPStatus.OK,
            request_id="req_no_msg",
            output={"code": "SomeError"},  # No message field
        )

        with pytest.raises(typer.Exit):
            ensure_ok(rsp)

        captured = capsys.readouterr()
        # Should show improved fallback message
        # (normalize all whitespace for Rich formatting)
        normalized_err = " ".join(captured.err.split())
        assert "API returned error code without message" in normalized_err

    def test_none_output(self, capsys):
        """Test when output is None despite HTTP 200."""
        rsp = DashScopeAPIResponse(
            status_code=HTTPStatus.OK,
            request_id="req_null",
            output=None,
        )

        with pytest.raises(typer.Exit):
            ensure_ok(rsp)

        captured = capsys.readouterr()
        assert "no output data" in captured.err

    def test_skip_business_error_check(self):
        """Test with check_business_error=False."""
        rsp = DashScopeAPIResponse(
            status_code=HTTPStatus.OK,
            output={
                "code": "AsyncTaskPending",
                "message": "Task is processing",
            },
        )

        # Should not raise even though there's a code in output
        result = ensure_ok(rsp, check_business_error=False)
        assert result == {
            "code": "AsyncTaskPending",
            "message": "Task is processing",
        }

    def test_object_output_with_error(self, capsys):
        """Test with object output containing error fields."""
        mock_output = Mock()
        mock_output.code = "ObjectError"
        mock_output.message = "Object-level error"

        rsp = DashScopeAPIResponse(
            status_code=HTTPStatus.OK,
            request_id="req_obj",
            output=mock_output,
        )

        with pytest.raises(typer.Exit):
            ensure_ok(rsp)

        captured = capsys.readouterr()
        assert "ObjectError" in captured.err


class TestHandleSdkError:
    """Test handle_sdk_error decorator."""

    def test_dashscope_exception_handling(self, capsys):
        """Test handling of DashScopeException."""

        @handle_sdk_error("Test action")
        def failing_function():
            # Create exception properly using __init__ with positional args
            exc = AuthenticationError()
            exc.request_id = "req_auth"
            exc.code = "AuthFailed"
            exc.message = "Invalid API key"
            raise exc

        with pytest.raises(typer.Exit):
            failing_function()

        captured = capsys.readouterr()
        assert "Test action" in captured.err
        assert "req_auth" in captured.err
        # Error code is kept in original camelCase format (exception type name)
        assert "AuthenticationError" in captured.err

    def test_generic_exception_handling(self, capsys):
        """Test handling of generic exceptions."""

        @handle_sdk_error("Generic test")
        def generic_failing_function():
            raise ValueError("Something went wrong")

        with pytest.raises(typer.Exit):
            generic_failing_function()

        captured = capsys.readouterr()
        assert "Generic test" in captured.err
        assert "Something went wrong" in captured.err

    def test_typer_exit_passthrough(self):
        """Test that typer.Exit is re-raised without modification."""

        @handle_sdk_error("Should not catch this")
        def intentional_exit():
            raise typer.Exit(code=2)

        with pytest.raises(typer.Exit) as exc_info:
            intentional_exit()

        assert exc_info.value.exit_code == 2

    def test_successful_function_passthrough(self):
        """Test that successful functions work normally."""

        @handle_sdk_error("Success test")
        def success_function():
            return "success"

        result = success_function()
        assert result == "success"


class TestHandleException:
    """Test _handle_exception error code, request id, and verbose traceback."""

    @pytest.fixture(autouse=True)
    def _reset_verbose(self):
        yield
        set_verbose_errors(False)

    @staticmethod
    def _fail(exc, action="Call failed"):
        """Report *exc* the way a real handler does: raised, then caught."""
        try:
            raise exc
        except Exception as raised:
            _handle_exception(raised, action, err_console)

    def test_registry_error_code_is_reported(self, capsys):
        """RL exceptions subclass plain Exception but carry error_code."""
        exc = OutputError("job rejected", response={"request_id": "req-rl"})

        with pytest.raises(typer.Exit) as exc_info:
            self._fail(exc, "Run failed")

        captured = capsys.readouterr()
        assert exc_info.value.exit_code == 1
        assert "sdk.agentic_rl.OutputError" in captured.err
        assert "req-rl" in captured.err
        assert "job rejected" in captured.err

    def test_request_id_read_from_response_dict(self, capsys):
        """request_id is attached to .response, not set as an attribute."""
        exc = OutputError("bad output")
        assert getattr(exc, "request_id", None) is None
        exc.response = {"request_id": "req-from-resp"}

        with pytest.raises(typer.Exit):
            self._fail(exc, "Run failed")

        assert "req-from-resp" in capsys.readouterr().err

    def test_core_exception_uses_class_name(self, capsys):
        """Core DashScopeException subclasses carry no error_code."""
        exc = AuthenticationError()
        exc.request_id = "req-core"

        with pytest.raises(typer.Exit):
            self._fail(exc)

        captured = capsys.readouterr()
        assert "AuthenticationError" in captured.err
        assert "req-core" in captured.err

    def test_plain_exception_has_no_code_block(self, capsys):
        with pytest.raises(typer.Exit):
            self._fail(ValueError("boom"))

        captured = capsys.readouterr()
        assert "boom" in captured.err
        assert "code:" not in captured.err

    def test_verbose_prints_traceback(self, capsys):
        set_verbose_errors(True)

        with pytest.raises(typer.Exit):
            self._fail(ValueError("boom"))

        assert "Traceback" in capsys.readouterr().err

    def test_without_verbose_no_traceback(self, capsys):
        with pytest.raises(typer.Exit):
            self._fail(ValueError("boom"))

        assert "Traceback" not in capsys.readouterr().err

    def test_decorator_shares_structured_reporting(self, capsys):
        @handle_sdk_error("Submit")
        def failing():
            raise OutputError("nope", response={"request_id": "req-deco"})

        with pytest.raises(typer.Exit):
            failing()

        captured = capsys.readouterr()
        assert "sdk.agentic_rl.OutputError" in captured.err
        assert "req-deco" in captured.err


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rsp(status_code, code="", message="", output=None, request_id="req-test"):
    return SimpleNamespace(
        status_code=status_code,
        code=code,
        message=message,
        output=output,
        request_id=request_id,
        usage=None,
    )


def _ok_rsp(output=None):
    return _rsp(200, output=output or {"embeddings": []})


# ---------------------------------------------------------------------------
# _exit_code_for — pure unit tests
# ---------------------------------------------------------------------------


class TestExitCodeFor:
    def test_server_error_500(self):
        assert _exit_code_for(_rsp(500)) == 1

    def test_server_error_503(self):
        assert _exit_code_for(_rsp(503)) == 1

    def test_auth_401(self):
        assert _exit_code_for(_rsp(401)) == 2

    def test_auth_403(self):
        assert _exit_code_for(_rsp(403)) == 2

    def test_param_400(self):
        assert _exit_code_for(_rsp(400)) == 3

    def test_param_422(self):
        assert _exit_code_for(_rsp(422)) == 3

    def test_rate_limit_429(self):
        assert _exit_code_for(_rsp(429)) == 4

    def test_business_auth_unauthorized(self):
        assert _exit_code_for(_rsp(200, code="Unauthorized")) == 2

    def test_business_auth_access_denied(self):
        assert _exit_code_for(_rsp(200, code="AccessDenied")) == 2

    def test_business_auth_failure(self):
        assert _exit_code_for(_rsp(200, code="AuthFailure")) == 2

    def test_business_param_invalid(self):
        assert _exit_code_for(_rsp(200, code="InvalidParameter")) == 3

    def test_business_param_bad_request(self):
        assert _exit_code_for(_rsp(200, code="BadRequest")) == 3

    def test_fallback_unknown(self):
        assert _exit_code_for(_rsp(200, code="SomeUnknownError")) == 1


# ---------------------------------------------------------------------------
# ensure_ok — via embeddings CLI app (uses ensure_ok internally)
# ---------------------------------------------------------------------------

_EMBED_ARGS = [
    "create",
    "--model",
    "text-embedding-v3",
    "--input",
    "hello",
]


class TestEnsureOkViaCli:
    runner = CliRunner()

    def _invoke(self, mock_rsp, monkeypatch):
        monkeypatch.setattr(
            embeddings.dashscope.TextEmbedding,
            "call",
            lambda **_: mock_rsp,
        )
        return self.runner.invoke(embeddings.app, _EMBED_ARGS)

    # exit 0 — success
    def test_exit_0_success(self, monkeypatch):
        out = {"embeddings": [{"text_index": 0, "embedding": [0.1]}]}
        rsp = _rsp(200, output=out)
        r = self._invoke(rsp, monkeypatch)
        assert r.exit_code == 0

    # exit 1 — server error
    def test_exit_1_http_500(self, monkeypatch):
        r = self._invoke(_rsp(500), monkeypatch)
        assert r.exit_code == 1

    def test_exit_1_null_output(self, monkeypatch):
        r = self._invoke(_rsp(200, output=None), monkeypatch)
        assert r.exit_code == 1

    # exit 2 — auth error
    def test_exit_2_http_401(self, monkeypatch):
        r = self._invoke(_rsp(401), monkeypatch)
        assert r.exit_code == 2

    def test_exit_2_http_403(self, monkeypatch):
        r = self._invoke(_rsp(403), monkeypatch)
        assert r.exit_code == 2

    def test_exit_2_business_unauthorized(self, monkeypatch):
        rsp = _rsp(
            200,
            code="Unauthorized",
            output={
                "embeddings": [],
                "code": "Unauthorized",
                "message": "bad key",
            },
        )
        r = self._invoke(rsp, monkeypatch)
        assert r.exit_code == 2

    # exit 3 — param error
    def test_exit_3_http_400(self, monkeypatch):
        r = self._invoke(_rsp(400), monkeypatch)
        assert r.exit_code == 3

    def test_exit_3_http_422(self, monkeypatch):
        r = self._invoke(_rsp(422), monkeypatch)
        assert r.exit_code == 3

    def test_exit_3_business_invalid_param(self, monkeypatch):
        rsp = _rsp(
            200,
            code="InvalidParameter",
            output={
                "embeddings": [],
                "code": "InvalidParameter",
                "message": "bad param",
            },
        )
        r = self._invoke(rsp, monkeypatch)
        assert r.exit_code == 3

    # exit 4 — rate limit
    def test_exit_4_http_429(self, monkeypatch):
        r = self._invoke(_rsp(429), monkeypatch)
        assert r.exit_code == 4
