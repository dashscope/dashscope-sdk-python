# -*- coding: utf-8 -*-
"""Test cases for dashscope/cli/common.py error handling improvements."""
import pytest
from http import HTTPStatus
from unittest.mock import Mock, patch
import typer

from dashscope.cli.common import (
    print_failed_message,
    ensure_ok,
    handle_sdk_error,
)
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.common.error import DashScopeException, AuthenticationError


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
        assert captured.err.count("Failed") == 1

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
        assert "Business Error" in normalized_err
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
        # Should show improved fallback message (normalize all whitespace for Rich formatting)
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
            output={"code": "AsyncTaskPending", "message": "Task is processing"},
        )
        
        # Should not raise even though there's a code in output
        result = ensure_ok(rsp, check_business_error=False)
        assert result == {"code": "AsyncTaskPending", "message": "Task is processing"}

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
        assert "AuthFailed" in captured.err

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
