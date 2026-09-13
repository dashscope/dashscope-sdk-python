# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Failures must be reported with the status they actually mean.

Three classification defects: a 403 WebSocket handshake was reported as a 401
AuthenticationError, telling the caller to rotate a valid API key; the
``run()`` workflow log was unreachable because every inner failure arrives
already wrapped as a DashScopeException and was re-raised before logging; and
an async call silently ignored a caller-supplied session it could not reuse.
"""
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dashscope.api_entities import http_request as http_request_module
from dashscope.api_entities.http_request import HttpRequest
from dashscope.api_entities.websocket_request import _handshake_error
from dashscope.common.constants import HTTPMethod
from dashscope.common.error import DashScopeException
from dashscope.common.error_registry import (
    AUTH_FAILED,
    INTERNAL_ERROR,
    PERMISSION_DENIED,
    SERVICE_UNAVAILABLE,
)
from dashscope.finetune.agentic_rl import AgenticRL


class TestHandshakeStatusClassification:
    @pytest.mark.parametrize(
        "status, expected",
        [
            (HTTPStatus.UNAUTHORIZED, AUTH_FAILED),
            (HTTPStatus.FORBIDDEN, PERMISSION_DENIED),
            (HTTPStatus.SERVICE_UNAVAILABLE, SERVICE_UNAVAILABLE),
            (HTTPStatus.INTERNAL_SERVER_ERROR, INTERNAL_ERROR),
        ],
    )
    def test_classified_statuses(self, status, expected):
        assert _handshake_error(status) is expected

    def test_forbidden_is_not_reported_as_an_auth_failure(self):
        """A 403 must not carry a 401 status or the AuthenticationError
        code, or the caller is told to replace credentials that work."""
        denied = _handshake_error(HTTPStatus.FORBIDDEN)

        assert denied.status_code == 403
        assert denied.error_code == "PermissionDeniedError"

    @pytest.mark.parametrize("status", [HTTPStatus.BAD_REQUEST, 418])
    def test_unclassified_status_returns_none(self, status):
        """None means the caller re-raises instead of inventing a response."""
        assert _handshake_error(status) is None


class TestRunLogsWorkflowFailure:
    @pytest.mark.asyncio
    @patch(
        "dashscope.finetune.agentic_rl.AgenticRLTuning.__init__",
        return_value=None,
    )
    @patch("dashscope.finetune.agentic_rl.logger")
    async def test_passthrough_failure_is_still_logged(
        self,
        mock_logger,
        _mock_parent_init,
    ):
        """Inner calls always hand run() a DashScopeException, so logging
        after the passthrough check meant never logging at all."""
        original = DashScopeException("Inner error")
        original.status_code = 400
        original.error_code = "BadRequestError"

        agent = AgenticRL.__new__(AgenticRL)
        object.__setattr__(agent, "tuning", MagicMock())
        object.__setattr__(
            agent,
            "register_functions",
            AsyncMock(side_effect=original),
        )

        with pytest.raises(DashScopeException) as exc_info:
            await agent.run()

        assert exc_info.value is original
        assert (
            mock_logger.error.call_args[0][1]
            == "sdk.agentic_rl.WorkflowFailed"
        )
        rendered = mock_logger.error.call_args[0][2]
        assert "BadRequestError" in rendered
        assert "{" not in rendered


class TestAsyncCallReportsIgnoredSession:
    @pytest.mark.asyncio
    async def test_warns_when_the_supplied_session_cannot_be_reused(self):
        """A session that is not an aiohttp.ClientSession is stored for sync
        calls only; an async call must say it is falling back rather than
        leaving a custom connector silently unapplied."""
        request = HttpRequest(
            url="https://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=object(),
        )

        with patch.object(
            http_request_module,
            "get_shared_aio_session",
            new=AsyncMock(side_effect=RuntimeError("stop here")),
        ):
            with patch.object(http_request_module, "logger") as mock_logger:
                with pytest.raises(RuntimeError):
                    async for _ in request._handle_aio_request():
                        pass

        assert any(
            "ignoring the supplied" in call.args[0]
            and "object" in call.args[1:]
            for call in mock_logger.warning.call_args_list
            if call.args
        )

    @pytest.mark.asyncio
    async def test_no_warning_without_a_supplied_session(self):
        request = HttpRequest(
            url="https://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        with patch.object(
            http_request_module,
            "get_shared_aio_session",
            new=AsyncMock(side_effect=RuntimeError("stop here")),
        ):
            with patch.object(http_request_module, "logger") as mock_logger:
                with pytest.raises(RuntimeError):
                    async for _ in request._handle_aio_request():
                        pass

        assert not mock_logger.warning.call_args_list
