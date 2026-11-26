# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for error-code report P2-1 (SDK side).

1. ``timeout=`` on legacy APIs is an alias of ``request_timeout`` and
   must arm the client timeout instead of being forwarded to the
   service as a model parameter.
2. A bare gateway error response (empty body, no content-type) must
   surface a diagnostic message and the ``x-request-id`` header, while
   keeping the server's real status code.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

import dashscope
from dashscope import Generation
from dashscope.legacy_1x.api_entities.api_request_factory import (
    _build_api_request,
)
from dashscope.legacy_1x.common.utils import (
    _handle_aiohttp_failed_response,
    _handle_http_failed_response,
)


def _build(timeout_kwargs):
    return _build_api_request(
        model="qwen-plus",
        input={"messages": [{"role": "user", "content": "hi"}]},
        task_group="aigc",
        task="text-generation",
        function="generation",
        api_key="sk-test",
        **timeout_kwargs,
    )


class TestTimeoutAlias:
    def test_timeout_arms_client_timeout(self):
        request = _build({"timeout": 7})
        assert request.timeout == 7
        assert "timeout" not in request.data.parameters

    def test_request_timeout_wins_over_alias(self):
        request = _build({"timeout": 7, "request_timeout": 5})
        assert request.timeout == 5
        assert "timeout" not in request.data.parameters

    def test_no_timeout_keeps_default(self):
        request = _build({})
        assert request.timeout == 300
        assert "timeout" not in request.data.parameters


def _bare_response(status=404, request_id="rid-bare"):
    response = requests.Response()
    response.status_code = status
    response.headers["x-request-id"] = request_id
    response._content = b""  # pylint: disable=protected-access
    return response


class TestBareErrorResponse:
    def test_sync_bare_404_diagnostic(self):
        rsp = _handle_http_failed_response(_bare_response())
        assert rsp.status_code == 404
        assert rsp.code == "Unknown"
        assert rsp.request_id == "rid-bare"
        assert "empty response body" in rsp.message

    def test_sync_non_json_body_preserved(self):
        response = _bare_response()
        response._content = (
            b"<html>oops</html>"  # pylint: disable=protected-access
        )
        rsp = _handle_http_failed_response(response)
        assert rsp.message == "<html>oops</html>"
        assert rsp.request_id == "rid-bare"

    def test_aio_bare_404_diagnostic(self):
        class _StubAioResponse:
            status = 404
            content_type = ""
            headers = {"x-request-id": "rid-aio"}

            async def text(self):
                return ""

        rsp = asyncio.run(
            _handle_aiohttp_failed_response(_StubAioResponse()),
        )
        assert rsp.status_code == 404
        assert rsp.code == "Unknown"
        assert rsp.request_id == "rid-aio"
        assert "empty response body" in rsp.message


class _CapturingHandler(BaseHTTPRequestHandler):
    """Records the request body; behavior set via server attributes."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.captured.append(json.loads(body))
        if self.server.stall_seconds:
            time.sleep(self.server.stall_seconds)
            payload = json.dumps(
                {"output": {"text": "ok"}, "request_id": "rid-slow"},
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            # Bare gateway 404: empty body, no content-type.
            self.send_response(404)
            self.send_header("x-request-id", "rid-bare-404")
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server(monkeypatch):
    servers = []

    def _start(stall_seconds=0):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _CapturingHandler)
        srv.captured = []
        srv.stall_seconds = stall_seconds
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        monkeypatch.setattr(
            dashscope,
            "base_http_api_url",
            f"http://127.0.0.1:{srv.server_port}/api/v1",
        )
        monkeypatch.setattr(dashscope, "api_key", "sk-test")
        return srv

    yield _start
    for srv in servers:
        srv.shutdown()


class TestGenerationEndToEnd:
    def test_timeout_alias_raises_on_slow_response(self, local_server):
        local_server(stall_seconds=3)
        start = time.time()
        with pytest.raises(requests.exceptions.Timeout):
            Generation.call(
                model="qwen-plus",
                messages=[{"role": "user", "content": "hi"}],
                timeout=1,
            )
        assert time.time() - start < 2.5

    def test_bare_404_surfaces_diagnostics(self, local_server):
        srv = local_server()
        rsp = Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1500,
            timeout=1,
        )
        assert rsp.status_code == 404
        assert rsp.code == "Unknown"
        assert rsp.request_id == "rid-bare-404"
        assert "empty response body" in rsp.message
        # The alias must not leak into the wire body parameters.
        assert "timeout" not in srv.captured[0]["parameters"]
