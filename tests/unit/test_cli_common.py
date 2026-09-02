# -*- coding: utf-8 -*-
"""Unit tests for CLI exit-code classification in dashscope.cli.common."""
from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.cli.common import _exit_code_for
from dashscope.cli import embeddings


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
