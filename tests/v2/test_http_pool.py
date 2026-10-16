# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""HTTP keep-alive pool: settings plumbing + real TCP-reuse proof."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dashscope.v2._config import resolve_config, settings
from dashscope.v2.client import Client


@pytest.fixture
def clean(monkeypatch):
    for name in (
        "DASHSCOPE_HTTP_MAX_CONNECTIONS",
        "DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS",
        "DASHSCOPE_HTTP_KEEPALIVE_EXPIRY",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


def test_http_pool_defaults(clean):
    config = resolve_config()
    assert config.http_max_connections == 100
    assert config.http_max_keepalive_connections == 20
    assert config.http_keepalive_expiry == 5.0


def test_http_pool_settings_override(clean):
    settings.http_max_connections = 50
    settings.http_max_keepalive_connections = 7
    settings.http_keepalive_expiry = 1.5
    config = resolve_config()
    assert config.http_max_connections == 50
    assert config.http_max_keepalive_connections == 7
    assert config.http_keepalive_expiry == 1.5


def test_http_pool_env_override(clean, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_HTTP_MAX_CONNECTIONS", "33")
    monkeypatch.setenv("DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS", "3")
    monkeypatch.setenv("DASHSCOPE_HTTP_KEEPALIVE_EXPIRY", "0.5")
    config = resolve_config()
    assert config.http_max_connections == 33
    assert config.http_max_keepalive_connections == 3
    assert config.http_keepalive_expiry == 0.5


def test_http_pool_config_file(clean, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "http_max_connections = 44\n"
        "http_max_keepalive_connections = 4\n"
        "http_keepalive_expiry = 2.5\n",
        encoding="utf-8",
    )
    config = resolve_config(config_path=config_file)
    assert config.http_max_connections == 44
    assert config.http_max_keepalive_connections == 4
    assert config.http_keepalive_expiry == 2.5


def test_http_pool_validation(clean, monkeypatch):
    with pytest.raises(ValueError, match="http_max_connections"):
        settings.http_max_connections = 0
    with pytest.raises(ValueError, match="http_max_keepalive_connections"):
        settings.http_max_keepalive_connections = -1
    with pytest.raises(ValueError, match="http_keepalive_expiry"):
        settings.http_keepalive_expiry = 0
    monkeypatch.setenv("DASHSCOPE_HTTP_MAX_CONNECTIONS", "0")
    with pytest.raises(ValueError, match="http_max_connections"):
        resolve_config()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.server.client_ports.append(self.client_address[1])
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.client_ports = []
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_http_keepalive_reuses_tcp_connection(clean):
    """Sequential requests on one client ride one TCP connection."""
    server = _serve()
    try:
        base = f"http://127.0.0.1:{server.server_port}/api/v1"
        with Client(api_key="sk-test", base_url=base) as client:
            for _ in range(3):
                response = client.transport.request("GET", "/services/x")
                assert response.status_code == 200
        assert len(server.client_ports) == 3
        assert len(set(server.client_ports)) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_http_keepalive_disabled_opens_new_connections(clean):
    """keepalive=0 flows settings -> limits -> httpx end to end."""
    settings.http_max_keepalive_connections = 0
    server = _serve()
    try:
        base = f"http://127.0.0.1:{server.server_port}/api/v1"
        with Client(api_key="sk-test", base_url=base) as client:
            for _ in range(3):
                response = client.transport.request("GET", "/services/x")
                assert response.status_code == 200
        assert len(set(server.client_ports)) == 3
    finally:
        server.shutdown()
        server.server_close()
