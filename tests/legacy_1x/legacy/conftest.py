# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import multiprocessing
import time

import pytest

import dashscope
from dashscope.legacy_1x.common.constants import DASHSCOPE_DISABLE_DATA_INSPECTION_ENV
from tests.legacy_1x.unit.mock_server import create_mock_server, run_server


@pytest.fixture(scope="session", autouse=True)
def set_test_api_key():
    """Set a dummy api_key and base URLs for all legacy tests."""
    original_api_key = dashscope.api_key
    original_base_http = dashscope.base_http_api_url
    original_base_ws = dashscope.base_websocket_api_url
    dashscope.api_key = "test-api-key"
    dashscope.base_http_api_url = "http://localhost:8080/api/v1"
    yield
    dashscope.api_key = original_api_key
    dashscope.base_http_api_url = original_base_http
    dashscope.base_websocket_api_url = original_base_ws


@pytest.fixture
def mock_disable_data_inspection_env(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_DISABLE_DATA_INSPECTION_ENV, "true")


@pytest.fixture
def mock_enable_data_inspection_env(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_DISABLE_DATA_INSPECTION_ENV, "false")


@pytest.fixture(scope="session", autouse=True)
def http_server(request):
    print("starting legacy server!!!!!!!!!")
    proc = multiprocessing.Process(target=run_server)
    proc.start()
    time.sleep(2)

    def stop_server():
        proc.terminate()
        print("Stopping legacy server")

    request.addfinalizer(stop_server)
    return proc


@pytest.fixture(scope="class")
def mock_server(request):
    print("Mock starting legacy server!!!!!!!!!")
    return create_mock_server(request)
