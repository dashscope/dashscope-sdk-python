# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import multiprocessing
import time

import pytest

from dashscope.common.constants import DASHSCOPE_DISABLE_DATA_INSPECTION_ENV
from tests.mock_server import create_app, create_mock_server, run_server


@pytest.fixture
def mock_disable_data_inspection_env(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_DISABLE_DATA_INSPECTION_ENV, "true")


@pytest.fixture
def mock_enable_data_inspection_env(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_DISABLE_DATA_INSPECTION_ENV, "false")


@pytest.fixture(scope="session")
def http_server(request):
    print("starting server!!!!!!!!!")
    runner = create_app()
    proc = multiprocessing.Process(target=run_server, args=(runner,))
    proc.start()
    time.sleep(2)

    def stop_server():
        proc.terminate()
        print("Stopping server")

    request.addfinalizer(stop_server)
    return proc


@pytest.fixture(scope="class")
def mock_server(request):
    print("Mock starting server!!!!!!!!!")

    return create_mock_server(request)
