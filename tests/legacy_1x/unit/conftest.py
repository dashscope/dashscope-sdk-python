# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import multiprocessing

import pytest

from dashscope.legacy_1x.common.constants import (
    DASHSCOPE_DISABLE_DATA_INSPECTION_ENV,
)
from tests.legacy_1x.unit.mock_server import (
    HTTP_SERVER_PORT,
    stop_process,
    create_mock_server,
    run_server,
    wait_for_port,
)


@pytest.fixture
def mock_disable_data_inspection_env(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_DISABLE_DATA_INSPECTION_ENV, "true")


@pytest.fixture
def mock_enable_data_inspection_env(monkeypatch):
    monkeypatch.setenv(DASHSCOPE_DISABLE_DATA_INSPECTION_ENV, "false")


@pytest.fixture(scope="session")
def http_server(request):
    print("starting server!!!!!!!!!")
    # Create the app inside the child process to avoid pickling AppRunner
    proc = multiprocessing.Process(target=run_server)
    proc.start()

    def stop_server():
        stop_process(proc)
        print("Stopping server")

    # Registered before the readiness poll: wait_for_port raises on
    # timeout, and a finalizer that never got registered leaves a
    # non-daemon child for multiprocessing's atexit join to hang on.
    request.addfinalizer(stop_server)
    wait_for_port(HTTP_SERVER_PORT)
    return proc


@pytest.fixture(scope="class")
def mock_server(request):
    print("Mock starting server!!!!!!!!!")

    return create_mock_server(request)
