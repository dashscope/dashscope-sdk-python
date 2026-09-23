# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import multiprocessing
import time

import pytest

from dashscope.common.constants import DASHSCOPE_DISABLE_DATA_INSPECTION_ENV
from tests.unit.mock_server import create_mock_server, run_server


@pytest.fixture(autouse=True)
def isolated_generation_path_cache(monkeypatch, tmp_path):
    """Keep TongyiProvider's learned model→path cache off the real ~/.acli.

    The cache is persisted across processes by design, so without this a test
    run would write bindings into the developer's home directory — and would
    read back whatever is already there, making the probe-order assertions
    depend on machine state instead of on the code.
    """
    from dashscope.acli import config as config_module
    from dashscope.acli.providers.tongyi import TongyiProvider

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path / "acli-home")
    # Class-level state survives across tests; monkeypatch restores the
    # previous value on teardown, so each test starts from "nothing learned".
    monkeypatch.setattr(TongyiProvider, "_path_cache", None)


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
