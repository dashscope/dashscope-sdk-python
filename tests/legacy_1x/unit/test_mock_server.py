# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The mock-server fixtures must never leave a child process behind.

``create_mock_server`` starts a non-daemon ``multiprocessing`` child, and
``multiprocessing``'s atexit handler joins every one of them. A child
nobody terminated therefore hangs the whole pytest run at exit -- after
the results have been printed, so it looks like a stuck terminal rather
than a failure. The session-scoped ``http_server`` fixtures in the two
conftest files start theirs the same way.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.legacy_1x.unit import mock_server as servers


class _FakeProcess:
    """Stands in for ``multiprocessing.Process`` so nothing is spawned."""

    def __init__(self) -> None:
        self.started = False
        self.terminated = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def join(self, timeout: float = None) -> None:
        return None


@pytest.fixture
def harness(monkeypatch):
    """Fake out the process and the readiness poll, recording the order."""
    spawned = []
    events = []
    finalizers = []

    def fake_process(target=None, args=()):
        process = _FakeProcess()
        spawned.append(process)
        return process

    def addfinalizer(function):
        finalizers.append(function)
        events.append("addfinalizer")

    monkeypatch.setattr(servers.multiprocessing, "Process", fake_process)
    monkeypatch.setattr(
        servers,
        "stop_process",
        lambda proc, timeout=10.0: (
            events.append("stop_process"),
            proc.terminate(),
        ),
    )
    return SimpleNamespace(
        spawned=spawned,
        events=events,
        finalizers=finalizers,
        request=SimpleNamespace(addfinalizer=addfinalizer),
        never_ready=lambda port, timeout=60.0: (
            events.append("wait_for_port"),
            (_ for _ in ()).throw(RuntimeError("never came up")),
        ),
        ready=lambda port, timeout=60.0: events.append("wait_for_port"),
        monkeypatch=monkeypatch,
    )


class TestCreateMockServer:
    def test_cleanup_is_registered_before_the_readiness_poll(self, harness):
        """The bug: ``wait_for_port`` raised, so ``addfinalizer`` never ran
        and the child was left for the atexit join to hang on."""
        harness.monkeypatch.setattr(
            servers,
            "wait_for_port",
            harness.never_ready,
        )
        with pytest.raises(RuntimeError):
            servers.create_mock_server(harness.request)

        assert harness.spawned[0].started
        assert harness.events.index("addfinalizer") < harness.events.index(
            "wait_for_port",
        )
        harness.finalizers[0]()
        assert "stop_process" in harness.events
        assert not harness.spawned[0].is_alive()

    def test_a_ready_server_is_returned_with_one_finalizer(self, harness):
        harness.monkeypatch.setattr(servers, "wait_for_port", harness.ready)
        server = servers.create_mock_server(harness.request)
        assert hasattr(server, "requests")
        assert len(harness.finalizers) == 1
        harness.finalizers[0]()
        assert not harness.spawned[0].is_alive()
