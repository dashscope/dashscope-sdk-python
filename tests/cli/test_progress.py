# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``--wait`` delegates to the SDK's poller instead of reimplementing it."""

import io
from types import SimpleNamespace

import pytest

from dashscope.cli.errors import CliError
from dashscope.cli.output import JSON, Renderer
from dashscope.cli.progress import validate_wait_timeout, wait_job


class _RecordingWait:
    """Stand-in for ``client.fine_tunes`` capturing the ``wait`` call."""

    def __init__(self, result=None, exc=None):
        self.calls = []
        self._result = result
        self._exc = exc

    def wait(self, job, **kwargs):
        self.calls.append((job, kwargs))
        if self._exc is not None:
            raise self._exc
        return self._result


def _client(waiter):
    return SimpleNamespace(fine_tunes=waiter)


class _TtyStderr(io.StringIO):
    def isatty(self):
        return True


# Both fixtures return an installer rather than patching directly: pytest
# re-applies its own capture around the *call* phase, which would undo a
# sys.stderr patch made during fixture setup.
@pytest.fixture
def tty_stderr(monkeypatch):
    def install() -> _TtyStderr:
        stream = _TtyStderr()
        monkeypatch.setattr("sys.stderr", stream)
        return stream

    return install


@pytest.fixture
def quiet_stderr(monkeypatch):
    def install() -> io.StringIO:
        stream = io.StringIO()
        monkeypatch.setattr("sys.stderr", stream)
        return stream

    return install


class TestDelegation:
    def test_returns_what_wait_returns(self):
        job = SimpleNamespace(job_id="job-1")
        waiter = _RecordingWait(result=job)
        assert wait_job(_client(waiter), "job-1", timeout=5) is job

    def test_job_object_or_id_both_work(self):
        job = SimpleNamespace(job_id="job-1")
        waiter = _RecordingWait(result=job)
        wait_job(_client(waiter), job, timeout=5)
        assert waiter.calls[0][0] is job

    def test_timeout_is_forwarded(self):
        waiter = _RecordingWait(result=None)
        wait_job(_client(waiter), "job-1", timeout=42)
        assert waiter.calls[0][1]["timeout"] == 42

    def test_timeout_is_omitted_when_unset(self):
        """So the SDK's own default applies rather than a CLI-invented one."""
        waiter = _RecordingWait(result=None)
        wait_job(_client(waiter), "job-1")
        assert "timeout" not in waiter.calls[0][1]

    def test_failure_propagates(self):
        """The CLI must not turn a failed job into a success exit code."""
        boom = RuntimeError("job failed")
        waiter = _RecordingWait(exc=boom)
        with pytest.raises(RuntimeError) as info:
            wait_job(_client(waiter), "job-1", timeout=5)
        assert info.value is boom


class TestProgressVisibility:
    def test_no_renderer_means_no_sleep_hook(self):
        waiter = _RecordingWait(result=None)
        wait_job(_client(waiter), "job-1", timeout=5)
        assert "_sleep" not in waiter.calls[0][1]

    def test_machine_format_stays_silent(self, tty_stderr):
        """A spinner would corrupt a JSON stream."""
        stream = tty_stderr()
        waiter = _RecordingWait(result=None)
        wait_job(
            _client(waiter),
            "job-1",
            timeout=5,
            renderer=Renderer(JSON),
        )
        assert "_sleep" not in waiter.calls[0][1]
        assert stream.getvalue() == ""

    def test_table_format_without_a_terminal_stays_silent(self, quiet_stderr):
        quiet_stderr()
        waiter = _RecordingWait(result=None)
        wait_job(
            _client(waiter),
            "job-1",
            timeout=5,
            renderer=Renderer("table"),
        )
        assert "_sleep" not in waiter.calls[0][1]

    def test_table_format_on_a_terminal_installs_the_hook(self, tty_stderr):
        tty_stderr()
        waiter = _RecordingWait(result=None)
        wait_job(
            _client(waiter),
            "job-1",
            timeout=5,
            renderer=Renderer("table"),
        )
        assert callable(waiter.calls[0][1]["_sleep"])


class TestSleepHook:
    def test_sleeps_for_the_requested_interval(self, tty_stderr, monkeypatch):
        tty_stderr()
        slept = []
        monkeypatch.setattr("dashscope.cli.progress.time.sleep", slept.append)
        waiter = _RecordingWait(result=None)
        wait_job(
            _client(waiter),
            "job-1",
            timeout=5,
            renderer=Renderer("table"),
        )
        waiter.calls[0][1]["_sleep"](1.5)
        assert slept == [1.5]

    def test_mentions_the_job_id(self, tty_stderr, monkeypatch):
        stream = tty_stderr()
        monkeypatch.setattr(
            "dashscope.cli.progress.time.sleep",
            lambda s: None,
        )
        waiter = _RecordingWait(result=None)
        wait_job(
            _client(waiter),
            SimpleNamespace(job_id="ft-20260101"),
            timeout=5,
            renderer=Renderer("table"),
        )
        waiter.calls[0][1]["_sleep"](0)
        assert "ft-20260101" in stream.getvalue()

    def test_repeated_polls_do_not_stack_the_suffix(
        self,
        tty_stderr,
        monkeypatch,
    ):
        stream = tty_stderr()
        monkeypatch.setattr(
            "dashscope.cli.progress.time.sleep",
            lambda s: None,
        )
        waiter = _RecordingWait(result=None)
        wait_job(
            _client(waiter),
            "job-1",
            timeout=5,
            renderer=Renderer("table"),
        )
        sleep = waiter.calls[0][1]["_sleep"]
        sleep(0)
        sleep(0)
        text = stream.getvalue()
        assert text.count("elapsed) waiting for") == 0

    def test_custom_label(self, tty_stderr, monkeypatch):
        stream = tty_stderr()
        monkeypatch.setattr(
            "dashscope.cli.progress.time.sleep",
            lambda s: None,
        )
        waiter = _RecordingWait(result=None)
        wait_job(
            _client(waiter),
            "dep-1",
            timeout=5,
            renderer=Renderer("table"),
            label="deployment",
        )
        waiter.calls[0][1]["_sleep"](0)
        assert "deployment dep-1" in stream.getvalue()


class TestValidateWaitTimeout:
    """A timeout the poller could never honour must fail before submitting."""

    @pytest.mark.parametrize("value", [None, 0.5, 1, 3600.0, 1e9])
    def test_accepts_a_usable_timeout(self, value):
        validate_wait_timeout(value)

    @pytest.mark.parametrize("value", [-1, -0.5, 0, 0.0])
    def test_rejects_a_non_positive_timeout(self, value):
        with pytest.raises(CliError) as info:
            validate_wait_timeout(value)
        assert info.value.category == "usage"
        assert "--timeout" in info.value.message

    def test_rejects_nan(self):
        """nan is false against every comparison, so the poll never ends."""
        with pytest.raises(CliError) as info:
            validate_wait_timeout(float("nan"))
        assert "finite" in info.value.message

    def test_rejects_infinity(self):
        with pytest.raises(CliError):
            validate_wait_timeout(float("inf"))

    def test_wait_job_checks_it_too(self):
        """The submit sites check first; this is the net underneath them."""
        waiter = _RecordingWait(result=None)
        with pytest.raises(CliError):
            wait_job(_client(waiter), "ft-1", timeout=float("nan"))
        assert waiter.calls == []
