# -*- coding: utf-8 -*-
"""Unit tests for the oneshot usage report an external harness reads back.

The Terminal-Bench adapter reports ``total_input_tokens`` /
``total_output_tokens`` for every trial, and until now those were the
hardcoded zeros ``AbstractInstalledAgent.perform_task`` returns — so eleven
bench runs produced no cost data at all. acli flushes the real totals to
``ACLI_USAGE_FILE``; these tests pin both the payload the adapter parses and
when it lands, since a flush that waits on the output stream never fires for
the wedged run a benchmark most needs measured.
"""
# pylint: disable=redefined-outer-name,protected-access

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from dashscope.acli.cli import runners
from dashscope.acli.cli.runners import _write_usage_file
from dashscope.acli.executor import Executor


@pytest.fixture
def executor():
    """An executor with a known usage history."""
    ex = Executor()
    ex.record_api_call(
        {
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "cached_tokens": 80,
        },
    )
    ex.record_api_call(
        {
            "input_tokens": 500,
            "output_tokens": 50,
            "total_tokens": 550,
            "cached_tokens": 0,
        },
    )
    return ex


def test_the_payload_carries_the_keys_the_adapter_reads(tmp_path, executor):
    # The adapter does ``usage.get("input_tokens")`` /
    # ``usage.get("output_tokens")``; a rename here silently turns every
    # future bench run back into zeros rather than failing loudly.
    path = tmp_path / "usage.json"
    _write_usage_file(str(path), executor)
    usage = json.loads(path.read_text(encoding="utf-8"))
    assert usage["input_tokens"] == 1500
    assert usage["output_tokens"] == 250


def test_the_totals_accumulate_across_api_calls(tmp_path, executor):
    path = tmp_path / "usage.json"
    _write_usage_file(str(path), executor)
    usage = json.loads(path.read_text(encoding="utf-8"))
    assert usage["total_tokens"] == 1750
    assert usage["cached_tokens"] == 80
    assert usage["api_calls"] == 2


def test_a_run_that_never_called_the_api_reports_zeros(tmp_path):
    # The install can fail, or the trial can be killed before the first
    # flush; the harness must still be able to parse what it finds.
    path = tmp_path / "usage.json"
    _write_usage_file(str(path), Executor())
    usage = json.loads(path.read_text(encoding="utf-8"))
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


def test_a_reflush_overwrites_rather_than_appends(tmp_path, executor):
    # The file is rewritten every few seconds so a timeout-killed run still
    # leaves totals behind; that only works if each write replaces the last.
    path = tmp_path / "usage.json"
    _write_usage_file(str(path), executor)
    executor.record_api_call({"input_tokens": 10, "output_tokens": 1})
    _write_usage_file(str(path), executor)
    usage = json.loads(path.read_text(encoding="utf-8"))
    assert usage["input_tokens"] == 1510
    assert usage["api_calls"] == 3


def test_an_unwritable_path_is_swallowed(executor):
    # Observability must never be the reason a run fails: a read-only mount
    # or a vanished /tmp costs the measurement, not the trial.
    _write_usage_file("/nonexistent-dir-xyz/usage.json", executor)


@pytest.fixture
def ticking_flusher(
    monkeypatch,
) -> Iterator[Callable[[str, Executor], Callable[[], None]]]:
    """Start flushers on a test-scale interval and join them in teardown.

    Joined here rather than by each test: one that fails mid-body would
    otherwise leave a daemon thread writing into an already-torn-down
    tmp_path, and a later test counting live threads would read that leak.
    """
    monkeypatch.setattr(runners, "_USAGE_FLUSH_SEC", 0.05)
    stops: list[Callable[[], None]] = []

    def _start(path: str, executor: Executor) -> Callable[[], None]:
        stop = runners._start_usage_flusher(path, executor)
        stops.append(stop)
        return stop

    yield _start
    while stops:
        stops.pop()()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_flusher_writes_while_nothing_is_streaming(
    tmp_path,
    ticking_flusher,
):
    # The regression this file exists for: the flush used to hang off the
    # oneshot output loop, so a run wedged inside run_stream -- precisely the
    # run a benchmark most needs measured -- never wrote and the harness read
    # back zeros for the whole trial.
    path = tmp_path / "usage.json"
    executor = Executor()
    ticking_flusher(str(path), executor)
    executor.record_api_call({"input_tokens": 111, "output_tokens": 222})
    time.sleep(0.3)
    usage = _read(path)
    assert usage["api_calls"] == 1
    assert usage["input_tokens"] == 111
    assert usage["output_tokens"] == 222


def test_duration_keeps_advancing_after_the_run_goes_quiet(
    tmp_path,
    ticking_flusher,
):
    # This is what separates "wedged" from "process gone" in a bench run: a
    # dead process stops the clock, a wedged one keeps paying for silence.
    path = tmp_path / "usage.json"
    executor = Executor()
    _write_usage_file(str(path), executor)
    first = _read(path)["duration_sec"]
    ticking_flusher(str(path), executor)
    time.sleep(0.3)
    assert _read(path)["duration_sec"] > first + 0.2


def test_stop_waits_for_a_tick_that_is_already_in_flight(
    tmp_path,
    ticking_flusher,
    monkeypatch,
):
    # Signalling the event is not enough, and merely watching for "no write
    # after stop()" cannot tell the difference: at a short interval the thread
    # has usually exited on its own before anything looks. So hold a tick in
    # flight. The caller writes once more after stop(), and two writers
    # sharing one temp path would clobber each other, so stop() must outlive
    # the tick -- which only a join guarantees.
    path = tmp_path / "usage.json"
    executor = Executor()
    in_tick = threading.Event()
    release = threading.Event()
    real_write = runners._write_usage_file

    def _blocking_write(target: str, owner: Executor) -> None:
        in_tick.set()
        release.wait(5.0)
        real_write(target, owner)

    monkeypatch.setattr(runners, "_write_usage_file", _blocking_write)
    stop = ticking_flusher(str(path), executor)
    assert in_tick.wait(5.0), "the flusher never ticked"

    stopped = threading.Event()

    def _stop() -> None:
        stop()
        stopped.set()

    joiner = threading.Thread(target=_stop, daemon=True)
    joiner.start()
    assert not stopped.wait(0.3), "stop() returned while a tick was in flight"
    release.set()
    assert stopped.wait(5.0), "stop() never returned"
    joiner.join(timeout=5.0)
    assert _read(path)["api_calls"] == 0


def test_the_usage_file_appears_only_through_a_rename(
    tmp_path,
    executor,
    monkeypatch,
):
    # The harness reads this from the host while acli is still running, and
    # keeps reading it after the trial is abandoned, so a torn write costs the
    # whole trial's numbers rather than one tick's. Only the rename makes it
    # atomic: checking for a leftover temp file would pass just as well if the
    # destination were written in place, which is the thing under test.
    path = tmp_path / "usage.json"
    renamed: list[str] = []
    real_replace = os.replace

    def _spy(*args: Any, **kwargs: Any) -> Any:
        renamed.append(str(args[1]))
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(runners.os, "replace", _spy)
    _write_usage_file(str(path), executor)
    assert renamed == [str(path)]
    assert _read(path)["api_calls"] == 2
    assert not (tmp_path / "usage.json.tmp").exists()


class RacyExecutor(Executor):
    """An executor whose first reads race the way the real one can."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures_left = failures

    def get_stats(self) -> dict:
        # get_stats() copies dicts the main thread is still mutating, so a
        # tick can genuinely raise rather than merely return stale numbers.
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("dictionary changed size during iteration")
        return super().get_stats()


def test_a_tick_that_raises_does_not_stop_the_flusher(
    tmp_path,
    ticking_flusher,
):
    # Losing one tick costs a measurement; killing the run costs the trial.
    path = tmp_path / "usage.json"
    executor = RacyExecutor(2)
    ticking_flusher(str(path), executor)
    time.sleep(0.4)
    assert executor.failures_left == 0
    assert _read(path)["duration_sec"] > 0.2
