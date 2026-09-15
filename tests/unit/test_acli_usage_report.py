# -*- coding: utf-8 -*-
"""Unit tests for the oneshot usage report an external harness reads back.

The Terminal-Bench adapter reports ``total_input_tokens`` /
``total_output_tokens`` for every trial, and until now those were the
hardcoded zeros ``AbstractInstalledAgent.perform_task`` returns — so eleven
bench runs produced no cost data at all. acli flushes the real totals to
``ACLI_USAGE_FILE``; these tests pin the payload the adapter parses.
"""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import json

import pytest

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
