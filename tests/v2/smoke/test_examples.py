# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Smoke tests that keep ``examples`` from going stale.

Two offline gates: every example file must compile, and the examples
built on mock transports / local subprocesses must run to completion,
so a breaking SDK change turns red here instead of being reported by
users. When adding a new offline-capable example, list it in
``OFFLINE_EXAMPLES`` below.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples"

# Examples that run fully offline (mock transport / local subprocess),
# as paths relative to examples.
OFFLINE_EXAMPLES = [
    "02_advanced_engineering/local_mock_debugging.py",
    "02_advanced_engineering/circuit_breaker_retry.py",
    "03_agent_recipes/mcp_server_demo/client_demo.py",
    "04_real_world_scenarios/production_ready_rag_harness.py",
    "04_real_world_scenarios/realtime_audio_meeting_harness.py",
    "04_real_world_scenarios/ticket_triage_harness.py",
]


def test_all_examples_compile():
    sources = sorted(EXAMPLES.rglob("*.py"))
    assert sources, "no example files found under examples"
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("relative", OFFLINE_EXAMPLES)
def test_offline_example_runs(relative):
    # A dummy key keeps construction hermetic; the offline examples
    # never reach the network (mock transport / local subprocess only).
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "DASHSCOPE_API_KEY": "sk-smoke",
    }
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / relative)],
        cwd=ROOT,
        env=env,
        timeout=180,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
