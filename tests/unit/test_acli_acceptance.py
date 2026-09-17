# -*- coding: utf-8 -*-
"""Unit tests for dashscope.acli.acceptance — the oneshot acceptance gate.

Covers:
  * ``discover_acceptance_command`` — which project layouts earn a check, and
    the interpreter the discovered command names.
  * ``AcceptanceGate.from_env`` — the ``ACLI_ACCEPTANCE*`` switches, including
    the interactive-mode exemption.
  * The verdict rules: pass, regression, still-failing, a check that could
    not run at all, one that is insensitive to the change, and an exhausted
    retry budget.
  * Baseline capture — lazily, once per turn, shared across a parallel tool
    batch, and cancelled by a per-turn reset.
"""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dashscope.acli.acceptance import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    DISCOVERED,
    EXPLICIT,
    AcceptanceGate,
    discover_acceptance_command,
)

PASS = "3 passed in 0.42s\n[exit code: 0]"


def fail(detail="FAILED tests/test_x.py::test_a - assert 1 == 2"):
    """A shell output shaped like the one ``run_command`` really returns.

    The ``[exit code: N]`` trailer is not decoration: ``classify_tool_text``
    reads the verdict from it, so a fake without it would be classified as a
    success and every failing-gate test would pass for the wrong reason.
    """
    return f"{detail}\n[exit code: 1]"


def exited(code, detail=""):
    """A shell output carrying an arbitrary exit status.

    ``fail`` hardcodes 1 because that is what a genuine test failure produces.
    The liveness probe reads the *other* statuses — the shell's 127 and
    pytest's collection-error 2 — so they have to be expressible too.
    """
    return f"{detail}\n[exit code: {code}]"


class _Shell:
    """Scripted stand-in for the subprocess the gate drives."""

    def __init__(self):
        self.audit = MagicMock()
        self.calls: list[tuple[str, int | None]] = []
        self.outputs: list[str] = []

    def script(self, *outputs: str) -> None:
        self.outputs.extend(outputs)


@pytest.fixture
def fake_shell(monkeypatch):
    """Route the gate's ``run_command`` to a script of shell outputs.

    The gate goes through the real ``run_command`` so it keeps the
    blocked-pattern checks, the optional sandbox and the output caps; only the
    subprocess is faked. An unscripted call raises rather than silently
    passing, so a gate that runs when it should not is a failure.
    """
    fake = _Shell()

    async def _run_command(command, timeout=None):
        fake.calls.append((command, timeout))
        return fake.outputs.pop(0)

    monkeypatch.setattr("dashscope.acli.tools.shell.run_command", _run_command)

    def _audit_logger():
        return fake.audit

    monkeypatch.setattr("dashscope.acli.audit.get_audit_logger", _audit_logger)
    return fake


def _gate(**kwargs) -> AcceptanceGate:
    kwargs.setdefault("command", "pytest -q")
    return AcceptanceGate(**kwargs)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_pytest_ini_alone_is_enough(self, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        assert discover_acceptance_command(tmp_path).endswith("-m pytest -q")

    def test_pyproject_tool_pytest_section(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\naddopts = '-q'\n",
            encoding="utf-8",
        )
        assert discover_acceptance_command(tmp_path).endswith("-m pytest -q")

    def test_setup_cfg_section(self, tmp_path):
        (tmp_path / "setup.cfg").write_text(
            "[tool:pytest]\ntestpaths = tests\n",
            encoding="utf-8",
        )
        assert discover_acceptance_command(tmp_path).endswith("-m pytest -q")

    def test_tox_ini_section(self, tmp_path):
        (tmp_path / "tox.ini").write_text("[pytest]\n", encoding="utf-8")
        assert discover_acceptance_command(tmp_path).endswith("-m pytest -q")

    def test_tests_dir_beside_pyproject_is_the_default_layout(self, tmp_path):
        # Most Python repos never declare pytest config at all, and bare
        # `pytest` still collects tests/ — refusing to gate those would leave
        # the common case uncovered.
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'x'\n",
            encoding="utf-8",
        )
        (tmp_path / "tests").mkdir()
        assert discover_acceptance_command(tmp_path).endswith("-m pytest -q")

    def test_a_manifest_with_no_tests_is_not_a_test_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'x'\n",
            encoding="utf-8",
        )
        assert discover_acceptance_command(tmp_path) is None

    def test_an_unreadable_config_is_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\n",
            encoding="utf-8",
        )
        (tmp_path / "tests").mkdir()

        def _denied(self, *args, **kwargs):
            raise PermissionError(13, "denied")

        monkeypatch.setattr(Path, "open", _denied)
        # The tests/ + manifest fallback still recognises the project, so a
        # config file this process cannot read costs nothing.
        assert discover_acceptance_command(tmp_path).endswith("-m pytest -q")

    def test_cargo_manifest(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        assert discover_acceptance_command(tmp_path) == "cargo test --quiet"

    def test_go_module(self, tmp_path):
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        assert discover_acceptance_command(tmp_path) == "go test ./..."

    def test_pytest_wins_when_several_manifests_are_present(self, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        assert discover_acceptance_command(tmp_path).endswith("-m pytest -q")

    def test_nothing_recognisable_discovers_nothing(self, tmp_path):
        (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
        assert discover_acceptance_command(tmp_path) is None

    def test_a_path_that_is_not_a_directory_discovers_nothing(self, tmp_path):
        assert discover_acceptance_command(tmp_path / "absent") is None

    def test_the_discovered_command_names_this_interpreter(self, tmp_path):
        # A bare `pytest` resolves against whatever is first on PATH, which is
        # not necessarily the environment the agent's own run_command calls
        # use — a green gate from the wrong interpreter proves nothing.
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        assert discover_acceptance_command(tmp_path).startswith(
            shlex.quote(sys.executable),
        )


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


class TestFromEnv:
    @pytest.fixture(autouse=True)
    def armed(self, monkeypatch):
        """Undo the suite-wide disarm: here the environment is the subject."""
        monkeypatch.setenv("ACLI_ACCEPTANCE", "1")

    def test_interactive_runs_are_never_gated(self, monkeypatch, tmp_path):
        # Interactively the user is the acceptance criterion and is reading the
        # output, so minutes of test suite before every reply is a bad trade.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "pytest -q")
        assert not AcceptanceGate.from_env(oneshot=False).active

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "No"])
    def test_explicit_off_switch(self, monkeypatch, value):
        monkeypatch.setenv("ACLI_ACCEPTANCE", value)
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "pytest -q")
        assert not AcceptanceGate.from_env(oneshot=True).active

    def test_an_explicit_command_beats_discovery(self, monkeypatch, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "make check")
        gate = AcceptanceGate.from_env(oneshot=True)
        assert (gate.command, gate.source) == ("make check", EXPLICIT)

    def test_discovery_fills_in_when_nothing_is_declared(
        self,
        monkeypatch,
        tmp_path,
    ):
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        gate = AcceptanceGate.from_env(oneshot=True)
        assert gate.source == DISCOVERED
        assert gate.command.endswith("-m pytest -q")

    def test_a_blank_command_falls_through_to_discovery(
        self,
        monkeypatch,
        tmp_path,
    ):
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "   ")
        assert AcceptanceGate.from_env(oneshot=True).source == DISCOVERED

    def test_nothing_discoverable_leaves_the_gate_inactive(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        assert not AcceptanceGate.from_env(oneshot=True).active

    def test_retries_and_timeout_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "pytest -q")
        monkeypatch.setenv("ACLI_ACCEPTANCE_RETRIES", "3")
        monkeypatch.setenv("ACLI_ACCEPTANCE_TIMEOUT", "45")
        gate = AcceptanceGate.from_env(oneshot=True)
        assert (gate.retries, gate.timeout) == (3, 45)

    def test_unparsable_numbers_fall_back_to_the_defaults(self, monkeypatch):
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "pytest -q")
        monkeypatch.setenv("ACLI_ACCEPTANCE_RETRIES", "lots")
        monkeypatch.setenv("ACLI_ACCEPTANCE_TIMEOUT", "")
        gate = AcceptanceGate.from_env(oneshot=True)
        assert gate.retries == DEFAULT_RETRIES
        assert gate.timeout == DEFAULT_TIMEOUT

    def test_the_gate_is_on_by_default_in_oneshot(self, monkeypatch):
        # Opt-out, not opt-in: a benchmark that wants raw model output sets
        # ACLI_ACCEPTANCE=0, but the default run should be verified.
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "pytest -q")
        assert AcceptanceGate.from_env(oneshot=True).active

    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_an_unset_or_blank_switch_also_defaults_to_on(
        self,
        monkeypatch,
        value,
    ):
        # The common case is a run that has never heard of ACLI_ACCEPTANCE;
        # that must land on the same side as an explicit "1".
        if value is None:
            monkeypatch.delenv("ACLI_ACCEPTANCE", raising=False)
        else:
            monkeypatch.setenv("ACLI_ACCEPTANCE", value)
        monkeypatch.setenv("ACLI_ACCEPTANCE_CMD", "pytest -q")
        assert AcceptanceGate.from_env(oneshot=True).active


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


class TestCheck:
    @pytest.mark.asyncio
    async def test_a_gate_with_no_command_is_never_applicable(
        self,
        fake_shell,
    ):
        assert await AcceptanceGate().check() is None
        assert fake_shell.calls == []

    @pytest.mark.asyncio
    async def test_nothing_mutated_means_nothing_to_accept(self, fake_shell):
        gate = _gate()
        assert await gate.check() is None
        assert fake_shell.calls == []

    @pytest.mark.asyncio
    async def test_a_passing_check_accepts_the_answer(self, fake_shell):
        gate = _gate()
        fake_shell.script(PASS, PASS)
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.run.ok
        assert verdict.rejection == ""
        assert verdict.note == ""

    @pytest.mark.asyncio
    async def test_failing_after_a_passing_baseline_is_called_a_regression(
        self,
        fake_shell,
    ):
        # This is the case the baseline exists for: "the suite is red" and
        # "you made it red" need different wording, and only the second one
        # is the agent's to fix.
        gate = _gate()
        fake_shell.script(PASS, fail())
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert "REGRESSION" in verdict.rejection
        assert verdict.note == ""

    @pytest.mark.asyncio
    async def test_failing_after_a_failing_baseline_is_not_a_regression(
        self,
        fake_shell,
    ):
        gate = _gate()
        fake_shell.script(fail("2 failed in 0.31s"), fail("3 failed in 0.44s"))
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.rejection
        assert "REGRESSION" not in verdict.rejection
        assert "still failing" in verdict.rejection

    @pytest.mark.asyncio
    async def test_the_rejection_quotes_the_raw_output(self, fake_shell):
        gate = _gate()
        fake_shell.script(PASS, fail())
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert "FAILED tests/test_x.py::test_a" in verdict.rejection
        assert "pytest -q" in verdict.rejection

    @pytest.mark.asyncio
    async def test_the_rejection_tells_the_model_not_to_attack_the_check(
        self,
        fake_shell,
    ):
        # The cheapest way to make a gate pass is to delete it.
        gate = _gate()
        fake_shell.script(PASS, fail())
        await gate.ensure_baseline()
        assert "withheld, not discarded" in (await gate.check()).rejection

    @pytest.mark.asyncio
    async def test_backticked_output_does_not_break_out_of_its_fence(
        self,
        fake_shell,
    ):
        # A fixed three-backtick fence would be closed early by output that
        # quotes markdown, leaking the rest of the rejection into the block.
        gate = _gate()
        fake_shell.script(
            PASS,
            fail("compiler said ```` quote ```` then died"),
        )
        await gate.ensure_baseline()
        before, quoted, after = (await gate.check()).rejection.split("`````")
        assert "pytest -q" in before
        assert "```` quote ````" in quoted
        assert after.strip().startswith("Your answer")

    @pytest.mark.asyncio
    async def test_a_huge_failure_is_quoted_in_bound(self, fake_shell):
        gate = _gate()
        fake_shell.script(PASS, fail("x" * 200000))
        await gate.ensure_baseline()
        rejection = (await gate.check()).rejection
        assert len(rejection) < 4000
        assert "omitted" in rejection

    @pytest.mark.asyncio
    async def test_a_check_that_fails_identically_is_reported_not_rejected(
        self,
        fake_shell,
    ):
        # Usually a command that does not run at all here: no pytest
        # installed, wrong interpreter. Sending the model back to fix that
        # burns the turn on a problem it neither created nor can solve.
        dead = fail("/bin/sh: pytest: command not found")
        gate = _gate()
        fake_shell.script(dead, dead)
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.rejection == ""
        assert "identically" in verdict.note

    @pytest.mark.asyncio
    async def test_a_real_failure_is_never_mistaken_for_a_dead_check(
        self,
        fake_shell,
    ):
        # Byte-identity is the whole guard: a suite that genuinely fails
        # prints timings, so two runs differ and still get rejected.
        gate = _gate()
        fake_shell.script(fail("1 failed in 0.31s"), fail("1 failed in 0.47s"))
        await gate.ensure_baseline()
        assert (await gate.check()).rejection

    @pytest.mark.asyncio
    async def test_a_command_the_shell_cannot_find_is_not_blamed_on_the_change(
        self,
        fake_shell,
    ):
        # 127 means there was no check to run. The two outputs differ here so
        # the byte-identity fallback cannot be what saves this — the exit
        # status alone has to be enough.
        gate = _gate(command="/opt/acli/bin/pytest -q")
        fake_shell.script(
            exited(127, "/bin/sh: 1: pytest: not found"),
            exited(127, "/bin/sh: 1: /opt/acli/bin/pytest: not found"),
        )
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.rejection == ""
        assert "never reached the tests" in verdict.note

    @pytest.mark.asyncio
    async def test_a_dependency_missing_at_collection_is_not_a_change_failure(
        self,
        fake_shell,
    ):
        # pytest exits 2 on a collection error and prints a timing line, so a
        # project whose dependencies are installed only by the grading step
        # fails differently every run. Without the probe that reads as "still
        # failing" and withholds an answer the check never evaluated.
        gate = _gate(command=f"{sys.executable} -m pytest -q")
        missing = "ModuleNotFoundError: No module named 'astropy'"
        fake_shell.script(
            exited(2, f"{missing}\n1 error in 0.05s"),
            exited(2, f"{missing}\n1 error in 0.07s"),
        )
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.rejection == ""
        assert "never reached the tests" in verdict.note

    @pytest.mark.asyncio
    async def test_a_green_baseline_keeps_a_collection_error_a_regression(
        self,
        fake_shell,
    ):
        # The other half of the probe: the baseline passing is proof the check
        # does run here, so the agent breaking an import is its own doing.
        gate = _gate()
        broken = "ModuleNotFoundError: No module named 'app'"
        fake_shell.script(PASS, exited(2, f"{broken}\n1 error in 0.06s"))
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.note == ""
        assert "REGRESSION" in verdict.rejection

    @pytest.mark.asyncio
    async def test_a_repaired_environment_still_holds_the_answer_to_the_check(
        self,
        fake_shell,
    ):
        # The check could not run before the turn and can now, which means the
        # model fixed the environment. Its exit 1 is a real verdict at that
        # point, so the inert baseline stops excusing anything.
        gate = _gate()
        fake_shell.script(
            exited(2, "1 error in 0.05s"),
            fail("1 failed in 0.31s"),
        )
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.note == ""
        assert "still failing" in verdict.rejection

    @pytest.mark.asyncio
    async def test_pytest_statuses_are_not_read_into_another_ecosystem(
        self,
        fake_shell,
    ):
        # 2 is only "never collected anything" for pytest. cargo reuses it for
        # a build failure, which is exactly the kind of result worth rejecting.
        gate = _gate(command="cargo test --quiet")
        fake_shell.script(
            exited(2, "error: could not compile `app`"),
            exited(2, "error: could not compile `app` (2 warnings)"),
        )
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.note == ""
        assert verdict.rejection

    @pytest.mark.asyncio
    async def test_the_retry_budget_is_spent_then_the_answer_goes_out_flagged(
        self,
        fake_shell,
    ):
        gate = _gate(retries=1)
        fake_shell.script(PASS, fail("a"), fail("b"))
        await gate.ensure_baseline()
        first = await gate.check()
        assert first.rejection and not first.note
        second = await gate.check()
        assert second.note and not second.rejection
        assert "retry budget (1) is spent" in second.note

    @pytest.mark.asyncio
    async def test_zero_retries_never_sends_the_answer_back(self, fake_shell):
        gate = _gate(retries=0)
        fake_shell.script(PASS, fail())
        await gate.ensure_baseline()
        verdict = await gate.check()
        assert verdict.rejection == ""
        assert "Unverified" in verdict.note

    @pytest.mark.asyncio
    async def test_the_configured_timeout_reaches_the_shell(self, fake_shell):
        gate = _gate(timeout=42)
        fake_shell.script(PASS)
        await gate.ensure_baseline()
        assert fake_shell.calls == [("pytest -q", 42)]

    @pytest.mark.asyncio
    async def test_every_gate_run_is_audited_with_its_source(self, fake_shell):
        gate = _gate(source=DISCOVERED)
        fake_shell.script(PASS, PASS)
        await gate.ensure_baseline()
        await gate.check()
        logged = fake_shell.audit.log_tool_call.call_args_list
        assert [c.args[0] for c in logged] == ["acceptance_gate"] * 2
        assert all(c.kwargs["decision"] == "allowed" for c in logged)
        assert "discovered" in logged[0].kwargs["reason"]


# ---------------------------------------------------------------------------
# The pre-change anchor
# ---------------------------------------------------------------------------


class TestBaseline:
    @pytest.mark.asyncio
    async def test_an_inactive_gate_never_runs_a_command(self, fake_shell):
        gate = AcceptanceGate()
        await gate.ensure_baseline()
        assert fake_shell.calls == []
        assert gate.baseline is None

    @pytest.mark.asyncio
    async def test_a_parallel_batch_captures_exactly_one_baseline(
        self,
        fake_shell,
    ):
        gate = _gate()
        fake_shell.script(PASS)
        await asyncio.gather(*[gate.ensure_baseline() for _ in range(4)])
        assert len(fake_shell.calls) == 1
        assert gate.baseline.ok

    @pytest.mark.asyncio
    async def test_a_caller_returns_only_once_the_baseline_has_landed(
        self,
        monkeypatch,
    ):
        # The reason the task is shared rather than flagged: a mutating call
        # that returned from ensure_baseline early would change the workspace
        # the "pre-change" verdict is still describing.
        release = asyncio.Event()
        order: list[str] = []

        async def _slow(**_kwargs):  # called by keyword, like run_command
            order.append("check")
            await release.wait()
            order.append("landed")
            return PASS

        monkeypatch.setattr("dashscope.acli.tools.shell.run_command", _slow)
        monkeypatch.setattr("dashscope.acli.audit.get_audit_logger", MagicMock)

        gate = _gate()
        first = asyncio.ensure_future(gate.ensure_baseline())
        second = asyncio.ensure_future(gate.ensure_baseline())
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        release.set()
        await asyncio.gather(first, second)
        assert order == ["check", "landed"]
        assert gate.baseline.ok

    @pytest.mark.asyncio
    async def test_reset_drops_the_anchor_and_the_retry_budget(
        self,
        fake_shell,
    ):
        gate = _gate()
        fake_shell.script(PASS, fail(), PASS)
        await gate.ensure_baseline()
        await gate.check()
        assert gate.attempts == 1
        gate.reset()
        assert gate.baseline is None
        assert gate.attempts == 0
        await gate.ensure_baseline()
        assert len(fake_shell.calls) == 3

    @pytest.mark.asyncio
    async def test_reset_cancels_a_capture_left_in_flight(self, monkeypatch):
        # An aborted turn can leave the baseline running; left alone it would
        # write the previous turn's verdict into the next turn's anchor.
        release = asyncio.Event()
        landed: list[str] = []

        async def _slow(command=None, **_rest):
            await release.wait()
            landed.append(command)
            return PASS

        monkeypatch.setattr("dashscope.acli.tools.shell.run_command", _slow)
        monkeypatch.setattr("dashscope.acli.audit.get_audit_logger", MagicMock)

        gate = _gate()
        pending = asyncio.ensure_future(gate.ensure_baseline())
        await asyncio.sleep(0)
        gate.reset()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        await asyncio.sleep(0)
        assert not landed
        assert gate.baseline is None
