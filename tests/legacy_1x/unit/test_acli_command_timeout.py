# -*- coding: utf-8 -*-
"""run_command's timeout in the vendored tree: the limit and what it kills.

Two defects, both observed while acli was resolving a GitHub PR conflict:

* The default was 30s -- shorter than a project's test or build command, and
  invisible to the model. `sleep 150 && gh pr checks 162` came back "command
  timed out (30s)" and acli spent further turns rediscovering by experiment
  that `sleep 25` fits. `/info` made it worse by printing an unrelated
  ``config.timeout`` under the bare label "Timeout".
* On timeout only the direct child was killed. ``proc.kill()`` reaches the
  shell, not the pipeline it started, so the work kept running with its
  stdout pipe still open -- and asyncio later logged "Exception ignored in
  BaseSubprocessTransport.__del__ ... Event loop is closed".

The orphan tests are marker-based, not pid-based: a SIGKILLed child nobody
reaps is a zombie, and ``os.kill(pid, 0)`` still succeeds on a zombie, so the
pid alone says nothing about whether the tree died.
"""

# pylint: disable=protected-access,redefined-outer-name

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dashscope.acli.tools import shell


def _fake_proc(stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = 0
    return proc


@pytest.fixture
def marker(tmp_path):
    """Path a surviving grandchild would create."""
    path = tmp_path / "orphan-marker"
    yield path
    if path.exists():
        path.unlink()


def _orphan_command(marker) -> str:
    """Backgrounds a writer that only fires if the kill misses the group.

    Non-interactive sh leaves background jobs in its own process group, so a
    group kill takes the writer with it and a child-only kill does not.
    """
    return f"(sleep 2; touch '{marker}') & sleep 30"


@pytest.fixture
def spy_wait_for(monkeypatch):
    """Record the timeout run_command hands to asyncio.wait_for."""
    seen: dict = {}
    real = asyncio.wait_for

    async def spy(awaitable, timeout=None):
        seen["timeout"] = timeout
        return await real(awaitable, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", spy)
    return seen


class TestDefaultTimeout:
    def test_default_covers_a_real_test_run(self, monkeypatch):
        """The regression: 30s killed ordinary test and build commands."""
        monkeypatch.delenv(shell.TIMEOUT_ENV, raising=False)
        assert shell.default_timeout() >= 300

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv(shell.TIMEOUT_ENV, "45")
        assert shell.default_timeout() == 45

    @pytest.mark.parametrize("raw", ["", "bogus", "0", "-5"])
    def test_unusable_override_falls_back(self, monkeypatch, raw):
        """A bad override must not become a zero or negative deadline."""
        monkeypatch.setenv(shell.TIMEOUT_ENV, raw)
        assert shell.default_timeout() == shell.DEFAULT_TIMEOUT

    async def test_argument_overrides_default(self, spy_wait_for):
        await shell.run_command("true", timeout=7)
        assert spy_wait_for["timeout"] == 7

    @pytest.mark.parametrize("value", [None, "", 0, "auto"])
    async def test_absent_argument_uses_default(
        self,
        monkeypatch,
        spy_wait_for,
        value,
    ):
        monkeypatch.delenv(shell.TIMEOUT_ENV, raising=False)
        await shell.run_command("true", timeout=value)
        assert spy_wait_for["timeout"] == shell.DEFAULT_TIMEOUT


class TestTimeoutIsDiscoverable:
    def test_tool_description_advertises_the_argument(self):
        """The model cannot pass a knob it has never been told exists."""
        description = shell.run_command._tool_definition.description
        assert "timeout" in description

    async def test_timeout_message_names_the_escape_hatch(self, monkeypatch):
        """A bare "timed out (30s)" left the model guessing magic numbers."""
        monkeypatch.delenv(shell.TIMEOUT_ENV, raising=False)
        out = await shell.run_command("sleep 30", timeout=1)
        assert "timed out" in out
        assert "timeout argument" in out
        assert f"{shell.DEFAULT_TIMEOUT}s" in out


class TestChildGetsItsOwnProcessGroup:
    async def test_posix_shell_branch_starts_a_new_session(self, monkeypatch):
        monkeypatch.setattr(shell, "IS_WINDOWS", False)
        with patch(
            "asyncio.create_subprocess_shell",
            new=AsyncMock(return_value=_fake_proc()),
        ) as spawn:
            await shell.run_command("true")
        assert spawn.await_args.kwargs["start_new_session"] is True

    async def test_sandbox_branch_starts_a_new_session(self, monkeypatch):
        monkeypatch.setattr(shell, "IS_WINDOWS", False)
        with (
            patch("dashscope.acli.sandbox.is_enabled", return_value=True),
            patch(
                "dashscope.acli.sandbox.build_argv",
                return_value=["/usr/bin/sandbox-exec", "-f", "p", "true"],
            ),
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=_fake_proc()),
            ) as spawn,
        ):
            await shell.run_command("true")
        assert spawn.await_args.kwargs["start_new_session"] is True

    async def test_windows_branch_creates_a_new_group(self, monkeypatch):
        """Without the flag, taskkill /T has no tree to walk."""
        monkeypatch.setattr(shell, "IS_WINDOWS", True)
        monkeypatch.setattr(shell, "_NEW_GROUP_FLAGS", 0x200)
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_fake_proc()),
        ) as spawn:
            await shell.run_command("true")
        assert spawn.await_args.kwargs["creationflags"] == 0x200


class TestTimeoutKillsTheWholeTree:
    async def test_no_orphan_survives_a_timeout(self, marker):
        out = await shell.run_command(_orphan_command(marker), timeout=1)
        assert "timed out" in out
        # past t=2s, when the backgrounded writer would fire if it survived
        await asyncio.sleep(2.5)
        assert not marker.exists()

    async def test_cancellation_also_kills_the_tree(self, marker):
        """Ctrl-C must not leave the command running either."""
        task = asyncio.create_task(shell.run_command(_orphan_command(marker)))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(2.5)
        assert not marker.exists()


class TestKillIsBestEffort:
    def test_gone_process_group_falls_back_to_child_kill(self):
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 4242
        with patch("os.killpg", side_effect=ProcessLookupError):
            shell._signal_tree(proc)
        proc.kill.assert_called_once_with()

    def test_already_exited_process_is_left_alone(self):
        proc = MagicMock()
        proc.returncode = 0
        with patch("os.killpg") as killpg:
            shell._signal_tree(proc)
        killpg.assert_not_called()
        proc.kill.assert_not_called()

    def test_windows_kill_uses_taskkill_with_the_tree_flag(self, monkeypatch):
        monkeypatch.setattr(shell, "IS_WINDOWS", True)
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 4242
        with patch("subprocess.run") as run:
            shell._signal_tree(proc)
        argv = run.call_args.args[0]
        assert argv[0] == "taskkill"
        assert "/F" in argv and "/T" in argv
        assert argv[-1] == "4242"

    async def test_spawn_failure_still_returns_an_error(self):
        """proc stays None, so the cleanup path must tolerate having nothing."""
        with patch(
            "asyncio.create_subprocess_shell",
            new=AsyncMock(side_effect=OSError("no such shell")),
        ):
            out = await shell.run_command("true")
        assert "execution failed" in out


class TestInfoReportsTheEnforcedTimeout:
    """/info used to print config.timeout as "Timeout".

    That field is the LLM request timeout consumed by providers/profile.py.
    run_command enforced a different number entirely, so the one place a user
    could look reported a limit nothing used.
    """

    @staticmethod
    def _config(**overrides):
        from types import SimpleNamespace

        defaults = {
            "provider": "tongyi",
            "model": "qwen3.7-max",
            "protocol": "openai",
            "base_url": "",
            "user_name": "tester",
            "memory_enabled": True,
            "enabled_capabilities": ["bailian.mcp"],
            "loop_mode": "auto",
            "max_turns": 1000,
            "timeout": 30,
            "tui": True,
            "privacy_mode": False,
            "tongyi_api_key": "",
            "anthropic_api_key": "",
            "openai_api_key": "",
            "fallback_providers": None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @staticmethod
    def _agent():
        from types import SimpleNamespace

        return SimpleNamespace(
            provider_name="tongyi",
            model_name="qwen3.7-max",
            json_mode=False,
        )

    def test_command_timeout_is_shown_separately(self, monkeypatch):
        from dashscope.acli.cli.dispatch import _handle_slash_command

        monkeypatch.setenv(shell.TIMEOUT_ENV, "45")
        console = MagicMock()
        monkeypatch.setattr("dashscope.acli.cli.dispatch.console", console)

        _handle_slash_command("/info", self._agent(), self._config())

        printed = " ".join(str(c) for c in console.print.call_args_list)
        assert "LLM timeout" in printed
        assert "Cmd timeout" in printed
        assert "45s" in printed
