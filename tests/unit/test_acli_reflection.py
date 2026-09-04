# -*- coding: utf-8 -*-
"""Unit tests for dashscope.acli.memory.reflection."""

import pytest

from dashscope.acli.memory.reflection import (
    ReflectionTracker,
    StagnationTracker,
    convergence_hint,
    is_readonly_tool_call,
)


# ── is_readonly_tool_call ────────────────────────────────────────────────────


class TestIsReadonlyToolCall:  # pylint: disable=too-many-public-methods
    """Tests for the is_readonly_tool_call classifier."""

    # -- Known read-only tools --

    @pytest.mark.parametrize(
        "tool_name",
        ["read_file", "search_files", "list_directory", "memory_search"],
    )
    def test_known_readonly_tools(self, tool_name):
        assert is_readonly_tool_call(tool_name, {}) is True

    def test_write_file_is_not_readonly(self):
        assert is_readonly_tool_call("write_file", {"path": "x.txt"}) is False

    def test_delete_file_is_not_readonly(self):
        assert is_readonly_tool_call("delete_file", {"path": "x.txt"}) is False

    def test_mcp_tool_is_not_readonly(self):
        assert is_readonly_tool_call("mcp_something", {}) is False

    def test_unknown_tool_is_not_readonly(self):
        assert is_readonly_tool_call("some_unknown_tool", {}) is False

    # -- run_command: read-only shell commands --

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat /etc/hosts",
            "head -n 10 file.txt",
            "tail -f log.txt",
            "grep -r 'pattern' src/",
            "find . -name '*.py'",
            "wc -l main.py",
            "ps aux",
            "df -h",
            "du -sh .",
            "whoami",
            "hostname",
            "uname -a",
            "pwd",
            "date",
            "nproc",
            "uptime",
        ],
    )
    def test_readonly_commands(self, command):
        assert (
            is_readonly_tool_call("run_command", {"command": command}) is True
        )

    # -- run_command: mutating commands --
    # Note: `git` is in _WRITE_MARKERS (conservative: all git treated as mutating)

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /tmp/test",
            "mv src.py dst.py",
            "cp src.py dst.py",
            "mkdir new_dir",
            "touch new_file.txt",
            "chmod +x script.sh",
            "pip install requests",
            "npm install",
            "git push",
            "git commit -m 'fix'",
            "git status",
            "git log --oneline -5",
            "git diff HEAD~1",
            "git branch -a",
            "python script.py",
            "python3 -c 'print(1)'",
            "node server.js",
            "make build",
            "cargo build",
            "go run main.go",
            "sed -i 's/a/b/' file.txt",
            "tee output.txt",
            "curl https://example.com",
            "wget https://example.com",
        ],
    )
    def test_mutating_commands(self, command):
        assert (
            is_readonly_tool_call("run_command", {"command": command}) is False
        )

    # -- Pipe chains --

    def test_readonly_pipe_chain(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "cat file.txt | grep foo | wc -l"},
            )
            is True
        )

    def test_mutating_in_pipe(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "cat file.txt | tee backup.txt | wc -l"},
            )
            is False
        )

    # -- Semicolon / && separated commands --

    def test_all_readonly_segments(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "ls -la; pwd; date"},
            )
            is True
        )

    def test_mutating_segment_among_readonly(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "ls -la; rm -f tmp.txt; pwd"},
            )
            is False
        )

    def test_and_chain_with_mutating(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "ls && rm -f tmp.txt"},
            )
            is False
        )

    def test_or_chain_all_readonly(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "ls || pwd"},
            )
            is True
        )

    # -- Redirections --

    def test_output_redirect_is_mutating(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "echo hello > output.txt"},
            )
            is False
        )

    def test_append_redirect_is_mutating(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "echo hello >> log.txt"},
            )
            is False
        )

    def test_devnull_redirect_is_benign(self):
        """Redirecting stderr to /dev/null should not make it mutating."""
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "ls 2>/dev/null"},
            )
            is True
        )

    def test_2_to_1_redirect_is_benign(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "grep foo file 2>&1"},
            )
            is True
        )

    # -- Container subcommands --

    @pytest.mark.parametrize(
        "command",
        [
            "docker ps",
            "docker images",
            "docker logs mycontainer",
            "docker inspect abc123",
            "podman ps",
            "colima status",
        ],
    )
    def test_container_readonly_subcmds(self, command):
        assert (
            is_readonly_tool_call("run_command", {"command": command}) is True
        )

    def test_container_mutating_subcmd(self):
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "docker run nginx"},
            )
            is False
        )

    # -- Edge cases --

    def test_empty_command(self):
        assert is_readonly_tool_call("run_command", {"command": ""}) is False

    def test_no_arguments(self):
        assert is_readonly_tool_call("run_command", None) is False

    def test_non_string_command(self):
        assert is_readonly_tool_call("run_command", {"command": 123}) is False

    def test_command_with_flags_only(self):
        """A segment with only flags and no verb is not readonly."""
        assert (
            is_readonly_tool_call("run_command", {"command": "-la"}) is False
        )

    def test_env_prefix_stripped(self):
        """`env VAR=val cmd` should classify by `cmd`."""
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "env FOO=bar ls -la"},
            )
            is True
        )

    def test_env_with_assignment_only_is_benign(self):
        """`env VAR=val` with no command after stripping → empty stage → readonly."""
        assert (
            is_readonly_tool_call(
                "run_command",
                {"command": "env FOO=bar"},
            )
            is True
        )


# ── ReflectionTracker ────────────────────────────────────────────────────────


class TestReflectionTracker:
    def test_starts_below_threshold(self):
        tracker = ReflectionTracker(threshold=3)
        assert tracker.needs_reflection() is False

    def test_below_threshold_no_hint(self):
        tracker = ReflectionTracker(threshold=3)
        tracker.record_failure("read_file")
        tracker.record_failure("read_file")
        assert tracker.needs_reflection() is False
        assert tracker.get_reflection_hint() == ""

    def test_reaches_threshold(self):
        tracker = ReflectionTracker(threshold=3)
        tracker.record_failure("write_file")
        tracker.record_failure("write_file")
        tracker.record_failure("run_command")
        assert tracker.needs_reflection() is True

    def test_success_resets(self):
        tracker = ReflectionTracker(threshold=3)
        tracker.record_failure("write_file")
        tracker.record_failure("write_file")
        tracker.record_success()
        assert tracker.needs_reflection() is False
        assert tracker.consecutive_failures == 0

    def test_hint_contains_tool_names(self):
        tracker = ReflectionTracker(threshold=2)
        tracker.record_failure("read_file")
        tracker.record_failure("run_command")
        hint = tracker.get_reflection_hint()
        assert "read_file" in hint
        assert "run_command" in hint
        assert "2 consecutive" in hint

    def test_hint_deduplicates_tool_names(self):
        tracker = ReflectionTracker(threshold=2)
        tracker.record_failure("read_file")
        tracker.record_failure("read_file")
        hint = tracker.get_reflection_hint()
        # "read_file" should appear once in the joined set
        assert hint.count("read_file") == 1

    def test_record_tool_execution_routes_success(self):
        tracker = ReflectionTracker(threshold=3)
        tracker.record_failure("x")
        tracker.record_tool_execution("y", success=True)
        assert tracker.consecutive_failures == 0

    def test_record_tool_execution_routes_failure(self):
        tracker = ReflectionTracker(threshold=3)
        tracker.record_tool_execution("x", success=False)
        assert tracker.consecutive_failures == 1

    def test_reset(self):
        tracker = ReflectionTracker(threshold=3)
        tracker.record_failure("x")
        tracker.record_failure("y")
        tracker.reset()
        assert tracker.consecutive_failures == 0
        assert not tracker.last_failed_tools
        assert tracker.needs_reflection() is False

    def test_failure_lesson_below_threshold(self):
        tracker = ReflectionTracker(threshold=3)
        tracker.record_failure("x")
        assert tracker.get_failure_lesson() == ""

    def test_failure_lesson_at_threshold(self):
        tracker = ReflectionTracker(threshold=2)
        tracker.record_failure("read_file")
        tracker.record_failure("read_file")
        lesson = tracker.get_failure_lesson()
        assert "2 consecutive failures" in lesson
        assert "read_file" in lesson


# ── StagnationTracker ────────────────────────────────────────────────────────


class TestStagnationTracker:
    def test_starts_clean(self):
        tracker = StagnationTracker(threshold=8)
        assert tracker.needs_nudge() is False
        assert tracker.get_stagnation_hint() == ""

    def test_below_threshold(self):
        tracker = StagnationTracker(threshold=8)
        for _ in range(7):
            tracker.record(readonly=True)
        assert tracker.needs_nudge() is False

    def test_reaches_threshold(self):
        tracker = StagnationTracker(threshold=8)
        for _ in range(8):
            tracker.record(readonly=True)
        assert tracker.needs_nudge() is True

    def test_mutating_resets_streak(self):
        tracker = StagnationTracker(threshold=8)
        for _ in range(7):
            tracker.record(readonly=True)
        tracker.record(readonly=False)
        assert tracker.readonly_streak == 0
        assert tracker.needs_nudge() is False

    def test_hint_contains_streak_count(self):
        tracker = StagnationTracker(threshold=3)
        for _ in range(5):
            tracker.record(readonly=True)
        hint = tracker.get_stagnation_hint()
        assert "5 consecutive" in hint

    def test_hint_with_hard_cap(self):
        tracker = StagnationTracker(threshold=3)
        for _ in range(5):
            tracker.record(readonly=True)
        hint = tracker.get_stagnation_hint(hard_cap=10)
        assert "Hard stop in 5 more" in hint

    def test_hint_without_hard_cap(self):
        tracker = StagnationTracker(threshold=3)
        for _ in range(5):
            tracker.record(readonly=True)
        hint = tracker.get_stagnation_hint(hard_cap=None)
        assert "Hard stop" not in hint

    def test_hint_hard_cap_at_streak(self):
        """When streak == hard_cap, remaining is 0 → no hard stop line."""
        tracker = StagnationTracker(threshold=3)
        for _ in range(5):
            tracker.record(readonly=True)
        hint = tracker.get_stagnation_hint(hard_cap=5)
        # hard_cap > n is False (5 > 5 is False), so no hard stop line
        assert "Hard stop" not in hint

    def test_reset(self):
        tracker = StagnationTracker(threshold=3)
        for _ in range(5):
            tracker.record(readonly=True)
        tracker.reset()
        assert tracker.readonly_streak == 0
        assert tracker.needs_nudge() is False

    def test_mixed_sequence(self):
        """Interleaved reads and writes should reset properly."""
        tracker = StagnationTracker(threshold=3)
        tracker.record(True)
        tracker.record(True)
        tracker.record(False)  # reset
        tracker.record(True)
        tracker.record(True)
        assert tracker.readonly_streak == 2
        assert tracker.needs_nudge() is False


# ── convergence_hint ──────────────────────────────────────────────────────────


class TestConvergenceHint:
    """Note: used = loop_index + 1, remaining = max(0, max_turns - used)."""

    def test_below_soft_returns_empty(self):
        # loop=0, max=100 → used=1, frac=0.01 < 0.6
        assert convergence_hint(0, 100) == ""

    def test_at_soft_boundary(self):
        # loop=59, max=100 → used=60, frac=0.60 → fires soft
        hint = convergence_hint(59, 100)
        assert "Budget check" in hint

    def test_between_soft_and_hard(self):
        # loop=70, max=100 → used=71, frac=0.71 → soft band, remaining=29
        hint = convergence_hint(70, 100, soft_ratio=0.6, hard_ratio=0.85)
        assert "Budget check" in hint
        assert "29 left" in hint

    def test_at_hard_boundary(self):
        # loop=84, max=100 → used=85, frac=0.85 → fires hard
        hint = convergence_hint(84, 100, soft_ratio=0.6, hard_ratio=0.85)
        assert "converge now" in hint

    def test_past_hard(self):
        # loop=95, max=100 → used=96, frac=0.96 → hard, remaining=4
        hint = convergence_hint(95, 100, soft_ratio=0.6, hard_ratio=0.85)
        assert "converge now" in hint
        assert "4 left" in hint

    def test_max_turns_zero_returns_empty(self):
        assert convergence_hint(10, 0) == ""

    def test_soft_ratio_ge_1_disables(self):
        assert convergence_hint(99, 100, soft_ratio=1.0) == ""
        assert convergence_hint(99, 100, soft_ratio=2.0) == ""

    def test_custom_ratios(self):
        # loop=4, max=10 → used=5, frac=0.5; soft=0.4 → fires soft
        hint = convergence_hint(4, 10, soft_ratio=0.4, hard_ratio=0.8)
        assert "Budget check" in hint

    def test_custom_ratios_hard(self):
        # loop=8, max=10 → used=9, frac=0.9; hard=0.8 → fires hard
        hint = convergence_hint(8, 10, soft_ratio=0.4, hard_ratio=0.8)
        assert "converge now" in hint

    def test_remaining_is_non_negative(self):
        # loop=100, max=100 → used=101, remaining=max(0,-1)=0
        hint = convergence_hint(100, 100, soft_ratio=0.6, hard_ratio=0.85)
        assert "0 left" in hint

    def test_soft_hint_contains_switch_advice(self):
        hint = convergence_hint(70, 100)
        assert "Switch to a structurally different strategy" in hint

    def test_hard_hint_says_no_new_approach(self):
        hint = convergence_hint(90, 100)
        assert "Do NOT start a new approach" in hint
