# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope ft`` -- the second pattern command group."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from dashscope import errors as sdk_errors
from dashscope.cli.app import build_app
from dashscope.resources.fine_tunes import (
    FineTune,
    FineTuneEvent,
    FineTuneList,
    JobLogs,
)
from dashscope.types.common import JobStatus, Usage


def _job(**overrides: Any) -> FineTune:
    data: Dict[str, Any] = {
        "job_id": "ft-202601011234-abc1",
        "status": "SUCCEEDED",
        "model": "qwen-plus",
        "finetuned_model": "qwen-plus-ft-abc1",
        "created_at": "2026-01-01 12:34:56",
        "usage": Usage(input_tokens=10, output_tokens=0, total_tokens=10),
    }
    data.update(overrides)
    return FineTune(**data)


class _FineTunes:
    """Stand-in for ``client.fine_tunes`` recording every call."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def _record(self, name: str, *args, **kwargs):
        self.calls.append({"method": name, "args": args, "kwargs": kwargs})
        result = self._results.get(name)
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result

    def create(self, **kwargs):
        return self._record("create", **kwargs)

    def get(self, job_id, **kwargs):
        return self._record("get", job_id, **kwargs)

    def list(self, **kwargs):
        return self._record("list", **kwargs)

    def cancel(self, job_id, **kwargs):
        return self._record("cancel", job_id, **kwargs)

    def delete(self, job_id, **kwargs):
        return self._record("delete", job_id, **kwargs)

    def stream_events(self, job_id, **kwargs):
        self._record("stream_events", job_id, **kwargs)
        return self._results["stream_events"]

    def logs(self, job_id, **kwargs):
        return self._record("logs", job_id, **kwargs)

    def wait(self, job, **kwargs):
        self._record("wait", job, **kwargs)
        return self._results.get("wait", job)


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def fine_tunes(make_client):
    """Install a fake client and hand back its ``fine_tunes`` recorder."""

    def install(**results: Any) -> _FineTunes:
        resource = _FineTunes(**results)
        make_client(fine_tunes=resource)
        return resource

    return install


def _as_terminal_tty(monkeypatch, answer: bool):
    monkeypatch.setattr("dashscope.cli.confirm.stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        "dashscope.cli.confirm.typer.confirm",
        lambda *a, **k: answer,
    )


class TestCreate:
    def test_required_options_are_enforced(self, invoke, fine_tunes):
        fine_tunes(create=_job())
        assert invoke(["ft", "create"]).exit_code == 2
        assert invoke(["ft", "create", "-m", "qwen-plus"]).exit_code == 2

    def test_the_model_does_not_fall_back_to_default_model(
        self,
        invoke,
        fine_tunes,
        clean_env,
    ):
        """A chat default is not a fine-tune base model."""
        from dashscope.cli import config_store

        config_store.set_value("default_model", "qwen-plus")
        result = invoke(["ft", "create", "-t", "file-1"])
        assert result.exit_code == 2

    def test_ids_and_model_reach_the_sdk(self, invoke, fine_tunes):
        resource = fine_tunes(create=_job())
        assert (
            invoke(
                [
                    "ft",
                    "create",
                    "-m",
                    "qwen-plus",
                    "-t",
                    "file-1",
                    "-t",
                    "file-2",
                    "-v",
                    "file-3",
                ],
            ).exit_code
            == 0
        )
        call = resource.calls[0]
        assert call["kwargs"]["model"] == "qwen-plus"
        assert call["kwargs"]["training_file_ids"] == ["file-1", "file-2"]
        assert call["kwargs"]["validation_file_ids"] == ["file-3"]

    def test_shortcut_flags_become_hyper_parameters(self, invoke, fine_tunes):
        resource = fine_tunes(create=_job())
        invoke(
            [
                "ft",
                "create",
                "-m",
                "qwen-plus",
                "-t",
                "f",
                "-e",
                "3",
                "-b",
                "8",
                "-l",
                "0.0001",
                "-p",
                "0.5",
            ],
        )
        assert resource.calls[0]["kwargs"]["hyper_parameters"] == {
            "n_epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "prompt_loss": 0.5,
        }

    def test_explicit_hyper_parameters_pass_through(self, invoke, fine_tunes):
        resource = fine_tunes(create=_job())
        invoke(
            [
                "ft",
                "create",
                "-m",
                "qwen-plus",
                "-t",
                "f",
                "--hyper-parameters",
                '{"n_epochs": 2}',
            ],
        )
        assert resource.calls[0]["kwargs"]["hyper_parameters"] == {
            "n_epochs": 2,
        }

    def test_the_two_spellings_of_one_key_are_refused(
        self,
        invoke,
        fine_tunes,
    ):
        """Silently picking a winner would hide a typo in the loser."""
        resource = fine_tunes(create=_job())
        result = invoke(
            [
                "ft",
                "create",
                "-m",
                "qwen-plus",
                "-t",
                "f",
                "--hyper-parameters",
                '{"n_epochs": 2}',
                "-e",
                "3",
            ],
        )
        assert result.exit_code == 2
        assert "n_epochs" in result.stderr
        assert resource.calls == []

    def test_invalid_hyper_parameters_json_is_a_usage_error(
        self,
        invoke,
        fine_tunes,
    ):
        fine_tunes(create=_job())
        result = invoke(
            [
                "ft",
                "create",
                "-m",
                "qwen-plus",
                "-t",
                "f",
                "--hyper-parameters",
                "[",
            ],
        )
        assert result.exit_code == 2
        assert "--hyper-parameters" in result.stderr

    def test_table_output_names_the_job(self, invoke, fine_tunes):
        fine_tunes(create=_job())
        result = invoke(["ft", "create", "-m", "qwen-plus", "-t", "f"])
        assert "ft-202601011234-abc1" in result.stdout
        assert "SUCCEEDED" in result.stdout

    def test_json_output_is_the_whole_job(self, invoke, fine_tunes):
        fine_tunes(create=_job())
        result = invoke(
            ["ft", "create", "-m", "qwen-plus", "-t", "f"],
            global_args=["-o", "json"],
        )
        payload = json.loads(result.stdout)
        assert payload["job_id"] == "ft-202601011234-abc1"
        assert payload["finetuned_model"] == "qwen-plus-ft-abc1"


class TestWait:
    def test_wait_polls_through_the_sdk(self, invoke, fine_tunes):
        resource = fine_tunes(create=_job(status="RUNNING"), wait=_job())
        assert (
            invoke(
                ["ft", "create", "-m", "qwen-plus", "-t", "f", "--wait"],
            ).exit_code
            == 0
        )
        methods = [call["method"] for call in resource.calls]
        assert methods == ["create", "wait"]

    def test_wait_renders_the_final_job_not_the_initial_one(
        self,
        invoke,
        fine_tunes,
    ):
        fine_tunes(
            create=_job(status="RUNNING", finetuned_model=None),
            wait=_job(status="SUCCEEDED", finetuned_model="qwen-plus-ft-done"),
        )
        result = invoke(
            ["ft", "create", "-m", "qwen-plus", "-t", "f", "--wait"],
        )
        assert "SUCCEEDED" in result.stdout
        assert "qwen-plus-ft-done" in result.stdout
        assert "RUNNING" not in result.stdout

    def test_without_wait_the_sdk_is_not_polled(self, invoke, fine_tunes):
        resource = fine_tunes(create=_job(status="RUNNING"))
        invoke(["ft", "create", "-m", "qwen-plus", "-t", "f"])
        assert [call["method"] for call in resource.calls] == ["create"]

    def test_the_timeout_is_forwarded(self, invoke, fine_tunes):
        resource = fine_tunes(create=_job(), wait=_job())
        invoke(
            [
                "ft",
                "create",
                "-m",
                "qwen-plus",
                "-t",
                "f",
                "--wait",
                "--timeout",
                "30",
            ],
        )
        wait_call = next(c for c in resource.calls if c["method"] == "wait")
        assert wait_call["kwargs"]["timeout"] == 30

    def test_a_failed_job_exits_nonzero(self, invoke, fine_tunes):
        """The 1.x `_wait_for_job` printed the failure and exited 0."""
        fine_tunes(
            create=_job(status="RUNNING"),
            wait=sdk_errors.InternalServerError("training diverged"),
        )
        result = invoke(
            ["ft", "create", "-m", "qwen-plus", "-t", "f", "--wait"],
        )
        assert result.exit_code == 1
        assert "Error [server]: training diverged" in result.stderr


class TestWaitTimeoutIsCheckedBeforeSubmitting:
    """``--wait --timeout -1`` used to create the job and then give up on it."""

    @pytest.mark.parametrize("timeout", ["-1", "0", "nan", "inf"])
    def test_no_job_is_created(self, invoke, fine_tunes, timeout):
        resource = fine_tunes(create=_job(), wait=_job())
        result = invoke(
            [
                "ft",
                "create",
                "-m",
                "qwen-plus",
                "-t",
                "f",
                "--wait",
                "--timeout",
                timeout,
            ],
        )
        assert result.exit_code == 2
        assert resource.calls == []
        assert "--timeout" in result.stderr

    def test_an_unused_timeout_is_not_newly_rejected(
        self,
        invoke,
        fine_tunes,
    ):
        """``--timeout`` means nothing without ``--wait``."""
        resource = fine_tunes(create=_job())
        result = invoke(
            ["ft", "create", "-m", "qwen-plus", "-t", "f", "--timeout", "-1"],
        )
        assert result.exit_code == 0
        assert [call["method"] for call in resource.calls] == ["create"]

    @pytest.mark.parametrize("timeout", ["0", "nan"])
    def test_ft_wait_is_rejected_before_polling(
        self,
        invoke,
        fine_tunes,
        timeout,
    ):
        """``nan`` is false against every comparison, so the poll never ends."""
        resource = fine_tunes(wait=_job())
        result = invoke(["ft", "wait", "ft-1", "--timeout", timeout])
        assert result.exit_code == 2
        assert "--timeout" in result.stderr
        assert resource.calls == []


class TestWaitCommand:
    """``ft wait``: block on a job this process did not create."""

    def test_it_polls_the_id_and_nothing_else(self, invoke, fine_tunes):
        resource = fine_tunes(wait=_job())
        assert invoke(["ft", "wait", "ft-1"]).exit_code == 0
        assert [call["method"] for call in resource.calls] == ["wait"]
        assert resource.calls[0]["args"] == ("ft-1",)

    def test_the_timeout_defaults_to_an_hour(self, invoke, fine_tunes):
        resource = fine_tunes(wait=_job())
        invoke(["ft", "wait", "ft-1"])
        assert resource.calls[0]["kwargs"] == {"timeout": 3600.0}

    def test_the_timeout_is_forwarded(self, invoke, fine_tunes):
        resource = fine_tunes(wait=_job())
        invoke(["ft", "wait", "ft-1", "--timeout", "30"])
        assert resource.calls[0]["kwargs"] == {"timeout": 30.0}

    def test_table_shows_the_settled_job(self, invoke, fine_tunes):
        fine_tunes(wait=_job(status="SUCCEEDED"))
        result = invoke(["ft", "wait", "ft-1"])
        assert "SUCCEEDED" in result.stdout

    def test_json_output_is_the_whole_job(self, invoke, fine_tunes):
        fine_tunes(wait=_job())
        result = invoke(["ft", "wait", "ft-1"], global_args=["-o", "json"])
        payload = json.loads(result.stdout)
        assert payload["job_id"] == "ft-202601011234-abc1"
        assert payload["status"] == "SUCCEEDED"

    def test_a_failed_job_exits_nonzero(self, invoke, fine_tunes):
        """Same rule as `create --wait`: a failure is never exit 0."""
        fine_tunes(wait=sdk_errors.InternalServerError("training diverged"))
        result = invoke(["ft", "wait", "ft-1"])
        assert result.exit_code == 1
        assert "Error [server]: training diverged" in result.stderr

    def test_a_missing_job_exits_four(self, invoke, fine_tunes):
        fine_tunes(wait=sdk_errors.NotFoundError("job not found"))
        assert invoke(["ft", "wait", "ft-nope"]).exit_code == 4

    def test_the_job_id_is_required(self, invoke, fine_tunes):
        resource = fine_tunes(wait=_job())
        assert invoke(["ft", "wait"]).exit_code == 2
        assert resource.calls == []


class TestGet:
    def test_table_shows_the_real_created_at(self, invoke, fine_tunes):
        """Regression pin: the 1.x `rl get` printed ``creator`` for
        ``created_at`` (``agentic_rl.py:618``)."""
        fine_tunes(
            get=_job(created_at="2026-01-01 12:34:56", creator="user-999"),
        )
        result = invoke(["ft", "get", "ft-1"])
        assert "2026-01-01 12:34:56" in result.stdout
        assert "user-999" not in result.stdout

    def test_the_job_id_is_forwarded(self, invoke, fine_tunes):
        resource = fine_tunes(get=_job())
        invoke(["ft", "get", "ft-abc"])
        assert resource.calls[0]["args"] == ("ft-abc",)

    def test_a_missing_job_exits_four(self, invoke, fine_tunes):
        fine_tunes(get=sdk_errors.NotFoundError("job not found"))
        result = invoke(["ft", "get", "ft-nope"])
        assert result.exit_code == 4
        assert "Error [notfound]: job not found" in result.stderr

    def test_json_output_is_the_whole_job(self, invoke, fine_tunes):
        fine_tunes(get=_job(creator="user-999"))
        result = invoke(["ft", "get", "ft-1"], global_args=["-o", "json"])
        payload = json.loads(result.stdout)
        assert payload["created_at"] == "2026-01-01 12:34:56"
        # Machine output keeps every field; only the table narrows it.
        assert payload["creator"] == "user-999"


class TestList:
    @staticmethod
    def _page(jobs, total=None, page_no=1, page_size=10):
        return FineTuneList(
            page_no=page_no,
            page_size=page_size,
            total_count=len(jobs) if total is None else total,
            jobs=jobs,
        )

    def test_pagination_is_forwarded(self, invoke, fine_tunes):
        resource = fine_tunes(list=self._page([]))
        invoke(["ft", "list", "--page", "3", "--size", "5"])
        assert resource.calls[0]["kwargs"] == {"page_no": 3, "page_size": 5}

    def test_table_shows_rows_and_a_footer(self, invoke, fine_tunes):
        fine_tunes(
            list=self._page(
                [_job(job_id="ft-a"), _job(job_id="ft-b")],
                total=25,
                page_no=2,
                page_size=10,
            ),
        )
        result = invoke(["ft", "list"])
        assert "ft-a" in result.stdout and "ft-b" in result.stdout
        # The footer is a diagnostic, so it stays off stdout.
        assert "page 2/3, total 25" in result.stderr
        assert "page 2/3" not in result.stdout

    def test_json_output_is_the_whole_envelope(self, invoke, fine_tunes):
        """A caller must be able to page without a second request."""
        fine_tunes(list=self._page([_job(job_id="ft-a")], total=25))
        result = invoke(["ft", "list"], global_args=["-o", "json"])
        payload = json.loads(result.stdout)
        assert payload["total_count"] == 25
        assert payload["jobs"][0]["job_id"] == "ft-a"


class TestStream:
    @staticmethod
    def _events(*outputs):
        return iter([FineTuneEvent(output=o) for o in outputs])

    def test_table_prints_text_frames_verbatim(self, invoke, fine_tunes):
        fine_tunes(stream_events=self._events("epoch 1", "epoch 2"))
        result = invoke(["ft", "stream", "ft-1"])
        assert result.exit_code == 0
        assert result.stdout.splitlines() == ["epoch 1", "epoch 2"]

    def test_table_compresses_dict_frames_to_one_line(
        self,
        invoke,
        fine_tunes,
    ):
        fine_tunes(stream_events=self._events({"loss": 0.5, "step": 10}))
        result = invoke(["ft", "stream", "ft-1"])
        lines = result.stdout.splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"loss": 0.5, "step": 10}

    def test_json_emits_one_frame_per_line(self, invoke, fine_tunes):
        fine_tunes(stream_events=self._events("a", {"loss": 1}))
        result = invoke(["ft", "stream", "ft-1"], global_args=["-o", "json"])
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 2
        assert [json.loads(line)["output"] for line in lines] == [
            "a",
            {"loss": 1},
        ]

    def test_a_server_error_frame_exits_nonzero(self, invoke, fine_tunes):
        """No ``except`` in the streaming path, so this cannot exit 0.

        The 1.x tree caught the error, printed it and returned -- the bug
        the design doc calls out.
        """

        def events():
            yield FineTuneEvent(output="starting")
            raise sdk_errors.InternalServerError("stream broke")

        fine_tunes(stream_events=events())
        result = invoke(["ft", "stream", "ft-1"])
        assert result.exit_code == 1
        assert "Error [server]: stream broke" in result.stderr
        assert "starting" in result.stdout

    def test_a_missing_job_exits_four(self, invoke, fine_tunes):
        fine_tunes(stream_events=sdk_errors.NotFoundError("job not found"))
        result = invoke(["ft", "stream", "ft-nope"])
        assert result.exit_code == 4


class TestLogs:
    def test_prints_the_log_text_verbatim(self, invoke, fine_tunes):
        """Raw text, not a table: a redirect must capture the trainer's own
        output and nothing else."""
        resource = fine_tunes(
            logs=JobLogs(job_id="ft-1", content="epoch 1\nepoch 2\n"),
        )
        result = invoke(["ft", "logs", "ft-1"])
        assert result.exit_code == 0
        assert result.stdout == "epoch 1\nepoch 2\n\n"
        assert resource.calls[0] == {
            "method": "logs",
            "args": ("ft-1",),
            "kwargs": {"offset": 1, "lines": 1000},
        }

    def test_the_window_is_adjustable(self, invoke, fine_tunes):
        resource = fine_tunes(logs=JobLogs(job_id="ft-1", content="tail"))
        invoke(["ft", "logs", "ft-1", "--offset", "40", "--lines", "20"])
        assert resource.calls[0]["kwargs"] == {"offset": 40, "lines": 20}

    def test_json_output_keeps_the_envelope(self, invoke, fine_tunes):
        fine_tunes(
            logs=JobLogs(
                job_id="ft-1",
                content="epoch 1\n",
                request_id="req-1",
            ),
        )
        result = invoke(["ft", "logs", "ft-1"], global_args=["-o", "json"])
        doc = json.loads(result.stdout)
        assert doc["job_id"] == "ft-1"
        assert doc["content"] == "epoch 1\n"
        assert doc["request_id"] == "req-1"

    def test_a_line_array_is_joined_by_the_sdk(self, invoke, fine_tunes):
        """The service may send lines instead of one string."""
        fine_tunes(logs=JobLogs(job_id="ft-1", content=["epoch 1", "epoch 2"]))
        result = invoke(["ft", "logs", "ft-1"])
        assert result.exit_code == 0
        assert "epoch 1\nepoch 2" in result.stdout

    def test_an_empty_log_is_not_an_error(self, invoke, fine_tunes):
        fine_tunes(logs=JobLogs(job_id="ft-1", content=""))
        result = invoke(["ft", "logs", "ft-1"])
        assert result.exit_code == 0
        assert result.stdout == "\n"

    def test_the_job_id_is_required(self, invoke, fine_tunes):
        fine_tunes(logs=JobLogs(job_id="ft-1", content="x"))
        assert invoke(["ft", "logs"]).exit_code == 2

    def test_a_missing_job_exits_four(self, invoke, fine_tunes):
        fine_tunes(logs=sdk_errors.NotFoundError("job not found"))
        result = invoke(["ft", "logs", "ft-nope"])
        assert result.exit_code == 4


class TestDestructiveCommands:
    @pytest.mark.parametrize("command", ["cancel", "delete"])
    def test_yes_skips_the_prompt(self, invoke, fine_tunes, command):
        resource = fine_tunes(**{command: JobStatus(status="CANCELED")})
        result = invoke(["ft", command, "ft-1", "--yes"])
        assert result.exit_code == 0
        assert [call["method"] for call in resource.calls] == [command]

    @pytest.mark.parametrize("command", ["cancel", "delete"])
    def test_short_flag_skips_the_prompt(self, invoke, fine_tunes, command):
        fine_tunes(**{command: JobStatus(status="CANCELED")})
        assert invoke(["ft", command, "ft-1", "-y"]).exit_code == 0

    @pytest.mark.parametrize("command", ["cancel", "delete"])
    def test_no_tty_without_yes_is_refused(self, invoke, fine_tunes, command):
        """A cron job that forgot ``--yes`` must fail loudly, not act."""
        resource = fine_tunes(**{command: JobStatus(status="CANCELED")})
        result = invoke(["ft", command, "ft-1"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        assert resource.calls == []

    @pytest.mark.parametrize(
        "command,answer",
        [("cancel", True), ("delete", True)],
    )
    def test_an_interactive_yes_proceeds(
        self,
        invoke,
        fine_tunes,
        monkeypatch,
        command,
        answer,
    ):
        _as_terminal_tty(monkeypatch, answer)
        resource = fine_tunes(**{command: JobStatus(status="CANCELED")})
        assert invoke(["ft", command, "ft-1"]).exit_code == 0
        assert [call["method"] for call in resource.calls] == [command]

    @pytest.mark.parametrize("command", ["cancel", "delete"])
    def test_an_interactive_no_aborts(
        self,
        invoke,
        fine_tunes,
        monkeypatch,
        command,
    ):
        _as_terminal_tty(monkeypatch, False)
        resource = fine_tunes(**{command: JobStatus(status="CANCELED")})
        result = invoke(["ft", command, "ft-1"])
        assert result.exit_code == 2
        assert "aborted by user" in result.stderr
        assert resource.calls == []

    def test_the_result_is_rendered(self, invoke, fine_tunes):
        fine_tunes(delete=JobStatus(status="DELETED"))
        result = invoke(
            ["ft", "delete", "ft-1", "--yes"],
            global_args=["-o", "json"],
        )
        assert json.loads(result.stdout)["status"] == "DELETED"

    def test_an_interactive_prompt_leaves_json_stdout_parseable(
        self,
        invoke,
        fine_tunes,
        monkeypatch,
    ):
        """The real click prompt, on a terminal, with ``-o json``.

        The prompt used to be written to stdout ahead of the JSON, so a
        machine consumer got neither.
        """
        monkeypatch.setattr(
            "dashscope.cli.confirm.stdin_is_tty",
            lambda: True,
        )
        fine_tunes(delete=JobStatus(status="DELETED"))
        result = invoke(
            ["ft", "delete", "ft-1"],
            global_args=["-o", "json"],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "delete fine-tune job ft-1?" not in result.stdout
        assert "delete fine-tune job ft-1?" in result.stderr
        # CliRunner echoes the keystroke into stdout; a real terminal
        # echoes it on the tty instead, so only the JSON is captured.
        payload = result.stdout[result.stdout.index("{") :]  # noqa: E203
        assert json.loads(payload)["status"] == "DELETED"

    def test_a_missing_job_exits_four(self, invoke, fine_tunes):
        fine_tunes(delete=sdk_errors.NotFoundError("job not found"))
        result = invoke(["ft", "delete", "ft-1", "--yes"])
        assert result.exit_code == 4


class TestGroupShape:
    def test_the_hidden_alias_runs_the_same_commands(self, invoke, fine_tunes):
        resource = fine_tunes(get=_job())
        assert invoke(["fine-tunes", "get", "ft-1"]).exit_code == 0
        assert resource.calls[0]["method"] == "get"

    def test_the_alias_is_hidden_from_the_root_help(self, app, runner):
        text = runner.invoke(app, ["--help"]).stdout
        assert "ft" in text
        assert "fine-tunes" not in text

    def test_bare_ft_shows_help(self, app, runner):
        assert "Usage" in runner.invoke(app, ["ft"]).stdout

    def test_help_does_not_leak_a_backslash(self, app, runner):
        assert "\\" not in runner.invoke(app, ["ft", "--help"]).stdout

    def test_the_router_sends_both_names_to_the_new_tree(self):
        from dashscope.cli.main import NEW, classify

        assert classify(["ft", "list"]) == NEW
        assert classify(["fine-tunes", "list"]) == NEW
