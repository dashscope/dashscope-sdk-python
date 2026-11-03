# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 CLI example: driving the CLI from a script or a CI job.

The command line is not a demo toy -- it is the interface your pipeline
talks to. That comes with a set of unglamorous requirements: parse
stdout instead of scraping a table, branch on the exit code instead of
grepping for the word "error", retry only the tiers that can succeed on
a second attempt, keep a job's configuration away from the developer's
``~/.dashscope``, and never print a secret into a build log.

This file writes all of that once, and proves it works by running the
**real** CLI as a subprocess against a **local mock gateway** built from
the standard library: real transport, real protocol, real exit codes,
zero network and zero tokens.

Scenarios:

1. Machine-readable output: parse ``--output json``, and why the table
   is for humans only (its diagnostics go to stderr).
2. Exit-code branching: 1 retryable vs 2/3/4 terminal, plus a retry
   wrapper that backs off and gives up.
3. Streaming consumed as NDJSON, line by line, as frames arrive.
4. Configuration isolation: a per-job ``DASHSCOPE_CONFIG``, the
   ``default_model`` fallback, api_key masking and the 0600 file mode.
5. ``-c KEY=VALUE`` overrides, including the fact that a misspelled key
   fails even on a command that never touches the service.
6. Long-running jobs: ``ft list`` pagination, ``ft stream`` events, and
   the destructive commands that refuse to run without a terminal.
7. CI hygiene: colour, and the environment worth setting in a job.

Run it offline (the default)::

    python3 examples/05_cli/cli_ci_harness.py

Or point the read-only scenarios at the real gateway::

    CLI_HARNESS_LIVE=1 DASHSCOPE_API_KEY=sk-xxx \
        python3 examples/05_cli/cli_ci_harness.py

Live mode skips every scenario that injects a fault, writes a
configuration file, or deletes a job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

# ---- configuration (override via env) ---- #
# Offline by default: the whole point is that a pipeline can rehearse
# its error handling without spending a token.
LIVE = os.environ.get("CLI_HARNESS_LIVE") == "1"
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
# A real job id, for the read-only `ft get`/`ft stream` in live mode.
DEFAULT_JOB_ID = "ft-20260101-demo"
JOB_ID = os.environ.get("DASHSCOPE_JOB_ID", DEFAULT_JOB_ID)
# Live mode only touches a job the caller named explicitly: guessing an
# id against a real account is how a demo deletes something important.
OWN_JOB = os.environ.get("DASHSCOPE_JOB_ID") is not None
RETRY_ATTEMPTS = 3

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir),
)

PROMPT = "请用一句话介绍杭州"
ANSWER = "杭州是浙江省省会，以西湖与龙井茶闻名。"
REQUEST_ID = "req-mock-0001"
USAGE = {"input_tokens": 18, "output_tokens": 22, "total_tokens": 40}

# The four tiers, mirroring dashscope/cli2/errors.py.
EXIT_OK = 0
EXIT_SERVER = 1  # server, ratelimit, network: worth retrying
EXIT_USAGE = 2  # the command line was wrong: retrying cannot help
EXIT_AUTH = 3  # bad credentials: retrying cannot help
EXIT_NOT_FOUND = 4  # the id does not exist: retrying cannot help


# ===================== the offline mock gateway ===================== #
# Fault injection is driven by a marker inside the request body, the
# way 02_advanced_engineering/local_mock_debugging.py does it, so one
# server can serve every scenario and the marker is visible in the
# command line that triggered it.
_FAULTS: Dict[str, Tuple[int, str, str]] = {
    "trigger-ratelimit": (429, "Throttling.RateQuota", "Rate limit exceeded."),
    "trigger-auth": (401, "InvalidApiKey", "Invalid API-key provided."),
    "trigger-server": (500, "InternalError", "Internal server error."),
}
_ONCE_MARKER = "trigger-ratelimit-once"


def _envelope(content: str, model: str) -> Dict[str, Any]:
    """The native chat-completion envelope the real gateway returns."""
    return {
        "model": model,
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                },
            ],
        },
        "usage": dict(USAGE),
        "request_id": REQUEST_ID,
    }


def _requested_model(body: str) -> str:
    """The model the request asked for, echoed back in the response.

    Echoing it is what lets the configuration demos prove something: the
    model in the response is the one that actually reached the wire, so
    ``default_model`` and ``-c`` can be seen to take effect rather than
    merely asserted to.
    """
    try:
        return str(json.loads(body).get("model") or MODEL)
    except ValueError:
        return MODEL


def _sse(events: List[Dict[str, Any]]) -> bytes:
    """Server-sent events: ``id:`` + ``data:`` per frame, blank-line ended."""
    out = []
    for position, event in enumerate(events, 1):
        payload = json.dumps(event, ensure_ascii=False)
        out.append(f"id: {position}\ndata: {payload}\n\n")
    return "".join(out).encode("utf-8")


def _chat_frames(content: str, model: str) -> List[Dict[str, Any]]:
    """Incremental chat frames, then a final frame carrying the usage."""
    pieces = [content[i : i + 6] for i in range(0, len(content), 6)]
    frames = [
        {
            "model": model,
            "output": {
                "choices": [
                    {"finish_reason": None, "message": {"content": piece}},
                ],
            },
            "request_id": "req-mock-0002",
        }
        for piece in pieces
    ]
    frames.append(
        {
            "model": model,
            "output": {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}},
                ],
            },
            "usage": dict(USAGE),
            "request_id": "req-mock-0002",
        },
    )
    return frames


def _job(job_id: str, status: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": status,
        "model": MODEL,
        "finetuned_output": f"{MODEL}-ft-demo",
        "created_at": "2026-01-01T10:00:00Z",
        "usage": dict(USAGE),
    }


class _Gateway(BaseHTTPRequestHandler):
    """Routes by path, serves canned envelopes, injects faults on demand."""

    protocol_version = "HTTP/1.1"
    # Shared across handler instances: how many times the "once" marker
    # has already been served. A dict keeps it mutable at class level.
    counters: Dict[str, int] = {}

    def log_message(self, *args) -> None:
        """Drop the access log; the example prints its own narration."""

    # -- plumbing ----------------------------------------------------
    def _body(self) -> str:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        return raw.decode("utf-8", "replace")

    def _send(
        self,
        status: int,
        payload: Any,
        content_type: str = "application/json",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        raw = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(raw)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(raw)

    def _fault(self, body: str) -> bool:
        """Answer with a scripted failure if the body asked for one."""
        if _ONCE_MARKER in body:
            seen = type(self).counters.get(_ONCE_MARKER, 0)
            type(self).counters[_ONCE_MARKER] = seen + 1
            if seen:
                return False
            self._send(
                429,
                {"code": "Throttling.RateQuota", "message": "Slow down."},
                headers={"Retry-After": "0"},
            )
            return True
        for marker, (status, code, message) in _FAULTS.items():
            if marker in body:
                self._send(
                    status,
                    {"code": code, "message": message},
                    headers={"Retry-After": "0"} if status == 429 else None,
                )
                return True
        return False

    # -- routes ------------------------------------------------------
    def do_POST(self) -> None:
        body = self._body()
        if self._fault(body):
            return
        if "text-generation/generation" in self.path:
            model = _requested_model(body)
            if self.headers.get("X-DashScope-SSE") == "enable":
                self._send(
                    200,
                    _sse(_chat_frames(ANSWER, model)),
                    content_type="text/event-stream",
                )
                return
            self._send(200, _envelope(ANSWER, model))
            return
        if self.path.startswith("/fine-tunes"):
            self._send(
                200,
                {
                    "output": _job(JOB_ID, "RUNNING"),
                    "request_id": "req-mock-0003",
                },
            )
            return
        self._send(404, {"code": "NotFound", "message": self.path})

    def do_GET(self) -> None:
        if not self.path.startswith("/fine-tunes"):
            self._send(404, {"code": "NotFound", "message": self.path})
            return
        route = urlsplit(self.path)
        if route.path == "/fine-tunes":
            query = parse_qs(route.query)
            size = int(query.get("page_size", ["10"])[0])
            jobs = [_job("ft-aaa", "SUCCEEDED"), _job("ft-bbb", "RUNNING")]
            self._send(
                200,
                {
                    "output": {
                        "jobs": jobs[:size],
                        "page_no": 1,
                        "page_size": size,
                        "total": 2,
                    },
                    "request_id": "req-mock-0004",
                },
            )
            return
        if route.path.endswith("/stream"):
            events = [
                {"status": "RUNNING", "progress": 40},
                {"status": "SUCCEEDED", "progress": 100},
            ]
            self._send(
                200,
                _sse(events),
                content_type="text/event-stream",
            )
            return
        job_id = route.path.rsplit("/", 1)[-1]
        if job_id == "ft-missing":
            self._send(
                404,
                {"code": "NotFound", "message": "The job does not exist."},
            )
            return
        self._send(
            200,
            {
                "output": _job(job_id, "SUCCEEDED"),
                "request_id": "req-mock-0005",
            },
        )

    def do_DELETE(self) -> None:
        if self.path.startswith("/fine-tunes"):
            self._send(
                200,
                {
                    "output": {"status": "DELETED"},
                    "request_id": "req-mock-0006",
                },
            )
            return
        self._send(404, {"code": "NotFound", "message": self.path})


def _start_gateway() -> Tuple[ThreadingHTTPServer, str]:
    """Serve on an ephemeral port; a fixed one collides on shared CI."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


# ===================== running the real CLI ===================== #
def _cli(*args: str) -> List[str]:
    """The command prefix that runs the real CLI in a subprocess.

    ``python -m dashscope.cli2`` rather than the ``dashscope`` console
    script: the script only exists once the package is installed, and an
    example has to run from a source checkout too.
    """
    return [sys.executable, "-m", "dashscope.cli2", *args]


def _child_env(config_path: str) -> Dict[str, str]:
    """The environment for every subprocess.

    ``DASHSCOPE_CONFIG`` points at a throwaway file so the run neither
    reads nor writes the developer's real ``~/.dashscope/config.toml``,
    and ``PYTHONPATH`` keeps the checkout importable when the package is
    not installed.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["NO_COLOR"] = "1"
    env["DASHSCOPE_CONFIG"] = config_path
    if not LIVE:
        env["DASHSCOPE_API_KEY"] = "sk-local-demo"
    # The endpoint travels as `--base-url` on every command line below, so
    # one inherited from the developer's shell must not leak in and take
    # over silently.
    env.pop("DASHSCOPE_BASE_URL", None)
    return env


def _global(base_url: Optional[str], *options: str) -> List[str]:
    """Global options, which must precede the subcommand."""
    prefix: List[str] = []
    if base_url:
        prefix += ["--base-url", base_url]
    # In-process retries would hide the tiers the caller has to handle,
    # so the offline runs turn them off and do their own.
    if not LIVE:
        prefix += ["-c", "max_retries=0"]
    return [*prefix, *options]


def _chat_argv(
    base_url: Optional[str],
    output: Optional[str],
    *extra: str,
) -> List[str]:
    """Full argv for one ``chat create``, with the global options first.

    ``output`` is the ``-o`` format, or None to leave the default table.
    Every scenario goes through here, which keeps the "global options
    must precede the subcommand" rule visible in one place.
    """
    options = ["-o", output] if output else []
    return _cli(*_global(base_url, *options), "chat", "create", *extra)


def _show(channel: str, text: str, limit: int = 12) -> None:
    """Print one captured channel, prefixed so the two stay separable."""
    body = (text or "").rstrip("\n")
    if not body:
        print(f"  {channel}: (empty)")
        return
    lines = body.splitlines()
    for line in lines[:limit]:
        print(f"  {channel}| {line}")
    if len(lines) > limit:
        print(f"  {channel}| ... {len(lines) - limit} more line(s)")


FAILURES: List[str] = []


def _run(
    label: str,
    args: List[str],
    expect: int,
    env: Dict[str, str],
    stdin_text: Optional[str] = None,
):
    """Run the real CLI, echo all three channels, check the exit code."""
    print(f"\n$ dashscope {' '.join(args[3:])}")
    completed = subprocess.run(
        args,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env=env,
        timeout=120,
    )
    _show("out", completed.stdout)
    _show("err", completed.stderr)
    verdict = "ok" if completed.returncode == expect else "FAIL"
    print(f"  exit={completed.returncode} expected={expect} -> {verdict}")
    if verdict == "FAIL":
        FAILURES.append(label)
    return completed


def _retry_cli(args: List[str], env: Dict[str, str]):
    """Run the CLI, retrying only the tier that can succeed later.

    Exit 1 covers server, rate-limit and network failures -- all of them
    transient. Exits 2 (your command line), 3 (your credentials) and 4
    (your id) will not fix themselves, so retrying them only burns the
    quota window and delays the red build.
    """
    completed = None
    delay = 0.5
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=env,
            timeout=120,
        )
        if completed.returncode != EXIT_SERVER:
            print(f"  attempt {attempt}: exit={completed.returncode}")
            return completed
        print(
            f"  attempt {attempt}: exit=1 (transient), "
            f"backing off {delay:.1f}s",
        )
        time.sleep(delay)
        delay *= 2
    print(f"  gave up after {RETRY_ATTEMPTS} attempts")
    return completed


# ===================== scenarios ===================== #
def _scenario_machine_output(base_url, env) -> None:
    print("\n===== 1. parse the machine formats, never the table =====")
    chat = ["-m", MODEL, "--prompt", PROMPT]
    result = _run("json", _chat_argv(base_url, "json", *chat), EXIT_OK, env)
    payload = json.loads(result.stdout)
    print(
        "  stdout is one JSON document:",
        payload["choices"][0]["message"]["content"],
    )
    print(f"  usage: {payload['usage']} | request_id: {payload['request_id']}")
    print(
        "  The same call in table mode: stdout carries the answer and",
        "nothing else, while reasoning and usage go to stderr. That is",
        "what makes `> answer.txt` safe -- and what makes scraping the",
        "table unnecessary.",
    )
    _run("table", _chat_argv(base_url, None, *chat), EXIT_OK, env)


def _scenario_exit_codes(base_url, env) -> None:
    print("\n===== 2. branch on the exit code, not on the text =====")
    chat = ["-m", MODEL]
    throttled = "trigger-ratelimit 请介绍杭州"
    bad_key = "trigger-auth 请介绍杭州"
    broken = "trigger-server 请介绍杭州"

    argv = _chat_argv(base_url, "json", *chat, "--prompt", throttled)
    _run("ratelimit", argv, EXIT_SERVER, env)
    print("  exit 1 with category ratelimit: transient, retry it.")

    argv = _chat_argv(base_url, None, *chat, "--prompt", bad_key)
    _run("auth", argv, EXIT_AUTH, env)
    print("  exit 3: the key is wrong. Retrying is pointless -- fail the job.")

    argv = _chat_argv(base_url, None, *chat, "--prompt", broken)
    _run("server", argv, EXIT_SERVER, env)

    argv = _cli(*_global(base_url), "ft", "get", "ft-missing")
    _run("notfound", argv, EXIT_NOT_FOUND, env)
    print("  exit 4: the id does not exist. Fix the input, do not retry.")

    argv = _chat_argv(base_url, "csv", *chat, "--prompt", PROMPT)
    _run("usage", argv, EXIT_USAGE, env)
    print("  exit 2: the command line itself is wrong.")
    print(
        "\n  A wrapper that retries only exit 1. The marker makes the",
        "gateway throttle the first attempt and serve the second:",
    )
    retry_prompt = f"{_ONCE_MARKER} 请介绍杭州"
    argv = _chat_argv(base_url, "json", *chat, "--prompt", retry_prompt)
    retried = _retry_cli(argv, env)
    if retried is not None and retried.returncode == EXIT_OK:
        answer = json.loads(retried.stdout)["choices"][0]["message"]["content"]
        print(f"  recovered after a throttle: {answer}")


def _scenario_streaming(base_url, env) -> None:
    print("\n===== 3. consume a stream as NDJSON, line by line =====")
    tail = ["-m", MODEL, "--prompt", PROMPT, "--stream"]
    args = _chat_argv(base_url, "json", *tail)
    print(f"\n$ dashscope {' '.join(args[3:])}")
    parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    # Popen rather than run: a stream is worth reading as it arrives,
    # which is the whole reason to ask for one. Frames land here in a
    # burst because the mock answers at once; a real gateway sends them
    # as the model produces them.
    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env,
    ) as proc:
        for line in proc.stdout:
            if not line.strip():
                continue
            frame = json.loads(line)
            for choice in frame.get("choices", []):
                parts.append(choice.get("delta", {}).get("content") or "")
            if frame.get("usage"):
                usage = frame["usage"]
        proc.wait(timeout=60)
    print(f"  exit={proc.returncode}")
    print(f"  reassembled from NDJSON: {''.join(parts)}")
    print(f"  usage from the final frame: {usage}")
    print(
        "  `-o yaml` streams NDJSON too -- one JSON object per line, not",
        "one YAML document.",
    )


def _scenario_config(config_path: str, base_url, env) -> None:
    print("\n===== 4. give the job its own configuration =====")
    print(f"  DASHSCOPE_CONFIG={config_path}")
    _run(
        "config-set-model",
        _cli("config", "set", "default_model", MODEL),
        EXIT_OK,
        env,
    )
    _run(
        "config-set-key",
        _cli("config", "set", "api_key", "sk-local-demo-1234567890"),
        EXIT_OK,
        env,
    )
    print(
        "  `config set` never echoes the value: a terminal recording or",
        "a build log should not capture a key that was just typed in.",
    )
    _run("config-get-masked", _cli("config", "get", "api_key"), EXIT_OK, env)
    argv = _cli("config", "get", "api_key", "--reveal")
    _run("config-get-revealed", argv, EXIT_OK, env)
    _run("config-list", _cli("config", "list"), EXIT_OK, env)
    print(
        "  `config get` reads the file alone; `config list` reports the",
        "effective value plus the layer that won. That is why api_key shows",
        "the file's masked value above but `env DASHSCOPE_API_KEY` below:",
        "this harness exports a key for the child process, and the",
        "environment outranks the file.",
    )
    mode = os.stat(config_path).st_mode & 0o777
    print(f"  the file mode after a secret was written: {oct(mode)}")
    print(
        "  With default_model configured, `-m` becomes optional -- note",
        "the model in the response is the configured one:",
    )
    argv = _chat_argv(base_url, "json", "--prompt", PROMPT)
    result = _run("default-model", argv, EXIT_OK, env)
    print(
        f"  model resolved from config: {json.loads(result.stdout)['model']}",
    )


def _scenario_overrides(base_url, env) -> None:
    print("\n===== 5. -c overrides one key for one invocation =====")
    print(
        "  Precedence is flag > -c > environment > config file > default.",
        "Here -c supplies the model for a single call, without writing",
        "anything down:",
    )
    argv = _cli(
        *_global(base_url, "-o", "json", "-c", "default_model=qwen-max"),
        "chat",
        "create",
        "--prompt",
        PROMPT,
    )
    result = _run("c-default-model", argv, EXIT_OK, env)
    print(f"  model: {json.loads(result.stdout)['model']}")
    print(
        "  A misspelled key fails immediately -- even on `version`, which",
        "never touches the service. That is deliberate: a typo in a",
        "pipeline should surface on the cheapest command there is.",
    )
    _run(
        "c-unknown-key",
        _cli(*_global(base_url), "-c", "defualt_model=qwen-max", "version"),
        EXIT_USAGE,
        env,
    )


def _scenario_jobs(base_url, env) -> None:
    print("\n===== 6. long-running jobs and destructive commands =====")
    _run(
        "ft-list-table",
        _cli(*_global(base_url), "ft", "list", "--size", "1"),
        EXIT_OK,
        env,
    )
    print(
        "  The `page x/y, total n` line is on stderr, so the table stays",
        "the only thing on stdout. In json mode the pagination is part of",
        "the document instead:",
    )
    result = _run(
        "ft-list-json",
        _cli(*_global(base_url, "-o", "json"), "ft", "list", "--size", "1"),
        EXIT_OK,
        env,
    )
    listed = json.loads(result.stdout)
    print(f"  total={listed['total_count']} page={listed['page_no']}")
    if LIVE:
        if not OWN_JOB:
            print(
                "  live mode: skipping `ft stream` and `ft delete` -- set",
                "DASHSCOPE_JOB_ID to a job you own to run the stream.",
            )
            return
        _run(
            "ft-stream",
            _cli(*_global(base_url), "ft", "stream", JOB_ID),
            EXIT_OK,
            env,
        )
        print("  live mode never runs the destructive commands.")
        return
    _run(
        "ft-stream",
        _cli(*_global(base_url), "ft", "stream", JOB_ID),
        EXIT_OK,
        env,
    )
    print(
        "  Cancel and delete are destructive. With no terminal attached --",
        "which is what a CI job looks like -- they refuse at exit 2",
        "instead of guessing:",
    )
    _run(
        "ft-delete-no-tty",
        _cli(*_global(base_url), "ft", "delete", JOB_ID),
        EXIT_USAGE,
        env,
    )
    _run(
        "ft-delete-yes",
        _cli(*_global(base_url), "ft", "delete", JOB_ID, "--yes"),
        EXIT_OK,
        env,
    )


def _scenario_ci_hygiene(base_url, env) -> None:
    print("\n===== 7. what else to set in a CI job =====")
    result = _run(
        "no-color",
        _cli(*_global(base_url), "ft", "list", "--size", "1"),
        EXIT_OK,
        env,
    )
    escaped = "\x1b" in result.stdout or "\x1b" in result.stderr
    print(
        f"  ANSI escapes in the captured output: {escaped}. Rich already",
        "detects the pipe, so NO_COLOR=1 matters mainly for runners that",
        "allocate a pseudo-terminal and then dump it into a log.",
    )
    print(
        "  Also worth setting: DASHSCOPE_CLI_NO_DEPRECATION=1 to silence",
        "the notice printed when a 1.x command is delegated to the legacy",
        "tree, and DASHSCOPE_CONFIG to a job-local path so parallel jobs",
        "never share one file.",
    )


# ===================== entry point ===================== #
def main() -> int:
    """Run every scenario; return non-zero if the CLI's contract moved."""
    server: Optional[ThreadingHTTPServer] = None
    base_url: Optional[str] = None
    with tempfile.TemporaryDirectory(prefix="dashscope-cli-harness-") as tmp:
        config_path = os.path.join(tmp, "config.toml")
        try:
            if LIVE:
                print(
                    "== CLI_HARNESS_LIVE=1: read-only scenarios against the",
                    "real gateway. Fault injection, config writes and",
                    "destructive commands are skipped. ==",
                )
                if not os.environ.get("DASHSCOPE_API_KEY"):
                    print("FAIL: DASHSCOPE_API_KEY is not set.")
                    return 1
            else:
                server, base_url = _start_gateway()
                print(f"== offline: mock gateway on {base_url} ==")
            env = _child_env(config_path)

            _scenario_machine_output(base_url, env)
            if not LIVE:
                _scenario_exit_codes(base_url, env)
            _scenario_streaming(base_url, env)
            if not LIVE:
                _scenario_config(config_path, base_url, env)
                _scenario_overrides(base_url, env)
            _scenario_jobs(base_url, env)
            _scenario_ci_hygiene(base_url, env)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()

    print("\n===== summary =====")
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} invocation(s) missed their exit code:")
        for label in FAILURES:
            print(f"  - {label}")
        return 1
    print("every scenario produced the exit code the CLI promises.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --- Production springboard (Best Practices & Patterns) -------
# CLI-side scripting above; the SDK-side equivalents are:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   CLI contract walkthrough: 05_cli/cli_output_and_errors.py
#   Recipes and reference: 05_cli/README.md
# -------------------------------------------------------------
