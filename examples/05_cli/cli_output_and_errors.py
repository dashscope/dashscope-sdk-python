# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 CLI example: output formats, streaming frames and the
exit-code tiers -- fully offline, with the SDK client swapped for a fake.

The command line is a contract surface of its own, separate from the
Python API: three output formats, a tiered exit code, and a rule about
which channel carries what. This file walks that contract end to end
without a network call, without an API key and without touching your
``~/.dashscope/config.toml``, by replacing the CLI's single SDK seam
(``dashscope.cli2.sdk.build_client``) with a fake client that returns
canned -- but genuinely typed -- SDK responses.

Five demonstrations:

1. **Output formats**: one completion rendered as table / json / yaml,
   showing that table mode keeps stdout clean (the answer only) and
   moves ``reasoning_content`` and ``usage`` to stderr, while the
   machine formats put the whole envelope on stdout and stay parseable.
2. **Streaming**: ``--stream`` writes incremental text in table mode
   and NDJSON -- one JSON object per line -- in *both* json and yaml
   mode, so ``--stream --output json | jq`` works.
3. **Input conventions**: ``@file`` and ``-`` (stdin) for ``--prompt``,
   and JSON array / JSONL / ``{"messages": [...]}`` for ``--messages``.
4. **Exit-code tiers**: usage 2, auth 3, not-found 4, and 1 for the
   rate-limit / server / network failures -- each raised as the real SDK
   error type, so the mapping in ``dashscope/cli2/errors.py`` is what
   actually runs rather than a description of it.
5. **Destructive commands**: with no terminal attached, ``ft delete``
   refuses at exit 2 unless ``--yes`` is given -- the classic CI
   failure, reproduced here on purpose.

Run it::

    python3 examples/05_cli/cli_output_and_errors.py

Every invocation is checked against the exit code it is supposed to
produce, so a silent contract change in the CLI turns this example red
instead of quietly teaching the wrong thing.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from typer.testing import CliRunner

from dashscope import errors as sdk_errors
from dashscope.cli2 import sdk as cli_sdk
from dashscope.cli2.app import build_app
from dashscope.resources.chat.completions import ChatStream
from dashscope.resources.fine_tunes import FineTune
from dashscope.types.common import JobStatus, Usage
from dashscope.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    Delta,
    StreamChoice,
)

# ---- configuration (override via env) ---- #
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")

PROMPT = "请用一句话介绍杭州"
ANSWER = "杭州是浙江省省会，以西湖与龙井茶闻名。"
REASONING = "这是一个地理常识问题，直接给一句话结论即可。"
USAGE = {"input_tokens": 18, "output_tokens": 22, "total_tokens": 40}
JOB_ID = "ft-20260101-demo"

# The four tiers, mirroring dashscope/cli2/errors.py.
EXIT_OK = 0
EXIT_SERVER = 1  # server, ratelimit, network: "not your bug, try again"
EXIT_USAGE = 2  # the command line was wrong
EXIT_AUTH = 3  # bad credentials or insufficient permission
EXIT_NOT_FOUND = 4  # the named resource does not exist

# `chat create` without a prompt: the --messages demos fill it in.
CHAT = ["chat", "create", "-m", MODEL, "--prompt", PROMPT]
CHAT_BARE = ["chat", "create", "-m", MODEL]


# ===================== canned, typed SDK responses ===================== #
def _completion() -> ChatCompletion:
    """A real ``ChatCompletion``, not a duck-typed stand-in.

    ``--output json`` serialises through pydantic's ``model_dump``; a
    ``SimpleNamespace`` would degrade to its own repr and teach the
    wrong lesson about what the machine formats emit.
    """
    return ChatCompletion(
        model=MODEL,
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant",
                    content=ANSWER,
                    reasoning_content=REASONING,
                ),
                finish_reason="stop",
            ),
        ],
        usage=Usage(**USAGE),
        request_id="req-demo-0001",
        # Transport metadata: stripped from machine output unless -v.
        headers={"x-request-id": "req-demo-0001"},
    )


def _chunk(text: str, *, final: bool = False) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        model=MODEL,
        choices=[
            StreamChoice(
                delta=Delta(role="assistant", content=text),
                finish_reason="stop" if final else None,
            ),
        ],
        usage=Usage(**USAGE) if final else None,
        request_id="req-demo-0002",
    )


def _chunks() -> List[ChatCompletionChunk]:
    """Incremental frames, the way the native wire sends them."""
    pieces = ["杭州", "是浙江省", "省会，", "以西湖", "闻名。"]
    frames = [_chunk(piece) for piece in pieces]
    frames.append(_chunk("", final=True))
    return frames


class _FakeInner:
    """Stand-in for the transport's SSE stream: iterable and closable."""

    def __init__(self, frames: List[ChatCompletionChunk]) -> None:
        self._frames = frames
        self.headers: Dict[str, str] = {}
        self.closed = False

    def __iter__(self):
        return iter(self._frames)

    def close(self) -> None:
        self.closed = True


class _FakeCompletions:
    """``client.chat.completions``: records calls, returns canned data.

    Setting ``error`` makes the next call raise instead, which is how the
    exit-code demos drive the CLI's error handler with real SDK types.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.error: Optional[Exception] = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            # The real stream wrapper: --stream must see .text_stream,
            # iteration, and .close(), exactly as the SDK provides them.
            return ChatStream(_FakeInner(_chunks()))
        return _completion()


class _FakeFineTunes:
    """``client.fine_tunes``: the two calls the demos below need."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.error: Optional[Exception] = None

    def _enter(self, name: str, job_id: str) -> None:
        """Record the call, then raise whatever the demo injected."""
        self.calls.append(f"{name} {job_id}")
        if self.error is not None:
            raise self.error

    def get(self, job_id, **_):
        self._enter("get", job_id)
        return FineTune(
            job_id=job_id,
            status="SUCCEEDED",
            model=MODEL,
            finetuned_output=f"{MODEL}-ft-demo",
            created_at="2026-01-01T10:00:00Z",
            usage=Usage(**USAGE),
        )

    def delete(self, job_id, **_):
        self._enter("delete", job_id)
        return JobStatus(status="DELETED", request_id="req-demo-0003")


_COMPLETIONS = _FakeCompletions()
_FINE_TUNES = _FakeFineTunes()
_CLIENT_KWARGS: List[Dict[str, Any]] = []


def _fake_build_client(overrides):
    """Replacement for ``dashscope.cli2.sdk.build_client``.

    ``CliContext.client`` calls the seam through the module attribute,
    so swapping it needs no HTTP mock, no server and no key. The
    resolved global flags arrive as ``overrides`` -- recording them shows
    exactly what ``-k`` / ``--timeout`` / ``-c`` turned into.
    """
    _CLIENT_KWARGS.append(dict(overrides))
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_COMPLETIONS),
        fine_tunes=_FINE_TUNES,
        close=lambda: None,
    )


# ===================== invocation helpers ===================== #
def _make_runner() -> CliRunner:
    """A runner that captures stdout and stderr separately.

    ``mix_stderr`` was removed in click 8.2, where the two streams are
    already separate, and the SDK pins only ``typer>=0.9`` -- so the flag
    is passed where it still exists and skipped where it does not.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # click >= 8.2
        return CliRunner()


def _stderr_of(result) -> str:
    try:
        return result.stderr
    except ValueError:  # click < 8.2 constructed with mixed stderr
        return "(this click version mixes stderr into stdout)"


def _show(channel: str, text: str) -> None:
    """Print one captured channel, prefixed so the two stay separable."""
    body = (text or "").rstrip("\n")
    if not body:
        print(f"  {channel}: (empty)")
        return
    for line in body.splitlines():
        print(f"  {channel}| {line}")


FAILURES: List[str] = []


def _run(label, args, expect, input_text=None, error=None):
    """One invocation: echo the command, show both channels, check exit."""
    print(f"\n$ dashscope {' '.join(args)}")
    if error is not None:
        # The command line is fine in these cases; the failure is
        # injected, so name the SDK error type the CLI is being handed.
        print(f"# injected failure: {type(error).__name__}")
        _COMPLETIONS.error = error
        _FINE_TUNES.error = error
    try:
        result = RUNNER.invoke(APP, list(args), input=input_text)
    finally:
        _COMPLETIONS.error = None
        _FINE_TUNES.error = None
    _show("out", result.stdout)
    _show("err", _stderr_of(result))
    verdict = "ok" if result.exit_code == expect else "FAIL"
    print(f"  exit={result.exit_code} expected={expect} -> {verdict}")
    if verdict == "FAIL":
        FAILURES.append(label)
    return result


def _global(*options) -> List[str]:
    """Global options, which must precede the subcommand."""
    return ["-k", "sk-demo", *options]


def _last_conversation() -> str:
    return json.dumps(
        _COMPLETIONS.calls[-1]["messages"],
        ensure_ascii=False,
    )


# ===================== A. the three output formats ===================== #
def _demo_output_formats() -> None:
    print("\n===== A. one completion, three output formats =====")
    print(
        "Table mode writes the answer to stdout and moves reasoning and\n"
        "usage to stderr, so `dashscope chat create ... > answer.txt`\n"
        "captures nothing but the answer. The machine formats put the\n"
        "whole envelope on stdout and never render Rich markup.",
    )
    _run("table", _global() + CHAT, EXIT_OK)
    result = _run("json", _global("-o", "json") + CHAT, EXIT_OK)
    payload = json.loads(result.stdout)
    print(
        "  stdout parsed with json.loads ->",
        payload["choices"][0]["message"]["content"],
    )
    print(
        "  (reasoning is in the same document:",
        bool(payload["choices"][0]["message"].get("reasoning_content")),
        "| headers stripped:",
        "headers" not in payload,
        ")",
    )
    _run("yaml", _global("-o", "yaml") + CHAT, EXIT_OK)
    print(
        "  -v keeps the transport headers that are otherwise dropped,",
        "so a support ticket can carry them:",
    )
    _run("json-verbose", _global("-o", "json", "-v") + CHAT, EXIT_OK)
    print("  the resolved global flags became Client(...) kwargs:")
    print(f"    {_CLIENT_KWARGS[-1]}")


# ===================== B. streaming ===================== #
def _demo_streaming() -> None:
    print("\n===== B. streaming: incremental text vs NDJSON =====")
    _run("stream-table", _global() + CHAT + ["--stream"], EXIT_OK)
    result = _run(
        "stream-json",
        _global("-o", "json") + CHAT + ["--stream"],
        EXIT_OK,
    )
    frames = [
        json.loads(line) for line in result.stdout.splitlines() if line.strip()
    ]
    text = "".join(
        frame["choices"][0]["delta"].get("content", "") for frame in frames
    )
    usage = next(
        (frame["usage"] for frame in frames if frame.get("usage")),
        None,
    )
    print(f"  {len(frames)} NDJSON frames read back from stdout")
    print(f"  reassembled: {text}")
    print(f"  usage from the final frame: {usage}")
    print(
        "  yaml streams as NDJSON too -- a stream is not one YAML",
        "document, it is one JSON object per line:",
    )
    _run(
        "stream-yaml",
        _global("-o", "yaml") + CHAT + ["--stream"],
        EXIT_OK,
    )


# ===================== C. input conventions ===================== #
def _demo_input_conventions() -> None:
    print("\n===== C. input: @file, stdin, and three --messages shapes =====")
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write("用一句话解释什么是熔断器\n")
        path = handle.name
    try:
        from_file = ["chat", "create", "-m", MODEL, "--prompt", f"@{path}"]
        _run("prompt-at-file", _global() + from_file, EXIT_OK)
    finally:
        os.unlink(path)
    print(f"  the CLI read the prompt from the file: {_last_conversation()}")

    _run(
        "prompt-stdin",
        _global() + ["chat", "create", "-m", MODEL, "--prompt", "-"],
        EXIT_OK,
        input_text="从管道读进来的问题\n",
    )
    print(f"  `--prompt -` took stdin: {_last_conversation()}")

    _run(
        "piped-stdin",
        _global() + CHAT_BARE,
        EXIT_OK,
        input_text="没有 --prompt，直接管道进来\n",
    )
    print(f"  bare piped stdin became the user turn: {_last_conversation()}")

    multi = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": PROMPT},
    ]
    as_json = json.dumps(multi, ensure_ascii=False)
    as_jsonl = "\n".join(json.dumps(m, ensure_ascii=False) for m in multi)
    as_object = json.dumps({"messages": multi}, ensure_ascii=False)

    _run(
        "messages-json",
        _global() + CHAT_BARE + ["--messages", as_json],
        EXIT_OK,
    )
    _run(
        "messages-jsonl",
        _global() + CHAT_BARE + ["--messages", as_jsonl],
        EXIT_OK,
    )
    print("  JSONL -- one message object per line -- built the same thing:")
    print(f"    {_last_conversation()}")
    _run(
        "messages-object",
        _global() + CHAT_BARE + ["--messages", as_object],
        EXIT_OK,
    )
    print(
        "  --messages wins over --prompt, and --system is prepended to",
        "--prompt only:",
    )
    with_system = [
        "chat",
        "create",
        "-m",
        MODEL,
        "--system",
        "你是一个简洁的助手。",
        "--prompt",
        PROMPT,
    ]
    _run("system-plus-prompt", _global() + with_system, EXIT_OK)
    print(f"    {_last_conversation()}")


# ===================== D. the exit-code tiers ===================== #
def _demo_usage_tier() -> None:
    print("\n----- D1. exit 2: the command line was wrong -----")
    _run(
        "bad-output-format",
        ["-k", "sk-demo", "-o", "csv"] + CHAT,
        EXIT_USAGE,
    )
    print(
        "  no model and no configured default_model (this example points",
        "DASHSCOPE_CONFIG at an empty temp file, so nothing leaks in):",
    )
    _run(
        "no-model",
        _global() + ["chat", "create", "--prompt", PROMPT],
        EXIT_USAGE,
    )
    print("  a global option placed AFTER the subcommand is not global:")
    _run(
        "global-option-after-subcommand",
        CHAT + ["-o", "json"],
        EXIT_USAGE,
    )
    _run(
        "tools-not-an-array",
        _global() + CHAT + ["--tools", '{"type": "function"}'],
        EXIT_USAGE,
    )


def _demo_service_tiers() -> None:
    print("\n----- D2. exits 1 / 3 / 4: what the service said -----")
    _run(
        "auth-401",
        _global() + CHAT,
        EXIT_AUTH,
        error=sdk_errors.AuthenticationError(
            "Invalid API-key provided.",
            request_id="req-demo-0401",
        ),
    )
    _run(
        "notfound-404",
        _global() + ["ft", "get", "ft-does-not-exist"],
        EXIT_NOT_FOUND,
        error=sdk_errors.NotFoundError(
            "The fine-tune job does not exist.",
            request_id="req-demo-0404",
        ),
    )
    _run(
        "ratelimit-429",
        _global() + CHAT,
        EXIT_SERVER,
        error=sdk_errors.RateLimitError(
            "Requests rate limit exceeded, please try again later.",
        ),
    )
    _run(
        "server-500",
        _global() + CHAT,
        EXIT_SERVER,
        error=sdk_errors.InternalServerError("Internal server error."),
    )
    print(
        "  network failures share exit 1 with server errors, but the",
        "category on stderr tells them apart -- and only the network and",
        "rate-limit tiers are worth an automatic retry:",
    )
    _run(
        "network",
        _global() + CHAT,
        EXIT_SERVER,
        error=sdk_errors.APIConnectionError(
            "Connection refused by the gateway.",
        ),
    )
    print(
        "\n  tier summary: 0 ok | 1 server/ratelimit/network (retry)",
        "| 2 usage (fix the command) | 3 auth (fix the key)",
        "| 4 not found (fix the id)",
    )


# ===================== E. destructive commands ===================== #
def _demo_destructive() -> None:
    print("\n===== E. destructive commands need --yes in CI =====")
    print(
        "There is no terminal here, which is exactly the situation a CI",
        "job is in: cancel/delete refuse rather than guess.",
    )
    _run("delete-no-tty", _global() + ["ft", "delete", JOB_ID], EXIT_USAGE)
    _run(
        "delete-yes",
        _global() + ["ft", "delete", JOB_ID, "--yes"],
        EXIT_OK,
    )
    print(f"  calls that reached the service: {_FINE_TUNES.calls}")


# ===================== entry point ===================== #
def _isolated_env():
    """Point the config file at an empty temp path, and disable colour.

    Without this the demos that depend on "no default_model configured"
    or on machine-readable output would vary with whatever the developer
    happens to have in ``~/.dashscope/config.toml``. The real file is
    never read and never written.
    """
    directory = tempfile.mkdtemp(prefix="dashscope-cli-example-")
    saved = {
        key: os.environ.get(key) for key in ("DASHSCOPE_CONFIG", "NO_COLOR")
    }
    os.environ["DASHSCOPE_CONFIG"] = os.path.join(directory, "config.toml")
    os.environ["NO_COLOR"] = "1"
    return directory, saved


def _restore_env(directory: str, saved: Dict[str, Optional[str]]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    config = os.path.join(directory, "config.toml")
    if os.path.exists(config):
        os.unlink(config)
    os.rmdir(directory)


APP = build_app()
RUNNER = _make_runner()


def main() -> int:
    """Run every demonstration; return non-zero if a contract drifted."""
    directory, saved_env = _isolated_env()
    original_build_client = cli_sdk.build_client
    cli_sdk.build_client = _fake_build_client
    try:
        _demo_output_formats()
        _demo_streaming()
        _demo_input_conventions()
        _demo_usage_tier()
        _demo_service_tiers()
        _demo_destructive()
    finally:
        cli_sdk.build_client = original_build_client
        _restore_env(directory, saved_env)

    print("\n===== summary =====")
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} invocation(s) missed their exit code:")
        for label in FAILURES:
            print(f"  - {label}")
        return 1
    print("every invocation produced the exit code its tier promises.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --- Production springboard (Best Practices & Patterns) -------
# Contract-level walkthrough above; to drive the CLI for real, see:
#   Shell/CI scripting: 05_cli/cli_ci_harness.py (local mock gateway +
#   subprocess, retry by exit code, config isolation, secret masking)
#   Recipes and reference: 05_cli/README.md
#   SDK-side offline debugging: 02_advanced_engineering/local_mock_debugging.py
# -------------------------------------------------------------
