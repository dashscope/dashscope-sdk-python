# -*- coding: utf-8 -*-
"""Hard acceptance gate for autonomous (oneshot) runs.

The system prompt already asks the model to verify its work before reporting
completion, and the model already ignores that. Aider's ``--auto-test`` and
Agentless' three-stage filter make verification a property of the loop rather
than a request: a non-zero exit means the answer is not accepted yet.
OpenHands shipping the advice without the gate is the counterexample.

Only oneshot runs are gated. Interactively the user *is* the acceptance
criterion and is reading the output, so spending minutes on a test suite
before every reply costs more than the near-miss it prevents.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from dashscope.acli.utils.text import markdown_fence, truncate_head_tail

# Quoted verbatim into the next prompt, so: large enough to contain the
# failing test id and its assertion, small enough to read. Tail-biased,
# because that is where pytest and cargo put the verdict.
_OUTPUT_CHARS = 3000

DEFAULT_RETRIES = 1
DEFAULT_TIMEOUT = 600

EXPLICIT = "explicit"
DISCOVERED = "discovered"

# Exit statuses meaning the check never reached the code under test, so its
# output describes the environment rather than the change. 127 is the shell's
# command-not-found; 2/3/4/5 are pytest's interrupted (also what a collection
# error such as a missing dependency reports), internal-error, usage-error and
# nothing-collected. pytest's 1 is deliberately absent: tests ran and one
# failed, which is the verdict this whole module exists to act on.
_NOT_FOUND = 127
_PYTEST_NEVER_RAN = frozenset({2, 3, 4, 5})


def _never_ran(run: AcceptanceRun) -> bool:
    """True when ``run``'s exit status shows the check could not execute."""
    if run.exit_code == _NOT_FOUND:
        return True
    return "pytest" in run.command and run.exit_code in _PYTEST_NEVER_RAN


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _read_head(path: Path, limit: int = 65536) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def _is_pytest_project(root: Path) -> bool:
    if (root / "pytest.ini").is_file():
        return True
    for name, marker in (
        ("pyproject.toml", "[tool.pytest"),
        ("setup.cfg", "[tool:pytest]"),
        ("tox.ini", "[pytest]"),
    ):
        path = root / name
        if path.is_file() and marker in _read_head(path):
            return True
    # Most Python repos never declare pytest config at all. A tests/ directory
    # beside a project file is pytest's default layout, and is what bare
    # `pytest` collects.
    return (root / "tests").is_dir() and (root / "pyproject.toml").is_file()


def discover_acceptance_command(
    cwd: str | os.PathLike[str] | None = None,
) -> str | None:
    """The conventional test entry point for the project at ``cwd``, if any.

    The allowlist is deliberately short. Every entry runs code from the
    repository — ``conftest.py`` at collection time, ``build.rs``, ``TestMain``
    — so "conservative" here means "a standard test entry point for an
    ecosystem whose manifest is sitting right here", not "harmless".
    ``npm test`` and ``make test`` are excluded because their bodies are
    arbitrary user scripts with no conventional meaning; declaring them via
    ``ACLI_ACCEPTANCE_CMD`` is how you ask for those.
    """
    root = Path(cwd or os.getcwd())
    if not root.is_dir():
        return None
    if _is_pytest_project(root):
        # This interpreter, not a bare `pytest`: the gate must run against the
        # same environment the agent's own run_command calls use.
        return f"{shlex.quote(sys.executable)} -m pytest -q"
    if (root / "Cargo.toml").is_file():
        return "cargo test --quiet"
    if (root / "go.mod").is_file():
        return "go test ./..."
    return None


@dataclass(frozen=True)
class AcceptanceRun:
    """One execution of the acceptance check."""

    command: str
    source: str
    ok: bool
    exit_code: int | None
    output: str


@dataclass(frozen=True)
class GateVerdict:
    """What the loop should do about a check that did not pass.

    At most one of ``rejection`` and ``note`` is non-empty. ``rejection``
    withholds the answer and sends the model back with the raw output;
    ``note`` is appended to the final answer when sending it back would not
    help.
    """

    run: AcceptanceRun
    rejection: str = ""
    note: str = ""


class AcceptanceGate:
    """Runs the acceptance check and decides whether the answer stands.

    Two runs per turn at most: a baseline captured before the first mutation,
    and the check at the loop's exit. The baseline is what turns "the suite is
    red" into "you made it red" — Agentless reproduces the issue on the
    unpatched repository for the same reason, and AlphaCodium's test anchors
    are the same obligation stated positively.
    """

    def __init__(
        self,
        command: str = "",
        source: str = EXPLICIT,
        retries: int = DEFAULT_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.command = command
        self.source = source
        self.retries = retries
        self.timeout = timeout
        self.baseline: AcceptanceRun | None = None
        self.attempts = 0
        self._baseline_task: asyncio.Task | None = None

    @classmethod
    def from_env(cls, oneshot: bool) -> AcceptanceGate:
        """Build the gate from the ``ACLI_ACCEPTANCE*`` environment variables.

        An explicit command wins over discovery. Setting ``ACLI_ACCEPTANCE=0``
        turns the gate off, which is what a benchmark measuring raw model
        output wants.
        """
        if not (oneshot and _env_flag("ACLI_ACCEPTANCE", True)):
            return cls()
        command = os.environ.get("ACLI_ACCEPTANCE_CMD", "").strip()
        source = EXPLICIT
        if not command:
            command = discover_acceptance_command() or ""
            source = DISCOVERED
        return cls(
            command=command,
            source=source,
            retries=_env_int("ACLI_ACCEPTANCE_RETRIES", DEFAULT_RETRIES),
            timeout=_env_int("ACLI_ACCEPTANCE_TIMEOUT", DEFAULT_TIMEOUT),
        )

    @property
    def active(self) -> bool:
        """An empty command is the off switch."""
        return bool(self.command)

    def reset(self) -> None:
        """Drop the previous turn's anchor and retry budget.

        The baseline has to come from the workspace as *this* turn found it,
        and the retry budget is per-turn so one slow turn cannot spend the
        next one's. An in-flight capture is cancelled for the same reason:
        left running, it would write the previous turn's verdict into this
        turn's ``baseline``.
        """
        if self._baseline_task is not None and not self._baseline_task.done():
            self._baseline_task.cancel()
        self._baseline_task = None
        self.baseline = None
        self.attempts = 0

    async def ensure_baseline(self) -> None:
        """Capture the pre-change verdict, once per turn.

        Called before the first mutating tool call rather than at turn start:
        a read-only or pure-answer turn never pays for a test suite, and the
        baseline still comes from a workspace the agent has not touched.

        Concurrent callers share one task rather than checking a flag. The
        flag form lets the second mutating call in a parallel batch proceed
        while the baseline is still running, so the "pre-change" verdict would
        be captured against a workspace that had already changed.
        """
        if not self.active:
            return
        if self._baseline_task is None:
            self._baseline_task = asyncio.ensure_future(self._capture())
        await self._baseline_task

    async def _capture(self) -> None:
        self.baseline = await self._run()

    async def check(self) -> GateVerdict | None:
        """Run the check at the loop's exit and rule on the answer.

        ``None`` means "not applicable" — gate off, no command found, or
        nothing was mutated this turn, in which case there is no work to
        accept.
        """
        if not self.active or self.baseline is None:
            return None
        run = await self._run()
        self.attempts += 1
        if run.ok:
            return GateVerdict(run)
        if _never_ran(run) and self._baseline_never_ran():
            return GateVerdict(run, note=self._unrunnable_note(run))
        if self._unchanged_since_baseline(run):
            return GateVerdict(run, note=self._insensitive_note(run))
        if self.attempts > self.retries:
            return GateVerdict(run, note=self._unverified_note(run))
        return GateVerdict(run, rejection=self._rejection(run))

    def _baseline_never_ran(self) -> bool:
        """True when the pre-change check could not execute either.

        This is what separates a broken environment from a broken change. The
        baseline is captured before the agent touches anything, so a check that
        already could not run then was not made unrunnable by the change.
        Requiring it also keeps the sharp edge: a *green* baseline proves the
        check does run here, so a collection error the agent introduced stays
        on the rejection path instead of being excused.
        """
        base = self.baseline
        return bool(base and not base.ok and _never_ran(base))

    def _unchanged_since_baseline(self, run: AcceptanceRun) -> bool:
        """True when the check came out byte-identical to a failing baseline.

        The catch-all behind ``_baseline_never_ran``, for a check whose exit
        statuses this module does not know — cargo, go, an immediate timeout —
        where reproducing the baseline exactly is the only evidence available
        that the break is environmental. Sending the model back to fix that
        burns the turn on a problem it neither created nor can solve, so the
        gate reports it instead of rejecting.

        Byte-identical is the point: a suite that genuinely fails prints
        timings, so real failures differ between runs and still get rejected.
        Only a stable, environment-level break reproduces exactly.
        """
        base = self.baseline
        return bool(
            base
            and not base.ok
            and base.exit_code == run.exit_code
            and base.output == run.output,
        )

    async def _run(self) -> AcceptanceRun:
        # run_command rather than a raw subprocess: it carries the
        # blocked-pattern checks, the optional OS sandbox, the output caps, and
        # the exit-code trailer that the rest of the loop already classifies.
        from dashscope.acli.audit import get_audit_logger
        from dashscope.acli.executor import classify_tool_text
        from dashscope.acli.tools.shell import run_command

        get_audit_logger().log_tool_call(
            "acceptance_gate",
            {"command": self.command},
            decision="allowed",
            reason=f"source: {self.source}",
        )
        text = await run_command(command=self.command, timeout=self.timeout)
        outcome = classify_tool_text(text, {"command": self.command})
        return AcceptanceRun(
            command=self.command,
            source=self.source,
            ok=outcome.ok,
            exit_code=outcome.exit_code,
            output=outcome.text,
        )

    def _rejection(self, run: AcceptanceRun) -> str:
        quoted = truncate_head_tail(run.output, _OUTPUT_CHARS, ratio=0.25)
        fence = markdown_fence(quoted)
        if self.baseline and self.baseline.ok:
            verdict = (
                "a REGRESSION: it passed before anything in this turn "
                "changed, so the change broke it"
            )
        else:
            verdict = "still failing"
        return (
            "\n## ⛔ Not accepted: the verification check is "
            f"{verdict}\n"
            f"`{run.command}` exited {run.exit_code}.\n"
            f"{fence}\n{quoted}\n{fence}\n"
            "Your answer is being withheld, not discarded: fix what this "
            "output shows and the turn ends by itself. Work from the output "
            "above — do not restate your plan, and do not declare the check "
            "wrong unless this output says so.\n"
        )

    def _unverified_note(self, run: AcceptanceRun) -> str:
        return (
            f"\n[Unverified: `{run.command}` exited {run.exit_code} and the "
            f"retry budget ({self.retries}) is spent, so the answer below is "
            "reported as-is rather than withheld.]\n"
        )

    def _unrunnable_note(self, run: AcceptanceRun) -> str:
        return (
            f"\n[Unverified: `{run.command}` exited {run.exit_code} before "
            "and after these changes, a status that means it never reached "
            "the tests. The check does not run in this environment, so it "
            "says nothing about the answer below.]\n"
        )

    def _insensitive_note(self, run: AcceptanceRun) -> str:
        return (
            f"\n[Unverified: `{run.command}` failed identically before and "
            "after these changes, so it says nothing about them — most likely "
            "the check itself does not run in this environment.]\n"
        )
