# -*- coding: utf-8 -*-
"""Hooks system — event-driven callbacks around tool calls and responses.

Loads ``.acli/hooks.toml`` and invokes registered callbacks at lifecycle
points: before_tool_call, after_tool_call, on_error, on_response.

Supported actions:
  run    — execute a shell command (with template variables)
  block  — prevent the tool call from running
  warn   — print a warning but continue
  alert  — print a warning; if the tool failed, escalate
  log    — print a silent log line
  confirm— treat the tool call as requiring user confirmation
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dashscope.acli.config import WORKSPACE_DIR
from dashscope.acli.utils import load_toml
from dashscope.acli.utils.template import render_brace_template


@dataclass
class HookContext:
    """Context passed to every hook callback."""

    event: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    success: bool | None = None


@dataclass
class HookActionResult:
    """Result of a single hook action."""

    blocked: bool = False
    confirm: bool = False
    warning: str = ""
    alert: str = ""
    log: str = ""
    output: str = ""
    error: str = ""


@dataclass
class HookDispatchResult:
    """Aggregated result of dispatching an event to all matching hooks."""

    blocked: bool = False
    confirm: bool = False
    warnings: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def merge(self, action: HookActionResult) -> None:
        self.blocked = self.blocked or action.blocked
        self.confirm = self.confirm or action.confirm
        if action.warning:
            self.warnings.append(action.warning)
        if action.alert:
            self.alerts.append(action.alert)
        if action.log:
            self.logs.append(action.log)
        if action.output:
            self.outputs.append(action.output)


def _resolve_path(arguments: dict[str, Any]) -> str:
    """Best-effort file path extraction from tool arguments."""
    for key in ("path", "file_path", "file", "src", "dst", "target"):
        val = arguments.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _build_variables(ctx: HookContext) -> dict[str, str]:
    """Build template variables for hook scripts."""
    path = _resolve_path(ctx.arguments)
    p = Path(path) if path else None
    variables = {
        "tool_name": ctx.tool_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "path": path,
        "filename": p.name if p else "",
        "filename_stem": p.stem if p else "",
        "exit_code": "",
        "content": "",
        "args": json.dumps(ctx.arguments, ensure_ascii=False)
        if ctx.arguments
        else "",
        "result": (ctx.result or "")[:1000],
        "error": (ctx.result or "")[:1000] if ctx.success is False else "",
    }

    if ctx.tool_name == "run_command":
        cmd = ctx.arguments.get("command", "")
        if isinstance(cmd, str):
            # Try to extract exit code from result like "exit code: 1\n..."
            if ctx.result and "exit code:" in ctx.result:
                try:
                    line = ctx.result.split("exit code:", 1)[1].split("\n", 1)[
                        0
                    ]
                    variables["exit_code"] = line.strip()
                except Exception:
                    pass
    elif ctx.tool_name == "write_file":
        content = ctx.arguments.get("content", "")
        if isinstance(content, str):
            variables["content"] = content[:500]

    return variables


def _match_condition(condition: str | None, path: str) -> bool:
    """Glob match a condition against a path."""
    if not condition:
        return True
    if not path:
        return False
    return fnmatch.fnmatch(path, condition) or fnmatch.fnmatch(
        Path(path).name,
        condition,
    )


_HOOK_TIMEOUT = 60  # seconds; shared by sync and async shell runners

# Shell wrapper for the async runner — keeps the same semantics as
# subprocess.run(..., shell=True).
_SHELL_PREFIX = ["cmd", "/c"] if os.name == "nt" else ["/bin/sh", "-c"]


def _run_shell(command: str) -> tuple[int, str, str]:
    """Run a shell command and return (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_HOOK_TIMEOUT,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "hook command timed out"
    except Exception as e:
        return 1, "", f"hook command failed: {e}"


async def _run_shell_async(command: str) -> tuple[int, str, str]:
    """Async variant of _run_shell — same timeout/kill semantics, but awaits
    the subprocess so the event loop is not blocked."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *_SHELL_PREFIX,
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=_HOOK_TIMEOUT,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            return 124, "", "hook command timed out"
        return (
            proc.returncode if proc.returncode is not None else 0,
            stdout.decode(errors="replace").strip(),
            stderr.decode(errors="replace").strip(),
        )
    except Exception as e:
        return 1, "", f"hook command failed: {e}"


class Hook:
    """A single hook rule loaded from hooks.toml.

    ``category`` distinguishes governance hooks (block/confirm/alert —
    security-critical, permission/audit/compliance) from plugin hooks
    (run/warn/log — functional extensions like tool before/after processing).
    Governance hooks are dispatched first so security decisions take
    precedence over functional extensions.
    """

    # Actions that are security-critical and run in the governance phase.
    _GOVERNANCE_ACTIONS = frozenset({"block", "confirm", "alert"})

    def __init__(self, event: str, spec: dict[str, Any]):
        self.event = event
        self.match = spec.get("match", "*")
        self.condition = spec.get("condition")
        self.content_pattern = spec.get("content_pattern")
        self.action = spec.get("action", "run")
        self.run = spec.get("run", "")
        self.message = spec.get("message", "")
        self.silent = spec.get("silent", False)
        self.on_fail = spec.get("on_fail", "warn")  # warn | alert | block
        # Governance hooks (block/confirm/alert) run before plugin hooks
        # (run/warn/log) so security decisions are never bypassed.
        self.category = (
            "governance"
            if self.action in self._GOVERNANCE_ACTIONS
            else "plugin"
        )

    def applies_to(self, ctx: HookContext) -> bool:
        if self.event != ctx.event:
            return False
        if self.match != "*" and not fnmatch.fnmatch(
            ctx.tool_name,
            self.match,
        ):
            return False
        if self.condition:
            path = _resolve_path(ctx.arguments)
            if not _match_condition(self.condition, path):
                return False
        if self.content_pattern:
            text = ""
            if ctx.event == "on_error":
                text = ctx.result or ""
            elif ctx.event == "on_response":
                # The reply body is passed via arguments["content"] (result
                # stays unset); fall back to result for direct dispatchers.
                content = ctx.arguments.get("content")
                text = ctx.result or (
                    content if isinstance(content, str) else ""
                )
            elif ctx.event == "on_message":
                # The user message is passed via arguments["input"].
                content = ctx.arguments.get("input")
                text = content if isinstance(content, str) else ""
            elif ctx.event in ("before_tool_call", "after_tool_call"):
                text = json.dumps(ctx.arguments, ensure_ascii=False)
            if not fnmatch.fnmatch(text, self.content_pattern):
                return False
        return True

    def _execute_simple(
        self,
        ctx: HookContext,
        variables: dict[str, str],
    ) -> HookActionResult | None:
        """Handle non-run actions; return None for 'run'/unknown actions."""
        result = HookActionResult()

        if self.action == "block":
            result.blocked = True
            result.warning = render_brace_template(
                self.message or "operation blocked by hook",
                variables,
            )
            return result

        if self.action == "confirm":
            result.confirm = True
            result.warning = render_brace_template(
                self.message or "confirmation required",
                variables,
            )
            return result

        if self.action in ("warn", "alert"):
            msg = render_brace_template(
                self.message or f"{self.action} from hook",
                variables,
            )
            if self.action == "warn":
                result.warning = msg
            else:
                result.alert = msg
            return result

        if self.action == "log":
            result.log = render_brace_template(
                self.message or f"{ctx.event} {ctx.tool_name}",
                variables,
            )
            return result

        return None

    def _run_command_result(
        self,
        code: int,
        stdout: str,
        stderr: str,
    ) -> HookActionResult:
        """Build the action result from a finished run-action command."""
        result = HookActionResult()
        output = stdout
        if stderr:
            output += "\n" + stderr if output else stderr
        if code != 0:
            if self.on_fail == "block":
                result.blocked = True
            elif self.on_fail == "alert":
                result.alert = (
                    f"hook command failed (exit {code}): " f"{output[:200]}"
                )
            else:
                result.warning = (
                    f"hook command failed (exit {code}): " f"{output[:200]}"
                )
        else:
            result.output = output
        if not self.silent and output:
            result.log = output[:500]
        return result

    def execute(self, ctx: HookContext) -> HookActionResult:
        variables = _build_variables(ctx)
        simple = self._execute_simple(ctx, variables)
        if simple is not None:
            return simple

        if self.action == "run":
            if not self.run:
                return HookActionResult()
            command = render_brace_template(self.run, variables)
            code, stdout, stderr = _run_shell(command)
            return self._run_command_result(code, stdout, stderr)

        # Unknown action — no-op
        return HookActionResult()

    async def execute_async(self, ctx: HookContext) -> HookActionResult:
        """Async variant of execute: run-actions await an asyncio subprocess
        instead of blocking the event loop for up to the hook timeout."""
        variables = _build_variables(ctx)
        simple = self._execute_simple(ctx, variables)
        if simple is not None:
            return simple

        if self.action == "run":
            if not self.run:
                return HookActionResult()
            command = render_brace_template(self.run, variables)
            code, stdout, stderr = await _run_shell_async(command)
            return self._run_command_result(code, stdout, stderr)

        # Unknown action — no-op
        return HookActionResult()


class HookBus:
    """Central event bus for hooks."""

    def __init__(self, hooks_file: Path | None = None):
        self.hooks: list[Hook] = []
        self._load_hooks(hooks_file or WORKSPACE_DIR / "hooks.toml")

    def _load_hooks(self, path: Path) -> None:
        data = load_toml(path)
        if data is None:
            return
        hooks_section = data.get("hooks", {})
        for event_name, entries in hooks_section.items():
            if not isinstance(entries, list):
                continue
            for spec in entries:
                if isinstance(spec, dict):
                    self.hooks.append(Hook(event_name, spec))

    def register(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def unregister(self, hook: Hook) -> None:
        """Remove a previously registered hook (by identity)."""
        self.hooks = [h for h in self.hooks if h is not hook]

    def dispatch(self, ctx: HookContext) -> HookDispatchResult:
        result = HookDispatchResult()
        # Governance hooks (block/confirm/alert) run before plugin hooks
        # (run/warn/log) so security decisions are never bypassed by a
        # functional extension.
        applicable = [h for h in self.hooks if h.applies_to(ctx)]
        applicable.sort(key=lambda h: 0 if h.category == "governance" else 1)
        for hook in applicable:
            result.merge(hook.execute(ctx))
            # Stop at first block
            if result.blocked:
                break
        return result

    async def dispatch_async(self, ctx: HookContext) -> HookDispatchResult:
        """Async variant of dispatch — identical ordering/semantics, but
        command hooks run as awaited asyncio subprocesses so the event loop
        is not blocked."""
        result = HookDispatchResult()
        applicable = [h for h in self.hooks if h.applies_to(ctx)]
        applicable.sort(key=lambda h: 0 if h.category == "governance" else 1)
        for hook in applicable:
            result.merge(await hook.execute_async(ctx))
            # Stop at first block
            if result.blocked:
                break
        return result


def create_hook_bus(hooks_file: Path | None = None) -> HookBus:
    return HookBus(hooks_file)
