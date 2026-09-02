# -*- coding: utf-8 -*-
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements
from __future__ import annotations

import inspect
import json
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from dashscope.acli.permission import get_permission_policy
from dashscope.acli.tools.registry import PermissionLevel, ToolDefinition
from dashscope.acli.tools.shell import is_safe_readonly
from dashscope.acli.utils.exceptions import UserAbortedTurn, UserSupplement
from dashscope.acli.utils.spinner import StderrSpinner
from dashscope.acli.utils.text import truncate_value
from dashscope.acli.utils.validation import coerce_types, missing_required_args

console = Console()


PERMISSION_STYLES = {
    PermissionLevel.AUTO: "green",
    PermissionLevel.CONFIRM: "yellow",
    PermissionLevel.DANGEROUS: "red bold",
}

# Canonical prefix for tool-execution failure results. Agent code classifies
# tool outcomes by this marker — keep it in sync with agent's
# _TOOL_ERROR_PREFIXES.
EXEC_ERROR_PREFIX = "Error"


class Executor:
    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve
        # Trust cache scoped to ONE conversation turn (a single user prompt
        # plus the agent loop that answers it). Agent.run / run_stream clear
        # these in a finally block on completion or abort. DANGEROUS tools
        # never enter _always_allow (see _check_permission below).
        self._always_allow: set[str] = set()
        self._always_deny: set[str] = set()
        # Optional async callback for TUI mode confirmation
        self._confirm_callback = None

        # Usage stats
        self._session_start_time: float = time.time()
        self._tool_call_counts: dict[str, int] = {}
        self._total_tool_calls: int = 0
        self._skill_calls: int = 0
        self._skill_counts: dict[str, int] = {}
        self._api_calls: int = 0
        self._token_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        self._prompt_composition: dict[str, int] = {
            "system": 0,
            "user": 0,
            "assistant": 0,
            "tools": 0,
        }
        self._errors: int = 0

    def clear_session_trust(self) -> None:
        """Called by Agent at the end of each turn to reset trust grants."""
        self._always_allow.clear()
        self._always_deny.clear()

    def record_api_call(self, usage: dict | None = None) -> None:
        """Record an API call and its token usage."""
        self._api_calls += 1
        if usage:
            self._token_usage["input_tokens"] += usage.get("input_tokens", 0)
            self._token_usage["output_tokens"] += usage.get("output_tokens", 0)
            self._token_usage["total_tokens"] += usage.get("total_tokens", 0)
            self._token_usage["cached_tokens"] += usage.get("cached_tokens", 0)

    def record_error(self) -> None:
        """Record an error occurrence."""
        self._errors += 1

    def record_skills(self, names: list[str]) -> None:
        """Record skill packages activated for a turn."""
        for name in names:
            self._skill_calls += 1
            self._skill_counts[name] = self._skill_counts.get(name, 0) + 1

    def record_prompt_composition(
        self,
        messages: list[dict],
        tools_schema: list[dict] | None = None,
    ) -> None:
        """Accumulate prompt size (chars) by category for one LLM call."""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, str):
                n = len(content)
            elif content:
                n = len(json.dumps(content, ensure_ascii=False))
            else:
                n = 0
            if role in ("system", "user", "assistant"):
                self._prompt_composition[role] += n
            elif role == "tool":
                self._prompt_composition["tools"] += n
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    args = (tc.get("function") or {}).get("arguments") or ""
                    self._prompt_composition["tools"] += len(args)
        if tools_schema:
            self._prompt_composition["tools"] += len(
                json.dumps(tools_schema, ensure_ascii=False),
            )

    async def execute(self, tool_def: ToolDefinition, arguments: dict) -> str:
        # Validate required params BEFORE asking for permission — otherwise
        # the model can emit `{}` and the user is forced to confirm a call
        # that will immediately fail validation anyway.
        missing = missing_required_args(tool_def, arguments)
        if missing:
            return (
                f"Error: tool {tool_def.name} missing required arguments: "
                f"{', '.join(missing)}. Call again following the schema "
                f"(required: {tool_def.parameters.get('required', [])})"
            )

        if not await self._async_check_permission(tool_def, arguments):
            from dashscope.acli.audit import get_audit_logger

            get_audit_logger().log_tool_call(
                tool_def.name,
                arguments,
                decision="denied",
                reason="user declined",
            )
            return "Cancelled"

        try:
            arguments = coerce_types(tool_def.func, arguments)
            with StderrSpinner(f"Running {tool_def.name}..."):
                result = tool_def.func(**arguments)
                if inspect.isawaitable(result):
                    result = await result
            self._total_tool_calls += 1
            self._tool_call_counts[tool_def.name] = (
                self._tool_call_counts.get(tool_def.name, 0) + 1
            )
            from dashscope.acli.audit import get_audit_logger

            get_audit_logger().log_tool_call(
                tool_def.name,
                arguments,
                decision="executed",
            )
            return str(result)
        except Exception as e:
            self.record_error()
            from dashscope.acli.audit import get_audit_logger

            get_audit_logger().log_tool_call(
                tool_def.name,
                arguments,
                decision="failed",
                reason=str(e),
            )
            return f"{EXEC_ERROR_PREFIX}: {type(e).__name__}: {e}"

    def get_stats(self) -> dict:
        """Return session usage statistics."""
        session_duration = time.time() - self._session_start_time
        return {
            "total_tool_calls": self._total_tool_calls,
            "tool_counts": dict(self._tool_call_counts),
            "skill_calls": self._skill_calls,
            "skill_counts": dict(self._skill_counts),
            "api_calls": self._api_calls,
            "token_usage": dict(self._token_usage),
            "prompt_composition": dict(self._prompt_composition),
            "errors": self._errors,
            "session_duration": session_duration,
        }

    async def _async_check_permission(
        self,
        tool_def: ToolDefinition,
        arguments: dict,
    ) -> bool:
        """Async version of _check_permission. When _confirm_callback is
        set (TUI mode), delegates to the callback; otherwise falls back
        to sync _check_permission.
        """
        if self._confirm_callback:
            if (
                tool_def.permission == PermissionLevel.AUTO
                or self.auto_approve
            ):
                return True
            # Policy deny rules run before the readonly fast-path: an admin
            # deny must always win, even over read-only auto-approval.
            policy = get_permission_policy()
            if policy.check_tool(tool_def.name) == "deny":
                return False
            if tool_def.name == "run_command":
                cmd = arguments.get("command", "")
                if (
                    isinstance(cmd, str)
                    and policy.check_command(cmd) == "deny"
                ):
                    return False
            # Auto-pass read-only shell commands
            if tool_def.name == "run_command":
                cmd = arguments.get("command", "")
                if isinstance(cmd, str) and is_safe_readonly(cmd):
                    return True
            # Consult PermissionPolicy (admin-level rules)
            policy_decision = policy.check_tool(tool_def.name)
            if policy_decision == "allow":
                return True
            if policy_decision == "deny":
                return False
            if tool_def.name == "run_command":
                cmd = arguments.get("command", "")
                if isinstance(cmd, str):
                    cmd_decision = policy.check_command(cmd)
                    if cmd_decision == "allow":
                        return True
                    if cmd_decision == "deny":
                        return False
            # Consult turn-scoped trust cache
            if tool_def.name in self._always_deny:
                return False
            if tool_def.name in self._always_allow:
                return True
            # Delegate to TUI callback
            is_dangerous = tool_def.permission == PermissionLevel.DANGEROUS
            # pylint: disable=not-callable
            result = await self._confirm_callback(
                tool_def,
                arguments,
                is_dangerous,
            )
            # pylint: enable=not-callable
            if result == "a":
                # DANGEROUS never enters the trust cache (sync path: y/n only)
                if not is_dangerous:
                    self._always_allow.add(tool_def.name)
                return True
            if result == "s":
                raise UserAbortedTurn("User aborted this turn")
            # [u]pdate should be handled by the confirmation callback itself
            # (it must prompt for supplement and raise UserSupplement). If a
            # callback returns "u" without doing that, treat it as abort.
            if result == "u":
                raise UserAbortedTurn("No supplementary info provided")
            return result == "y"
        return self._check_permission(tool_def, arguments)

    def _check_permission(
        self,
        tool_def: ToolDefinition,
        arguments: dict,
    ) -> bool:
        if tool_def.permission == PermissionLevel.AUTO or self.auto_approve:
            return True

        # Policy deny rules run before the readonly fast-path: an admin
        # deny must always win, even over read-only auto-approval.
        policy = get_permission_policy()
        if policy.check_tool(tool_def.name) == "deny":
            console.print(
                f"[dim red]✗ {tool_def.name} (policy denied)[/dim red]",
            )
            return False
        if tool_def.name == "run_command":
            cmd = arguments.get("command", "")
            if isinstance(cmd, str) and policy.check_command(cmd) == "deny":
                console.print("[dim red]✗ command denied by policy[/dim red]")
                return False

        # Auto-pass read-only shell commands (grep, ls, find, git status, …).
        # The classifier lives in shell.py since it owns the shell semantics;
        # see is_safe_readonly for the allow/deny rules. Anything not
        # recognized stays in the CONFIRM lane.
        if tool_def.name == "run_command":
            cmd = arguments.get("command", "")
            if isinstance(cmd, str) and is_safe_readonly(cmd):
                preview = cmd if len(cmd) <= 80 else cmd[:80] + "…"
                console.print(f"[dim green]→ {preview}[/dim green]")
                return True

        # Consult PermissionPolicy (admin-level rules)
        policy_decision = policy.check_tool(tool_def.name)
        if policy_decision == "allow":
            return True
        if policy_decision == "deny":
            console.print(
                f"[dim red]✗ {tool_def.name} (policy denied)[/dim red]",
            )
            return False
        if tool_def.name == "run_command":
            cmd = arguments.get("command", "")
            if isinstance(cmd, str):
                cmd_decision = policy.check_command(cmd)
                if cmd_decision == "allow":
                    return True
                if cmd_decision == "deny":
                    console.print(
                        "[dim red]✗ command denied by policy[/dim red]",
                    )
                    return False

        # Consult turn-scoped trust cache (CONFIRM only; DANGEROUS never
        # caches).
        if tool_def.name in self._always_deny:
            console.print(
                f"[dim red]✗ {tool_def.name} (denied this turn)[/dim red]",
            )
            return False
        if tool_def.name in self._always_allow:
            console.print(
                f"[dim green]✓ {tool_def.name} "
                f"(trusted this turn)[/dim green]",
            )
            return True

        style = PERMISSION_STYLES[tool_def.permission]
        title = (
            "⚠️  Dangerous operation"
            if tool_def.permission == PermissionLevel.DANGEROUS
            else "Confirmation required"
        )

        args_display = "\n".join(
            f"  {k}: {truncate_value(v)}" for k, v in arguments.items()
        )
        content = f"Tool: {tool_def.name}\nArguments:\n{args_display}"

        console.print()
        console.print(Panel(content, title=title, border_style=style))

        # DANGEROUS sticks with binary y/n. Caching delete-style operations
        # crosses a line nobody benefits from.
        if tool_def.permission == PermissionLevel.DANGEROUS:
            try:
                return Confirm.ask("Execute?", default=False)
            except (KeyboardInterrupt, EOFError):
                # Ctrl-C / Ctrl-D at the confirmation prompt = abort the turn.
                # asyncio + prompt_toolkit can swallow the signal mid-prompt,
                # so we explicitly translate it here.
                raise UserAbortedTurn("Ctrl-C aborted this turn") from None

        # CONFIRM gets the four-way prompt with turn-scoped memory.
        # Brackets are escaped (\[…\]) so Rich doesn't parse them as markup
        # tags and eat the first letter — see issue with "es / o / lways /
        # top" rendering previously.
        try:
            choice = Prompt.ask(
                r"Execute? \[y]es / \[n]o / \[u]pdate (add info, replan) / "
                r"\[a]lways (allow this tool for the turn) / \[s]top (abort)",
                choices=["y", "n", "u", "a", "s"],
                default="y",
                show_choices=False,
            )
        except (KeyboardInterrupt, EOFError):
            raise UserAbortedTurn("Ctrl-C aborted this turn") from None
        if choice == "a":
            self._always_allow.add(tool_def.name)
            return True
        if choice == "u":
            supplement = Prompt.ask("[dim]Supplementary info[/dim]")
            if supplement.strip():
                raise UserSupplement(supplement.strip())
            return True  # empty supplement = proceed
        if choice == "s":
            raise UserAbortedTurn("User aborted this turn")
        return choice == "y"
