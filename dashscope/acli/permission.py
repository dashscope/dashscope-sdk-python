# -*- coding: utf-8 -*-
"""Per-tool and per-domain permission policy.

The interactive permission protocol between ``Executor`` and the TUI/CLI
confirm paths uses raw strings (``"y"``/``"n"``/``"u"``/``"a"``/``"s"``).

``PermissionPolicy`` enables per-tool and per-domain rules so administrators
can express policies like "always allow git status" or "always deny curl to
external hosts" without relying on the per-turn trust cache.

Rules are loaded from ``~/.acli/permissions.toml`` (global) and
``.acli/permissions.toml`` (workspace, wins on conflicts)::

    [tools]
    write_file = "confirm"
    delete_file = "deny"

    [commands]
    "git status" = "allow"
    "git push" = "deny"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


def _normalize_command(command: str) -> str:
    """Collapse whitespace runs so rule prefixes match ``git\\tpush`` too."""
    return " ".join(command.split())


@dataclass
class PermissionPolicy:
    """Per-tool and per-domain permission rules.

    Rules are checked before the interactive prompt. For command rules the
    most specific (longest) matching prefix wins, so a narrow ``deny``
    always overrides a broad ``allow``. If no rule matches, the normal
    ``PermissionLevel`` flow applies.
    """

    # tool_name -> "allow" | "deny" | "confirm"
    tool_rules: dict[str, str] = field(default_factory=dict)
    # shell-command-prefix -> "allow" | "deny" | "confirm"
    command_rules: dict[str, str] = field(default_factory=dict)

    def check_tool(self, tool_name: str) -> str | None:
        """Return the policy decision for a tool, or None to fall through."""
        return self.tool_rules.get(tool_name)

    def check_command(self, command: str) -> str | None:
        """Return the policy decision for a shell command, or None."""
        normalized = _normalize_command(command)
        best: tuple[int, str] | None = None
        for prefix, decision in self.command_rules.items():
            if normalized.startswith(prefix):
                # pylint: disable=unsubscriptable-object
                if best is None or len(prefix) > best[0]:
                    best = (len(prefix), decision)
        return best[1] if best else None

    def add_tool_rule(self, tool_name: str, decision: str) -> None:
        if decision not in ("allow", "deny", "confirm"):
            raise ValueError(
                f"decision must be allow/deny/confirm, got {decision}",
            )
        self.tool_rules[tool_name] = decision

    def add_command_rule(self, prefix: str, decision: str) -> None:
        if decision not in ("allow", "deny", "confirm"):
            raise ValueError(
                f"decision must be allow/deny/confirm, got {decision}",
            )
        normalized = _normalize_command(prefix)
        if not normalized:
            raise ValueError("command rule prefix must not be empty")
        self.command_rules[normalized] = decision


def load_permissions_file(policy: PermissionPolicy, path: Path) -> list[str]:
    """Merge a permissions.toml into ``policy``; returns load errors."""
    from dashscope.acli.utils import load_toml

    errors: list[str] = []
    data = load_toml(path)
    if data is None:
        return errors
    for section, add in (
        ("tools", policy.add_tool_rule),
        ("commands", policy.add_command_rule),
    ):
        entries = data.get(section, {})
        if not isinstance(entries, dict):
            errors.append(f"{path}: [{section}] must be a table")
            continue
        for key, decision in entries.items():
            try:
                add(str(key), str(decision))
            except ValueError as e:
                errors.append(f"{path}: [{section}] {key!r}: {e}")
    return errors


# ── Global policy ──────────────────────────────────────────────────────

_policy = PermissionPolicy()


def get_permission_policy() -> PermissionPolicy:
    return _policy


def set_permission_policy(policy: PermissionPolicy) -> None:
    global _policy
    _policy = policy


def configure_permission_policy() -> PermissionPolicy:
    """Load permissions.toml (global + workspace) into the global policy.

    Called once at startup alongside ``configure_audit_logger``. Workspace
    rules are applied last and win on identical keys. Invalid entries are
    skipped with a warning — a malformed rule must never widen permissions
    silently, so bad decisions are dropped, not defaulted to allow.
    """
    from dashscope.acli.config import CONFIG_DIR, WORKSPACE_DIR

    policy = PermissionPolicy()
    for path in (
        CONFIG_DIR / "permissions.toml",
        WORKSPACE_DIR / "permissions.toml",
    ):
        for err in load_permissions_file(policy, path):
            log.warning("%s", err)
    set_permission_policy(policy)
    return policy
