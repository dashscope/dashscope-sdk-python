# -*- coding: utf-8 -*-
"""Audit logging for user input and agent decisions.

The design doc places ``AuditLog`` in the Governance layer, receiving events
from both ``Interface`` (user input) and ``Agent`` (tool decisions). This
module provides the ``AuditEvent`` value type and ``AuditLogger`` that
persists events as JSONL to ``~/.acli/audit.log``.

Wiring:
  - User input: ``log_user_input(text)`` called from the REPL/TUI before
    each turn.
  - Tool decisions: ``log_tool_call(name, args, decision, reason)`` called
    from ``Executor`` after permission resolution.
  - Query: ``/audit query`` slash command reads back events.

Privacy: when ``privacy_mode`` is on, the audit log still records
*decisions* (which tool was approved/denied) but redacts argument values
that may contain user data. The tool name and decision are always kept so
the audit trail remains useful for compliance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashscope.acli.config import WORKSPACE_DIR

_AUDIT_FILE = WORKSPACE_DIR / "audit.log"

# Tool argument keys whose values are redacted in privacy mode.
_SENSITIVE_ARG_KEYS = frozenset(
    {"content", "command", "query", "text", "prompt", "url", "path", "file"},
)

# Credential-like keys are redacted in full regardless of length — even a
# short token must never land in the log.
_CREDENTIAL_ARG_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "key",
        "token",
        "secret",
        "password",
        "passwd",
        "authorization",
        "auth",
        "credential",
        "cookie",
    },
)


@dataclass
class AuditEvent:
    """One auditable action in the system."""

    timestamp: str
    source: str  # "user" | "agent" | "scheduler" | "system"
    action: str  # "input" | "tool_call" | "permission" | "cron" | "mcp"
    subject: str  # tool name, user input preview, etc.
    decision: str  # "approved" | "denied" | "executed" | "skipped" | "auto"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        source: str,
        action: str,
        subject: str,
        decision: str,
        reason: str = "",
        **details,
    ) -> "AuditEvent":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            action=action,
            subject=subject,
            decision=decision,
            reason=reason,
            details=details,
        )


def _redact_args(args: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive argument values for privacy mode (recursive)."""
    redacted: dict[str, Any] = {}
    for k, v in args.items():
        key = k.lower()
        if key in _CREDENTIAL_ARG_KEYS:
            redacted[k] = "…(redacted)"
        elif isinstance(v, dict):
            redacted[k] = _redact_args(v)
        elif key in _SENSITIVE_ARG_KEYS and isinstance(v, str) and len(v) > 20:
            redacted[k] = v[:20] + "…(redacted)"
        else:
            redacted[k] = v
    return redacted


def _redact_text(text: str, max_len: int = 20) -> str:
    """Truncate free-form text for privacy mode."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…(redacted)"


class AuditLogger:
    """Append-only JSONL audit log."""

    def __init__(self, path: Path | None = None):
        self._path = path or _AUDIT_FILE
        self._privacy_mode = False

    def set_privacy_mode(self, enabled: bool) -> None:
        self._privacy_mode = enabled

    def log(self, event: AuditEvent) -> None:
        """Append one event. Swallows I/O errors so audit never breaks
        the agent."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            record = asdict(event)
            if self._privacy_mode:
                record["details"] = _redact_args(record.get("details", {}))
                if record.get("source") == "user":
                    record["subject"] = _redact_text(record.get("subject", ""))
            with self._path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n",
                )
        except Exception:
            pass  # audit must never break the agent loop

    def log_user_input(self, text: str, preview_len: int = 100) -> None:
        preview = text[:preview_len] + ("…" if len(text) > preview_len else "")
        self.log(AuditEvent.now("user", "input", preview, "received"))

    def log_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        decision: str,
        reason: str = "",
    ) -> None:
        self.log(
            AuditEvent.now(
                "agent",
                "tool_call",
                tool_name,
                decision,
                reason,
                **args,
            ),
        )

    def log_cron(self, job_id: str, decision: str, reason: str = "") -> None:
        self.log(AuditEvent.now("scheduler", "cron", job_id, decision, reason))

    def query(
        self,
        source: str | None = None,
        action: str | None = None,
        subject: str | None = None,
        decision: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Read back events matching the given filters."""
        if not self._path.exists():
            return []
        results: list[dict[str, Any]] = []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if source and record.get("source") != source:
                        continue
                    if action and record.get("action") != action:
                        continue
                    if subject and record.get("subject") != subject:
                        continue
                    if decision and record.get("decision") != decision:
                        continue
                    results.append(record)
                    if len(results) >= limit:
                        break
        except Exception:
            pass  # unreadable log file → empty result, query must not raise
        return results

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the last N events (tail of the log)."""
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").strip().split("\n")
        except Exception:
            return []
        records: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            pass  # best-effort: a locked/readonly log file is not fatal


# ── Global instance ────────────────────────────────────────────────────

_logger = AuditLogger()


def get_audit_logger() -> AuditLogger:
    return _logger


def set_audit_logger(logger: AuditLogger) -> None:
    global _logger
    _logger = logger


def configure_audit_logger(config) -> AuditLogger:
    """Wire config settings into the global audit logger at startup.

    Re-arms privacy redaction after a restart: the logger is constructed at
    import time with privacy off, so without this the persisted
    ``privacy_mode = true`` would not redact until the user re-ran
    ``/privacy on``.
    """
    _logger.set_privacy_mode(bool(getattr(config, "privacy_mode", False)))
    return _logger
