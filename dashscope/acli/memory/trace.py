# -*- coding: utf-8 -*-
"""
Trace logging for agent execution.
Records LLM calls, tool executions, and decision points to JSONL files.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


class TraceLogger:
    """Logs agent execution traces to .acli/traces/<session_id>.jsonl"""

    def __init__(self, workspace_dir: Path):
        self.traces_dir = workspace_dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_file = self.traces_dir / f"{self.session_id}.jsonl"
        self._session_start = time.time()

    def log(self, event_type: str, data: dict[str, Any]) -> None:
        """Write a trace event to the JSONL file."""
        from dashscope.acli.utils import sanitize

        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": int((time.time() - self._session_start) * 1000),
            "event": event_type,
            **sanitize(data),
        }
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Trace logging should never break the main flow
            pass

    def log_llm_call(
        self,
        messages: list[dict],
        tools: list[dict],
        response: dict,
        duration_ms: int = 0,
        model: str = "",
    ) -> None:
        """Log an LLM API call.

        Lightweight fields (counts, usage, duration) are always recorded;
        the full prompt payload is included only in debug mode (/debug on).
        """
        data: dict[str, Any] = {
            "model": model,
            "message_count": len(messages),
            "tool_count": len(tools),
            "duration_ms": duration_ms,
            "response_content_length": len(response.get("content", "")),
            "tool_calls": response.get("tool_calls", []),
            "usage": response.get("usage"),
        }
        from dashscope.acli import debuglog

        if debuglog.debug_enabled():
            data["messages"] = messages
            data["tools"] = [t.get("name", "?") for t in tools or []]
        self.log("llm_call", data)

    def log_tool_execution(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        duration_ms: int,
        success: bool,
    ) -> None:
        """Log a tool execution."""
        self.log(
            "tool_execution",
            {
                "tool": tool_name,
                "arguments": arguments,
                "result_length": len(result),
                "result_preview": result[:200] if result else "",
                "duration_ms": duration_ms,
                "success": success,
            },
        )

    def log_decision(self, decision_type: str, details: dict) -> None:
        """Log a decision point (e.g., plan created, compression triggered)."""
        self.log(
            "decision",
            {
                "type": decision_type,
                **details,
            },
        )

    def iter_events(self) -> Iterator[dict[str, Any]]:
        """Yield all trace events of this session, oldest first."""
        if not self.trace_file.exists():
            return
        try:
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return

    def tail_events(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(self.iter_events())[-limit:]

    def search_events(
        self,
        keyword: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        kw = keyword.lower()
        matches = [
            e
            for e in self.iter_events()
            if kw in json.dumps(e, ensure_ascii=False, default=str).lower()
        ]
        return matches[-limit:]

    def clear(self) -> None:
        try:
            self.trace_file.unlink(missing_ok=True)
        except Exception:
            pass  # best-effort: a locked/readonly trace file is not fatal


def generate_report(trace_logger: TraceLogger | None) -> dict | None:
    """Generate a performance report from trace logs."""
    if trace_logger is None or not trace_logger.trace_file.exists():
        return None

    llm_calls = 0
    tool_calls = 0
    tool_successes = 0
    total_duration_ms = 0
    duration_count = 0
    tool_counts: dict[str, int] = {}

    try:
        with open(trace_logger.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = entry.get("event")
                if event == "llm_call":
                    llm_calls += 1
                elif event == "tool_execution":
                    tool_calls += 1
                    if entry.get("success"):
                        tool_successes += 1
                    dur = entry.get("duration_ms", 0)
                    if dur:
                        total_duration_ms += dur
                        duration_count += 1
                    name = entry.get("tool", "unknown")
                    tool_counts[name] = tool_counts.get(name, 0) + 1
    except OSError:
        return None

    if llm_calls == 0 and tool_calls == 0:
        return None

    avg_response_time = (
        (total_duration_ms / duration_count / 1000) if duration_count else 0.0
    )
    top_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_llm_calls": llm_calls,
        "total_tool_calls": tool_calls,
        "tool_success_rate": (tool_successes / tool_calls)
        if tool_calls
        else 0.0,
        "avg_response_time": avg_response_time,
        "top_tools": top_tools,
    }
