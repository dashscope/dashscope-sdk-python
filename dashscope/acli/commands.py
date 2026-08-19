# -*- coding: utf-8 -*-
"""Shared command definitions and utilities used by both CLI and TUI.

This module contains UI-agnostic pieces of slash-command handling so that
command behavior only needs to be updated in one place.
"""

from __future__ import annotations

import os
import subprocess as _sp

# Single source of truth for the /help menu. CLI and TUI both render from
# this structure so that adding or renaming a command only requires one edit.
HELP_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Session",
        [
            ("/help", "Show help"),
            ("/clear", "Clear conversation history"),
            ("/info", "Show runtime info (provider, model, config)"),
            ("/stats", "Show session stats (tool calls, model info)"),
            ("/voice", "Voice input (on/off/model/silence/max/threshold)"),
            (
                "/tts",
                "Voice output (on/off/status/model/voice/speed/say/last)",
            ),
            ("/camera capture [file]", "Take a photo with the camera"),
            ("/camera record [duration] [file]", "Record video (default 5s)"),
            ("/copy", "Copy the last reply to the clipboard"),
            (
                "/save [path]",
                "Save the last reply to a file "
                "(default acli_output_<time>.md)",
            ),
            (
                "/json on|off",
                "JSON output mode (replies forced to JSON when on)",
            ),
            ("/compress", "Compress context (LLM summary replaces history)"),
            ("/history", "Conversation history (stats/list/export/clear)"),
            (
                "/feedback good|bad",
                "Rate task satisfaction (stored in experience memory)",
            ),
            ("/report", "Generate a trace performance report"),
            (
                "/log",
                "View LLM prompts logged in debug mode "
                "(tail [N]/search <keyword>/clear), paged",
            ),
            (
                "/trace",
                "View execution traces (calls/latency/data flow) "
                "(tail [N]/search <keyword>/clear), paged",
            ),
            ("/exit", "Exit"),
        ],
    ),
    (
        "Config",
        [
            (
                "/setup",
                "Set up the workspace (user, Provider, model, capabilities)",
            ),
            (
                "/capability",
                "Capability toggles (list/enable/disable/reload/config)",
            ),
            (
                "/subagents",
                "Subagent management (list/reload/enable/disable/config)",
            ),
            (
                "/provider",
                "Configure Provider / Key / model / protocol (Q&A style)",
            ),
            (
                "/trust",
                "Tool trust/deny cache for this session "
                "(list/clear/allow/deny)",
            ),
            (
                "/rule",
                "Long-term user rules (list/add/remove/edit/clear), "
                "injected into the system prompt each turn",
            ),
            (
                "/privacy",
                "Privacy mode (on/off/status); data stays local when on",
            ),
            (
                "/debug",
                "Debug mode (on/off/status); logs the final LLM prompt",
            ),
            ("/theme", "Theme settings (list/set/custom colors)"),
            (
                "/directives",
                "Directives auto-learning proposals (proposals/accept/reject)",
            ),
        ],
    ),
    (
        "Capabilities",
        [
            ("/profile", "User profile (list/search/add/remove/clear)"),
            ("/memory", "Chat history (list/search/remove <id|num>/clear)"),
            ("/session", "Session management (new/list/switch/rename/remove)"),
            (
                "/summarize",
                "Summarize the current task; record key steps and lessons",
            ),
            ("/mcp", "MCP services (list/add/remove)"),
            (
                "/skill",
                "Skill invocation and management (list/add/remove/install/"
                "uninstall/enable/disable/update)",
            ),
            ("/cron", "Scheduled tasks (add/list/remove/pause/resume)"),
            ("/audit", "Audit log (recent [N]/query/clear)"),
        ],
    ),
    (
        "Dev / Extensions",
        [
            ("/dev", "Overview (model registry + extension guides)"),
            (
                "/dev model add|list|remove <provider> [<name>]",
                "Register/list/remove models for a provider "
                "(persisted to workspace)",
            ),
            (
                "/dev provider add|list|remove [name]",
                "Layer-1 LLM Provider extension (OpenAI-compatible), "
                "written to custom-extensions.toml",
            ),
            (
                "/dev capability add|list|remove [key]",
                "Layer-1 HTTP tool capability extension, scaffold + edit",
            ),
            (
                "/dev skill add|list|remove [name]",
                "Layer-1 custom Skill (prompt template), "
                "written to custom-extensions.toml",
            ),
            (
                "/dev tool add|list|remove [name]",
                "Layer-1 custom Shell tool (wrap a command as an LLM tool)",
            ),
            (
                "/dev debug tools|schema|call|prompt",
                "Debug: registered tools / param schemas / manual call / "
                "system prompt",
            ),
            (
                "/dev test provider <name> | reload | log",
                "Test provider connectivity / hot-reload extensions / "
                "tool registration stats",
            ),
            (
                "/dev platform | tool | skill",
                "Layer-2 guide for real Python module extensions "
                "(prints steps)",
            ),
            ("/example", "List available example projects"),
            (
                "/example download <name>",
                "Merge an example into ./.acli/ (conflicts auto-backed "
                "up, restore undoes)",
            ),
            (
                "/example restore",
                "Restore the .acli/backup/ backup (undo the last merge)",
            ),
        ],
    ),
]

# Examples shown at the bottom of /help.
_HELP_EXAMPLES = [
    "List files in the current directory",
    "Create a test.txt containing hello world",
    "/mcp add code-interpreter",
    "/cron add every 5m /skill my-skill arg1",
    "/history export history.json --format json",
    "/json on",
    "/save output.md",
]


def render_help_text() -> str:
    """Return the /help content as Rich-tagged text."""
    lines = ["[bold]Available commands[/bold]"]
    for title, items in HELP_SECTIONS:
        lines.append(f"\n[bold yellow]{title}[/bold yellow]")
        for cmd_text, desc in items:
            lines.append(f"  [cyan]{cmd_text}[/cyan] [dim]—[/dim] {desc}")
    lines.append("\n[bold]Examples[/bold]")
    for ex in _HELP_EXAMPLES:
        lines.append(f"  [dim]·[/dim] {ex}")
    return "\n".join(lines)


def handle_shell_escape(shell_cmd: str) -> tuple[str, str, int]:
    """Execute a shell escape command and return (stdout, stderr, rc).

    Used by the TUI where output must be captured and rendered into the
    RichLog. The CLI keeps its own direct-terminal implementation for
    streaming interactive commands.
    """
    env = os.environ.copy()
    env["ACLI_CLI"] = "1"
    try:
        proc = _sp.run(
            shell_cmd,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except Exception as e:
        return "", str(e), 1
