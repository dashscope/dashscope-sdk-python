# -*- coding: utf-8 -*-
"""Deliverable surfacing — detect and present files produced by tool calls.

When an agent turn writes files (via write_file, shell, etc.), this module
identifies the resulting paths and prints a compact summary so the user can
open or inspect them without manually copying paths from the tool trail.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

from rich.console import Console

console = Console()

# Tools whose results commonly contain generated file paths.
_DELIVERABLE_TOOL_NAMES = {
    "write_file",
    "move_file",
    "copy_file",
    "run_command",
    "shell",
}

# File extensions we consider "openable" deliverables.
_OPENABLE_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".html",
    ".css",
    ".sh",
    ".sql",
    ".log",
    ".csv",
}

# Path-like tokens: /foo/bar, ./foo/bar, foo/bar.txt (with extension).
_PATH_PATTERN = re.compile(
    r"(?:^|[\s\[\](){}<>|`\"'\n])"
    r"((?:/|(?:\.{1,2}/)|[A-Za-z]:\\)?"
    r"[\w\-./\\]+(?:\.[A-Za-z0-9\-_]+)+)"
    r"(?=$|[\s\[\](){}<>|`\"'\n])",
)


def _looks_like_path(token: str) -> bool:
    token = token.strip()
    if not token or len(token) < 3:
        return False
    # Skip obvious non-paths.
    if token.startswith(("http://", "https://", "<", ">", "`")):
        return False
    if "/" not in token and "\\" not in token and "." not in token:
        return False
    # Must have a reasonable extension or start with / or ./
    if not (token.startswith(("/", "./", "../")) or Path(token).suffix):
        return False
    return True


def _candidate_paths(text: str) -> Iterable[str]:
    """Yield path-looking tokens from tool result text."""
    seen: set[str] = set()
    for match in _PATH_PATTERN.finditer(text):
        token = match.group(1)
        if not _looks_like_path(token):
            continue
        # Normalize relative paths against cwd.
        if token.startswith("./"):
            token = token[2:]
        elif token.startswith("../"):
            pass  # keep as-is; resolve below
        if token in seen:
            continue
        seen.add(token)
        yield token


def collect_deliverables(messages: list[dict]) -> list[Path]:
    """Scan tool messages and return existing file paths produced this turn."""
    deliverables: list[Path] = []
    seen: set[Path] = set()
    cwd = Path.cwd()

    for msg in messages:
        if msg.get("role") != "tool":
            continue
        if msg.get("name", "") not in _DELIVERABLE_TOOL_NAMES:
            continue

        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )

        for token in _candidate_paths(content):
            path = Path(token)
            if not path.is_absolute():
                path = cwd / path
            try:
                path = path.resolve()
            except OSError:
                continue
            if path in seen:
                continue
            seen.add(path)
            if path.exists() and path.is_file():
                deliverables.append(path)

    return deliverables


def _open_command() -> str | None:
    """Return the platform-appropriate open command, if available."""
    if os.name == "nt":
        return "start"
    if os.uname().sysname == "Darwin":
        return "open"
    return "xdg-open"


def surface_deliverables(
    deliverables: list[Path],
    auto_open: bool = False,
) -> None:
    """Print a summary of generated files and optionally open them."""
    if not deliverables:
        return

    console.print()
    console.print("[dim]─ Generated files ─[/dim]")
    for path in deliverables:
        rel = (
            path.relative_to(Path.cwd())
            if path.is_relative_to(Path.cwd())
            else path
        )
        suffix = path.suffix.lower()
        icon = "📄" if suffix in _OPENABLE_EXTENSIONS else "📦"
        console.print(f"{icon} [cyan]{rel}[/cyan]")

        if auto_open and suffix in _OPENABLE_EXTENSIONS:
            cmd = _open_command()
            if cmd:
                try:
                    subprocess.Popen(  # pylint: disable=consider-using-with
                        [cmd, str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

    console.print(
        "[dim]  Tip: use /open <path> or view in your file manager[/dim]",
    )
