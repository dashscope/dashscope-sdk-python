# -*- coding: utf-8 -*-
from __future__ import annotations

import fnmatch
import os
import shutil

from dashscope.acli.tools.registry import PermissionLevel, tool
from dashscope.acli.utils.paths import (
    SENSITIVE_NAMES,
    validate_path,
    validate_write_path,
)

# Maximum directory depth for search_files traversal.
_MAX_SEARCH_DEPTH = 8

# Maximum file content size for write_file (50 MB).
_MAX_WRITE_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="read_file",
    description="Read file contents. Optional start line and line count.",
    permission=PermissionLevel.AUTO,
)
def read_file(
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not os.path.isfile(path):
        return f"Error: file not found - {path}"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    start = offset or 0
    end = start + limit if limit else len(lines)
    selected = lines[start:end]

    result_lines = []
    for i, line in enumerate(selected, start=start + 1):
        result_lines.append(f"{i:>4}\t{line.rstrip()}")
    return "\n".join(result_lines)


@tool(
    name="write_file",
    description=(
        "Write content to a file; overwrites if it exists. "
        "The result includes a unified diff."
    ),
    permission=PermissionLevel.CONFIRM,
)
def write_file(path: str, content: str) -> str:
    import difflib

    try:
        path = validate_write_path(path)
    except ValueError as e:
        return f"Error: {e}"

    if len(content) > _MAX_WRITE_SIZE:
        return (
            f"Error: content too large ({len(content)} bytes); "
            f"limit is {_MAX_WRITE_SIZE // (1024*1024)} MB"
        )

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Capture before-state for diff (best-effort — binary or read errors
    # fall through to "new file" treatment).
    old_content = ""
    existed = os.path.isfile(path)
    if existed:
        try:
            with open(path, "r", encoding="utf-8") as f:
                old_content = f.read()
        except (OSError, UnicodeDecodeError):
            existed = False  # treat as new for diff purposes

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    if not existed:
        return f"Created file: {path} ({len(content)} chars)"

    if old_content == content:
        return f"Content unchanged: {path}"

    # Cap displayed diff so a 5000-line full rewrite doesn't drown the UI;
    # full content is in messages so LLM still sees everything it wrote.
    # Don't pass lineterm="" — default "\n" gives proper newlines on the
    # ---, +++, @@ control lines so the diff is readable when joined.
    diff_lines = list(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        ),
    )
    MAX_DIFF_LINES = 80
    truncated = ""
    if len(diff_lines) > MAX_DIFF_LINES:
        total_added = sum(
            1
            for dline in diff_lines
            if dline.startswith("+") and not dline.startswith("+++")
        )
        diff_lines = diff_lines[:MAX_DIFF_LINES]
        truncated = (
            f"\n... (diff truncated: {total_added} lines added, "
            f"showing first {MAX_DIFF_LINES})"
        )
    diff_text = "".join(diff_lines)

    return (
        f"Wrote file: {path} ({len(content)} chars)\n"
        f"--- diff ---\n"
        f"{diff_text}{truncated}"
    )


@tool(
    name="list_directory",
    description="List directory contents with file type and size.",
    permission=PermissionLevel.AUTO,
)
def list_directory(path: str) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not os.path.isdir(path):
        return f"Error: directory not found - {path}"

    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            entries.append(f"  [dir]  {name}/")
        else:
            size = os.path.getsize(full)
            entries.append(f"  [file] {name}  ({_format_size(size)})")

    header = f"Directory: {path} ({len(entries)} entries)"
    return header + "\n" + "\n".join(entries)


@tool(
    name="create_directory",
    description="Create a directory, including nested levels.",
    permission=PermissionLevel.CONFIRM,
)
def create_directory(path: str) -> str:
    try:
        path = validate_write_path(path)
    except ValueError as e:
        return f"Error: {e}"
    os.makedirs(path, exist_ok=True)
    return f"Created directory: {path}"


@tool(
    name="delete_file",
    description="Delete the specified file. This cannot be undone.",
    permission=PermissionLevel.DANGEROUS,
)
def delete_file(path: str) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not os.path.isfile(path):
        return f"Error: file not found - {path}"
    os.remove(path)
    return f"Deleted file: {path}"


@tool(
    name="delete_directory",
    description=(
        "Delete a directory and all its contents. " "This cannot be undone."
    ),
    permission=PermissionLevel.DANGEROUS,
)
def delete_directory(path: str) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not os.path.isdir(path):
        return f"Error: directory not found - {path}"
    # Prevent deleting cwd itself
    abs_path = os.path.abspath(path)
    cwd = os.path.abspath(".")
    if abs_path == cwd:
        return "Error: cannot delete the current working directory"
    shutil.rmtree(path)
    return f"Deleted directory: {path}"


@tool(
    name="search_files",
    description=(
        "Search files by name pattern. "
        "Supports wildcards like *.py, test_*."
    ),
    permission=PermissionLevel.AUTO,
)
def search_files(pattern: str, path: str | None = None) -> str:
    search_root = os.path.expanduser(path) if path else os.getcwd()
    try:
        search_root = validate_path(search_root)
    except ValueError as e:
        return f"Error: {e}"
    if not os.path.isdir(search_root):
        return f"Error: search path not found - {search_root}"

    matches = []
    root_depth = search_root.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(search_root):
        # Enforce depth limit
        current_depth = root.rstrip(os.sep).count(os.sep) - root_depth
        if current_depth >= _MAX_SEARCH_DEPTH:
            dirs[:] = []
        # Skip hidden and sensitive directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and d not in SENSITIVE_NAMES
        ]
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if name in SENSITIVE_NAMES:
                    continue
                matches.append(os.path.join(root, name))
                if len(matches) >= 100:
                    break
        if len(matches) >= 100:
            break

    if not matches:
        return f"No files match '{pattern}'"
    result = f"Found {len(matches)} matching files:\n"
    result += "\n".join(f"  {m}" for m in matches)
    return result


@tool(
    name="move_file",
    description="Move or rename a file/directory.",
    permission=PermissionLevel.CONFIRM,
)
def move_file(src: str, dst: str) -> str:
    try:
        src = validate_path(src)
        dst = validate_write_path(dst)
    except ValueError as e:
        return f"Error: {e}"
    if not os.path.exists(src):
        return f"Error: source path not found - {src}"
    shutil.move(src, dst)
    return f"Moved: {src} → {dst}"


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
