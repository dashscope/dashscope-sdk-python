# -*- coding: utf-8 -*-
"""File checkpoint/undo support for mutating file tools.

Backups live under ``<workspace>/.acli/checkpoints/`` next to a JSONL
index. ``snapshot()`` records the pre-mutation state of a file and
``undo()`` reverses the most recent recorded mutation.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

from rich.console import Console

console = Console()

# Maximum number of checkpoint entries kept in the index.
_MAX_ENTRIES = 50

# Size ceiling for backing up a directory tree: a node_modules-scale delete
# would otherwise copy gigabytes into the checkpoint store on every call.
# Oversized trees are recorded with no backup, so undo reports the gap
# instead of pretending the delete is reversible.
_MAX_DIR_BACKUP_BYTES = 256 * 1024 * 1024

_INDEX_NAME = "index.jsonl"

# Maps a checkpoint action to the tool name shown in undo messages.
_TOOL_BY_ACTION = {
    "overwrite": "write_file",
    "create": "write_file",
    "delete": "delete_file",
    "delete_dir": "delete_directory",
}


def _remove_backup(cp_dir: Path, name: str) -> None:
    """Drop one backup payload, file (``.bak``) or tree (``.bakdir``)."""
    victim = cp_dir / name
    try:
        if name.endswith(".bakdir"):
            shutil.rmtree(victim)
        else:
            os.remove(victim)
    except OSError:
        pass


def _tree_size(path: str) -> int:
    """Total bytes of every file under *path*; unreadable files count 0."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _checkpoint_dir() -> Path:
    """Directory holding backup files and the JSONL index."""
    # Lazy import so loading this module never triggers config cycles.
    from dashscope.acli.config import WORKSPACE_DIR

    return Path(WORKSPACE_DIR) / "checkpoints"


def _read_entries(cp_dir: Path) -> list[dict]:
    """Load index entries, skipping blank or malformed lines."""
    index = cp_dir / _INDEX_NAME
    if not index.is_file():
        return []
    entries: list[dict] = []
    with open(index, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _write_entries(cp_dir: Path, entries: list[dict]) -> None:
    """Rewrite the index atomically (temp file + rename)."""
    index = cp_dir / _INDEX_NAME
    tmp = index.with_name(_INDEX_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, index)


def snapshot(path: str, action: str) -> None:
    """Record a pre-mutation checkpoint for *path*.

    *action* is one of ``"overwrite"``, ``"create"`` or ``"delete"``.
    Never raises: a checkpoint failure must not break the write tool.
    """
    try:
        _snapshot(path, action)
    except Exception:
        # Checkpointing is best-effort; skip silently on any error.
        pass


def _snapshot(path: str, action: str) -> None:
    """Internal snapshot implementation (may raise)."""
    abs_path = os.path.abspath(path)
    cp_dir = _checkpoint_dir()
    cp_dir.mkdir(parents=True, exist_ok=True)
    entry_id = uuid.uuid4().hex
    backup = None
    note = None
    if os.path.isfile(abs_path):
        backup = entry_id + ".bak"
        shutil.copy2(abs_path, cp_dir / backup)
    elif os.path.isdir(abs_path):
        if _tree_size(abs_path) <= _MAX_DIR_BACKUP_BYTES:
            backup = entry_id + ".bakdir"
            shutil.copytree(abs_path, cp_dir / backup)
        else:
            note = "directory too large to back up"
    entry = {
        "id": entry_id,
        "path": abs_path,
        "backup": backup,
        "action": action,
        "ts": time.time(),
    }
    if note:
        entry["note"] = note
    entries = _read_entries(cp_dir)
    entries.append(entry)
    overflow = len(entries) - _MAX_ENTRIES
    dropped = entries[:overflow] if overflow > 0 else []
    _write_entries(cp_dir, entries[-_MAX_ENTRIES:])
    for old in dropped:
        name = old.get("backup")
        if not name:
            continue
        _remove_backup(cp_dir, name)


def undo() -> str:
    """Undo the most recent checkpointed file mutation.

    Returns a human-readable result string; never raises.
    """
    try:
        return _undo()
    except Exception as e:
        return f"Error: undo failed - {e}"


def _restore_backup(
    cp_dir: Path,
    backup: str,
    path: str,
    tool_name: str,
) -> str:
    """Restore one backup payload (file or tree) to *path*."""
    src = cp_dir / backup
    if backup.endswith(".bakdir"):
        if not src.is_dir():
            return f"Error: backup directory missing for {path}"
        shutil.copytree(src, path)
        shutil.rmtree(src)
        return f"Undid {tool_name}: restored {path}"
    if not src.is_file():
        return f"Error: backup file missing for {path}"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy2(src, path)
    try:
        os.remove(src)
    except OSError:
        pass
    return f"Undid {tool_name}: restored {path}"


def _undo() -> str:
    """Internal undo implementation (may raise)."""
    cp_dir = _checkpoint_dir()
    entries = _read_entries(cp_dir)
    if not entries:
        return "Nothing to undo"
    entry = entries.pop()
    _write_entries(cp_dir, entries)

    path = entry.get("path") or ""
    action = entry.get("action") or ""
    backup = entry.get("backup")
    tool_name = _TOOL_BY_ACTION.get(action, action or "unknown")

    if backup:
        return _restore_backup(cp_dir, backup, path, tool_name)

    if action == "create":
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return f"Undid {tool_name}: removed created {path}"

    return f"Error: no backup recorded for {path}"


def handle_undo_command() -> None:
    """CLI-facing wrapper: undo the last change and print the result."""
    console.print(undo())
