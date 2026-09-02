# -*- coding: utf-8 -*-
"""Append-only session event log (roadmap P2-Phase2, first increment).

An event sidecar that records session lifecycle (and, in later
increments, turn) events as an append-only JSONL stream, without
touching the existing ``history.json`` read/write path. This lays the
groundwork for future projection / fork / resume built on an immutable
event source.

Storage layout (per topic)::

    .acli/session/<topic>/events.jsonl

Each line is a self-describing event::

    {"v": 1, "seq": 3, "ts": "<iso>", "type": "topic/created",
     "data": {...}}

Writes are best-effort: an event-log failure never breaks the session,
mirroring the checkpoint/history philosophy.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# Bump when the on-disk event schema changes in a breaking way.
SCHEMA_VERSION = 1

EVENTS_FILENAME = "events.jsonl"


def latest_snapshot_messages(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the message list from the newest ``messages/snapshot``.

    Scans *entries* (as produced by :meth:`SessionEventLog.read_raw`)
    backwards; returns an empty list when no snapshot is present.
    """
    for entry in reversed(entries):
        if entry.get("type") != "messages/snapshot":
            continue
        data = entry.get("data") or {}
        msgs = data.get("messages")
        if isinstance(msgs, list):
            return msgs
    return []


class SessionEventLog:
    """Append-only JSONL event log bound to one file.

    Events are appended, never edited in place — with one exception:
    :meth:`compact_snapshots` atomically rewrites the file to drop
    stale full-history snapshots (all other events are preserved).
    ``seq`` is a 1-based monotonically increasing position derived
    from the current line count, so it stays correct even if another
    process appends.
    """

    def __init__(self, events_file: Path):
        self._file = Path(events_file)

    @property
    def path(self) -> Path:
        return self._file

    def append(self, event_type: str, data: dict | None = None) -> None:
        """Append one event. Never raises (best-effort sidecar)."""
        try:
            self._append(event_type, data or {})
        except Exception:
            # The event log is advisory; never break the caller.
            pass

    def _append(self, event_type: str, data: dict) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "v": SCHEMA_VERSION,
            "seq": self._next_seq(),
            "ts": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _next_seq(self) -> int:
        """1 + number of non-blank lines currently in the log."""
        if not self._file.exists():
            return 1
        count = 0
        with open(self._file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count + 1

    def read(
        self,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return events in order, optionally filtered by type.

        Malformed or wrong-schema lines are skipped silently.
        """
        if not self._file.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(self._file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("v") != SCHEMA_VERSION:
                    continue
                if event_type is not None and entry.get("type") != event_type:
                    continue
                events.append(entry)
        return events

    def read_raw(self) -> list[dict[str, Any]]:
        """Return every well-formed entry, any schema version.

        Unlike :meth:`read` (which filters to the current schema), this
        yields all parseable entries — used when projecting large events
        such as ``messages/snapshot``.
        """
        if not self._file.exists():
            return []
        entries: list[dict[str, Any]] = []
        with open(self._file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
        return entries

    def compact_snapshots(
        self,
        keep: int = 2,
        snapshot_type: str = "messages/snapshot",
    ) -> bool:
        """Drop all but the newest ``keep`` snapshot events, atomically.

        Snapshots carry a full copy of the message list, so retaining
        every one grows the log quadratically in history length.
        Compaction rewrites the file with all non-snapshot events plus
        the newest ``keep`` snapshots, renumbering ``seq`` to stay
        line-ordered. The rewrite goes through a tmp file and
        ``os.replace``, so a crash mid-compaction leaves the original
        log intact.

        Best-effort: returns True only when the file was rewritten.
        """
        try:
            entries = self.read_raw()
        except OSError:
            return False
        snapshot_idx = [
            i for i, e in enumerate(entries) if e.get("type") == snapshot_type
        ]
        if len(snapshot_idx) <= keep:
            return False
        drop = set(snapshot_idx[: len(snapshot_idx) - keep])
        kept = [e for i, e in enumerate(entries) if i not in drop]
        tmp = self._file.with_suffix(self._file.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for seq, entry in enumerate(kept, start=1):
                    entry["seq"] = seq
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            os.replace(tmp, self._file)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        return True

    def __len__(self) -> int:
        return len(self.read())
