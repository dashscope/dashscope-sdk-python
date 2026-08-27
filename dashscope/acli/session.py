# -*- coding: utf-8 -*-
"""Session manager — handles multi-topic session persistence.

Each topic gets its own directory under WORKSPACE_DIR/session/<topic>/:
  - history.json: message history
  - input-history.txt: command input history
  - meta.json: metadata (created, last_accessed, message_count)
  - scene.md: persistent scene memory (topic-scoped notes injected into
    the system prompt every turn)
  - events.jsonl: append-only event sidecar (lifecycle; see
    session_events.SessionEventLog)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dashscope.acli.config import WORKSPACE_DIR
from dashscope.acli.session_events import (
    EVENTS_FILENAME,
    SessionEventLog,
    latest_snapshot_messages,
)

DEFAULT_TOPIC = "default"
SCENE_FILENAME = "scene.md"

# Snapshot pruning (event-log completeness Phase 2): snapshots carry a
# full message copy, so unbounded retention grows the log quadratically.
# Compact once the log holds more than _SNAPSHOT_COMPACT_AT snapshots,
# keeping the newest _SNAPSHOT_KEEP.
_SNAPSHOT_KEEP = 2
_SNAPSHOT_COMPACT_AT = 8


@dataclass
class SessionMeta:
    """Metadata for a session topic."""

    topic: str
    created: str
    last_accessed: str
    message_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SessionMeta:
        return cls(
            topic=data.get("topic", DEFAULT_TOPIC),
            created=data.get("created", datetime.now().isoformat()),
            last_accessed=data.get(
                "last_accessed",
                datetime.now().isoformat(),
            ),
            message_count=data.get("message_count", 0),
        )


class SessionManager:
    """Manages multi-topic session persistence."""

    def __init__(self, workspace_dir: Path = WORKSPACE_DIR):
        self.workspace_dir = workspace_dir
        self.session_dir = workspace_dir / "session"
        self.current_topic: str = DEFAULT_TOPIC
        self._ensure_session_dir()

    def _ensure_session_dir(self) -> None:
        """Ensure the session directory and default topic exist."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        default_dir = self._topic_dir(DEFAULT_TOPIC)
        default_dir.mkdir(parents=True, exist_ok=True)
        # Ensure default topic has meta.json
        default_meta = self._meta_file(DEFAULT_TOPIC)
        if not default_meta.exists():
            self._save_meta(
                DEFAULT_TOPIC,
                SessionMeta(
                    topic=DEFAULT_TOPIC,
                    created=datetime.now().isoformat(),
                    last_accessed=datetime.now().isoformat(),
                ),
            )

    def _topic_dir(self, topic: str) -> Path:
        """Get the directory for a specific topic."""
        return self.session_dir / topic

    def _safe_topic_dir(self, topic: str) -> Path | None:
        """Get the topic directory, or None if the name is unsafe."""
        if (
            not topic
            or "/" in topic
            or "\\" in topic
            or ".." in topic
            or Path(topic).is_absolute()
        ):
            return None
        base = self.session_dir.resolve()
        resolved = (base / topic).resolve()
        if resolved != base and base not in resolved.parents:
            return None
        return self.session_dir / topic

    def _history_file(self, topic: str) -> Path:
        """Get the history.json path for a topic."""
        return self._topic_dir(topic) / "history.json"

    def _input_history_file(self, topic: str) -> Path:
        """Get the input-history path for a topic."""
        return self._topic_dir(topic) / "input-history.txt"

    def _meta_file(self, topic: str) -> Path:
        """Get the meta.json path for a topic."""
        return self._topic_dir(topic) / "meta.json"

    def _events_file(self, topic: str) -> Path:
        """Get the events.jsonl sidecar path for a topic."""
        return self._topic_dir(topic) / EVENTS_FILENAME

    def _scene_file(self, topic: str) -> Path:
        """Get the scene.md path for a topic."""
        return self._topic_dir(topic) / SCENE_FILENAME

    def get_scene(self, topic: str | None = None) -> str:
        """Return the scene memory text for a topic (default: current).

        Returns an empty string when the file is missing or unreadable.
        """
        path = self._scene_file(topic or self.current_topic)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def set_scene(self, text: str, topic: str | None = None) -> bool:
        """Replace the scene memory for a topic (default: current).

        An empty/whitespace-only *text* clears the scene file. Returns
        False when the topic name is unsafe or the write fails.
        """
        topic = topic or self.current_topic
        topic_dir = self._safe_topic_dir(topic)
        if topic_dir is None:
            return False
        path = self._scene_file(topic)
        content = text.strip()
        try:
            if not content:
                if path.exists():
                    path.unlink()
            else:
                topic_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(content + "\n", encoding="utf-8")
        except OSError:
            return False
        self.event_log(topic).append(
            "scene/updated",
            {"topic": topic, "chars": len(content)},
        )
        return True

    def append_scene(self, text: str, topic: str | None = None) -> bool:
        """Append a note line to the scene memory of a topic."""
        topic = topic or self.current_topic
        note = text.strip()
        if not note:
            return False
        existing = self.get_scene(topic)
        merged = f"{existing}\n{note}" if existing else note
        return self.set_scene(merged, topic)

    def event_log(self, topic: str | None = None) -> SessionEventLog:
        """Return the append-only event log for a topic (default: current).

        The sidecar is advisory: callers may append lifecycle/turn events
        without affecting the history.json read/write path.
        """
        return SessionEventLog(
            self._events_file(topic or self.current_topic),
        )

    def record_turn_event(
        self,
        user_text: str,
        assistant_text: str,
        tools_used: list[str] | None = None,
        outcome: str = "",
        topic: str | None = None,
    ) -> None:
        """Append a ``turn/end`` event to the topic's event sidecar.

        Records a compact summary of one turn (truncated user/assistant
        text, tools used, outcome). Best-effort: never raises, so event
        recording can't break the agent loop. This is a step toward the
        session-as-event-log direction (later: projection / fork /
        resume built on the event stream).
        """
        try:
            topic = topic or self.current_topic
            self.event_log(topic).append(
                "turn/end",
                {
                    "topic": topic,
                    "user": (user_text or "")[:200],
                    "assistant": (assistant_text or "")[:200],
                    "tools": list(tools_used or []),
                    "outcome": outcome,
                },
            )
        except Exception:
            pass

    def record_messages_snapshot(
        self,
        messages: list[dict[str, Any]],
        topic: str | None = None,
    ) -> None:
        """Append a full-fidelity ``messages/snapshot`` event.

        Stores the complete current message list so the session can later
        be reconstructed (resume/fork) from the event log alone. Old
        snapshots are pruned once more than ``_SNAPSHOT_COMPACT_AT``
        accumulate, keeping the newest ``_SNAPSHOT_KEEP``. Best-effort:
        never raises.
        """
        try:
            topic = topic or self.current_topic
            log = self.event_log(topic)
            log.append(
                "messages/snapshot",
                {"topic": topic, "messages": list(messages or [])},
            )
            if len(log.read("messages/snapshot")) > _SNAPSHOT_COMPACT_AT:
                log.compact_snapshots(keep=_SNAPSHOT_KEEP)
        except Exception:
            pass

    def resume_from_events(
        self,
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rebuild the message list from the latest snapshot event.

        Returns an empty list when the topic has no snapshot. This is the
        projection side of the event-sourced session: combined with the
        append-only log it enables resume / fork / crash recovery.
        """
        try:
            events = self.event_log(topic).read_raw()
        except Exception:
            return []
        return latest_snapshot_messages(events)

    def fork_topic(self, src: str, dst: str) -> bool:
        """Create topic *dst* seeded with a copy of *src*'s event log.

        The forked topic resumes from the same state as the source. Both
        names must be safe and the destination must not already exist.
        Returns False otherwise.
        """
        src_dir = self._safe_topic_dir(src)
        dst_dir = self._safe_topic_dir(dst)
        if src_dir is None or dst_dir is None:
            return False
        if not src_dir.exists() or dst_dir.exists():
            return False
        try:
            import shutil

            dst_dir.mkdir(parents=True)
            self._save_meta(
                dst,
                SessionMeta(
                    topic=dst,
                    created=datetime.now().isoformat(),
                    last_accessed=datetime.now().isoformat(),
                ),
            )
            src_events = self._events_file(src)
            if src_events.exists():
                shutil.copy2(src_events, self._events_file(dst))
            # Pre-snapshot sessions keep their messages only in
            # history.json; copy it too so the fork restores them.
            src_history = self._history_file(src)
            if src_history.exists():
                shutil.copy2(src_history, self._history_file(dst))
        except OSError:
            return False
        self.event_log(dst).append(
            "topic/forked",
            {"from": src, "to": dst},
        )
        return True

    def list_topics(self) -> list[SessionMeta]:
        """List all available topics with metadata."""
        topics = []
        if not self.session_dir.exists():
            return topics

        for topic_dir in self.session_dir.iterdir():
            if not topic_dir.is_dir():
                continue
            meta_file = self._meta_file(topic_dir.name)
            if meta_file.exists():
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    topics.append(SessionMeta.from_dict(data))
                except (json.JSONDecodeError, OSError):
                    # Fallback: create meta from directory
                    topics.append(
                        SessionMeta(
                            topic=topic_dir.name,
                            created=datetime.now().isoformat(),
                            last_accessed=datetime.now().isoformat(),
                        ),
                    )
            else:
                # No meta file — derive timestamps from directory mtime to
                # avoid non-deterministic sort order caused by auto-touching.
                try:
                    mtime = topic_dir.stat().st_mtime
                    from datetime import timezone as _tz

                    ts = (
                        datetime.fromtimestamp(mtime, tz=_tz.utc)
                        .replace(tzinfo=None)
                        .isoformat()
                    )
                except OSError:
                    ts = datetime.now().isoformat()
                topics.append(
                    SessionMeta(
                        topic=topic_dir.name,
                        created=ts,
                        last_accessed=ts,
                    ),
                )

        # Sort: default first, then by last_accessed descending.
        # Python sort is stable, so sort by secondary key first.
        topics.sort(key=lambda m: m.last_accessed, reverse=True)
        topics.sort(key=lambda m: m.topic != DEFAULT_TOPIC)
        return topics

    def get_current_topic(self) -> str:
        """Get the current active topic."""
        return self.current_topic

    def set_current_topic(self, topic: str) -> bool:
        """Set the current active topic. Returns True if successful."""
        topic_dir = self._safe_topic_dir(topic)
        if topic_dir is None or not topic_dir.exists():
            return False
        self.current_topic = topic
        self._update_last_accessed(topic)
        self.event_log(topic).append("topic/switched", {"topic": topic})
        return True

    def create_topic(self, topic: str) -> bool:
        """Create a new topic directory. Returns True if created."""
        topic_dir = self._safe_topic_dir(topic)
        if topic_dir is None or topic_dir.exists():
            return False
        topic_dir.mkdir(parents=True)
        meta = SessionMeta(
            topic=topic,
            created=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
        )
        self._save_meta(topic, meta)
        self.current_topic = topic
        self.event_log(topic).append("topic/created", {"topic": topic})
        return True

    def rename_topic(self, old_name: str, new_name: str) -> bool:
        """Rename a topic. Returns True if successful."""
        old_dir = self._safe_topic_dir(old_name)
        new_dir = self._safe_topic_dir(new_name)
        if old_dir is None or new_dir is None:
            return False

        if not old_dir.exists() or new_dir.exists():
            return False

        try:
            # Read meta before rename so we can update it after.
            meta_data: dict | None = None
            old_meta_file = old_dir / "meta.json"
            if old_meta_file.exists():
                try:
                    meta_data = json.loads(
                        old_meta_file.read_text(encoding="utf-8"),
                    )
                except (json.JSONDecodeError, OSError):
                    meta_data = None

            old_dir.rename(new_dir)

            # Update meta with new topic name.
            if meta_data is not None:
                meta_data["topic"] = new_name
                new_meta_file = new_dir / "meta.json"
                try:
                    new_meta_file.write_text(
                        json.dumps(meta_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except OSError:
                    # meta mismatch is tolerable; directory is already
                    # renamed
                    pass

            if self.current_topic == old_name:
                self.current_topic = new_name
            self.event_log(new_name).append(
                "topic/renamed",
                {"from": old_name, "to": new_name},
            )
            return True
        except OSError:
            return False

    def delete_topic(self, topic: str) -> bool:
        """Delete a topic and all its files. Returns True if successful."""
        if topic == DEFAULT_TOPIC:
            return False  # Cannot delete default

        topic_dir = self._safe_topic_dir(topic)
        if topic_dir is None or not topic_dir.exists():
            return False

        try:
            import shutil

            shutil.rmtree(topic_dir)
            if self.current_topic == topic:
                self.current_topic = DEFAULT_TOPIC
            return True
        except OSError:
            return False

    def get_history_path(self, topic: str | None = None) -> Path:
        """Get the history.json path for a topic (default: current)."""
        topic = topic or self.current_topic
        return self._history_file(topic)

    def get_input_history_path(self, topic: str | None = None) -> Path:
        """Get the input-history path for a topic (default: current)."""
        topic = topic or self.current_topic
        return self._input_history_file(topic)

    def load_messages(
        self,
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load stored chat messages for a topic (default: current).

        Returns an empty list when the history file is missing,
        unreadable, or not a JSON list.
        """
        path = self._history_file(topic or self.current_topic)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def update_message_count(
        self,
        count: int,
        topic: str | None = None,
    ) -> None:
        """Update the message count for a topic."""
        topic = topic or self.current_topic
        meta_file = self._meta_file(topic)
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                data["message_count"] = count
                data["last_accessed"] = datetime.now().isoformat()
                meta_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except (json.JSONDecodeError, OSError):
                pass

    def _save_meta(self, topic: str, meta: SessionMeta) -> None:
        """Save metadata for a topic."""
        meta_file = self._meta_file(topic)
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _update_last_accessed(self, topic: str) -> None:
        """Update the last_accessed timestamp for a topic."""
        meta_file = self._meta_file(topic)
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                data["last_accessed"] = datetime.now().isoformat()
                meta_file.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except (json.JSONDecodeError, OSError):
                pass

    def should_prompt_archive(
        self,
        topic: str | None = None,
        threshold: int = 100,
    ) -> bool:
        """Check if a topic has enough messages to suggest archiving."""
        topic = topic or self.current_topic
        if topic != DEFAULT_TOPIC:
            return False  # Only prompt for default topic

        meta_file = self._meta_file(topic)
        if not meta_file.exists():
            return False

        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            return data.get("message_count", 0) >= threshold
        except (json.JSONDecodeError, OSError):
            return False


# Global instance
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global _manager
    _manager = manager
