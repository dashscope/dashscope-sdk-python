# -*- coding: utf-8 -*-
"""Session manager — handles multi-topic session persistence.

Each topic gets its own directory under WORKSPACE_DIR/session/<topic>/:
  - history.json: message history
  - input-history.txt: command input history
  - meta.json: metadata (created, last_accessed, message_count)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dashscope.acli.config import WORKSPACE_DIR

DEFAULT_TOPIC = "default"


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
