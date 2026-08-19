# -*- coding: utf-8 -*-
"""Filesystem path validation helpers used by tools and the CLI."""

from __future__ import annotations

import os
from pathlib import Path

# Directory / file names considered sensitive. Any path containing these
# as a component is blocked for read, write, and delete operations.
SENSITIVE_NAMES = frozenset(
    {
        # SSH / GPG / TLS
        ".ssh",
        ".gnupg",
        ".gpg",
        ".pki",
        ".ssl",
        ".certs",
        # Cloud credentials
        ".aws",
        ".azure",
        ".gcloud",
        ".kube",
        # Secrets stores
        ".netrc",
        ".pgpass",
        ".my.cnf",
        "keychain.db",
        # VCS secrets
        ".git-credentials",
        # NPM / Node
        ".npmrc",
        ".env",
        # Python
        ".pypirc",
        # OS-level
        "shadow",
        "master.key",
        "passwd",
    },
)


def validate_path(path: str) -> str:
    """Resolve and validate a filesystem path.

    Returns the expanded path. Raises ValueError for sensitive paths.
    """
    path = os.path.expanduser(path)
    # realpath resolves symlinks, so a symlink inside the workspace that
    # points at a sensitive dir (e.g. ~/.ssh) is still caught.
    abs_path = os.path.realpath(path)
    parts = set(os.path.normpath(abs_path).split(os.sep))
    hit = parts & SENSITIVE_NAMES
    if hit:
        raise ValueError(
            f"Path '{path}' contains sensitive component {hit}, "
            "operation denied (safety guard)",
        )
    return path


def validate_write_path(path: str) -> str:
    """Validate a path for write operations (stricter than read).

    In addition to sensitive-path blocking, prevents writes outside cwd.
    """
    path = validate_path(path)
    # Resolve both sides with realpath: abspath doesn't follow symlinks, so
    # a symlink inside cwd pointing outside would otherwise pass the check.
    abs_path = os.path.realpath(path)
    cwd = os.path.realpath(".")
    if not abs_path.startswith(cwd + os.sep) and abs_path != cwd:
        raise ValueError(
            f"Write path '{path}' is outside the current working "
            "directory (path traversal guard)",
        )
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically.

    Writes to a temp file in the SAME directory, flush+fsync, then
    os.replace — a crash mid-write leaves the old file intact instead of
    truncating it.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
