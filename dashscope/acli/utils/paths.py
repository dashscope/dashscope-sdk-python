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


def _temp_write_roots() -> tuple[str, ...]:
    """Realpath'd system temp directories a write is allowed to land in.

    ``run_command`` can already write here — the shell is not path-guarded —
    ``delete_file`` reaches anywhere behind its own prompt, and the opt-in OS
    sandbox confines writes to "the workspace and temp areas". A cwd-only rule
    for write_file was therefore stricter than every layer around it without
    protecting anything, and it cost more than a refused call: models reach
    for /tmp for scratch files by instinct, so the retry put the scratch file
    in the user's repo instead, which is the worse outcome.

    ``tempfile.gettempdir()`` is what makes this portable — it honours
    TMPDIR/TEMP/TMP, which on macOS is a per-user /var/folders path rather than
    /tmp. /tmp is listed separately because that is what a model actually
    writes, and realpath is what ties it to /private/tmp there.
    """
    import tempfile

    candidates = [tempfile.gettempdir()]
    if os.name != "nt":
        candidates += ["/tmp", "/private/tmp"]
    roots: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = os.path.realpath(candidate)
        except OSError:
            continue
        if resolved not in roots and os.path.isdir(resolved):
            roots.append(resolved)
    return tuple(roots)


def validate_write_path(path: str) -> str:
    """Validate a path for write operations (stricter than read).

    In addition to sensitive-path blocking, confines writes to the working
    directory and the system temp dir.
    """
    path = validate_path(path)
    # Resolve both sides with realpath: abspath doesn't follow symlinks, so
    # a symlink inside cwd pointing outside would otherwise pass the check.
    abs_path = os.path.realpath(path)
    roots = (os.path.realpath("."),) + _temp_write_roots()
    inside = any(
        abs_path == root or abs_path.startswith(root + os.sep)
        for root in roots
    )
    if inside:
        return path
    raise ValueError(
        f"Write path '{path}' is outside the writable roots: the working "
        "directory and the system temp dir (path traversal guard)",
    )


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
