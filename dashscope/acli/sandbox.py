# -*- coding: utf-8 -*-
"""Optional OS-level sandbox for shell command execution (roadmap P0-L3).

Defense-in-depth layered on top of the permission engine and the
dangerous-command blocklist. When enabled *and* a sandbox backend is
present, ``run_command`` executes commands inside an OS sandbox that
confines filesystem writes to the current workspace.

Design constraints (see the roadmap):

* **Opt-in** — disabled by default (``sandbox = false`` in config). Local
  users already get the permission prompts + blocklist; the sandbox is an
  extra layer for ``auto_approve`` / untrusted-skill scenarios.
* **Graceful degradation** — if no backend is detected the command runs
  normally under the existing permission layer. The sandbox never blocks
  startup and never raises out of ``run_command``.
* **Best-effort boundary** — this raises the cost of a destructive or
  runaway command; it is *not* a hardened security boundary. Treat it as
  confinement, not isolation.

Backends:

* macOS: Seatbelt via ``sandbox-exec`` (built into macOS). The profile
  keeps the default-allow policy but denies filesystem writes outside the
  workspace and temp areas, so process execution, reads, and networking are
  unaffected and normal dev commands keep working.
* Linux: bubblewrap (``bwrap``) when installed. bwrap confines by
  remapping the filesystem, so it is more restrictive; commands that write
  outside the workspace/temp will fail.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Optional

# Backend identifiers.
SEATBELT = "seatbelt"
BWRAP = "bwrap"


def detect_backend() -> Optional[str]:
    """Return the available sandbox backend name, or ``None``.

    macOS prefers ``sandbox-exec`` (Seatbelt); Linux prefers ``bwrap``.
    Windows has no supported backend. Detection only checks for the tool's
    presence on PATH — it never starts a sandbox.
    """
    if sys.platform == "darwin":
        return SEATBELT if shutil.which("sandbox-exec") else None
    if sys.platform.startswith("linux"):
        return BWRAP if shutil.which("bwrap") else None
    return None


def available() -> bool:
    """True when a sandbox backend is detected on this machine."""
    return detect_backend() is not None


# Optional override (mainly for tests / explicit CLI control). ``None``
# means "read the setting from config".
_enabled_override: Optional[bool] = None


def set_enabled(value: Optional[bool]) -> None:
    """Override sandbox enablement; pass ``None`` to use the config value."""
    global _enabled_override
    _enabled_override = value


def is_enabled() -> bool:
    """Whether sandboxing is enabled (config-driven, overridable).

    Reads ``sandbox`` from the loaded config on each call so config
    changes take effect; degrades to ``False`` on any error so the
    sandbox can never break command execution.
    """
    if _enabled_override is not None:
        return bool(_enabled_override)
    try:
        from dashscope.acli.config import Config

        return bool(Config.load().sandbox)
    except Exception:
        return False


def seatbelt_profile(cwd: str) -> str:
    """Build a macOS Seatbelt profile confining writes to the workspace.

    The base policy stays default-allow; only filesystem writes are
    denied, then re-allowed for the workspace and standard temp/cache
    locations. Reads, process execution, and networking are untouched so
    ordinary development commands keep working.
    """
    safe_cwd = cwd.replace('"', '\\"')
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny file-write*)",
            f'(allow file-write* (subpath "{safe_cwd}"))',
            '(allow file-write* (subpath "/private/tmp"))',
            '(allow file-write* (subpath "/tmp"))',
            '(allow file-write* (subpath "/private/var/folders"))',
            '(allow file-write* (literal "/dev/null"))',
        ],
    )


def bwrap_argv(command: str, cwd: str) -> list[str]:
    """Build a ``bwrap`` argv that runs *command* with a read-only root.

    The whole filesystem is bound read-only, then the workspace and a
    fresh ``/tmp`` are made writable. Commands that need to write elsewhere
    (e.g. package caches) will fail — this is the intended confinement.
    """
    return [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        cwd,
        cwd,
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        cwd,
        "/bin/sh",
        "-c",
        command,
    ]


def build_argv(
    command: str,
    cwd: str,
    backend: Optional[str] = None,
) -> Optional[list[str]]:
    """Return an argv to run *command* inside the sandbox, or ``None``.

    ``None`` means "no sandbox available — run the command normally".
    ``backend`` may be passed explicitly (mainly for tests); otherwise it
    is detected.
    """
    backend = backend or detect_backend()
    if backend == SEATBELT:
        return [
            "sandbox-exec",
            "-p",
            seatbelt_profile(cwd),
            "/bin/sh",
            "-c",
            command,
        ]
    if backend == BWRAP:
        return bwrap_argv(command, cwd)
    return None


def is_sandboxed_path(path: str, cwd: str) -> bool:
    """True when *path* is inside the writable workspace or temp areas.

    Informational helper (e.g. for messages); not a security check.
    """
    try:
        abs_path = os.path.abspath(path)
    except (OSError, ValueError):
        return False
    for base in (cwd, "/tmp", "/private/tmp"):
        if abs_path == base or abs_path.startswith(base + os.sep):
            return True
    return False
