# -*- coding: utf-8 -*-
"""System clipboard helpers.

Used by the TUI to write mouse selections directly to the system clipboard:
the app captures the mouse (alt-screen), so the terminal's own Cmd+C only
sees the visible screen — multi-screen selections would be truncated there.
"""

from __future__ import annotations

import shutil
import subprocess

# (tool, argv) tried in order; first one present wins.
_CLIPBOARD_TOOLS: tuple[tuple[str, list[str]], ...] = (
    ("pbcopy", ["pbcopy"]),
    ("xclip", ["xclip", "-selection", "clipboard"]),
    ("wl-copy", ["wl-copy"]),
    ("clip", ["clip"]),
)


def copy_to_clipboard(text: str) -> str | None:
    """Write ``text`` to the system clipboard.

    Returns the tool name used, or ``None`` when no clipboard tool is
    available or the write failed. Never raises.
    """
    for name, cmd in _CLIPBOARD_TOOLS:
        if not shutil.which(name):
            continue
        try:
            subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            continue
        return name
    return None
