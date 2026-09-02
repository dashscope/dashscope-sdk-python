# -*- coding: utf-8 -*-
"""Image encoding helpers shared by the @image.png REPL flow and the
understand_image vision tool."""

from __future__ import annotations

import base64
from pathlib import Path

from dashscope.acli.cli.constants import _AT_IMAGE_MAX_BYTES, _IMAGE_MIME


def image_to_data_url(path: str | Path) -> str:
    """Read an image file and return a ``data:<mime>;base64,<b64>`` URL.

    Raises ``ValueError`` for unsupported extensions or files exceeding
    the 5 MB cap, and ``OSError`` (from ``Path.read_bytes``) when the
    file cannot be read.
    """
    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    mime = _IMAGE_MIME.get(suffix)
    if mime is None:
        raise ValueError(f"Unsupported image extension: {suffix}")
    data = p.read_bytes()
    if len(data) > _AT_IMAGE_MAX_BYTES:
        raise ValueError(
            f"Image too large ({len(data) // 1024} KB > "
            f"{_AT_IMAGE_MAX_BYTES // 1024} KB), ignored",
        )
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"
