# -*- coding: utf-8 -*-
"""MkDocs hooks: surface the SDK version from dashscope/version.py in pages.

Pages opt in with the ``{{ sdk_version }}`` placeholder; the value is read
from the source of truth at build time so docs never drift from the package.
"""
import re
from pathlib import Path


def _read_version(docs_dir: str) -> str:
    source = Path(docs_dir).parent / "dashscope" / "version.py"
    match = re.search(
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        source.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else "unknown"


def on_config(config):
    config.extra["sdk_version"] = _read_version(config.docs_dir)
    return config


def on_page_markdown(markdown, *, page, config, files):
    return markdown.replace(
        "{{ sdk_version }}",
        config.extra.get("sdk_version", "unknown"),
    )
