# -*- coding: utf-8 -*-
"""MkDocs hooks: surface the SDK version from dashscope/version.py in pages.

Pages opt in with the ``{{ sdk_version }}`` placeholder; the value is read
from the source of truth at build time so docs never drift from the package.
"""
import re
from pathlib import Path

from mkdocs.structure.toc import AnchorLink, TableOfContents

# Reference pages listing dozens of API symbols: even a section+symbol
# outline floods the sidebar (80-110 entries), so their "On this page"
# nav keeps only the h2 section titles. Applies to both locales.
_SECTION_ONLY_TOC_PAGES = {
    "reference/agentstudio.md",
    "reference/legacy.md",
    "reference/resources.md",
}


def _read_version(docs_dir: str) -> str:
    source = Path(docs_dir).parent / "dashscope" / "version.py"
    match = re.search(
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        source.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else "unknown"


def on_config(config):
    config.extra["sdk_version"] = _read_version(config.docs_dir)
    # mkdocstrings inserts its stylesheet on every per-locale on_config
    # pass, so the i18n rebuild emits the link twice; keep first occurrence.
    config.extra_css = list(dict.fromkeys(config.extra_css))
    return config


def on_page_content(html, *, page, config, files):
    # i18n folder structure prefixes src_uri with the locale (en/ or zh/)
    parts = page.file.src_uri.split("/")
    src = (
        "/".join(parts[1:]) if parts[0] in ("en", "zh") else page.file.src_uri
    )
    if src in _SECTION_ONLY_TOC_PAGES:
        # toc tokens nest sections under the leading h1; Material replaces
        # the nav with that h1's children, so prune from the same place.
        items = list(page.toc)
        if items and items[0].level == 1:
            items = items[0].children
        sections = [
            AnchorLink(item.title, item.id, item.level)
            for item in items
            if item.level == 2
        ]
        if sections:
            sections[0].active = True
        page.toc = TableOfContents(sections)
    return html


def on_page_markdown(markdown, *, page, config, files):
    return markdown.replace(
        "{{ sdk_version }}",
        config.extra.get("sdk_version", "unknown"),
    )
