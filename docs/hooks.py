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

# The theme's static 404.html template renders once per i18n pass and the
# last (zh) pass overwrites site/404.html, shipping a mixed-language shell.
# Stash the en pass render context and reuse it for every later pass.
_EN_404_CONTEXT = None


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


def _snapshot_404_context(context, config):
    """Snapshot a 404 render context with explicit language destinations."""
    # Plain-dict snapshot: Config/Theme are UserDict/MutableMapping whose
    # internal storage copy.copy would share, and i18n mutates it per pass.
    snapshot = dict(context)
    cfg = dict(config)
    cfg["theme"] = dict(config.theme)
    cfg["extra"] = dict(config.extra)
    # Static theme templates have no page context. Without this override, the
    # i18n plugin leaves the last content page's alternates behind.
    base_url = context.get("base_url", "/")
    cfg["extra"]["alternate"] = [
        {"link": base_url, "lang": "en", "name": "English"},
        {
            "link": base_url + "zh/404/",
            "lang": "zh",
            "name": "简体中文",
        },
    ]
    snapshot["config"] = cfg
    return snapshot


def on_template_context(context, *, template_name, config):
    global _EN_404_CONTEXT
    if template_name != "404.html":
        return context
    if config.theme.get("language", "en") == "en":
        _EN_404_CONTEXT = _snapshot_404_context(context, config)
        return _EN_404_CONTEXT
    if _EN_404_CONTEXT is None:
        # mkdocs serve can rebuild only the zh pass; static destinations must
        # not depend on an earlier en render having populated the cache.
        return _snapshot_404_context(context, config)
    # Fresh copy per pass: later plugin hooks may mutate the returned dict.
    return dict(_EN_404_CONTEXT)


def on_post_build(*, config):
    """Keep the localized 404 page out of the sitemap.

    mkdocs only special-cases the root ``404.html``; the zh 404 lives at
    ``zh/404/`` and would otherwise be listed as a regular page.
    """
    import gzip

    sitemap = Path(config.site_dir) / "sitemap.xml"
    if not sitemap.exists():
        return
    text = sitemap.read_text(encoding="utf-8")
    pruned = text
    for block in re.findall(r"<url>.*?</url>", text, re.S):
        if re.search(r"<loc>[^<]*/404/</loc>", block):
            pruned = pruned.replace(block, "", 1)
    if pruned == text:
        return
    sitemap.write_text(pruned, encoding="utf-8")
    compressed = sitemap.with_suffix(".xml.gz")
    if compressed.exists():
        compressed.write_bytes(gzip.compress(pruned.encode("utf-8")))
