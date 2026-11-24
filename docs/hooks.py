# -*- coding: utf-8 -*-
"""MkDocs hooks: build-time substitution for version and release notes.

Pages opt in with placeholders; values are read from their source of
truth at build time so docs never drift:

- ``{{ sdk_version }}``   <- dashscope/version.py
- ``{{ release_notes }}`` <- published GitHub Releases (latest-release
  table linking to each release page), so publishing a release is all it
  takes to refresh the docs.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

from mkdocs.structure.toc import AnchorLink, TableOfContents

# Reference pages listing dozens of API symbols: even a section+symbol
# outline floods the sidebar (80-110 entries), so their "On this page"
# nav keeps only the h2 section titles. Applies to both locales.
# release-notes.md joins them for a consistent section-only outline.
_SECTION_ONLY_TOC_PAGES = {
    "reference/agentstudio.md",
    "reference/legacy.md",
    "reference/resources.md",
    "release-notes.md",
}

_RELEASES_URL = (
    "https://api.github.com/repos/dashscope/dashscope-sdk-python"
    "/releases?per_page=20"
)
_RELEASES_PAGE = "https://github.com/dashscope/dashscope-sdk-python/releases"
_RELEASES_SHOWN = 5
# Fetch once per build process; the i18n plugin runs two locale passes.
_RELEASES_CACHE = None

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


def _fetch_releases():
    """Published releases, newest first; None when the API is unreachable.

    Drafts are excluded so the skeleton rendered by release.yml never
    leaks into the docs before a maintainer publishes it. GITHUB_TOKEN
    (set in CI) only lifts the anonymous rate limit; the data is public.
    """
    global _RELEASES_CACHE
    if _RELEASES_CACHE is not None:
        return _RELEASES_CACHE or None
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "dashscope-sdk-docs-build",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        request = urllib.request.Request(_RELEASES_URL, headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            releases = json.load(response)
        releases = [r for r in releases if not r.get("draft")]
        releases.sort(key=lambda r: r.get("published_at") or "", reverse=True)
        _RELEASES_CACHE = releases
        return releases or None
    except Exception:
        _RELEASES_CACHE = []
        return None


def _render_release_notes(lang: str) -> str:
    releases = _fetch_releases()
    zh = lang == "zh"
    if not releases:
        return (
            "构建时未能加载 GitHub Releases，完整变更日志请参阅"
            f"[GitHub Releases]({_RELEASES_PAGE})。"
            if zh
            else "Release notes could not be loaded at build time; see "
            f"[GitHub Releases]({_RELEASES_PAGE}) for the full changelog."
        )
    releases = releases[:_RELEASES_SHOWN]
    parts = ["## 近期发布" if zh else "## Recent releases", ""]
    parts.append("| Tag | 说明 |" if zh else "| Tag | Notes |")
    parts.append("| --- | --- |")
    for release in releases:
        tag = release.get("tag_name") or ""
        if release.get("prerelease"):
            tag += "（预发布）" if zh else " (pre-release)"
        date = (release.get("published_at") or "")[:10]
        label = "发布说明" if zh else "Release notes"
        parts.append(
            f"| {tag} | {date} · [{label}]({release.get('html_url')}) |",
        )
    return "\n".join(parts)


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
    markdown = markdown.replace(
        "{{ sdk_version }}",
        config.extra.get("sdk_version", "unknown"),
    )
    if "{{ release_notes }}" in markdown:
        # i18n folder structure prefixes src_uri with the locale (en/ zh/)
        lang = "zh" if page.file.src_uri.startswith("zh/") else "en"
        markdown = markdown.replace(
            "{{ release_notes }}",
            _render_release_notes(lang),
        )
    return markdown


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
