# -*- coding: utf-8 -*-
"""Web search tool — DuckDuckGo-based web search.

Uses the `duckduckgo-search` package (DDGS) as a free, keyless search backend.
Complements the browser tools (scrape_web, scrape_web_html) which handle
page content extraction. Together they form a complete web acquisition
pipeline:
  web_search  → find relevant URLs
  scrape_web  → extract content from those URLs

Security:
- Query length limits to prevent abuse
- Result count limits to avoid overwhelming context windows
- Lazy import of duckduckgo_search to keep startup fast
"""

from __future__ import annotations

import json
from typing import Any

from dashscope.acli.tools.registry import PermissionLevel, tool

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_DDGS_HINT = (
    "Error: duckduckgo-search is not installed. "
    "Run `pip install duckduckgo-search`"
)

_MAX_QUERY_LEN = 500
_DEFAULT_MAX_RESULTS = 10
_ABS_MAX_RESULTS = 30


def _get_ddgs_class():
    """Lazily import DDGS to avoid hard dependency at module load time."""
    try:
        from duckduckgo_search import DDGS

        return DDGS
    except ImportError as exc:
        raise ImportError(_DDGS_HINT) from exc


def _validate_query(query: str) -> str:
    """Sanitise and validate a search query."""
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty")
    if len(query) > _MAX_QUERY_LEN:
        raise ValueError(
            f"Search query too long ({len(query)} chars); "
            f"max {_MAX_QUERY_LEN} chars",
        )
    return query


def _validate_max_results(max_results: Any) -> int:
    """Coerce max_results to a safe integer."""
    try:
        n = int(max_results)
    except (TypeError, ValueError):
        n = _DEFAULT_MAX_RESULTS
    return max(1, min(n, _ABS_MAX_RESULTS))


def _validate_region(region: str) -> str:
    """Normalise region code (e.g. 'cn-zh', 'us-en', 'wt-wt' for global).

    Accepts both full DDGS region codes and common short country codes
    (e.g. 'cn' -> 'cn-zh', 'us' -> 'us-en') so the LLM does not crash on
    abbreviated region values.
    """
    if not region or not region.strip():
        return "wt-wt"
    region = region.strip().lower()
    if region == "wt-wt":
        return region

    # Map common short country codes to DDGS region-language pairs.
    short_map = {
        "cn": "cn-zh",
        "us": "us-en",
        "uk": "uk-en",
        "jp": "jp-jp",
        "kr": "kr-kr",
        "de": "de-de",
        "fr": "fr-fr",
        "ru": "ru-ru",
        "br": "br-pt",
        "in": "in-en",
        "tw": "tw-tzh",
        "hk": "hk-tzh",
    }
    if region in short_map:
        return short_map[region]

    # Basic format check: xx-xx or wt-wt
    if len(region) < 4 or len(region) > 10 or "-" not in region:
        raise ValueError(
            f"Invalid region code: '{region}'; "
            "examples: 'cn-zh', 'us-en', 'wt-wt'",
        )
    return region


def _format_results(results: list[dict], query: str) -> str:
    """Format search results into a readable string for the LLM."""
    if not results:
        return json.dumps(
            {"query": query, "results": [], "message": "No results found"},
            ensure_ascii=False,
        )

    formatted = []
    for i, r in enumerate(results, 1):
        entry = {
            "rank": i,
            "title": r.get("title", ""),
            "url": r.get("href", r.get("url", r.get("link", ""))),
            "snippet": r.get(
                "body",
                r.get("snippet", r.get("description", "")),
            ),
        }
        formatted.append(entry)

    return json.dumps(
        {"query": query, "results": formatted, "count": len(formatted)},
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# Tool: web_search
# ---------------------------------------------------------------------------


@tool(
    name="web_search",
    description=(
        "Search the web with DuckDuckGo. Returns a JSON list of titles, "
        "URLs, and snippets. Good for finding up-to-date info, docs, and "
        "tutorials. Use scrape_web afterwards to fetch full page content."
    ),
    permission=PermissionLevel.AUTO,
)
async def web_search(
    query: str,
    max_results: int = 10,
    region: str = "wt-wt",
) -> str:
    """Search the web using DuckDuckGo and return structured results.

    Args:
        query: search keywords, e.g. 'Python asyncio tutorial'
            or 'Rust ownership explained'
        max_results: number of results to return, default 10, max 30
        region: region code, e.g. 'cn-zh' (Chinese), 'us-en' (English),
            'wt-wt' (global, default)
    """
    query = _validate_query(query)
    max_results = _validate_max_results(max_results)
    region = _validate_region(region)

    DDGS = _get_ddgs_class()

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    region=region,
                    max_results=max_results,
                ),
            )
    except ImportError as e:
        return str(e)
    except Exception as e:
        return json.dumps(
            {
                "query": query,
                "error": f"Search failed: {type(e).__name__}: {e}",
                "results": [],
            },
            ensure_ascii=False,
        )

    return _format_results(results, query)
