# -*- coding: utf-8 -*-
"""Keyword extraction helpers for local search (Chinese + English)."""

from __future__ import annotations

import re


def extract_keywords(text: str) -> set[str]:
    """Extract searchable keywords from free text.

    Returns a set of Chinese character substrings (2+ chars) and English
    words (3+ chars), plus individual Chinese characters.
    """
    keywords: set[str] = set()
    lowered = text.lower()

    # Chinese chars: 2+ length substrings
    cn_chars = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    keywords.update(cn_chars)

    # English words: 3+ chars
    en_words = re.findall(r"[a-zA-Z]{3,}", lowered)
    keywords.update(w.lower() for w in en_words)

    # Single Chinese chars for broader matching
    for char in lowered:
        if "\u4e00" <= char <= "\u9fff":
            keywords.add(char)

    return keywords


def expand_scoring_terms(keywords: set[str]) -> set[str]:
    """Expand keywords into scoring terms with CJK bigram overlap.

    extract_keywords emits only the full contiguous CJK run ("登录报错",
    "login error"), which never substring-matches "登录接口错误" ("login
    API error"). Adding overlapping bigrams ("登录" / "login", "报错" /
    "error") keeps Chinese retrieval working without single-char
    noise. Falls back to the raw set when nothing multi-char exists.
    """
    terms: set[str] = set()
    for kw in keywords:
        if len(kw) < 2:
            continue
        terms.add(kw)
        for i in range(len(kw) - 1):
            bigram = kw[i : i + 2]
            if all("一" <= ch <= "鿿" for ch in bigram):
                terms.add(bigram)
    return terms or keywords
