# -*- coding: utf-8 -*-
"""Minimal template rendering helpers."""

from __future__ import annotations

import json
from typing import Any


def render_brace_template(template: str, variables: dict[str, str]) -> str:
    """Simple {var} substitution."""
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def render_mustache_template(template: str, params: dict[str, Any]) -> str:
    """Mustache-style {{var}} substitution.

    Values are JSON-encoded so they drop in cleanly inside a JSON body
    without manual quoting (a string "hello" becomes \"hello\", a number
    10 becomes 10, an array stays as array).
    """
    out = template
    for k, v in params.items():
        encoded = json.dumps(v, ensure_ascii=False)
        out = out.replace("{{" + k + "}}", encoded)
        # Also support {{ k }} with whitespace
        out = out.replace("{{ " + k + " }}", encoded)
    return out
