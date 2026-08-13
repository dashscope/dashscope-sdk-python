# -*- coding: utf-8 -*-
"""Argument validation and coercion helpers for tool execution."""

from __future__ import annotations

import sys
import types
from typing import Union, get_type_hints

from dashscope.acli.tools.registry import ToolDefinition


def parse_string_annotations(func) -> dict:
    """Extract base types from string annotations like 'int | None'.

    Fallback for Python <3.10 where get_type_hints() can't evaluate
    PEP 604 union syntax at runtime.
    """
    annotations = getattr(func, "__annotations__", {})
    hints = {}
    _TYPE_MAP = {"int": int, "float": float, "bool": bool, "str": str}
    for key, ann in annotations.items():
        if isinstance(ann, str):
            # Strip Optional / Union: "int | None" → "int"
            for part in ann.split("|"):
                part = part.strip()
                if part and part != "None":
                    t = _TYPE_MAP.get(part)
                    if t:
                        hints[key] = t
                    break
        elif isinstance(ann, type):
            hints[key] = ann
    return hints


def coerce_types(func, arguments: dict) -> dict:
    """Coerce string arguments to expected types based on function hints."""
    try:
        hints = get_type_hints(func, globalns=getattr(func, "__globals__", {}))
    except Exception:
        # Python <3.10: get_type_hints fails on `int | None` syntax from
        # `from __future__ import annotations`. Fall back to parsing raw
        # string annotations.
        hints = parse_string_annotations(func)

    coerced = {}
    for key, value in arguments.items():
        expected = hints.get(key)
        if expected is None or not isinstance(value, str):
            coerced[key] = value
            continue

        # Unwrap Optional[X] / X | None
        actual_type = expected
        origin = getattr(expected, "__origin__", None)
        is_union = origin is Union
        if sys.version_info >= (3, 10):
            is_union = is_union or isinstance(expected, types.UnionType)
        if is_union:
            args = getattr(expected, "__args__", ())
            actual_type = next(
                (a for a in args if a is not type(None)),
                expected,
            )

        # Convert string to target type. Some LLMs (DashScope qwen, in
        # particular) emit numeric args as decimal-looking strings even when
        # the schema says integer — fall back through float() so "120.0"
        # and "120" both land as 120 rather than getting passed through as
        # str and blowing up at the call site (asyncio.wait_for(timeout=str)
        # was the most recent victim).
        if actual_type is int:
            try:
                coerced[key] = int(value)
            except ValueError:
                try:
                    coerced[key] = int(float(value))
                except (ValueError, TypeError):
                    coerced[key] = value
        elif actual_type is float:
            try:
                coerced[key] = float(value)
            except ValueError:
                coerced[key] = value
        elif actual_type is bool:
            coerced[key] = value.lower() in ("true", "1", "yes")
        else:
            coerced[key] = value

    return coerced


def missing_required_args(
    tool_def: ToolDefinition,
    arguments: dict,
) -> list[str]:
    """Return the list of required-by-schema arg names that are missing,
    explicitly None, or — for string params — an empty/whitespace-only
    value. Models sometimes emit ``""`` when they mean "not given"; we
    surface that as a missing-arg error so the user doesn't get prompted
    to confirm a no-op call."""
    if not tool_def.parameters:
        return []
    required = tool_def.parameters.get("required", [])
    properties = tool_def.parameters.get("properties", {})
    missing = []
    for name in required:
        val = arguments.get(name)
        if val is None:
            missing.append(name)
            continue
        prop_type = properties.get(name, {}).get("type")
        if prop_type == "string" and isinstance(val, str) and not val.strip():
            missing.append(name)
    return missing
