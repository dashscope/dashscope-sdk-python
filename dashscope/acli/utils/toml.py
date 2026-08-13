# -*- coding: utf-8 -*-
"""Small TOML loader shim using tomllib (3.11+) or tomli."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_toml(path: Path) -> dict[str, Any] | None:
    """Load a TOML file, returning None on missing/invalid files."""
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return None


def loads_toml(text: str) -> dict[str, Any] | None:
    """Parse TOML from a string, returning None on invalid input."""
    try:
        return tomllib.load(BytesIO(text.encode("utf-8")))
    except Exception:
        return None


_TOML_BASIC_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def toml_str(value: Any) -> str:
    """Serialize *value* as a quoted TOML basic string.

    Escapes backslash, double quote, and control characters per the TOML
    spec so an arbitrary value (API key, path, prompt) cannot corrupt the
    config file it is written into.
    """
    out: list[str] = []
    for ch in str(value):
        esc = _TOML_BASIC_ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def parse_value(line: str) -> str:
    """Extract the value part from a 'key = value' line, stripping quotes."""
    _, _, value = line.partition("=")
    return value.strip().strip('"').strip("'")


def parse_value_raw(line: str) -> str:
    """Like parse_value but doesn't strip surrounding quotes — used when the
    right-hand side is structured (e.g. a JSON / TOML array literal)."""
    _, _, value = line.partition("=")
    return value.strip()


def parse_toml_inline_table(raw: str) -> dict | None:
    """Parse a TOML inline table: {key = "value", key2 = 42, key3 = true}."""
    raw = raw.strip()
    if not raw.startswith("{") or not raw.endswith("}"):
        return None
    inner = raw[1:-1].strip()
    if not inner:
        return {}
    result = {}
    for pair in inner.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, _, val = pair.partition("=")
        key = key.strip().strip('"').strip("'")
        val = val.strip()
        if not key:
            continue
        # Try bool / int / float / string
        if val.lower() == "true":
            result[key] = True
        elif val.lower() == "false":
            result[key] = False
        else:
            val_unquoted = val.strip('"').strip("'")
            try:
                result[key] = int(val_unquoted)
            except ValueError:
                try:
                    result[key] = float(val_unquoted)
                except ValueError:
                    result[key] = val_unquoted
    return result
