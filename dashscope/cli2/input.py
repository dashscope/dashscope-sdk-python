# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Command-line input parsing for the dashscope CLI.

Three spellings are accepted wherever a payload option takes text:

``value``      literal text
``@path``      read the file (``~`` expanded)
``-``          read stdin to EOF

The ``@`` form is what keeps a long ``--messages`` payload or a tool
schema out of shell quoting hell, and ``-`` is what makes the CLI
composable::

    cat messages.jsonl | dashscope chat create -m qwen-plus --messages -

Message payloads additionally accept JSONL, because that is how people
already store conversation dumps: one message object per line, with no
surrounding array.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashscope.cli2.errors import CliError, usage_error

STDIN_TOKEN = "-"


def stdin_is_piped() -> bool:
    """True when stdin is a pipe or file rather than a terminal."""
    try:
        return not sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def read_stdin() -> str:
    try:
        return sys.stdin.read()
    except OSError as exc:
        raise usage_error(f"could not read stdin: {exc}") from exc


def read_text_arg(raw: Optional[str], *, option_name: str) -> Optional[str]:
    """Resolve a payload option to its text.

    ``None`` passes through so an unset option stays unset; the caller
    decides what a missing value means.
    """
    if raw is None:
        return None
    if raw == STDIN_TOKEN:
        text = read_stdin()
        if not text.strip():
            raise usage_error(
                f"{option_name}: no input received on stdin "
                f"(pass a value, an @file, or pipe something in)",
            )
        return text
    if raw.startswith("@"):
        path_text = raw[1:].strip()
        if not path_text:
            raise usage_error(f"{option_name}: '@' must be followed by a path")
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise usage_error(f"{option_name}: file not found: {path}")
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise usage_error(
                f"{option_name}: cannot read {path}: {exc}",
            ) from exc
    return raw


def parse_json_arg(raw: Optional[str], *, option_name: str) -> Any:
    """``read_text_arg`` plus a JSON decode, with an actionable error."""
    text = read_text_arg(raw, option_name=option_name)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        source = (
            "stdin"
            if raw == STDIN_TOKEN
            else (
                raw[1:] if raw and raw.startswith("@") else "the given value"
            )
        )
        raise usage_error(
            f"{option_name}: invalid JSON in {source}: {exc.msg} "
            f"(line {exc.lineno} column {exc.colno})",
        ) from exc


def parse_json_object_arg(
    raw: Optional[str],
    *,
    option_name: str,
) -> Optional[Dict[str, Any]]:
    """Like :func:`parse_json_arg` but requires a JSON object."""
    value = parse_json_arg(raw, option_name=option_name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise usage_error(
            f"{option_name}: expected a JSON object, got "
            f"{type(value).__name__}",
        )
    return value


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(("{", "["))


def parse_messages_arg(raw: Optional[str]) -> List[Dict[str, Any]]:
    """Parse a conversation payload into a list of message objects.

    Accepts a JSON array of messages, a JSON object carrying a
    ``messages`` key (the OpenAI request shape people paste from docs),
    or JSONL with one message object per line.
    """
    text = read_text_arg(raw, option_name="--messages")
    if text is None:
        return []
    return messages_from_text(text)


def messages_from_text(text: str) -> List[Dict[str, Any]]:
    """The payload half of :func:`parse_messages_arg`, for text in hand.

    The JSON and JSONL shapes are told apart by trying the stricter one
    first: JSONL also opens with a brace, so the first character cannot
    decide it, and a whole-text parse of JSONL fails with "Extra data".
    """
    stripped = text.strip()
    if not stripped:
        raise usage_error("--messages: payload is empty")

    if _looks_like_json(stripped):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return _coerce_messages(value)

    messages: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise usage_error(
                f"--messages: invalid JSON on line {lineno}: {exc.msg}",
            ) from exc
        messages.extend(_coerce_messages(value))
    if not messages:
        raise usage_error("--messages: no messages found in payload")
    return messages


def _coerce_messages(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        inner = value.get("messages")
        if isinstance(inner, list):
            return _require_message_objects(inner)
        return _require_message_objects([value])
    if isinstance(value, list):
        return _require_message_objects(value)
    raise usage_error(
        "--messages: expected a message object, a list of them, or "
        f'{{"messages": [...]}}, got {type(value).__name__}',
    )


def _require_message_objects(items: List[Any]) -> List[Dict[str, Any]]:
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise usage_error(
                f"--messages: entry {index} is not a JSON object "
                f"(got {type(item).__name__})",
            )
    return list(items)


def messages_from_prompt(prompt: str, system: Optional[str] = None):
    """Build a message list from ``--prompt`` (+ optional ``--system``)."""
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def read_prompt_or_messages(
    *,
    prompt: Optional[str],
    messages: Optional[str],
    system: Optional[str] = None,
    allow_stdin_prompt: bool = True,
) -> Any:
    """Resolve the conversation for a generate-style command.

    Precedence: ``--messages`` > ``--prompt`` > piped stdin. Stdin is
    treated as a message payload when it parses as JSON and as prompt
    text otherwise, so both ``cat msgs.json | dashscope chat create -``
    and ``echo hi | dashscope chat create -m qwen-plus`` work.
    """
    if messages is not None:
        return parse_messages_arg(messages)
    if prompt is not None:
        text = read_text_arg(prompt, option_name="--prompt")
        return messages_from_prompt(text or "", system)
    if allow_stdin_prompt and stdin_is_piped():
        text = read_stdin().strip()
        if text:
            if _looks_like_json(text):
                try:
                    return messages_from_text(text)
                except CliError:
                    # Looks like JSON but is not a message payload: most
                    # often a prompt that happens to start with a brace,
                    # so fall back to text rather than failing.
                    pass
            return messages_from_prompt(text, system)
    raise usage_error(
        "no input given: pass --prompt, --messages, or pipe text on stdin",
    )
