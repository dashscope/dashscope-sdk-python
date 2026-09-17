# -*- coding: utf-8 -*-
"""Context compression upgrades for long conversations.

This module replaces the single message-count threshold with a token-budget
aware strategy:

* Estimate token usage from message text.
* Trigger compression at a soft threshold (50% of context window) and force it
  at a hard safety threshold (85%).
* Preserve the most recent messages so the agent does not lose the current
  task context.
* Truncate or summarize old tool outputs instead of keeping them verbatim.
"""

from __future__ import annotations

import json
from typing import Protocol

from dashscope.acli.config import DEFAULT_CONTEXT_WINDOW
from dashscope.acli.utils import message_text_for_compress, truncate_text

MIN_COMPRESS_MESSAGES = 4  # Never compress chats smaller than this
SOFT_THRESHOLD_RATIO = 0.50
HARD_THRESHOLD_RATIO = 0.85

# Keep the last N messages untouched during compression.  Recent user
# instruction + assistant tool calls + results are usually essential.
PRESERVE_RECENT_MESSAGES = 4

# Maximum characters to feed into the summarization prompt.  A very long
# conversation can itself exceed the window, so we cap the dump.
MAX_COMPRESS_DUMP_CHARS = 30000

# Old tool results beyond this are truncated in place by
# shrink_old_tool_messages before each LLM call.
OLD_TOOL_MESSAGE_MAX_CHARS = 2000

# Tools whose output is fully determined by their arguments and free to
# re-fetch, so a truncated result can tell the model how to read it again.
# run_command is deliberately excluded — re-running a shell command may have
# side effects — and so are web_search/image_search, which are paid calls
# whose results drift.  A "restore" hint for any of those is a false promise.
RESTORABLE_TOOLS = frozenset({"read_file", "search_files", "list_directory"})


def _tool_call_arguments(messages: list[dict]) -> dict[str, object]:
    """Map tool_call_id -> arguments, read off the assistant messages.

    A tool message carries a result but not the arguments that produced it;
    those live on the preceding assistant message's ``tool_calls``.
    """
    index: dict[str, object] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            call_id = tc.get("id")
            args = (tc.get("function") or {}).get("arguments")
            if call_id and args is not None:
                index[call_id] = args
    return index


def restore_note(name: str | None, arguments: object) -> str:
    """Marker suffix saying how to get the omitted content back.

    Empty for tools that are not RESTORABLE_TOOLS, so truncation never
    advertises a recovery path that would be a side effect or a paid call.
    """
    if not name or name not in RESTORABLE_TOOLS or not arguments:
        return ""
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return f" — re-run {name} with {arguments} to re-read"


def shrink_old_tool_messages(
    messages: list[dict],
    keep_recent: int = PRESERVE_RECENT_MESSAGES,
    max_chars: int = OLD_TOOL_MESSAGE_MAX_CHARS,
) -> int:
    """Truncate large tool results outside the recent tail, in place.

    Cheaper than full compression (no LLM call): a session full of stale
    read_file/run_command outputs otherwise re-sends them on every call.
    Results from RESTORABLE_TOOLS keep a note recording the original call, so
    the model re-reads instead of reasoning over a hole in its context.
    Idempotent; returns the number of messages shrunk.
    """
    from dashscope.acli.utils.text import (
        has_omitted_marker,
        truncate_head_tail,
    )

    shrunk = 0
    arguments_by_id: dict[str, object] | None = None
    boundary = max(len(messages) - keep_recent, 0)
    for i in range(boundary):
        m = messages[i]
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if (
            not isinstance(content, str)
            or len(content) <= max_chars
            or has_omitted_marker(content)  # already truncated — idempotent
        ):
            continue
        if arguments_by_id is None:
            arguments_by_id = _tool_call_arguments(messages)
        call_id = m.get("tool_call_id") or m.get("tool_use_id") or ""
        m["content"] = truncate_head_tail(
            content,
            max_chars,
            note=restore_note(m.get("name"), arguments_by_id.get(call_id)),
        )
        shrunk += 1
    return shrunk


class _ChatCallable(Protocol):
    async def __call__(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> object:
        ...


def _is_cjk(ch: str) -> bool:
    """Rough CJK detector: ideographs, kana, hangul and fullwidth forms."""
    return (
        "一" <= ch <= "鿿"  # CJK Unified Ideographs
        or "㐀" <= ch <= "䶿"  # Extension A
        or "豈" <= ch <= "﫿"  # Compatibility Ideographs
        or "぀" <= ch <= "ヿ"  # Hiragana + Katakana
        or "가" <= ch <= "힯"  # Hangul Syllables
        or "＀" <= ch <= "￯"  # Fullwidth Forms
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate.

    Counts CJK characters separately from ASCII: CJK text is ~1 token per
    character while ASCII is ~1 token per ~4 characters.  Other non-ASCII
    (accents, emoji, …) sits in between at ~1 token per ~2 characters.  This
    is intentionally conservative (slightly overestimates) for safety.
    """
    if not text:
        return 0
    cjk = ascii_ = other = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_ += 1
        elif _is_cjk(ch):
            cjk += 1
        else:
            other += 1
    return max(cjk + (other + 1) // 2 + (ascii_ + 3) // 4, 1)


def _estimate_text(msg: dict) -> str:
    """Full, untruncated text of a message for token estimation.

    Unlike message_text_for_compress (which truncates tool outputs for the
    summarization prompt), estimation must reflect the real payload size —
    a single huge tool result is exactly the overflow case we guard against.
    """
    content = msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    if msg.get("tool_calls"):
        extras = []
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            extras.append(f"{fn.get('name', '')}({args})")
        content = " ".join([content, *extras])
    return content


def estimate_message_tokens(messages: list[dict]) -> int:
    """Sum token estimates for all messages, including overhead per message."""
    total = 0
    for msg in messages:
        text = _estimate_text(msg)
        total += estimate_tokens(text) + 4  # small overhead for role/name keys
    return total


def _prepare_compress_dump(messages: list[dict]) -> str:
    """Build a cleaned, length-capped dump for the summarization prompt."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        text = message_text_for_compress(msg)
        # Old tool outputs are truncated before summarization; the summary
        # still captures the gist while the dump stays bounded.
        if role == "tool" and len(text) > 2000:
            text = truncate_text(text, 2000)
        lines.append(f"[{role}]: {text}")
    dump = "\n".join(lines)
    if len(dump) > MAX_COMPRESS_DUMP_CHARS:
        dump = truncate_text(dump, MAX_COMPRESS_DUMP_CHARS)
    return dump


async def _summarize(messages: list[dict], chat: _ChatCallable) -> str:
    """Ask the model to summarize the provided messages."""
    compress_msgs = [
        {
            "role": "system",
            "content": (
                "Compress the conversation below into a brief summary. "
                "Keep key decisions, code changes, file paths, user "
                "preferences, and tool results. Output only the summary; "
                "do not explain."
            ),
        },
        {"role": "user", "content": _prepare_compress_dump(messages)},
    ]
    resp = await chat(compress_msgs, tools=[])
    return resp.content if hasattr(resp, "content") else str(resp)


def should_compress(
    messages: list[dict],
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    extra_tokens: int = 0,
) -> tuple[bool, str]:
    """Return (should_compress, reason).

    Compression is triggered when the estimated tokens cross the soft
    threshold (50% of context window) or forced at the hard threshold (85%).
    A small minimum message count guards against compressing tiny chats;
    beyond it the token budget alone decides, so a few very large messages
    (e.g. huge tool outputs) still compress before overflowing the window.
    """

    if len(messages) < MIN_COMPRESS_MESSAGES:
        return False, ""

    estimated = estimate_message_tokens(messages) + extra_tokens
    soft = int(context_window * SOFT_THRESHOLD_RATIO)
    hard = int(context_window * HARD_THRESHOLD_RATIO)

    if estimated >= hard:
        return True, f"token hard threshold ({estimated} >= {hard})"
    if estimated >= soft:
        return True, f"token soft threshold ({estimated} >= {soft})"
    return False, ""


def _advance_to_safe_boundary(messages: list[dict], split: int) -> int:
    """Move ``split`` forward until the kept tail starts on a safe boundary.

    Safe means a user message, or an assistant message without tool_calls
    left unanswered inside the kept range.  A tail starting with ``tool``
    messages leaves them orphaned (their assistant tool_calls message was
    cut away), which the chat API rejects with a 400.
    """
    while split < len(messages):
        msg = messages[split]
        if msg.get("role") == "tool":
            split += 1
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            answered = {
                m.get("tool_call_id")
                for m in messages[split + 1 :]
                if m.get("role") == "tool"
            }
            pending = [
                tc for tc in msg["tool_calls"] if tc.get("id") not in answered
            ]
            if pending:
                split += 1
                continue
        break
    return split


def preserve_recent_messages(
    messages: list[dict],
    keep: int = PRESERVE_RECENT_MESSAGES,
):
    """Split messages into (older_compressible, recent_preserved).

    The split point is advanced past orphaned tool results / dangling
    tool_calls so the preserved tail never triggers an API 400.
    """
    if len(messages) <= keep:
        return [], messages
    split = _advance_to_safe_boundary(messages, len(messages) - keep)
    return messages[:split], messages[split:]


async def compress_messages(
    messages: list[dict],
    chat: _ChatCallable,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    keep_recent: int = PRESERVE_RECENT_MESSAGES,
    extra_tokens: int = 0,
) -> list[dict] | None:
    """Compress older messages while preserving the most recent ones.

    Returns a new message list with a summary of older messages followed by
    the preserved recent messages.  Returns ``None`` only when compression is
    not needed; if the summarizer itself fails, an omission marker takes the
    summary's place so overflow protection still kicks in.
    """
    should, _reason = should_compress(messages, context_window, extra_tokens)
    if not should:
        return None

    older, recent = preserve_recent_messages(messages, keep_recent)
    if not older:
        return None

    try:
        summary = await _summarize(older, chat)
        content = (
            "Summary of earlier conversation (auto-compressed, "
            "background context only):\n"
            f"{summary}"
        )
    except Exception:
        content = (
            f"Omitted {len(older)} earlier messages "
            "(auto-compress failed; keeping only recent "
            "messages to bound context)."
        )
    return [{"role": "user", "content": content}, *recent]


async def safety_compress_if_needed(
    messages: list[dict],
    chat: _ChatCallable,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    extra_tokens: int = 0,
) -> list[dict] | None:
    """Gateway-style safety compression right before an API call.

    If messages (plus the system-prompt/tools overhead in ``extra_tokens``)
    are near the hard threshold, compress older messages aggressively so the
    upcoming request does not exceed the model window.
    """
    estimated = estimate_message_tokens(messages) + extra_tokens
    hard = int(context_window * HARD_THRESHOLD_RATIO)
    if estimated < hard:
        return None
    return await compress_messages(
        messages,
        chat,
        context_window=context_window,
        keep_recent=max(2, PRESERVE_RECENT_MESSAGES - 1),
        extra_tokens=extra_tokens,
    )
