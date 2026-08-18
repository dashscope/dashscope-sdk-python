# -*- coding: utf-8 -*-
"""Shared helpers for converting and inspecting chat messages."""
# pylint: disable=too-many-branches

from __future__ import annotations

import json

# Markers that indicate a compressed/auto-generated message carrying raw
# tool-call scaffolding.  Such text should not be stored as a user-facing
# history summary because it poisons memory recall.
_TOOL_GARBAGE_MARKERS = ("<tool>", "</tool>", "[tools:")

# Tools whose results carry a diff / multi-line summary worth showing in full
# to the user (and the UI layer will pretty-print). Everything else stays
# capped at 200 chars to keep the trail readable.
_FULL_DISPLAY_TOOLS = frozenset({"write_file"})


def message_text_for_compress(msg: dict) -> str:
    """Return a human-readable text representation of a message for
    compression.

    Strips raw tool-call scaffolding so the summarization prompt stays clean
    and the resulting summary does not contain XML/noise that pollutes memory.
    """
    role = msg.get("role", "?")
    content = msg.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )

    if role == "assistant" and "tool_calls" in msg:
        parts = []
        if content:
            parts.append(content)
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            parts.append(f"[calls tool {name}({args})]")
        return " ".join(parts)

    if role == "tool":
        name = msg.get("name", "tool")
        result = content
        # Truncate long tool results to keep the compression prompt small.
        if len(result) > 500:
            result = result[:500] + "..."
        return f"[tool {name} returned] {result}"

    return content


def is_tool_garbage(text: str) -> bool:
    """Return True when text contains raw tool-call scaffolding."""
    return any(marker in text for marker in _TOOL_GARBAGE_MARKERS)


def text_of(content) -> str:
    """Flatten a message's content to plain text. Strings pass through;
    OpenAI-style content-block lists are joined by their ``text`` parts
    (images/audio are skipped). Used for memory recall/store where the backend
    needs a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        return " ".join(p for p in parts if p)
    return ""


def normalize_for_model(messages: list[dict], model: str) -> list[dict]:
    """Drop multimodal content blocks the current model can't parse.

    Vision-capable models get the messages as-is (image_url blocks pass
    through). Text-only models would otherwise hit "Unexpected item type
    in content" from the backend the moment session history contains a
    prior turn with images — e.g., user sent ``@pic.png`` to qwen-vl-max,
    then switched to qwen-plus via /provider (text-only). We flatten
    list content to a single string per message: text blocks
    concatenated, image blocks replaced by a short placeholder so the
    model still knows an image was there. ``self.messages`` itself is
    untouched — switching back to a vision model restores the original
    blocks."""
    from dashscope.acli.config import is_audio_model, is_vision_model

    supports_vision = is_vision_model(model)
    supports_audio = is_audio_model(model)

    # Fully multimodal model: pass messages through unchanged.
    if not model or (supports_vision and supports_audio):
        return messages

    out = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list):
            out.append(m)
            continue

        # No multimodal support at all: flatten to a single string.
        if not supports_vision and not supports_audio:
            parts = []
            for blk in c:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type")
                if t == "text":
                    txt = blk.get("text", "")
                    if txt:
                        parts.append(txt)
                elif t == "image_url":
                    parts.append(
                        "[image omitted — model does not support images]",
                    )
                elif t == "input_audio":
                    parts.append(
                        "[audio omitted — model does not support audio]",
                    )
            new_m = dict(m)
            new_m["content"] = "\n".join(parts) if parts else ""
            out.append(new_m)
            continue

        # Partial support (e.g., vision-only or audio-only): keep list format
        # and replace unsupported blocks with placeholder text.
        new_blocks = []
        for blk in c:
            if not isinstance(blk, dict):
                continue
            t = blk.get("type")
            if t == "text":
                new_blocks.append(blk)
            elif t == "image_url":
                if supports_vision:
                    new_blocks.append(blk)
                else:
                    new_blocks.append(
                        {
                            "type": "text",
                            "text": "[image omitted — model does "
                            "not support images]",
                        },
                    )
            elif t == "input_audio":
                if supports_audio:
                    new_blocks.append(blk)
                else:
                    new_blocks.append(
                        {
                            "type": "text",
                            "text": "[audio omitted — model does "
                            "not support audio]",
                        },
                    )
        new_m = dict(m)
        new_m["content"] = new_blocks
        out.append(new_m)
    return out


def tool_result_for_display(tool_name: str, result: str) -> str:
    """Truncate tool results for the [trail] line. Tools producing diffs
    (write_file) pass through uncapped (cli.py's _stream_response then
    renders them with Syntax highlighting)."""
    from dashscope.acli.utils import truncate

    if tool_name in _FULL_DISPLAY_TOOLS:
        return result
    return truncate(result, 200)


# Tool results larger than this are truncated before entering conversation
# history: every LLM call re-sends the full history, so one huge read_file /
# run_command output inflates input tokens for the rest of the session.
MAX_TOOL_RESULT_HISTORY_CHARS = 8000


def tool_result_for_history(
    result: str,
    max_chars: int = MAX_TOOL_RESULT_HISTORY_CHARS,
) -> str:
    """Cap a tool result before it is stored in history (head + tail kept)."""
    from dashscope.acli.utils.text import truncate_head_tail

    return truncate_head_tail(result, max_chars)
