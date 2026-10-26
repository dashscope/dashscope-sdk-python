# -*- coding: utf-8 -*-
"""Chat / completion / DashScope response → GenAI InputMessage /
OutputMessage conversion.

Tool-call ``arguments`` in GenAI ``ToolCall`` parts are sanitized before
export:

- Length: ``AGENTIC_RL_TOOL_CALL_ARGUMENTS_MAX_CHARS`` (default: 32 KiB
  per serialized arguments string).
- Masking: ``AGENTIC_RL_TOOL_CALL_ARGUMENTS_MASK``:
  - **Unset** → masking **on** by default (``deep_mask`` on
    JSON-decodable arguments).
  - **``false`` / ``0`` / ``no``** (and other non-truthy values) → masking
    **off** (plaintext export; truncation still applies).
  - **Truthy** ``true`` / ``1`` / ``yes`` / ``y`` / ``on`` → masking **on**
    (explicit; same behavior as unset).

Diagnostics: set ``AGENTIC_RL_DEBUG_LLM_TOOL_CALLS`` to ``1`` / ``true`` /
``yes`` / ``y`` / ``on`` to log per-message tool-call shapes and
per-``tool_calls`` item mapping (branch, argument types/lengths) without
dumping full payloads — for tracing ``"{}"`` / missing arguments issues.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dashscope.finetune.reinforcement.common.log import logger
from dashscope.finetune.reinforcement.common.utils import deep_mask
from dashscope.finetune.reinforcement.component.observability import genai

# pylint: disable=protected-access
_core = genai._core

# Default max length for serialized tool ``arguments`` written into GenAI
# ToolCall parts. Override with ``AGENTIC_RL_TOOL_CALL_ARGUMENTS_MAX_CHARS``
# (integer).
_DEFAULT_TOOL_CALL_ARGUMENTS_MAX_CHARS = 32 * 1024

_ENV_TOOL_ARGS_MAX = "AGENTIC_RL_TOOL_CALL_ARGUMENTS_MAX_CHARS"
_ENV_TOOL_ARGS_MASK = "AGENTIC_RL_TOOL_CALL_ARGUMENTS_MASK"
_ENV_DEBUG_LLM_TOOL_CALLS = "AGENTIC_RL_DEBUG_LLM_TOOL_CALLS"


def _debug_llm_tool_calls() -> bool:
    return os.environ.get(_ENV_DEBUG_LLM_TOOL_CALLS, "").strip().lower() in (
        "true",
        "1",
        "yes",
        "y",
        "on",
    )


def _env_truthy(name: str, *, default: bool = False) -> bool:
    """Match truthy set used by ``AGENTIC_RL_DEBUG_*`` / tool-module env
    flags."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "y", "on")


def _debug_llm_tool_calls_log_exc() -> None:
    """If diagnostic ``logger.info`` raises, record at DEBUG without affecting
    callers."""
    try:
        logger.debug(
            "[agentic_rl_debug_llm_tool_calls] diagnostic logging raised",
            exc_info=True,
        )
    except Exception:
        pass


def _tool_call_arguments_max_chars() -> int:
    raw = os.environ.get(_ENV_TOOL_ARGS_MAX, "").strip()
    if not raw:
        return _DEFAULT_TOOL_CALL_ARGUMENTS_MAX_CHARS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_TOOL_CALL_ARGUMENTS_MAX_CHARS


def _truncate_tool_arguments_string(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    suffix = "...[truncated]"
    head = max_chars - len(suffix)
    if head <= 0:
        # ``max_chars`` too small for the full suffix; avoid a chopped marker
        # string.
        return s[:max_chars]
    return s[:head] + suffix


def _coerce_tool_arguments_to_str(arguments: Any) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False, default=str)
    except Exception:
        return str(arguments)


def _sanitize_tool_call_arguments(arguments: Any) -> str:
    """Truncate + optional ``deep_mask`` for JSON-shaped arguments
    (best-effort)."""
    s = _coerce_tool_arguments_to_str(arguments)
    mask_on = _env_truthy(_ENV_TOOL_ARGS_MASK, default=True)
    if mask_on:
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            pass
        else:
            try:
                masked = deep_mask(parsed)
                s = json.dumps(masked, ensure_ascii=False, default=str)
            except Exception:
                logger.warning(
                    (
                        "agentic_rl: tool_call arguments masking failed; "
                        "exporting raw arguments"
                    ),
                    exc_info=True,
                )
    max_chars = _tool_call_arguments_max_chars()
    return _truncate_tool_arguments_string(s, max_chars)


def unwrap_openai_completion(  # pylint: disable=too-many-return-statements
    completion: Any,
) -> Any:
    """Best-effort unwrap OpenAI-compatible completion containers.

    Some clients return wrapper response objects (e.g. LegacyAPIResponse) where
    the parsed payload lives on `.parsed` or behind `.parse()`.
    """
    try:
        if completion is None:
            return None
        if isinstance(completion, Dict):
            return completion
        if getattr(completion, "choices", None) is not None:
            return completion
        parsed = getattr(completion, "parsed", None)
        if parsed is not None:
            return parsed
        parse_fn = getattr(completion, "parse", None)
        if callable(parse_fn):
            return parse_fn()
        return completion
    except Exception:
        return completion


def _shape_args_for_debug(arguments: Any) -> Dict[str, Any]:
    """Short, log-safe summary of tool-call payload before sanitize (no raw
    values)."""
    if arguments is None:
        return {"args_type": "None", "args_str_len": 0}
    if isinstance(arguments, str):
        return {"args_type": "str", "args_str_len": len(arguments)}
    if isinstance(arguments, (Dict, list)):
        return {
            "args_type": type(arguments).__name__,
            "args_json_len": len(json.dumps(arguments, default=str)),
        }
    return {
        "args_type": type(arguments).__name__,
        "args_repr_len": len(repr(arguments)),
    }


def _first_tool_call_keys(tc: Any) -> Any:
    if isinstance(tc, Dict):
        return sorted(tc.keys())
    out = []
    for k in ("function", "name", "args", "id", "type", "arguments"):
        if hasattr(tc, k):
            out.append(k)
    return out or type(tc).__name__


# pylint: disable=too-many-branches
def tool_calls_to_parts(
    tool_calls: Any,
) -> List[Any]:  # pylint: disable=too-many-branches
    """Convert common ``tool_calls`` shapes on OpenAI/DashScope messages to
    parts.

    Maps structures to ``ToolCall`` parts for the GenAI handler.
    """
    if not tool_calls or not _core.GENAI_AVAILABLE:
        if _debug_llm_tool_calls():
            try:
                logger.info(
                    (
                        "[agentic_rl_debug_llm_tool_calls] "
                        "tool_calls_to_parts:early_return "
                        "tool_calls_empty=%s genai_available=%s"
                    ),
                    not bool(tool_calls),
                    _core.GENAI_AVAILABLE,
                )
            except Exception:
                _debug_llm_tool_calls_log_exc()
        return []
    parts: List[Any] = []
    for i, tc in enumerate(tool_calls):
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, Dict):
            fn = tc.get("function")
        name = ""
        args: Any = "{}"
        tid = None
        branch = "nested_function"
        if fn is not None:
            # ``function`` may be an OpenAI SDK object or a plain ``dict``
            # (e.g. serialized ``messages`` history).  Never use
            # ``getattr(fn, "arguments", "{}")`` for dicts: ``getattr`` returns
            # the default literal ``"{}"``, which is truthy and prevents the
            # ``or fn.get("arguments", ...)`` fallback from running — real
            # arguments are lost.
            if isinstance(fn, Dict):
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
            else:
                name = getattr(fn, "name", "") or ""
                raw_args = getattr(fn, "arguments", None)
            args = "{}" if raw_args in (None, "") else raw_args
        else:
            branch = "top_level_lc"
            name = getattr(tc, "name", "") or (
                tc.get("name") if isinstance(tc, Dict) else ""
            )
            # LangChain ``ToolCall`` / LC-like dicts: payload lives in
            # top-level ``args``, not OpenAI's nested ``function.arguments``
            # string.
            if isinstance(tc, Dict):
                if tc.get("args") is not None:
                    args = tc["args"]
            else:
                lc_args = getattr(tc, "args", None)
                if lc_args is not None:
                    args = lc_args
        tid = (
            getattr(tc, "id", None)
            if not isinstance(tc, Dict)
            else tc.get("id")
        )
        args_str = _sanitize_tool_call_arguments(args)
        if _debug_llm_tool_calls():
            try:
                nested_fn_diag: Optional[Dict[str, Any]] = None
                if fn is not None:
                    raw_a = (
                        fn.get("arguments", "__sentinel__")
                        if isinstance(fn, Dict)
                        else getattr(fn, "arguments", "__sentinel__")
                    )
                    nested_fn_diag = {
                        "fn_arguments_absent": raw_a == "__sentinel__",
                        "fn_arguments_is_none": raw_a is None,
                        "fn_arguments_type": (
                            None
                            if raw_a in ("__sentinel__", None)
                            else type(raw_a).__name__
                        ),
                        "fn_arguments_str_len": (
                            len(raw_a) if isinstance(raw_a, str) else None
                        ),
                    }
                logger.info(
                    (
                        "[agentic_rl_debug_llm_tool_calls] "
                        "tool_calls_to_parts:map "
                        "idx=%s branch=%s tc_type=%s tc_keys=%s nested_fn=%s "
                        "pre_sanitize=%s post_sanitize_len=%s name_len=%s"
                    ),
                    i,
                    branch,
                    type(tc).__name__,
                    _first_tool_call_keys(tc),
                    nested_fn_diag,
                    _shape_args_for_debug(args),
                    len(args_str),
                    len(name or ""),
                )
            except Exception:
                _debug_llm_tool_calls_log_exc()
        parts.append(
            _core.ToolCall(
                name=name or "tool",
                arguments=args_str,
                id=tid,
            ),
        )
    return parts


def _extract_tool_calls_from_message_obj(m: Any) -> Any:
    tc = getattr(m, "tool_calls", None)
    if tc is not None:
        return tc
    ak = getattr(m, "additional_kwargs", None)
    if isinstance(ak, Dict):
        return ak.get("tool_calls")
    return None


def _should_include_tool_calls_in_input(
    raw_role: Any,
    tool_calls: Any,
) -> bool:
    if not tool_calls:
        return False
    if raw_role == "assistant":
        return True
    if raw_role is None or raw_role == "":
        return True
    return False


# pylint: disable-next=too-many-branches,too-many-statements
def openai_chat_messages_to_input_messages(
    messages: Any,
) -> List[Any]:
    """Convert an OpenAI-style ``messages`` list to a list of ``InputMessage``
    objects.

    Handles both dict-based messages (``{"role": ..., "content": ...}``) and
    object-based messages (attributes ``role`` / ``content``).  Multi-part
    content lists are JSON-serialised into a single ``Text`` part.

    Assistant messages may carry ``tool_calls`` with empty ``content``;
    those calls are appended as ``ToolCall`` parts (same mapping as
    completion output).

    When ``role`` is missing or empty but ``tool_calls`` is present (legacy
    / malformed history), ``tool_calls`` are still mapped and ``role`` is
    displayed as ``assistant`` for observability.
    """
    if not messages:
        return []
    dbg = _debug_llm_tool_calls()
    if dbg:
        try:
            logger.info(
                (
                    "[agentic_rl_debug_llm_tool_calls] "
                    "openai_chat_messages_to_input_messages:enter "
                    "msg_count=%s genai_available=%s"
                ),
                len(messages),
                _core.GENAI_AVAILABLE,
            )
        except Exception:
            _debug_llm_tool_calls_log_exc()
    out: List[Any] = []
    for idx, m in enumerate(messages):
        if isinstance(m, Dict):
            raw_role = m.get("role")
            content = m.get("content")
            tool_calls = m.get("tool_calls")
        else:
            raw_role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            tool_calls = _extract_tool_calls_from_message_obj(m)

        norm_role = raw_role if raw_role not in (None, "") else "user"
        include_tc = _should_include_tool_calls_in_input(raw_role, tool_calls)
        if dbg:
            try:
                tc_len = len(tool_calls) if tool_calls else 0
                first_keys = None
                if tool_calls and len(tool_calls) > 0:
                    first_keys = _first_tool_call_keys(tool_calls[0])
                logger.info(
                    (
                        "[agentic_rl_debug_llm_tool_calls] "
                        "openai_chat_messages_to_input_messages:msg "
                        "idx=%s raw_role=%r msg_type=%s tc_len=%s "
                        "include_tc=%s first_tc_keys=%s"
                    ),
                    idx,
                    raw_role,
                    type(m).__name__,
                    tc_len,
                    include_tc,
                    first_keys,
                )
            except Exception:
                _debug_llm_tool_calls_log_exc()
        if include_tc and (raw_role is None or raw_role == ""):
            norm_role = "assistant"

        parts: List[Any] = []
        if content is None:
            parts.append(_core.Text(content=""))
        elif isinstance(content, str):
            parts.append(_core.Text(content=content))
        elif isinstance(content, list):
            parts.append(
                _core.Text(
                    content=json.dumps(
                        content,
                        ensure_ascii=False,
                        default=str,
                    ),
                ),
            )
        else:
            parts.append(_core.Text(content=str(content)))
        if include_tc:
            parts.extend(tool_calls_to_parts(tool_calls))
        out.append(_core.InputMessage(role=norm_role, parts=parts))
    if dbg and _core.GENAI_AVAILABLE and _core.ToolCall is not None:
        try:
            per_msg: List[Dict[str, Any]] = []
            for idx, im in enumerate(out):
                n_tool = 0
                arg_lens: List[int] = []
                for p in getattr(im, "parts", None) or []:
                    if isinstance(p, _core.ToolCall):
                        n_tool += 1
                        arg_lens.append(len(getattr(p, "arguments", "") or ""))
                if n_tool:
                    per_msg.append(
                        {
                            "idx": idx,
                            "norm_role": getattr(im, "role", None),
                            "toolcall_parts": n_tool,
                            "arguments_str_lens": arg_lens,
                        },
                    )
            if per_msg:
                logger.info(
                    (
                        "[agentic_rl_debug_llm_tool_calls] "
                        "openai_chat_messages_to_input_messages:out "
                        "input_messages_with_tool_parts=%s"
                    ),
                    per_msg,
                )
        except Exception:
            _debug_llm_tool_calls_log_exc()
    return out


def openai_completion_to_output_messages(  # pylint: disable=too-many-branches
    completion: Any,
) -> List[Any]:
    """Convert an OpenAI ``ChatCompletion`` response to a list of
    ``OutputMessage`` objects.

    Each choice becomes one ``OutputMessage``.  Both text content and
    ``tool_calls`` are extracted; an empty ``Text`` part is appended when a
    choice has neither.
    """
    completion = unwrap_openai_completion(completion)

    # OpenAI-compatible SDKs sometimes return plain dicts (e.g. http clients or
    # wrappers). Handle both object- and dict-shaped responses.
    if isinstance(completion, Dict):
        choices = completion.get("choices") or []
    else:
        choices = getattr(completion, "choices", None) or []
    if not choices:
        return []
    result: List[Any] = []
    for ch in choices:
        if isinstance(ch, Dict):
            msg = ch.get("message")
            finish = ch.get("finish_reason") or "stop"
        else:
            msg = getattr(ch, "message", None)
            finish = getattr(ch, "finish_reason", None) or "stop"

        if isinstance(msg, Dict):
            role = msg.get("role") or "assistant"
        else:
            role = (
                getattr(msg, "role", "assistant")
                if msg is not None
                else "assistant"
            )
        parts: List[Any] = []
        if msg is not None:
            if isinstance(msg, Dict):
                content = msg.get("content")
            else:
                content = getattr(msg, "content", None)
            if content:
                parts.append(
                    _core.Text(
                        content=(
                            content
                            if isinstance(content, str)
                            else str(content)
                        ),
                    ),
                )
            if isinstance(msg, Dict):
                tool_calls = msg.get("tool_calls")
            else:
                tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                parts.extend(tool_calls_to_parts(tool_calls))
        if not parts:
            parts.append(_core.Text(content=""))
        result.append(
            _core.OutputMessage(role=role, parts=parts, finish_reason=finish),
        )
    return result


def dashscope_response_to_output_messages(response: Any) -> List[Any]:
    """Convert ``GenerationResponse`` from ``Generation.call`` to
    ``OutputMessage`` list.

    Non-streaming responses only.
    """
    if not _core.GENAI_AVAILABLE:
        return []
    output = getattr(response, "output", None)
    if output is None:
        return []
    choices = getattr(output, "choices", None)
    if choices:
        result: List[Any] = []
        for ch in choices:
            finish = getattr(ch, "finish_reason", None) or "stop"
            msg = getattr(ch, "message", None)
            role = "assistant"
            parts: List[Any] = []
            if msg is not None:
                if isinstance(msg, Dict):
                    role = msg.get("role") or "assistant"
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")
                else:
                    role = getattr(msg, "role", None) or "assistant"
                    content = getattr(msg, "content", None)
                    tool_calls = getattr(msg, "tool_calls", None)
                if content:
                    parts.append(
                        _core.Text(
                            content=(
                                content
                                if isinstance(content, str)
                                else str(content)
                            ),
                        ),
                    )
                if tool_calls:
                    parts.extend(tool_calls_to_parts(tool_calls))
            if not parts:
                parts.append(_core.Text(content=""))
            result.append(
                _core.OutputMessage(
                    role=role,
                    parts=parts,
                    finish_reason=finish,
                ),
            )
        return result
    text = getattr(output, "text", None)
    if text:
        return [
            _core.OutputMessage(
                role="assistant",
                parts=[
                    _core.Text(
                        content=(text if isinstance(text, str) else str(text)),
                    ),
                ],
                finish_reason="stop",
            ),
        ]
    return []
