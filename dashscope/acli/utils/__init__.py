# -*- coding: utf-8 -*-
"""acli.utils — shared utility modules."""

from __future__ import annotations

from dashscope.acli.utils.crypto import decrypt_value, encrypt_value
from dashscope.acli.utils.exceptions import UserAbortedTurn, UserSupplement
from dashscope.acli.utils.ids import now_iso, short_uuid
from dashscope.acli.utils.keywords import extract_keywords
from dashscope.acli.utils.messages import (
    is_tool_garbage,
    message_text_for_compress,
    normalize_for_model,
    text_of,
    tool_result_for_display,
    tool_result_for_history,
)
from dashscope.acli.utils.paths import (
    SENSITIVE_NAMES,
    atomic_write_text,
    validate_path,
    validate_write_path,
)
from dashscope.acli.utils.sanitizer import (
    is_secret_field,
    sanitize,
    sanitize_text,
)
from dashscope.acli.utils.spinner import AsyncSpinner, StderrSpinner
from dashscope.acli.utils.template import (
    render_brace_template,
    render_mustache_template,
)
from dashscope.acli.utils.text import (
    mask_secret,
    strip_frontmatter,
    truncate,
    truncate_head_tail,
    truncate_text,
    truncate_value,
)

# Re-export commonly used functions for convenience
from dashscope.acli.utils.toml import (
    load_toml,
    loads_toml,
    parse_toml_inline_table,
    parse_value,
    parse_value_raw,
    toml_str,
)
from dashscope.acli.utils.validation import (
    coerce_types,
    missing_required_args,
    parse_string_annotations,
)

__all__ = [
    "load_toml",
    "loads_toml",
    "parse_toml_inline_table",
    "parse_value",
    "parse_value_raw",
    "toml_str",
    "atomic_write_text",
    "is_tool_garbage",
    "message_text_for_compress",
    "normalize_for_model",
    "text_of",
    "tool_result_for_display",
    "tool_result_for_history",
    "is_secret_field",
    "sanitize",
    "sanitize_text",
    "mask_secret",
    "strip_frontmatter",
    "truncate",
    "truncate_head_tail",
    "truncate_text",
    "truncate_value",
    "AsyncSpinner",
    "StderrSpinner",
    "now_iso",
    "short_uuid",
    "extract_keywords",
    "render_brace_template",
    "render_mustache_template",
    "decrypt_value",
    "encrypt_value",
    "SENSITIVE_NAMES",
    "validate_path",
    "validate_write_path",
    "coerce_types",
    "missing_required_args",
    "parse_string_annotations",
    "UserAbortedTurn",
    "UserSupplement",
]
