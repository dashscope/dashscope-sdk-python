# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility re-export; the implementation moved to the 2.0 package.

The offline tokenizer factory lives only in
``dashscope.v2.tokenizers.tokenizer``, next to the bundled vocab, so both
stacks share one copy of the code and of the vocab. Import from there in
new code.

Note that :func:`get_tokenizer` raises ``ValueError`` for an unknown model;
it used to raise ``dashscope.legacy_1x.common.error.UnsupportedModel``, which the 2.0
leaf must not depend on.

``Tokenizer`` and ``QwenTokenizer`` are re-exported because the old module
imported them into its own namespace. The old module-level ``current_path``
and ``root_path`` are deliberately not restored: they only located the
vocab, which no longer ships next to this file.
"""

# pylint: disable-next=unused-import
from dashscope.v2.tokenizers.tokenizer import (  # noqa: F401
    QWEN_SERIALS,
    get_tokenizer,
    list_tokenizers,
)

# pylint: disable-next=unused-import
from dashscope.v2.tokenizers.qwen_tokenizer import (  # noqa: F401
    QwenTokenizer,
)

# pylint: disable-next=unused-import
from dashscope.v2.tokenizers.tokenizer_base import Tokenizer  # noqa: F401

__all__ = [
    "QWEN_SERIALS",
    "QwenTokenizer",
    "Tokenizer",
    "get_tokenizer",
    "list_tokenizers",
]
