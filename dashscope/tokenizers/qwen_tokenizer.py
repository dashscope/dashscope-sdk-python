# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility re-export; the implementation moved to the 2.0 package.

The Qwen tiktoken tokenizer and its special-token tables live only in
``dashscope.v2.tokenizers.qwen_tokenizer``, next to the bundled vocab, so
both stacks share one copy. Import from there in new code.

``Tokenizer`` is re-exported because the old module imported it into its
own namespace, making ``from dashscope.tokenizers.qwen_tokenizer import
Tokenizer`` a working (if incidental) path.
"""

# pylint: disable-next=unused-import
from dashscope.v2.tokenizers.qwen_tokenizer import (  # noqa: F401
    ENDOFTEXT,
    EXTRAS,
    IMEND,
    IMSTART,
    PAT_STR,
    SPECIAL_START_ID,
    SPECIAL_TOKENS,
    SPECIAL_TOKENS_SET,
    QwenTokenizer,
    Tokenizer,
)

__all__ = [
    "ENDOFTEXT",
    "EXTRAS",
    "IMEND",
    "IMSTART",
    "PAT_STR",
    "SPECIAL_START_ID",
    "SPECIAL_TOKENS",
    "SPECIAL_TOKENS_SET",
    "QwenTokenizer",
    "Tokenizer",
]
