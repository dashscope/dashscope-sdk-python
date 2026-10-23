# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offline tokenizers owned by the 2.0 SDK.

Pure local counting over the bundled tiktoken vocab -- no network, no
dependency on the 1.x package. ``dashscope.tokenizers`` re-exports these
so both stacks share one implementation and one copy of the vocab.
"""

from .tokenizer import get_tokenizer, list_tokenizers
from .tokenizer_base import Tokenizer

__all__ = [
    "Tokenizer",
    "get_tokenizer",
    "list_tokenizers",
]
