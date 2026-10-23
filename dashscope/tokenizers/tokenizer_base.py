# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compatibility re-export; the implementation moved to the 2.0 package.

The offline tokenizer base class lives only in
``dashscope.v2.tokenizers.tokenizer_base`` so both stacks share one copy.
Import from there in new code.
"""

from dashscope.v2.tokenizers.tokenizer_base import Tokenizer

__all__ = ["Tokenizer"]
