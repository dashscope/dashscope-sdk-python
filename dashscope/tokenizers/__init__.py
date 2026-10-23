# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from .tokenization import Tokenization

# The offline tokenizer implementation lives in the 2.0 package; both stacks
# share it so there is one copy of the code and of the bundled vocab. It is
# imported second on purpose: the 1.x chain above must finish loading before
# the v2 chain starts pulling modules in.
from dashscope.v2.tokenizers import (  # pylint: disable=wrong-import-order
    Tokenizer,
    get_tokenizer,
    list_tokenizers,
)

__all__ = [
    "Tokenization",
    "Tokenizer",
    "get_tokenizer",
    "list_tokenizers",
]
