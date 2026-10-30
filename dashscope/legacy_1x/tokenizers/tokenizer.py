# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import List

from dashscope.legacy_1x.common.error import UnsupportedModel
from .qwen_tokenizer import QwenTokenizer
from .tokenizer_base import Tokenizer

QWEN_SERIALS = ["qwen-7b-chat", "qwen-turbo", "qwen-plus", "qwen-max"]
# The vocab ships next to this module, so the leaf stays relocatable.
_VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "qwen.tiktoken",
)


def get_tokenizer(model: str) -> Tokenizer:
    """Get a tokenizer based on model name.

    Args:
        model (str): The model name.

    Raises:
        UnsupportedModel: Not support model

    Returns:
        Tokenizer: The  `Tokenizer` of the model.
    """
    if model.startswith("qwen"):
        return QwenTokenizer(_VOCAB_PATH)
    raise UnsupportedModel(
        f"Not support model: {model}, currently only support qwen models.",
    )


def list_tokenizers() -> List[str]:
    """List support models

    Returns:
        List[str]: The model list.
    """
    return QWEN_SERIALS
