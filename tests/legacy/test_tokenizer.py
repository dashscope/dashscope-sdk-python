# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os

from dashscope.tokenizers.tokenizer import get_tokenizer


class TestTokenization:
    @classmethod
    def setup_class(cls):
        # install tiktoken
        os.system("pip install tiktoken")

    def test_encode_decode(self):
        tokenizer = get_tokenizer("qwen-7b-chat")

        input_str = "这个是千问tokenizer"

        tokens = tokenizer.encode("这个是千问tokenizer")
        decoded_str = tokenizer.decode(tokens)
        assert input_str == decoded_str

        assert [151643] == tokenizer.encode(
            "<|endoftext|>",
            allowed_special={
                "<|endoftext|>",
            },
        )
        assert [151643] == tokenizer.encode(
            "<|endoftext|>",
            allowed_special="all",
        )
        assert [
            27,
            91,
            8691,
            723,
            427,
            91,
            29,
        ] == tokenizer.encode(
            "<|endoftext|>",
            allowed_special=set(),
        )
        assert [151643] == tokenizer.encode(
            "<|endoftext|>",
            disallowed_special=set(),
        )
