# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib
import os

import pytest

from dashscope.tokenizers.tokenizer import get_tokenizer
from dashscope.tokenizers.qwen_tokenizer import _CHUNK_SIZE


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

    def test_encode_chunk_size_exceed(self):
        # Test encoding functionality when text length exceeds _CHUNK_SIZE
        tokenizer = get_tokenizer("qwen-7b-chat")

        # Create a long text that exceeds _CHUNK_SIZE
        long_text = "Hello world! " * (
            _CHUNK_SIZE // 12 + 10
        )  # Ensure it exceeds the threshold

        # Encode long text, should not raise an exception
        tokens = tokenizer.encode(long_text)

        # Decoded text should match the original string
        decoded_str = tokenizer.decode(tokens)
        assert decoded_str == long_text

        # Ensure the return type is a list
        assert isinstance(tokens, list)

        # Test long text with special characters
        long_text_with_special = (
            "<|extra_0|> " * (_CHUNK_SIZE // 12 + 5)
        ) + "Normal text here."
        tokens_with_special = tokenizer.encode(
            long_text_with_special,
            allowed_special="all",
        )
        decoded_with_special = tokenizer.decode(tokens_with_special)
        assert decoded_with_special == long_text_with_special


class TestTokenizerPaths:
    """The root package owns one implementation and its published exports."""

    @pytest.mark.parametrize(
        "module_path,name",
        [
            ("dashscope.tokenizers.tokenizer", "get_tokenizer"),
            ("dashscope.tokenizers.tokenizer", "list_tokenizers"),
            ("dashscope.tokenizers.tokenizer", "QWEN_SERIALS"),
            ("dashscope.tokenizers.qwen_tokenizer", "QwenTokenizer"),
            ("dashscope.tokenizers.qwen_tokenizer", "SPECIAL_TOKENS_SET"),
            ("dashscope.tokenizers.tokenizer_base", "Tokenizer"),
        ],
    )
    def test_deep_imports_resolve(self, module_path, name):
        module = importlib.import_module(module_path)

        assert getattr(module, name) is not None

    def test_package_level_reexports(self):
        from dashscope import tokenizers
        from dashscope.tokenizers.tokenization import Tokenization
        from dashscope.tokenizers.tokenizer import list_tokenizers
        from dashscope.tokenizers.tokenizer_base import Tokenizer

        assert tokenizers.__all__ == [
            "Tokenization",
            "Tokenizer",
            "get_tokenizer",
            "list_tokenizers",
        ]
        assert tokenizers.Tokenization is Tokenization
        assert tokenizers.Tokenizer is Tokenizer
        assert tokenizers.get_tokenizer is get_tokenizer
        assert tokenizers.list_tokenizers is list_tokenizers

    def test_public_path_returns_a_working_tokenizer(self):
        tokenizer = get_tokenizer("qwen-turbo")
        tokens = tokenizer.encode("hello world")

        assert isinstance(tokens, list)
        assert tokenizer.decode(tokens) == "hello world"

    def test_from_import_syntax_resolves(self):
        from dashscope.tokenizers.qwen_tokenizer import QwenTokenizer
        from dashscope.tokenizers.tokenizer_base import Tokenizer

        assert issubclass(QwenTokenizer, Tokenizer)
