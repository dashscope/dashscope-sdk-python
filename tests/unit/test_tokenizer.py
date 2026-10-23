# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import importlib
import os

import pytest

from dashscope.v2.tokenizers.tokenizer import get_tokenizer
from dashscope.v2.tokenizers.qwen_tokenizer import _CHUNK_SIZE


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


class TestLegacyTokenizerPaths:
    """The published ``dashscope.tokenizers.*`` deep paths must keep working.

    The implementation moved to ``dashscope.v2.tokenizers``; the 1.x paths
    are thin re-export shims, so they have to resolve to the very same
    objects rather than to a second copy.
    """

    @pytest.mark.parametrize(
        "legacy_path,v2_path,name",
        [
            (
                "dashscope.tokenizers.tokenizer",
                "dashscope.v2.tokenizers.tokenizer",
                "get_tokenizer",
            ),
            (
                "dashscope.tokenizers.tokenizer",
                "dashscope.v2.tokenizers.tokenizer",
                "list_tokenizers",
            ),
            (
                "dashscope.tokenizers.tokenizer",
                "dashscope.v2.tokenizers.tokenizer",
                "QWEN_SERIALS",
            ),
            (
                "dashscope.tokenizers.qwen_tokenizer",
                "dashscope.v2.tokenizers.qwen_tokenizer",
                "QwenTokenizer",
            ),
            (
                "dashscope.tokenizers.qwen_tokenizer",
                "dashscope.v2.tokenizers.qwen_tokenizer",
                "SPECIAL_TOKENS_SET",
            ),
            (
                "dashscope.tokenizers.tokenizer_base",
                "dashscope.v2.tokenizers.tokenizer_base",
                "Tokenizer",
            ),
        ],
    )
    def test_deep_imports_resolve_to_the_v2_objects(
        self,
        legacy_path,
        v2_path,
        name,
    ):
        legacy = importlib.import_module(legacy_path)
        v2 = importlib.import_module(v2_path)

        assert getattr(legacy, name) is getattr(v2, name)

    def test_package_level_reexports_are_untouched(self):
        import dashscope.tokenizers as legacy_pkg
        import dashscope.v2.tokenizers as v2_pkg

        for name in v2_pkg.__all__:
            assert getattr(legacy_pkg, name) is getattr(v2_pkg, name)

    def test_legacy_path_returns_a_working_tokenizer(self):
        from dashscope.tokenizers.tokenizer import get_tokenizer as legacy

        tokenizer = legacy("qwen-turbo")
        tokens = tokenizer.encode("hello world")

        assert isinstance(tokens, list)
        assert tokenizer.decode(tokens) == "hello world"

    def test_from_import_syntax_resolves(self):
        from dashscope.tokenizers.qwen_tokenizer import QwenTokenizer
        from dashscope.tokenizers.tokenizer_base import Tokenizer

        assert issubclass(QwenTokenizer, Tokenizer)
