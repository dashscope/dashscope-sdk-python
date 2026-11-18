# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import os
import tempfile
import urllib.request
from typing import List

from dashscope.legacy_1x.common.error import UnsupportedModel
from .qwen_tokenizer import QwenTokenizer
from .tokenizer_base import Tokenizer

QWEN_SERIALS = ["qwen-7b-chat", "qwen-turbo", "qwen-plus", "qwen-max"]
_VOCAB_NAME = "qwen.tiktoken"
# The vocab is no longer bundled in the repository; it is downloaded from
# ModelScope on first use and cached under ~/.cache/dashscope/tokenizers.
_VOCAB_URL = (
    "https://www.modelscope.cn/models/Qwen/Qwen-7B-Chat/resolve/master/"
    + _VOCAB_NAME
)
_VOCAB_SHA256 = (
    "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"
)
_VOCAB_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "dashscope",
    "tokenizers",
)
_VOCAB_ENV = "DASHSCOPE_QWEN_VOCAB"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_vocab(dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(dest),
        prefix=_VOCAB_NAME + ".",
        suffix=".download",
    )
    os.close(fd)
    try:
        with urllib.request.urlopen(_VOCAB_URL, timeout=60) as resp:
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        if _sha256(tmp) != _VOCAB_SHA256:
            raise IOError(
                f"SHA256 mismatch for vocab downloaded from {_VOCAB_URL}",
            )
        os.replace(tmp, dest)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to obtain the Qwen tokenizer vocab ({_VOCAB_NAME}). "
            f"Download it manually from {_VOCAB_URL} and either place it "
            f"at {dest} or point {_VOCAB_ENV} at the file "
            f"(required for offline environments). Cause: {exc}",
        ) from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _resolve_vocab_path() -> str:
    override = os.environ.get(_VOCAB_ENV)
    if override:
        return os.path.expanduser(override)
    cached = os.path.join(_VOCAB_CACHE_DIR, _VOCAB_NAME)
    if os.path.isfile(cached) and _sha256(cached) == _VOCAB_SHA256:
        return cached
    _download_vocab(cached)
    return cached


def get_tokenizer(model: str) -> Tokenizer:
    """Get a tokenizer based on model name.

    The vocab file is downloaded on first use from ModelScope into
    ``~/.cache/dashscope/tokenizers``.
    Set ``DASHSCOPE_QWEN_VOCAB`` to a local file path for offline use.

    Args:
        model (str): The model name.

    Raises:
        UnsupportedModel: Not support model

    Returns:
        Tokenizer: The  `Tokenizer` of the model.
    """
    if model.startswith("qwen"):
        return QwenTokenizer(_resolve_vocab_path())
    raise UnsupportedModel(
        f"Not support model: {model}, currently only support qwen models.",
    )


def list_tokenizers() -> List[str]:
    """List support models

    Returns:
        List[str]: The model list.
    """
    return QWEN_SERIALS
