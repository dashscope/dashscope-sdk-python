# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Module-level quickstart entry points backed by a lazy singleton Client."""

from __future__ import annotations

import threading
from typing import Any, Optional

from dashscope.v2.client import Client
from dashscope.v2.types.completions import ChatCompletion

__all__ = ["chat"]

_lock = threading.Lock()
_client: Optional[Client] = None


def _get_client() -> Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = Client()
    return _client


def chat(model: str, prompt: str, **kwargs: Any) -> ChatCompletion:
    """One-call chat entry: ``dashscope.chat("qwen-plus", "Tell me a joke")``.

    Extra keyword arguments are forwarded to
    ``client.chat.completions.create``.
    """
    return _get_client().chat.completions.create(
        model=model,
        messages=[prompt],
        **kwargs,
    )
