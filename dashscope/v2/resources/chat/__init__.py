# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.chat resource."""

from dashscope.v2.resources.chat.completions import (
    AsyncCompletions,
    Completions,
)
from dashscope.v2.types.common import Usage
from dashscope.v2.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    Delta,
)

__all__ = [
    "AsyncChat",
    "AsyncCompletions",
    "Chat",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "Choice",
    "Completions",
    "Delta",
    "Usage",
]


class Chat:
    def __init__(self, client) -> None:
        self.completions = Completions(client)


class AsyncChat:
    def __init__(self, client) -> None:
        self.completions = AsyncCompletions(client)
