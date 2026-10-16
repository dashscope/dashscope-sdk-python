# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.multimodal resource."""

from dashscope.v2.resources.multimodal.chat import (
    AsyncMultimodalChat,
    MultimodalChat,
)
from dashscope.v2.resources.multimodal.embeddings import (
    AsyncMultimodalEmbeddings,
    MultimodalEmbeddings,
)

__all__ = [
    "AsyncMultimodal",
    "AsyncMultimodalEmbeddings",
    "Multimodal",
    "MultimodalEmbeddings",
]


class Multimodal:
    def __init__(self, client) -> None:
        self.chat = MultimodalChat(client)
        self.embeddings = MultimodalEmbeddings(client)


class AsyncMultimodal:
    def __init__(self, client) -> None:
        self.chat = AsyncMultimodalChat(client)
        self.embeddings = AsyncMultimodalEmbeddings(client)
