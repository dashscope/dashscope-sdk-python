# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
from logging import NullHandler

from dashscope.aigc.code_generation import CodeGeneration
from dashscope.aigc.conversation import Conversation, History, HistoryItem
from dashscope.aigc.generation import AioGeneration, Generation
from dashscope.aigc.image_synthesis import ImageSynthesis
from dashscope.aigc.multimodal_conversation import (
    MultiModalConversation,
    AioMultiModalConversation,
)
from dashscope.aigc.video_synthesis import VideoSynthesis
from dashscope.app.application import Application
from dashscope.assistants import Assistant, AssistantList, Assistants
from dashscope.assistants.assistant_types import AssistantFile, DeleteResponse
from dashscope.audio.asr.transcription import Transcription
from dashscope.audio.tts.speech_synthesizer import SpeechSynthesizer
from dashscope.common.api_key import save_api_key
from dashscope.common.env import (
    api_key,
    api_key_file_path,
    base_http_api_url,
    base_websocket_api_url,
)
from dashscope.common.aio_session_manager import AioSessionManager
from dashscope.common.session_manager import SessionManager
from dashscope.customize.deployments import Deployments
from dashscope.customize.finetunes import FineTunes
from dashscope.embeddings.batch_text_embedding import BatchTextEmbedding
from dashscope.embeddings.batch_text_embedding_response import (
    BatchTextEmbeddingResponse,
)
from dashscope.embeddings.multimodal_embedding import (
    MultiModalEmbedding,
    MultiModalEmbeddingItemAudio,
    MultiModalEmbeddingItemImage,
    MultiModalEmbeddingItemText,
    AioMultiModalEmbedding,
)
from dashscope.embeddings.text_embedding import TextEmbedding
from dashscope.files import Files
from dashscope.models import Models
from dashscope.nlp.understanding import Understanding
from dashscope.rerank.text_rerank import TextReRank
from dashscope.threads import (
    MessageFile,
    Messages,
    Run,
    RunList,
    Runs,
    RunStep,
    RunStepList,
    Steps,
    Thread,
    ThreadMessage,
    ThreadMessageList,
    Threads,
)
from dashscope.tokenizers import (
    Tokenization,
    Tokenizer,
    get_tokenizer,
    list_tokenizers,
)


def enable_http_connection_pool(
    pool_connections: int = None,
    pool_maxsize: int = None,
    max_retries: int = None,
    pool_block: bool = None,
):
    """
    启用 HTTP 连接池复用

    启用后，所有同步 HTTP 请求将复用连接，显著减少延迟。

    Args:
        pool_connections: 连接池大小，默认 10
            - 低并发（< 10 req/s）: 10
            - 中并发（10-50 req/s）: 20-30
            - 高并发（> 50 req/s）: 50-100

        pool_maxsize: 最大连接数，默认 20
            - 应该 >= pool_connections
            - 低并发: 20
            - 中并发: 50
            - 高并发: 100-200

        max_retries: 重试次数，默认 3
            - 网络稳定: 3
            - 网络不稳定: 5-10

        pool_block: 连接池满时是否阻塞，默认 False
            - False: 连接池满时创建新连接（推荐）
            - True: 连接池满时等待可用连接

    Examples:
        >>> import dashscope
        >>>
        >>> # 使用默认配置
        >>> dashscope.enable_http_connection_pool()
        >>>
        >>> # 自定义配置
        >>> dashscope.enable_http_connection_pool(
        ...     pool_connections=20,
        ...     pool_maxsize=50
        ... )
        >>>
        >>> # 之后的所有请求都会复用连接
        >>> Generation.call(model='qwen-turbo', prompt='Hello')
    """
    SessionManager.get_instance().enable(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=max_retries,
        pool_block=pool_block,
    )


def disable_http_connection_pool():
    """
    禁用 HTTP 连接池复用

    恢复到原有的每次请求创建新连接的行为。

    Example:
        >>> import dashscope
        >>> dashscope.disable_http_connection_pool()
    """
    SessionManager.get_instance().disable()


def reset_http_connection_pool():
    """
    重置 HTTP 连接池

    用于处理连接问题或网络切换场景。

    Example:
        >>> import dashscope
        >>> dashscope.reset_http_connection_pool()
    """
    SessionManager.get_instance().reset()


def configure_http_connection_pool(
    pool_connections: int = None,
    pool_maxsize: int = None,
    max_retries: int = None,
    pool_block: bool = None,
):
    """
    配置 HTTP 连接池参数

    运行时动态调整连接池配置。

    Args:
        pool_connections: 连接池大小
        pool_maxsize: 最大连接数
        max_retries: 重试次数
        pool_block: 连接池满时是否阻塞

    Examples:
        >>> import dashscope
        >>>
        >>> # 调整单个参数
        >>> dashscope.configure_http_connection_pool(pool_maxsize=100)
        >>>
        >>> # 调整多个参数
        >>> dashscope.configure_http_connection_pool(
        ...     pool_connections=50,
        ...     pool_maxsize=100
        ... )
    """
    SessionManager.get_instance().configure(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=max_retries,
        pool_block=pool_block,
    )


async def enable_aio_http_connection_pool(
    limit: int = None,
    limit_per_host: int = None,
    ttl_dns_cache: int = None,
    keepalive_timeout: int = None,
    force_close: bool = None,
):
    """
    启用异步 HTTP 连接池复用

    启用后，所有异步 HTTP 请求将复用连接，显著减少延迟。

    Args:
        limit: 总连接数限制，默认 100
            - 低并发（< 10 req/s）: 100
            - 中并发（10-50 req/s）: 200
            - 高并发（> 50 req/s）: 300-500

        limit_per_host: 每个主机的连接数限制，默认 30
            - 应该 <= limit
            - 低并发: 30
            - 中并发: 50
            - 高并发: 100

        ttl_dns_cache: DNS 缓存 TTL（秒），默认 300
            - DNS 稳定: 300-600
            - DNS 变化频繁: 60-120

        keepalive_timeout: Keep-Alive 超时（秒），默认 30
            - 短连接: 15-30
            - 长连接: 60-120

        force_close: 是否强制关闭连接，默认 False
            - False: 复用连接（推荐）
            - True: 每次关闭连接

    Examples:
        >>> import asyncio
        >>> import dashscope
        >>> from dashscope import AioGeneration
        >>>
        >>> async def main():
        ...     # 使用默认配置
        ...     await dashscope.enable_aio_http_connection_pool()
        ...
        ...     # 之后的所有异步请求都会复用连接
        ...     response = await AioGeneration.call(
        ...         model='qwen-turbo',
        ...         prompt='Hello'
        ...     )
        ...
        ...     # 自定义配置
        ...     await dashscope.enable_aio_http_connection_pool(
        ...         limit=200,
        ...         limit_per_host=50
        ...     )
        >>>
        >>> asyncio.run(main())
    """
    manager = await AioSessionManager.get_instance()
    await manager.enable(
        limit=limit,
        limit_per_host=limit_per_host,
        ttl_dns_cache=ttl_dns_cache,
        keepalive_timeout=keepalive_timeout,
        force_close=force_close,
    )


async def disable_aio_http_connection_pool():
    """
    禁用异步 HTTP 连接池复用

    恢复到原有的每次请求创建新连接的行为。

    Examples:
        >>> import asyncio
        >>> import dashscope
        >>>
        >>> async def main():
        ...     await dashscope.disable_aio_http_connection_pool()
        >>>
        >>> asyncio.run(main())
    """
    manager = await AioSessionManager.get_instance()
    await manager.disable()


async def reset_aio_http_connection_pool():
    """
    重置异步 HTTP 连接池

    用于处理连接问题或网络切换场景。

    Examples:
        >>> import asyncio
        >>> import dashscope
        >>>
        >>> async def main():
        ...     await dashscope.reset_aio_http_connection_pool()
        >>>
        >>> asyncio.run(main())
    """
    manager = await AioSessionManager.get_instance()
    await manager.reset()


async def configure_aio_http_connection_pool(
    limit: int = None,
    limit_per_host: int = None,
    ttl_dns_cache: int = None,
    keepalive_timeout: int = None,
    force_close: bool = None,
):
    """
    配置异步 HTTP 连接池参数

    运行时动态调整连接池配置。

    Args:
        limit: 总连接数限制
        limit_per_host: 每个主机的连接数限制
        ttl_dns_cache: DNS 缓存 TTL（秒）
        keepalive_timeout: Keep-Alive 超时（秒）
        force_close: 是否强制关闭连接

    Examples:
        >>> import asyncio
        >>> import dashscope
        >>>
        >>> async def main():
        ...     # 调整单个参数
        ...     await dashscope.configure_aio_http_connection_pool(limit=200)
        ...
        ...     # 调整多个参数
        ...     await dashscope.configure_aio_http_connection_pool(
        ...         limit=200,
        ...         limit_per_host=50
        ...     )
        >>>
        >>> asyncio.run(main())
    """
    manager = await AioSessionManager.get_instance()
    await manager.configure(
        limit=limit,
        limit_per_host=limit_per_host,
        ttl_dns_cache=ttl_dns_cache,
        keepalive_timeout=keepalive_timeout,
        force_close=force_close,
    )


__all__ = [
    "base_http_api_url",
    "base_websocket_api_url",
    "api_key",
    "api_key_file_path",
    "save_api_key",
    "AioGeneration",
    "Conversation",
    "Generation",
    "History",
    "HistoryItem",
    "ImageSynthesis",
    "Transcription",
    "Files",
    "Deployments",
    "FineTunes",
    "Models",
    "TextEmbedding",
    "MultiModalEmbedding",
    "AioMultiModalEmbedding",
    "MultiModalEmbeddingItemAudio",
    "MultiModalEmbeddingItemImage",
    "MultiModalEmbeddingItemText",
    "SpeechSynthesizer",
    "MultiModalConversation",
    "AioMultiModalConversation",
    "BatchTextEmbedding",
    "BatchTextEmbeddingResponse",
    "Understanding",
    "CodeGeneration",
    "Tokenization",
    "Tokenizer",
    "get_tokenizer",
    "list_tokenizers",
    "Application",
    "TextReRank",
    "Assistants",
    "Threads",
    "Messages",
    "Runs",
    "Assistant",
    "ThreadMessage",
    "Run",
    "Steps",
    "AssistantList",
    "ThreadMessageList",
    "RunList",
    "RunStepList",
    "Thread",
    "DeleteResponse",
    "RunStep",
    "MessageFile",
    "AssistantFile",
    "VideoSynthesis",
    "enable_http_connection_pool",
    "disable_http_connection_pool",
    "reset_http_connection_pool",
    "configure_http_connection_pool",
    "enable_aio_http_connection_pool",
    "disable_aio_http_connection_pool",
    "reset_aio_http_connection_pool",
    "configure_aio_http_connection_pool",
]

logging.getLogger(__name__).addHandler(NullHandler())
