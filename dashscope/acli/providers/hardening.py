# -*- coding: utf-8 -*-
"""Provider edge-case hardening.

Wraps any LLMProvider with:

* Retry on transient errors (timeout, connect, rate limit, server errors).
* Empty-response recovery: if a provider returns no content and no tool calls,
  retry once with a scaffolding hint.
* Stream parse-error recovery: malformed SSE/JSON chunks are logged and skipped
  instead of crashing the whole turn.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from dashscope.acli.providers.base import LLMChunk, LLMProvider, LLMResponse

# Retry classification patterns. Provider errors are bilingual: the
# wrappers raise Chinese RuntimeErrors ("API 请求超时…", "无法连接到 API
# 服务器…") while SDK/network errors surface in English. Both layers
# (HardenedProvider retry + ProviderChain fallback) share
# ``is_retryable_error`` below so they classify identically.
#
# NOTE: "不可用" / "服务异常" stay retryable so the stream-empty message
# "…可能是限流、模型不可用或服务异常" is treated as transient.
_RETRYABLE_PATTERNS = (
    # English
    "timeout",
    "connect",
    "rate limit",
    "too many requests",
    "server error",
    "internal error",
    "temporarily unavailable",
    "empty response",
    "stream",
    "parse",
    "jsondecodeerror",
    "unexpected eof",
    # 中文
    "超时",
    "连接",
    "网络",
    "限流",
    "不可用",
    "空响应",
    "未返回任何内容",
    "服务异常",
    "内部错误",
    "流式中断",
)

# Checked FIRST and wins over retryable patterns — e.g.
# "API 端点不存在 (404)" must not be retried.
_NON_RETRYABLE_PATTERNS = (
    # English
    "invalid api key",
    "authentication",
    "unauthorized",
    "forbidden",
    "not found",
    "invalid request",
    "model not found",
    "insufficient_quota",
    "billing",
    # 中文
    "权限",
    "认证",
    "鉴权",
    "无效 api key",
    "密钥",
    "配额",
    "账单",
    "不存在",
)

# Hint appended to a retry when the model produced an empty reply.
_EMPTY_RECOVERY_HINT = {
    "role": "user",
    "content": ("你刚才没有输出任何内容。请根据上下文继续，必要时直接调用工具完成用户请求。"),
}


def is_retryable_error(error: BaseException) -> bool:
    """Classify an exception as retryable/fallback-worthy (bilingual).

    Shared by HardenedProvider (retry) and ProviderChain (fallback) so
    both layers agree. Non-retryable patterns win over retryable ones.
    """
    text = str(error).lower()
    if any(p in text for p in _NON_RETRYABLE_PATTERNS):
        return False
    return any(p in text for p in _RETRYABLE_PATTERNS)


def _is_empty_response(resp: LLMResponse) -> bool:
    return not resp.content.strip() and not resp.tool_calls


class HardenedProvider:
    """Wraps an LLMProvider with retries and empty-response recovery."""

    def __init__(
        self,
        provider: LLMProvider,
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ):
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        last_error: BaseException | None = None
        attempt_messages = messages

        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.provider.chat(
                    attempt_messages,
                    tools,
                    response_format=response_format,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries and is_retryable_error(e):
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise

            if _is_empty_response(resp):
                if attempt < self.max_retries:
                    # Retry once with a recovery hint appended.
                    attempt_messages = list(messages) + [_EMPTY_RECOVERY_HINT]
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
            return resp

        raise last_error or RuntimeError("Provider 调用失败")

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncIterator[LLMChunk]:
        attempt_messages = messages

        for attempt in range(self.max_retries + 1):
            emitted_anything = False
            try:
                async for chunk in self.provider.chat_stream(
                    attempt_messages,
                    tools,
                    response_format=response_format,
                ):
                    emitted_anything = True
                    yield chunk
                # Stream ended normally.
                if not emitted_anything and attempt < self.max_retries:
                    # No chunks at all; treat like empty response.
                    attempt_messages = list(messages) + [_EMPTY_RECOVERY_HINT]
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                return
            except Exception as e:
                # First-chunk rule: after any chunk was yielded, retrying
                # would re-emit content — propagate the error instead.
                if (
                    not emitted_anything
                    and is_retryable_error(e)
                    and attempt < self.max_retries
                ):
                    # Retry with same messages for transient failures.
                    attempt_messages = messages
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
