# -*- coding: utf-8 -*-
"""The vendored provider stack must not multiply its own retry budget.

``HardenedProvider`` marks an error whose retries it already spent with
``RetriesExhausted``, so ``Agent`` does not pay that same budget a second
time. Against an endpoint that accepts the connection and never replies, the
unmarked version cost 12 HTTP attempts and ~819s of silent wall clock in one
turn, from a 60s ``request_timeout``.

The original message is carried verbatim on purpose. ``ProviderChain`` decides
whether to fall back to the next profile with ``is_retryable_error``, so a
wrapper that reworded the error would turn "this profile is exhausted" into
"abort the whole chain".
"""

# pylint: disable=redefined-outer-name,protected-access,unused-argument

from __future__ import annotations

from collections import deque

import pytest

from dashscope.acli.providers.base import LLMChunk
from dashscope.acli.providers.hardening import (
    HardenedProvider,
    RetriesExhausted,
    is_retryable_error,
)
from dashscope.acli.providers.profile import ProviderChain, ProviderProfile

# The literal message the tongyi provider raises on a read timeout. It
# classifies as retryable via "network", not "timeout": the pattern list has
# no entry for the "timed out" wording the provider actually uses.
STALLED = "API request timed out; check network or retry later"

MESSAGES = [{"role": "user", "content": "hi"}]


class _Scripted:
    """Pops one scripted outcome per call: an exception, or chunks to yield."""

    def __init__(self, chat=None, stream=None) -> None:
        self._chat_q = deque(chat or [])
        self._stream_q = deque(stream or [])
        self.chat_calls = 0
        self.stream_calls = 0

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ):
        value = self._chat_q.popleft()
        self.chat_calls += 1
        if isinstance(value, Exception):
            raise value
        return value

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ):
        value = self._stream_q.popleft()
        self.stream_calls += 1
        if isinstance(value, Exception):
            raise value
        for chunk in value:
            yield chunk


def _one_chunk_then_stall():
    """A stream outcome that yields one chunk and then fails."""
    yield LLMChunk(delta_content="hi")
    raise RuntimeError(STALLED)


def _profile(name: str, max_retries: int = 0) -> ProviderProfile:
    return ProviderProfile(
        name=name,
        provider="tongyi",
        model="m",
        api_key="k",
        max_retries=max_retries,
    )


async def _collect(stream) -> list:
    return [c async for c in stream]


async def test_an_exhausted_stream_is_marked_and_costs_exactly_the_budget():
    inner = _Scripted(stream=[RuntimeError(STALLED)] * 4)
    hardened = HardenedProvider(inner, max_retries=3, retry_delay=0.01)

    with pytest.raises(RetriesExhausted) as exc:
        await _collect(hardened.chat_stream(MESSAGES))

    # max_retries=3 means four attempts and no more: this layer must not
    # multiply internally the way the turn-level retry multiplied it.
    assert inner.stream_calls == 4
    assert not inner._stream_q
    assert str(exc.value) == STALLED
    assert isinstance(exc.value.error, RuntimeError)


async def test_an_exhausted_chat_is_marked_too():
    inner = _Scripted(chat=[RuntimeError(STALLED)] * 2)
    hardened = HardenedProvider(inner, max_retries=1, retry_delay=0.01)

    with pytest.raises(RetriesExhausted):
        await hardened.chat(MESSAGES)
    assert inner.chat_calls == 2


def test_the_marker_keeps_the_message_retryable():
    assert is_retryable_error(RetriesExhausted(RuntimeError(STALLED))) is True


async def test_the_chain_still_falls_back_after_a_stream_exhaustion():
    bad = _Scripted(stream=[RuntimeError(STALLED)] * 3)
    good = _Scripted(stream=[[LLMChunk(delta_content="fallback")]])
    chain = ProviderChain([_profile("p1", max_retries=2), _profile("p2")])
    chain._instances[0] = HardenedProvider(
        bad,
        max_retries=2,
        retry_delay=0.01,
    )
    chain._instances[1] = HardenedProvider(
        good,
        max_retries=0,
        retry_delay=0.01,
    )

    chunks = await _collect(chain.chat_stream(MESSAGES))

    assert [c.delta_content for c in chunks] == ["fallback"]
    assert bad.stream_calls == 3
    assert good.stream_calls == 1


async def test_a_non_retryable_error_stays_plain():
    # An auth failure was never retried, so no budget was spent on it and the
    # marker would be a lie that suppresses a legitimate upper-layer retry.
    inner = _Scripted(chat=[RuntimeError("invalid api key")])
    hardened = HardenedProvider(inner, max_retries=2, retry_delay=0.01)

    with pytest.raises(RuntimeError) as exc:
        await hardened.chat(MESSAGES)
    assert not isinstance(exc.value, RetriesExhausted)
    assert inner.chat_calls == 1


async def test_a_failure_after_the_first_chunk_stays_plain():
    # The first-chunk rule propagates at once instead of retrying, so this
    # layer spent no budget on it either.
    inner = _Scripted(stream=[_one_chunk_then_stall()])
    hardened = HardenedProvider(inner, max_retries=2, retry_delay=0.01)

    seen = []
    with pytest.raises(RuntimeError) as exc:
        async for chunk in hardened.chat_stream(MESSAGES):
            seen.append(chunk)
    assert [c.delta_content for c in seen] == ["hi"]
    assert not isinstance(exc.value, RetriesExhausted)
    assert inner.stream_calls == 1
