# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 local mock debugging: develop against a canned
gateway with zero network calls and zero token cost.

The 2.0 ``Client`` accepts an injected ``http_client``. Passing an
``httpx.Client`` wired to ``httpx.MockTransport`` serves canned
responses in-process, so you can exercise both the success path and
the error paths (bad parameter, rate limit, content filter)
deterministically -- before spending a single real token.

Swap ``_mock_client()`` for ``dashscope.Client()`` and the exact same
calling code runs against the real gateway.
"""

import httpx

import dashscope
from dashscope.v2 import errors

# A fake gateway URL. MockTransport intercepts every request, so no
# traffic ever leaves this process and the host is never resolved.
BASE_URL = "https://mock.local/api/v1"


def _chat_envelope(text, request_id="mock-req-1"):
    """The native chat-completion envelope the real gateway returns."""
    return {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                },
            ],
        },
        "usage": {
            "input_tokens": 8,
            "output_tokens": 4,
            "total_tokens": 12,
        },
        "request_id": request_id,
    }


def _handler(request):
    """Route by request body so one transport serves many scenarios.

    Send a message containing ``trigger-error`` to exercise the error
    path; anything else returns a normal completion.
    """
    body = request.content.decode("utf-8")
    if "trigger-error" in body:
        return httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "mocked bad request",
            },
        )
    return httpx.Response(200, json=_chat_envelope("你好，这是本地 Mock 返回。"))


def _mock_client():
    """A Client whose transport is fully served in-process."""
    transport = httpx.MockTransport(_handler)
    return dashscope.Client(
        api_key="sk-mock",  # never validated; no real call is made
        base_url=BASE_URL,
        http_client=httpx.Client(transport=transport, base_url=BASE_URL),
    )


if __name__ == "__main__":
    client = _mock_client()

    print("----- mocked success -----")
    resp = client.chat.completions.create(
        model="qwen3.7-plus",
        messages=[{"role": "user", "content": "你好"}],
    )
    print("output_text :", resp.output_text)
    print("request_id  :", resp.request_id)
    print("total_tokens:", resp.usage.total_tokens)

    # Error path 1: the SDK raises a typed exception by default.
    print("----- mocked error (raise_on_error=True) -----")
    try:
        client.chat.completions.create(
            model="qwen3.7-plus",
            messages=[{"role": "user", "content": "trigger-error"}],
        )
    except errors.DashScopeError as exc:
        print("caught      :", type(exc).__name__)
        print("code        :", exc.code)
        print("status_code :", exc.status_code)
        print("message     :", exc.message)

    # Error path 2: opt out of raising to inspect the error inline.
    print("----- mocked error (raise_on_error=False) -----")
    lenient = dashscope.Client(
        api_key="sk-mock",
        base_url=BASE_URL,
        http_client=httpx.Client(
            transport=httpx.MockTransport(_handler),
            base_url=BASE_URL,
        ),
        raise_on_error=False,
    )
    err = lenient.chat.completions.create(
        model="qwen3.7-plus",
        messages=[{"role": "user", "content": "trigger-error"}],
    )
    print("is_error    :", err.is_error)
    print("code        :", err.code)
    print("message     :", err.message)
