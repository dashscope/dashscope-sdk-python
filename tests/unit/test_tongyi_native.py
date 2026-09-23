# -*- coding: utf-8 -*-
"""TongyiProvider talks to the DashScope native generation routes.

The provider POSTs {model, input.messages, parameters} with
result_format=message. Each model is bound to exactly one aigc service
path (text-generation vs multimodal-generation); the provider falls back
on the gateway's "url error" 400 and caches the working path. Responses
and SSE chunks are reshaped to the OpenAI shape the parsing logic (and
the anthropic adapter path) was written against.
"""

# pylint: disable=redefined-outer-name,protected-access

import json

import pytest

from dashscope.acli.providers.tongyi import (
    _GENERATION_PATHS,
    _PATH_CACHE_NAME,
    TongyiProvider,
)

# Spelled out rather than destructured from _GENERATION_PATHS: a positional
# unpack silently swaps these two names the moment the probe order changes,
# inverting every assertion below while most of them still pass.
TEXT_GEN = "/api/v1/services/aigc/text-generation/generation"
MULTIMODAL_GEN = "/api/v1/services/aigc/multimodal-generation/generation"
ROOT = "https://dashscope.aliyuncs.com"

MESSAGES = [{"role": "user", "content": "hi"}]

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
]

NATIVE_RESPONSE = {
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello there"},
            },
        ],
    },
    "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    "request_id": "req-native-1",
}

# multimodal-generation returns content as a list of parts
MULTIMODAL_RESPONSE = {
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hello"}, {"text": " there"}],
                    "reasoning_content": "thinking...",
                },
            },
        ],
    },
    "usage": {
        "input_tokens": 62,
        "output_tokens": 28,
        "total_tokens": 90,
        "prompt_tokens_details": {"cached_tokens": 5},
    },
    "request_id": "req-mm-1",
}

URL_ERROR_BODY = {
    "code": "InvalidParameter",
    "message": "url error, please check url！ For details, see: "
    "https://help.aliyun.com/zh/model-studio/error-code#error-url",
    "request_id": "req-400",
}

NATIVE_TOOL_RESPONSE = {
    "output": {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "hz"}',
                            },
                        },
                    ],
                },
            },
        ],
    },
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    "request_id": "req-native-2",
}

NATIVE_SSE_LINES = [
    "id:1",
    "event:result",
    ":HTTP_STATUS/200",
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "content": "Hel"}}]}, "usage": {"input_tokens": '
    '11, "output_tokens": 1, "total_tokens": 12}, "request_id": "r1"}',
    "",
    "id:2",
    "event:result",
    ":HTTP_STATUS/200",
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "content": "lo"}}]}, "usage": {"input_tokens": '
    '11, "output_tokens": 2, "total_tokens": 13}, "request_id": "r1"}',
    'data:{"output": {"choices": [{"finish_reason": "stop", "message": '
    '{"role": "assistant", "content": ""}}]}, "usage": {"input_tokens": 11,'
    ' "output_tokens": 2, "total_tokens": 13}, "request_id": "r1"}',
]

# multimodal-generation streams content as part lists
MULTIMODAL_SSE_LINES = [
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "content": [], "reasoning_content": "th"}}]}, '
    '"request_id": "r5"}',
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "content": [{"text": "Hel"}]}}]}, '
    '"request_id": "r5"}',
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "content": [{"text": "lo"}]}}]}, '
    '"request_id": "r5"}',
    'data:{"output": {"choices": [{"finish_reason": "stop", "message": '
    '{"role": "assistant", "content": []}}]}, "usage": {"input_tokens": 5,'
    ' "output_tokens": 4, "total_tokens": 9}, "request_id": "r5"}',
]

NATIVE_SSE_TOOL_LINES = [
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "content": "", "tool_calls": [{"index": 0, "id":'
    ' "call_1", "type": "function", "function": {"name": "get_weather", '
    '"arguments": ""}}]}}]}, "request_id": "r2"}',
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "tool_calls": [{"index": 0, "function": '
    '{"arguments": "{\\"ci"}}]}}]}, "request_id": "r2"}',
    'data:{"output": {"choices": [{"finish_reason": "null", "message": '
    '{"role": "assistant", "tool_calls": [{"index": 0, "function": '
    '{"arguments": "ty\\": \\"hz\\"}"}}]}}]}, "request_id": "r2"}',
    'data:{"output": {"choices": [{"finish_reason": "tool_calls", '
    '"message": {"role": "assistant", "content": ""}}]}, "usage": '
    '{"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}, '
    '"request_id": "r2"}',
]


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = (
            json_data
            if isinstance(json_data, str)
            else json.dumps(json_data, ensure_ascii=False)
        )

    def json(self):
        if isinstance(self._json_data, str):
            raise ValueError("not json")
        return self._json_data


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200, body=""):
        self._lines = lines
        self.status_code = status_code
        self._body = body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body.encode()


class _FakeStreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class _FakeClient:
    """Stand-in for httpx.AsyncClient; queues canned responses."""

    requests = []
    _post_queue = []
    _stream_queue = []

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @classmethod
    def reset(cls):
        cls.requests = []
        cls._post_queue = []
        cls._stream_queue = []

    @classmethod
    def enqueue_post(cls, json_data, status_code=200):
        cls._post_queue.append(_FakeResponse(json_data, status_code))

    @classmethod
    def enqueue_stream(cls, lines, status_code=200, body=""):
        cls._stream_queue.append(_FakeStreamResponse(lines, status_code, body))

    async def post(self, url, json=None, headers=None):
        type(self).requests.append(
            {"url": url, "body": json, "headers": headers},
        )
        return type(self)._post_queue.pop(0)

    def stream(self, method, url, json=None, headers=None):
        type(self).requests.append(
            {
                "method": method,
                "url": url,
                "body": json,
                "headers": headers,
            },
        )
        return _FakeStreamCtx(type(self)._stream_queue.pop(0))


@pytest.fixture
def fake_http(monkeypatch):
    import httpx

    _FakeClient.reset()
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return _FakeClient


def _provider(**kwargs):
    kwargs.setdefault("model", "qwen-plus")
    kwargs.setdefault("api_key", "sk-x")
    return TongyiProvider(**kwargs)


# ---------------------------------------------------------------------------
# request shape
# ---------------------------------------------------------------------------


async def test_chat_posts_native_body(fake_http):
    fake_http.enqueue_post(NATIVE_RESPONSE)
    resp = await _provider().chat(MESSAGES, tools=TOOLS)

    cap = fake_http.requests[0]
    body = cap["body"]
    assert body["model"] == "qwen-plus"
    assert body["input"] == {"messages": MESSAGES}
    assert "messages" not in body and "tools" not in body
    params = body["parameters"]
    assert params["result_format"] == "message"
    assert "stream" not in params
    tool = params["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    # OpenAI-shaped tools pass through untouched
    assert resp.content == "hello there"


async def test_chat_stream_posts_native_body_with_sse_headers(fake_http):
    fake_http.enqueue_stream(NATIVE_SSE_LINES)
    provider = _provider()
    chunks = [c async for c in provider.chat_stream(MESSAGES, tools=TOOLS)]

    cap = fake_http.requests[0]
    params = cap["body"]["parameters"]
    assert params["result_format"] == "message"
    assert params["stream"] is True
    assert params["incremental_output"] is True
    assert params["tools"][0]["function"]["name"] == "get_weather"
    assert "stream_options" not in cap["body"]
    headers = cap["headers"]
    assert headers["X-DashScope-SSE"] == "enable"
    assert headers["Accept"] == "text/event-stream"
    assert headers["x-dashscope-sdk-client"].startswith("acli/")
    assert chunks


def test_openai_content_parts_converted_to_native():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAA"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "just "},
                {"type": "text", "text": "text"},
            ],
        },
    ]
    body = _provider()._build_request_body(  # pylint: disable=protected-access
        messages,
        None,
    )
    converted = body["input"]["messages"]
    # image list → native parts (multimodal-generation rejects image_url)
    assert converted[0]["content"] == [
        {"text": "what is this?"},
        {"image": "data:image/png;base64,AAA"},
    ]
    # text-only list flattens to a string (text-generation needs strings)
    assert converted[1]["content"] == "just text"


# ---------------------------------------------------------------------------
# base_url normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "given,expected",
    [
        (None, ROOT),
        ("https://dashscope.aliyuncs.com", ROOT),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", ROOT),
        ("https://dashscope.aliyuncs.com/api/v1", ROOT),
        ("https://dashscope.aliyuncs.com/", ROOT),
        ("http://127.0.0.1:9000", "http://127.0.0.1:9000"),
    ],
)
def test_base_url_normalized_to_service_root(given, expected):
    assert _provider(base_url=given).base_url == expected


# ---------------------------------------------------------------------------
# endpoint selection: multimodal-generation first, text-generation fallback
# ---------------------------------------------------------------------------


def test_multimodal_path_is_probed_first():
    """Pinned on purpose. The gateway binds each model to one aigc path and
    rejects the other with a 400 "url error" that lands in the prod log as
    InvalidParameter, so the probe order decides who pays it. acli's default
    model (qwen3.8-max) plus the vl/omni family are multimodal-bound; the
    classic text models are the minority. Flip this and every fallback
    assertion below flips with it."""
    assert _GENERATION_PATHS == (MULTIMODAL_GEN, TEXT_GEN)


async def test_chat_multimodal_model_needs_no_probe(fake_http):
    """The common case: first request already hits the right path, and the
    part-list content multimodal-generation returns is flattened."""
    fake_http.enqueue_post(MULTIMODAL_RESPONSE)
    resp = await _provider(model="qwen3.8-max").chat(MESSAGES)

    assert [r["url"] for r in fake_http.requests] == [ROOT + MULTIMODAL_GEN]
    assert resp.content == "hello there"
    assert resp.reasoning_content == "thinking..."
    assert resp.usage["cached_tokens"] == 5


async def test_stream_multimodal_model_needs_no_probe(fake_http):
    fake_http.enqueue_stream(MULTIMODAL_SSE_LINES)
    provider = _provider(model="qwen3.8-max")
    chunks = [c async for c in provider.chat_stream(MESSAGES)]

    assert [r["url"] for r in fake_http.requests] == [ROOT + MULTIMODAL_GEN]
    assert "th" in [c.delta_reasoning_content for c in chunks]
    assert [c.delta_content for c in chunks if c.delta_content] == [
        "Hel",
        "lo",
    ]
    assert chunks[-1].usage["total_tokens"] == 9


async def test_chat_falls_back_to_text_path(fake_http):
    fake_http.enqueue_post(URL_ERROR_BODY, status_code=400)
    fake_http.enqueue_post(NATIVE_RESPONSE)
    provider = _provider(model="qwen-plus")
    resp = await provider.chat(MESSAGES)

    assert [r["url"] for r in fake_http.requests] == [
        ROOT + MULTIMODAL_GEN,
        ROOT + TEXT_GEN,
    ]
    assert resp.content == "hello there"
    assert resp.usage["total_tokens"] == 18
    # working path is cached: the next call skips the 400 round-trip
    fake_http.enqueue_post(NATIVE_RESPONSE)
    await provider.chat(MESSAGES)
    assert len(fake_http.requests) == 3
    assert fake_http.requests[-1]["url"] == ROOT + TEXT_GEN


async def test_chat_no_fallback_on_other_400(fake_http):
    fake_http.enqueue_post(
        {"code": "InvalidParameter", "message": "bad temperature"},
        status_code=400,
    )
    with pytest.raises(RuntimeError, match="bad temperature"):
        await _provider().chat(MESSAGES)
    assert len(fake_http.requests) == 1


async def test_stream_falls_back_to_text_path(fake_http):
    fake_http.enqueue_stream(
        [],
        status_code=400,
        body=json.dumps(URL_ERROR_BODY, ensure_ascii=False),
    )
    fake_http.enqueue_stream(NATIVE_SSE_LINES)
    provider = _provider(model="qwen-plus")
    chunks = [c async for c in provider.chat_stream(MESSAGES)]

    assert [r["url"] for r in fake_http.requests] == [
        ROOT + MULTIMODAL_GEN,
        ROOT + TEXT_GEN,
    ]
    contents = [c.delta_content for c in chunks if c.delta_content]
    assert contents == ["Hel", "lo"]
    last = chunks[-1]
    assert last.finish_reason == "stop"
    assert last.usage["total_tokens"] == 13


async def test_stream_falls_back_on_sse_error_event(fake_http):
    """The url error also arrives as a 200 SSE error event (event:error),
    not a 400 — the fallback must trigger before any chunk is yielded."""
    fake_http.enqueue_stream(
        [
            "id:1",
            "event:error",
            ":HTTP_STATUS/400",
            "data:" + json.dumps(URL_ERROR_BODY, ensure_ascii=False),
        ],
    )
    fake_http.enqueue_stream(NATIVE_SSE_LINES)
    provider = _provider(model="qwen-plus")
    chunks = [c async for c in provider.chat_stream(MESSAGES)]

    assert [r["url"] for r in fake_http.requests] == [
        ROOT + MULTIMODAL_GEN,
        ROOT + TEXT_GEN,
    ]
    assert [c.delta_content for c in chunks if c.delta_content] == [
        "Hel",
        "lo",
    ]
    # the working path is cached only after a real chunk arrived
    assert provider._generation_path == TEXT_GEN


async def test_stream_sse_url_error_raises_after_paths_exhausted(fake_http):
    event = ["data:" + json.dumps(URL_ERROR_BODY, ensure_ascii=False)]
    fake_http.enqueue_stream(list(event))
    fake_http.enqueue_stream(list(event))
    provider = _provider()
    with pytest.raises(RuntimeError, match="url error"):
        async for _ in provider.chat_stream(MESSAGES):
            pass
    assert len(fake_http.requests) == 2


async def test_stream_no_fallback_after_content_started(fake_http):
    lines = NATIVE_SSE_LINES[:4] + [
        "data:" + json.dumps(URL_ERROR_BODY, ensure_ascii=False),
    ]
    fake_http.enqueue_stream(lines)
    provider = _provider()
    with pytest.raises(RuntimeError, match="url error"):
        async for _ in provider.chat_stream(MESSAGES):
            pass
    assert len(fake_http.requests) == 1


# ---------------------------------------------------------------------------
# learned-path cache: a probe costs a 400 in the prod gateway log, so the
# answer is remembered per model — in-process and on disk
# ---------------------------------------------------------------------------


async def test_learned_path_skips_the_probe_in_a_new_instance(fake_http):
    """A provider is constructed per session start, per model switch and per
    vision call; only the first of those may pay the probe."""
    fake_http.enqueue_post(URL_ERROR_BODY, status_code=400)
    fake_http.enqueue_post(NATIVE_RESPONSE)
    await _provider().chat(MESSAGES)
    assert len(fake_http.requests) == 2

    fake_http.enqueue_post(NATIVE_RESPONSE)
    await _provider().chat(MESSAGES)
    assert len(fake_http.requests) == 3
    assert fake_http.requests[-1]["url"] == ROOT + TEXT_GEN


async def test_learned_path_survives_a_new_process(fake_http, monkeypatch):
    """Bench and CI runs execute in throwaway containers, so the binding has
    to outlive the process to be worth anything."""
    fake_http.enqueue_post(URL_ERROR_BODY, status_code=400)
    fake_http.enqueue_post(NATIVE_RESPONSE)
    await _provider().chat(MESSAGES)
    assert len(fake_http.requests) == 2

    # Simulate a fresh process: empty in-process cache, same home directory.
    monkeypatch.setattr(TongyiProvider, "_path_cache", None)
    fake_http.enqueue_post(NATIVE_RESPONSE)
    await _provider().chat(MESSAGES)
    assert len(fake_http.requests) == 3
    assert fake_http.requests[-1]["url"] == ROOT + TEXT_GEN


async def test_path_cache_is_written_next_to_the_global_config(fake_http):
    from dashscope.acli import config as config_module

    fake_http.enqueue_post(URL_ERROR_BODY, status_code=400)
    fake_http.enqueue_post(NATIVE_RESPONSE)
    await _provider().chat(MESSAGES)

    cache_file = config_module.CONFIG_DIR / _PATH_CACHE_NAME
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {
        "qwen-plus": TEXT_GEN,
    }


def test_path_cache_drops_unknown_paths(monkeypatch):
    """A model the gateway rebinds, or a hand-edited cache, must fall back to
    probing rather than POST to a path that no longer exists."""
    from dashscope.acli import config as config_module

    cache_file = config_module.CONFIG_DIR / _PATH_CACHE_NAME
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"qwen-plus": "/api/v1/services/aigc/bogus/generation"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(TongyiProvider, "_path_cache", None)

    assert _provider()._generation_path == MULTIMODAL_GEN


def test_path_cache_survives_a_corrupt_file(monkeypatch):
    from dashscope.acli import config as config_module

    cache_file = config_module.CONFIG_DIR / _PATH_CACHE_NAME
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("not json at all", encoding="utf-8")
    monkeypatch.setattr(TongyiProvider, "_path_cache", None)

    assert _provider()._generation_path == MULTIMODAL_GEN


# ---------------------------------------------------------------------------
# non-stream response parsing
# ---------------------------------------------------------------------------


async def test_chat_parses_native_response(fake_http):
    fake_http.enqueue_post(NATIVE_RESPONSE)
    resp = await _provider().chat(MESSAGES)
    assert resp.content == "hello there"
    assert resp.tool_calls == []
    assert resp.usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
        "cached_tokens": 0,
    }


async def test_chat_parses_native_tool_calls(fake_http):
    fake_http.enqueue_post(NATIVE_TOOL_RESPONSE)
    resp = await _provider().chat(MESSAGES, tools=TOOLS)
    assert len(resp.tool_calls) == 1
    call = resp.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert call.arguments == {"city": "hz"}


async def test_chat_raises_on_native_error_body(fake_http):
    fake_http.enqueue_post(
        {"code": "InvalidApiKey", "message": "bad key", "request_id": "r3"},
    )
    with pytest.raises(RuntimeError, match="InvalidApiKey"):
        await _provider().chat(MESSAGES)


async def test_anthropic_protocol_over_native_route(fake_http):
    fake_http.enqueue_post(NATIVE_TOOL_RESPONSE)
    provider = _provider(protocol="anthropic")
    resp = await provider.chat(
        [{"role": "user", "content": "weather?"}],
        tools=TOOLS,
    )
    # request was converted anthropic->openai then wrapped natively
    body = fake_http.requests[0]["body"]
    assert body["input"]["messages"][0]["role"] == "user"
    assert body["parameters"]["tools"][0]["function"]["name"] == "get_weather"
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "hz"}


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


async def test_stream_parses_native_chunks(fake_http):
    fake_http.enqueue_stream(NATIVE_SSE_LINES)
    provider = _provider()
    chunks = [c async for c in provider.chat_stream(MESSAGES)]

    contents = [c.delta_content for c in chunks if c.delta_content]
    assert contents == ["Hel", "lo"]
    last = chunks[-1]
    assert last.finish_reason == "stop"
    assert last.usage == {
        "input_tokens": 11,
        "output_tokens": 2,
        "total_tokens": 13,
        "cached_tokens": 0,
    }


async def test_stream_accumulates_native_tool_calls(fake_http):
    fake_http.enqueue_stream(NATIVE_SSE_TOOL_LINES)
    provider = _provider()
    chunks = [c async for c in provider.chat_stream(MESSAGES, tools=TOOLS)]

    tool_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_chunks) == 1
    call = tool_chunks[0].tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "get_weather"
    assert call.arguments == {"city": "hz"}
    assert tool_chunks[0].finish_reason == "tool_calls"
    assert tool_chunks[0].usage["input_tokens"] == 5


async def test_stream_raises_on_midstream_error(fake_http):
    fake_http.enqueue_stream(
        [
            NATIVE_SSE_LINES[3],
            'data:{"code": "Throttling.AllocationQuotaExceeded", "message": '
            '"slow down", "request_id": "r9"}',
        ],
    )
    provider = _provider()
    with pytest.raises(RuntimeError, match="AllocationQuotaExceeded"):
        async for _ in provider.chat_stream(MESSAGES):
            pass


async def test_stream_flushes_orphan_tool_calls_on_truncation(fake_http):
    # stream ends (connection closed) without a finish chunk
    fake_http.enqueue_stream(NATIVE_SSE_TOOL_LINES[:3])
    provider = _provider()
    chunks = [c async for c in provider.chat_stream(MESSAGES, tools=TOOLS)]
    tool_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].tool_calls[0].arguments == {"city": "hz"}
    assert tool_chunks[0].finish_reason == "stop"


def test_request_body_is_json_serializable():
    body = _provider()._build_request_body(  # pylint: disable=protected-access
        MESSAGES,
        TOOLS,
        stream=True,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(json.dumps(body))
    assert parsed["parameters"]["response_format"] == {
        "type": "json_object",
    }
