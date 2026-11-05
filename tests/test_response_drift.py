# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end tests for fault-tolerant response parsing (T2).

These drive a real ``Client`` against a mock transport, so they cover the
whole path -- protocol adapter, ``parse_model``, the streaming wrapper and
the resource parse helpers -- rather than the model layer alone. The unit
behaviour of the salvage algorithm lives in ``tests/test_parse_model.py``.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from dashscope import AsyncClient, Client
from dashscope.errors import APIResponseValidationError, DashScopeError
from dashscope.resources._jobs import Jobs
from dashscope.transport.sse import AsyncStream, Stream

BASE_URL = "https://api.test.local/api/v1"

# The drift under test: ``usage`` retyped from an object to a string. It is
# the shape that originally escaped as a bare pydantic ValidationError.
NATIVE_DRIFT_BODY = {
    "request_id": "r-drift",
    "output": {"text": "hi"},
    "usage": "unlimited",
}
OPENAI_DRIFT_BODY = {
    "id": "chatcmpl-drift",
    "model": "qwen-plus",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        },
    ],
    "usage": "unlimited",
}
ANTHROPIC_DRIFT_BODY = {
    "id": "msg-drift",
    "model": "claude-x",
    "content": [{"type": "text", "text": "hi"}],
    "usage": "unlimited",
}


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def _sse(events):
    out = []
    for index, data in enumerate(events, 1):
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append(f"id: {index}\ndata: {payload}\n\n")
    return "".join(out)


def _json_response(body):
    return httpx.Response(200, json=body)


def _drift_records(caplog):
    return [r for r in caplog.records if "response drift" in r.getMessage()]


# ---------------------------------------------------------------------------
# every protocol degrades a drifted body instead of raising
# ---------------------------------------------------------------------------


def test_native_chat_degrades_a_drifted_usage(caplog):
    client = _make_client(lambda request: _json_response(NATIVE_DRIFT_BODY))
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
        )
    assert completion.output_text == "hi"
    assert completion.request_id == "r-drift"
    assert completion.usage is None
    assert not completion.is_error
    assert _drift_records(caplog), "degradation must be logged"


def test_openai_chat_degrades_a_drifted_usage(caplog):
    client = _make_client(
        lambda request: _json_response(OPENAI_DRIFT_BODY),
        protocol="openai",
    )
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
        )
    assert completion.output_text == "hi"
    assert completion.usage is None
    assert _drift_records(caplog)


def test_anthropic_chat_degrades_a_drifted_usage(caplog):
    client = _make_client(
        lambda request: _json_response(ANTHROPIC_DRIFT_BODY),
        protocol="anthropic",
    )
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        completion = client.chat.completions.create(
            model="claude-x",
            messages="hi",
        )
    assert completion.output_text == "hi"
    assert completion.usage is None
    assert _drift_records(caplog)


def test_a_clean_body_still_parses_and_logs_nothing(caplog):
    body = dict(NATIVE_DRIFT_BODY)
    body["usage"] = {"input_tokens": 1, "output_tokens": 2}
    client = _make_client(lambda request: _json_response(body))
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
        )
    assert completion.usage.input_tokens == 1
    assert completion.usage.output_tokens == 2
    assert not _drift_records(caplog)


# ---------------------------------------------------------------------------
# strict mode: the drift becomes an error the caller can catch
# ---------------------------------------------------------------------------


def test_strict_validation_raises_an_error_a_caller_can_catch(caplog):
    # The acceptance criterion for the whole feature: with the flag on, a
    # drifted body surfaces inside ``except DashScopeError`` rather than as
    # a pydantic ValidationError the caller has no reason to import.
    client = _make_client(
        lambda request: _json_response(NATIVE_DRIFT_BODY),
        strict_validation=True,
    )
    caught = None
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        try:
            client.chat.completions.create(model="qwen-plus", messages="hi")
        except DashScopeError as exc:
            caught = exc
    assert isinstance(caught, APIResponseValidationError)
    assert isinstance(caught.cause, ValidationError)
    assert caught.body["usage"] == "unlimited"
    # The chat path parses a dict built from the body, not the response
    # object, so the request id is recovered from the payload instead.
    assert caught.request_id == "r-drift"
    assert _drift_records(caplog)


# One drifted body per wire format, with the model name its adapter expects.
_DRIFT_CASES = [
    ("native", NATIVE_DRIFT_BODY, "qwen-plus"),
    ("openai", OPENAI_DRIFT_BODY, "qwen-plus"),
    ("anthropic", ANTHROPIC_DRIFT_BODY, "claude-x"),
]


@pytest.mark.parametrize("protocol, body, model", _DRIFT_CASES)
def test_strict_validation_applies_to_every_protocol(protocol, body, model):
    client = _make_client(
        lambda request: _json_response(body),
        protocol=protocol,
        strict_validation=True,
    )
    with pytest.raises(APIResponseValidationError):
        client.chat.completions.create(model=model, messages="hi")


def test_strict_validation_raises_even_under_raise_on_error_false():
    # Deliberate: the flag is an explicit opt-in for contract testing and is
    # orthogonal to the compat mode, which governs *service* errors only.
    client = _make_client(
        lambda request: _json_response(NATIVE_DRIFT_BODY),
        strict_validation=True,
        raise_on_error=False,
    )
    with pytest.raises(APIResponseValidationError):
        client.chat.completions.create(model="qwen-plus", messages="hi")


# ---------------------------------------------------------------------------
# streaming: one bad event must not end the iteration
# ---------------------------------------------------------------------------


_CHUNKED_EVENTS = [
    {
        "id": "c-drift",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "he"}},
        ],
    },
    # A number where the model declares ``Optional[str]``.
    {"id": "c-drift", "choices": [{"index": 0, "delta": {"content": 123}}]},
    {
        "id": "c-drift",
        "choices": [
            {"index": 0, "delta": {"content": "llo"}, "finish_reason": "stop"},
        ],
    },
    "[DONE]",
]


def _stream_handler(request):
    return httpx.Response(
        200,
        text=_sse(_CHUNKED_EVENTS),
        headers={"content-type": "text/event-stream"},
    )


def test_streaming_continues_past_a_corrupt_delta(caplog):
    client = _make_client(_stream_handler, protocol="openai")
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        stream = client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
            stream=True,
        )
        chunks = list(stream)
    # All three events arrived; the corrupt one degraded to an empty delta
    # instead of tearing down the iterator mid-flight.
    assert len(chunks) == 3
    assert chunks[0].choices[0].delta.content == "he"
    assert chunks[1].choices[0].delta.content is None
    assert chunks[2].choices[0].delta.content == "llo"
    assert _drift_records(caplog)


@pytest.mark.asyncio
async def test_async_streaming_continues_past_a_corrupt_delta(caplog):
    client = _make_async_client(_stream_handler, protocol="openai")
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        stream = await client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
            stream=True,
        )
        contents = [chunk.choices[0].delta.content async for chunk in stream]
    assert contents == ["he", None, "llo"]
    assert _drift_records(caplog)
    await client.aclose()


def test_get_final_message_survives_a_corrupt_delta():
    client = _make_client(_stream_handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    final = stream.get_final_message()
    assert final.output_text == "hello"
    assert isinstance(final.choices[0].message.role, str)


@pytest.mark.asyncio
async def test_async_get_final_message_survives_a_corrupt_delta():
    client = _make_async_client(_stream_handler, protocol="openai")
    stream = await client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    final = await stream.get_final_message()
    assert final.output_text == "hello"
    await client.aclose()


def test_strict_validation_raises_mid_stream():
    client = _make_client(
        _stream_handler,
        protocol="openai",
        strict_validation=True,
    )
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    with pytest.raises(APIResponseValidationError) as caught:
        list(stream)
    # The parser is bound to the stream's envelope, so a mid-stream
    # failure still names the response it came from.
    assert caught.value.response is not None
    assert caught.value.response.status_code == 200


# ---------------------------------------------------------------------------
# the transport-level net: a parser that validates on its own is converted
# ---------------------------------------------------------------------------


class _ValidatingModel(BaseModel):
    """Stands in for a resource whose parser has no salvage step."""

    total: int


def _validating_parse(body):
    return _ValidatingModel.model_validate(body)


def test_stream_converts_a_bare_validation_error():
    # Chat parsers degrade, so this net only fires for a parser that
    # validates directly (today: audio and tingwu streams). It still
    # matters -- mid-iteration, after earlier chunks were consumed, a raw
    # pydantic error would slip past the caller's ``except DashScopeError``.
    response = httpx.Response(
        200,
        text=_sse([{"total": 1}, {"total": "many"}]),
        headers={"content-type": "text/event-stream"},
    )
    with pytest.raises(APIResponseValidationError) as caught:
        list(Stream(response, _validating_parse))
    assert isinstance(caught.value, DashScopeError)
    assert caught.value.body == {"total": "many"}
    assert isinstance(caught.value.cause, ValidationError)


@pytest.mark.asyncio
async def test_async_stream_converts_a_bare_validation_error():
    response = httpx.Response(
        200,
        text=_sse([{"total": 1}, {"total": "many"}]),
        headers={"content-type": "text/event-stream"},
    )
    with pytest.raises(APIResponseValidationError) as caught:
        async for _ in AsyncStream(response, _validating_parse):
            pass
    assert caught.value.body == {"total": "many"}


# ---------------------------------------------------------------------------
# resources: models, files, jobs
# ---------------------------------------------------------------------------


def test_models_list_degrades_a_null_total_count(caplog):
    def handler(request):
        return _json_response(
            {
                "request_id": "r-models",
                "output": {
                    "page_no": 1,
                    "page_size": 10,
                    "total": None,
                    "models": [{"model": "qwen-plus", "open_source": "maybe"}],
                },
            },
        )

    client = _make_client(handler)
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        page = client.models.list()
    assert page.total_count == 0
    assert page.models[0].model == "qwen-plus"
    assert _drift_records(caplog)


def test_models_get_degrades_a_drifted_model(caplog):
    def handler(request):
        return _json_response(
            {
                "request_id": "r-model",
                "output": {
                    "models": [
                        {"model": "qwen-plus", "open_source": "maybe"},
                    ],
                },
            },
        )

    client = _make_client(handler)
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        model = client.models.get("qwen-plus")
    assert model.model == "qwen-plus"
    assert model.open_source is None
    assert _drift_records(caplog)


def test_files_list_degrades_a_drifted_size(caplog):
    def handler(request):
        return _json_response(
            {
                "request_id": "r-files",
                "data": {
                    "page_no": 1,
                    "page_size": 10,
                    "total": 1,
                    "files": [
                        {
                            "file_id": "f-1",
                            "name": "t.jsonl",
                            "size": "12kb",
                        },
                    ],
                },
            },
        )

    client = _make_client(handler)
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        listed = client.files.list()
    assert listed.files[0].file_id == "f-1"
    # ``size`` is Optional, so the repair is its declared default rather
    # than a zeroed int -- a missing count reads as unknown, not as empty.
    assert listed.files[0].size is None
    assert listed.total_count == 1
    assert _drift_records(caplog)


def test_files_get_degrades_a_drifted_uploaded_at(caplog):
    def handler(request):
        return _json_response(
            {
                "request_id": "r-file",
                "data": {
                    "file_id": "f-1",
                    "uploaded_at": 1700000000,
                    "purpose": "fine-tune",
                },
            },
        )

    client = _make_client(handler)
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        got = client.files.get("f-1")
    assert got.file_id == "f-1"
    assert got.uploaded_at is None
    assert _drift_records(caplog)


def test_job_fetch_degrades_a_drifted_task_status(caplog):
    def handler(request):
        return _json_response(
            {
                "request_id": "r-job",
                "output": {
                    "task_id": "task-1",
                    "task_status": 200,
                    "results": [{"url": "https://example.test/a.png"}],
                },
                "usage": {"input_tokens": "many"},
            },
        )

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        job = jobs.fetch("task-1")
    # ``status`` is aliased from ``task_status``, so the repair has to
    # resolve the alias to find the field it is fixing.
    assert job.task_id == "task-1"
    assert job.status == ""
    assert not job.succeeded
    assert job.usage.input_tokens == 0
    # A drifted sibling must not cost the caller the healthy fields.
    assert job.image_urls == ["https://example.test/a.png"]
    assert _drift_records(caplog)


def test_job_list_degrades_a_drifted_task(caplog):
    def handler(request):
        return _json_response(
            {
                "request_id": "r-jobs",
                "page_no": 1,
                "page_size": 10,
                "total": "many",
                "tasks": [
                    {"task_id": "task-1", "task_status": "SUCCEEDED"},
                    {"task_id": "task-2", "submit_time": 1700000000},
                ],
            },
        )

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        page = jobs.list()
    assert page.total == 0
    assert page.tasks[0].task_id == "task-1"
    assert page.tasks[0].succeeded
    # The drifted entry keeps its index: a caller paging by position would
    # be misaligned if the repair dropped it instead.
    assert page.tasks[1].task_id == "task-2"
    assert page.tasks[1].submitted_at is None
    assert _drift_records(caplog)


@pytest.mark.asyncio
async def test_async_resources_degrade_too(caplog):
    def handler(request):
        return _json_response(
            {
                "request_id": "r-files",
                "data": {"files": [{"file_id": "f-1", "size": "12kb"}]},
            },
        )

    client = _make_async_client(handler)
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        listed = await client.files.list()
    # ``size`` is Optional, so the repair is its declared default rather
    # than a zeroed int -- a missing count reads as unknown, not as empty.
    assert listed.files[0].size is None
    assert _drift_records(caplog)
    await client.aclose()


def test_strict_validation_covers_the_resource_layer():
    def handler(request):
        return _json_response(
            {
                "request_id": "r-files",
                "data": {"files": [{"file_id": "f-1", "size": "12kb"}]},
            },
        )

    client = _make_client(handler, strict_validation=True)
    with pytest.raises(APIResponseValidationError) as caught:
        client.files.list()
    # ``body`` is the payload handed to the model, which a resource helper
    # has already unwrapped out of the wire envelope -- still enough to see
    # exactly which field drifted.
    assert caught.value.body["files"][0]["size"] == "12kb"
    # The response is threaded through the helper too, so the caller can
    # read the status and headers of the body that failed.
    assert caught.value.response is not None
    assert caught.value.response.status_code == 200


# ---------------------------------------------------------------------------
# drift that used to escape as a bare ValueError / TypeError / AttributeError
# ---------------------------------------------------------------------------


def test_anthropic_streaming_survives_a_drifted_usage_bag():
    # ``dict.update()`` on a retyped bag raised ValueError (a string) or
    # TypeError (a number) mid-iteration, outside any handler the caller
    # could write -- and after earlier chunks were already consumed.
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg-1", "model": "m", "usage": "unlimited"},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "message_delta", "usage": 5},
        {"type": "message_stop"},
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, protocol="anthropic")
    stream = client.chat.completions.create(
        model="claude-x",
        messages="hi",
        stream=True,
    )
    assert "".join(stream.text_stream) == "hi"


def test_a_non_mapping_choice_element_degrades_instead_of_raising():
    # ``dict(raw)`` on a bare string raised ValueError before parse_model
    # ever saw the body.
    def handler(request):
        return _json_response(
            {
                "request_id": "r-drift",
                "output": {
                    "choices": [
                        "oops",
                        {
                            "index": 1,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        },
                    ],
                },
            },
        )

    client = _make_client(handler)
    result = client.chat.completions.create(model="qwen-plus", messages="hi")
    # The bad element keeps its slot rather than being dropped, so the
    # healthy choice stays at the index the server put it at.
    assert len(result.choices) == 2
    assert result.choices[1].message.content == "ok"


def test_a_non_mapping_choice_element_keeps_a_stream_positionally_aligned():
    def handler(request):
        return httpx.Response(
            200,
            text=_sse(
                [
                    {
                        "id": "c-drift",
                        "choices": [
                            "oops",
                            {"index": 1, "delta": {"content": "ok"}},
                        ],
                    },
                    "[DONE]",
                ],
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    (chunk,) = list(stream)
    assert [choice.index for choice in chunk.choices] == [0, 1]
    assert chunk.choices[1].delta.content == "ok"


def test_a_non_mapping_content_block_is_skipped_by_the_anthropic_adapter():
    def handler(request):
        return _json_response(
            {
                "id": "msg-drift",
                "model": "claude-x",
                "content": ["oops", {"type": "text", "text": "hi"}],
            },
        )

    client = _make_client(handler, protocol="anthropic")
    result = client.chat.completions.create(model="claude-x", messages="hi")
    assert result.output_text == "hi"


def test_a_non_mapping_output_envelope_degrades_instead_of_raising():
    def handler(request):
        return _json_response({"request_id": "r-drift", "output": "oops"})

    client = _make_client(handler)
    result = client.chat.completions.create(model="qwen-plus", messages="hi")
    assert result.request_id == "r-drift"
    assert result.choices == []


def test_strict_validation_reaches_the_image_sync_path():
    # ``images.generate`` answers synchronously for multimodal-protocol
    # models and builds its Job through a helper of its own; without the
    # flag threaded there, strict mode was silently ignored on that path.
    def handler(request):
        return _json_response(
            {
                "request_id": "r-img",
                "submit_time": 5,
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": [{"text": "hi"}]},
                        },
                    ],
                },
            },
        )

    client = _make_client(handler, strict_validation=True)
    with pytest.raises(APIResponseValidationError) as caught:
        client.images.generate(model="wan2.6-t2i", prompt="a cat")
    assert caught.value.response is not None

    degraded = _make_client(handler).images.generate(
        model="wan2.6-t2i",
        prompt="a cat",
    )
    assert degraded.succeeded
    assert degraded.submitted_at is None


# ---------------------------------------------------------------------------
# the whole resource layer, one drifted body each
# ---------------------------------------------------------------------------


def _resolve(obj, path):
    """Read ``"deployments[0].capacity"`` off a parsed model."""
    current = obj
    for part in path.replace("[", ".[").split("."):
        if not part:
            continue
        if part.startswith("["):
            current = current[int(part[1:-1])]
        else:
            current = getattr(current, part)
    return current


# Every resource that parses a wire body gets one row: the field named by
# ``path`` arrives with the wrong type, so ``parse_model`` has to replace it
# with the model's declared default. A bare ``pydantic.ValidationError``
# escaping here is the failure mode -- it is not a ``DashScopeError``, so no
# caller's ``except`` clause would catch it.
_RESOURCE_DRIFT_CASES = [
    (
        "deployments.get",
        lambda client: client.deployments.get("dep-1"),
        {
            "request_id": "req-1",
            "output": {
                "deployed_model": "dep-1",
                "model_name": "ft-model",
                "status": "RUNNING",
                "capacity": "lots",
            },
        },
        "capacity",
        None,
    ),
    (
        "deployments.list",
        lambda client: client.deployments.list(),
        {
            "request_id": "req-2",
            "output": {
                "page_no": 1,
                "page_size": 10,
                "total": "many",
                "deployments": [
                    {
                        "deployed_model": "dep-1",
                        "model_name": "ft-model",
                        "status": "RUNNING",
                    },
                ],
            },
        },
        "total_count",
        0,
    ),
    (
        "fine_tunes.get",
        lambda client: client.fine_tunes.get("ft-1"),
        {
            "request_id": "req-1",
            "output": {"job_id": "ft-1", "status": "RUNNING"},
            "usage": "unlimited",
        },
        "usage",
        None,
    ),
    (
        "fine_tunes.logs",
        lambda client: client.fine_tunes.logs("ft-1"),
        {"request_id": "req-logs", "output": {"logs": 123}},
        "content",
        "",
    ),
    (
        "rl.jobs.fetch",
        lambda client: client.rl.jobs.fetch("ft-1"),
        {
            "request_id": "req-3",
            "output": {"job_id": "ft-1", "status": "RUNNING"},
            "usage": "unlimited",
        },
        "usage",
        None,
    ),
    (
        "rerank.create",
        lambda client: client.rerank.create(
            model="gte-rerank-v2",
            query="q",
            documents=["a", "b"],
        ),
        {
            "request_id": "req-r",
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ],
            },
            "usage": "unlimited",
        },
        "usage",
        None,
    ),
    (
        "tokenization.count",
        lambda client: client.tokenization.count(
            model="qwen-plus",
            prompt="hello world",
        ),
        {
            "request_id": "req-t",
            "usage": {"input_tokens": 7},
            "output": {"token_ids": 5},
        },
        "token_ids",
        None,
    ),
    (
        "tingwu.call",
        lambda client: client.tingwu.call(
            model="tingwu-turbo",
            input={"source_language": "cn", "task": "summarization"},
        ),
        {
            "request_id": "req-1",
            "output": {"TaskId": "t-1", "Result": {"summary": "ok"}},
            "usage": "unlimited",
        },
        "usage",
        None,
    ),
    (
        "apps.completions.create",
        lambda client: client.apps.completions.create(
            app_id="app-1",
            prompt="hi",
        ),
        {
            "request_id": "req-1",
            "output": {"text": "Hello from app", "finish_reason": "stop"},
            "usage": "unlimited",
        },
        "usage",
        None,
    ),
    (
        "decision.create",
        lambda client: client.decision.create(
            model="decision-model-preview",
            state="text",
            questions={
                "escalate": {"type": "noul", "instructions": "?"},
            },
        ),
        {
            "model": "decision-model-preview",
            "request_id": "r-1",
            "answers": {"escalate": {"type": "noul", "noul": 0.99}},
            "latency_ms": "fast",
        },
        "latency_ms",
        None,
    ),
    (
        "audio.phrases.list",
        lambda client: client.audio.phrases.list(),
        {
            "request_id": "r-1",
            "output": {
                "jobs": [{"job_id": "p-1", "status": "SUCCEEDED"}],
                "total": "many",
            },
        },
        "total_count",
        0,
    ),
    (
        "audio.vocabularies.create",
        lambda client: client.audio.vocabularies.create(
            target_model="paraformer-v2",
            prefix="med",
            vocabulary=[{"text": "x", "weight": 4}],
        ),
        {
            "request_id": "req-v1",
            "output": {"vocabulary_id": "vocab-1", "vocabulary": "hot"},
        },
        "vocabulary",
        None,
    ),
    (
        "audio.tts.http.synthesize",
        lambda client: client.audio.tts.http.synthesize(
            "你好",
            model="qwen-audio-3.1-tts-flash",
            voice="longanfengyue_v3.1",
        ),
        {
            "request_id": "req-body-1",
            "output": {
                "audio": {
                    "url": "https://cdn.test/a.wav",
                    "id": "audio-1",
                    "expires_at": 1720000000,
                },
            },
            "message": 123,
        },
        "message",
        None,
    ),
]


@pytest.mark.parametrize(
    "call, body, path, expected",
    [case[1:] for case in _RESOURCE_DRIFT_CASES],
    ids=[case[0] for case in _RESOURCE_DRIFT_CASES],
)
def test_every_resource_degrades_a_drifted_field(
    call,
    body,
    path,
    expected,
    caplog,
):
    client = _make_client(lambda request: _json_response(body))
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        result = call(client)
    assert _resolve(result, path) == expected
    assert _drift_records(caplog), "degradation must be logged"


@pytest.mark.parametrize(
    "call, body, path, expected",
    [case[1:] for case in _RESOURCE_DRIFT_CASES],
    ids=[case[0] for case in _RESOURCE_DRIFT_CASES],
)
def test_every_resource_raises_in_strict_mode(call, body, path, expected):
    # The point of the flag: a contract test wants the drift surfaced, not
    # papered over -- and it must surface as something catchable.
    client = _make_client(
        lambda request: _json_response(body),
        strict_validation=True,
    )
    with pytest.raises(APIResponseValidationError) as caught:
        call(client)
    assert caught.value.response is not None
    assert caught.value.response.status_code == 200


@pytest.mark.parametrize(
    "call, body, path, expected",
    [case[1:] for case in _RESOURCE_DRIFT_CASES],
    ids=[case[0] for case in _RESOURCE_DRIFT_CASES],
)
def test_drift_still_degrades_under_raise_on_error_false(
    call,
    body,
    path,
    expected,
    caplog,
):
    # Compat mode is about *service* errors; a 200 body with a drifted field
    # must degrade the same way either way. This is also the only test that
    # notices a resource dropping ``client=`` when it calls ``request_api``,
    # which would silently reset ``raise_on_error`` to its default.
    client = _make_client(
        lambda request: _json_response(body),
        raise_on_error=False,
    )
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        result = call(client)
    assert not result.is_error
    assert _resolve(result, path) == expected
    assert _drift_records(caplog)


# ---------------------------------------------------------------------------
# streamed frames: a drift partway through must not end the iteration
# ---------------------------------------------------------------------------


def test_a_drifted_sse_frame_degrades_instead_of_breaking_the_stream(caplog):
    # ``stream_events`` parses each frame after the earlier ones were already
    # handed to the caller, so a bare ValidationError used to land
    # mid-iteration with nothing the caller could have caught up front.
    sse = (
        'id: 1\ndata: {"status": "RUNNING"}\n\n'
        # ``output`` is declared ``Union[Dict, str, None]``; a bare number
        # matches none of the three.
        "id: 2\ndata: 123\n\n"
        'id: 3\ndata: {"status": "SUCCEEDED"}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
        events = list(client.fine_tunes.stream_events("ft-1"))
    assert [event.output for event in events] == [
        {"status": "RUNNING"},
        None,
        {"status": "SUCCEEDED"},
    ]
    assert _drift_records(caplog)


def test_a_drifted_sse_frame_raises_in_strict_mode():
    sse = 'id: 1\ndata: {"status": "RUNNING"}\n\nid: 2\ndata: 123\n\n'

    def handler(request):
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, strict_validation=True)
    events = client.fine_tunes.stream_events("ft-1")
    assert next(events).output == {"status": "RUNNING"}
    # The first frame was already consumed, so this is exactly the case where
    # an uncatchable exception would have been worst.
    with pytest.raises(APIResponseValidationError) as caught:
        next(events)
    # The frame's response is the stream itself, which is still open.
    assert caught.value.response is not None


class _ScriptedSocket:
    """Websocket stub replaying a queued inbox; the task id is echoed back."""

    def __init__(self, inbox):
        self.sent = []
        self._inbox = list(inbox)
        self.closed = False

    def send(self, payload, opcode=1):
        self.sent.append(payload)

    def recv(self):
        if not self._inbox:
            return None
        raw = self._inbox.pop(0)
        if isinstance(raw, str):
            event = json.loads(raw)
            header = event.get("header") if isinstance(event, dict) else None
            if isinstance(header, dict) and header.get("task_id") == "t":
                sent = json.loads(self.sent[0])
                header["task_id"] = sent["header"]["task_id"]
                return json.dumps(event)
        return raw

    def close(self):
        self.closed = True

    def settimeout(self, seconds):
        pass


def _ws_event(name, payload=None):
    return json.dumps(
        {
            "header": {"event": name, "task_id": "t", "request_id": "r"},
            "payload": payload or {},
        },
    )


def _recognition(client, inbox):
    socket = _ScriptedSocket(inbox)
    return client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connector=lambda url, headers: socket,
    )


def _ws_client(**kwargs):
    # ``WsSession`` derives the websocket endpoint from the workspace, so the
    # WS tests need one even though no HTTP request is ever made.
    kwargs.setdefault("workspace", "ws-test")
    return _make_client(lambda request: _json_response({}), **kwargs)


def test_a_drifted_websocket_frame_degrades(caplog):
    # ``Recognition`` keeps the live config rather than a snapshot of the
    # flag, so a session opened before ``strict_validation`` changed still
    # reads the current setting at parse time.
    client = _ws_client()
    with _recognition(
        client,
        [
            _ws_event("task-started"),
            _ws_event("result-generated", {"output": "oops"}),
            _ws_event(
                "result-generated",
                {"output": {"sentence": {"text": "你好", "end_time": 9}}},
            ),
            _ws_event("task-finished"),
        ],
    ) as recognition:
        recognition.stop()
        with caplog.at_level(logging.DEBUG, logger="dashscope._logging"):
            results = list(recognition.results())
    assert results[0].output == {}
    assert results[1].get_sentence()["text"] == "你好"
    assert _drift_records(caplog)


def test_a_drifted_websocket_frame_raises_in_strict_mode():
    client = _ws_client(strict_validation=True)
    with _recognition(
        client,
        [
            _ws_event("task-started"),
            _ws_event("result-generated", {"output": "oops"}),
            _ws_event("task-finished"),
        ],
    ) as recognition:
        recognition.stop()
        with pytest.raises(APIResponseValidationError) as caught:
            list(recognition.results())
    # Deliberate: a websocket frame has no httpx.Response to carry, and
    # ``APIResponseValidationError.response=None`` is a supported shape.
    assert caught.value.response is None


def test_a_non_dict_websocket_payload_does_not_crash_the_session():
    # ``events()`` timed the first package off
    # ``payload["output"]["sentence"]`` and ``results()`` did
    # ``dict(payload)`` -- a payload that drifted to a bare string raised
    # AttributeError / ValueError before ``parse_model`` ever ran, and no
    # ``except DashScopeError`` catches either.
    client = _ws_client()
    with _recognition(
        client,
        [
            _ws_event("task-started"),
            json.dumps(
                {
                    "header": {
                        "event": "result-generated",
                        "task_id": "t",
                        "request_id": "r",
                    },
                    "payload": "oops",
                },
            ),
            _ws_event(
                "result-generated",
                {"output": {"sentence": {"text": "你好", "end_time": 9}}},
            ),
            _ws_event("task-finished"),
        ],
    ) as recognition:
        recognition.stop()
        results = list(recognition.results())
    assert results[0].output == {}
    assert results[1].get_sentence()["text"] == "你好"
