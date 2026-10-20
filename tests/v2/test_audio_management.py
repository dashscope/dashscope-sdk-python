# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the audio management plane (M5).

``client.audio.vocabularies`` / ``phrases`` / ``voices`` /
``voice_design`` -- 1.x ``VocabularyService`` / ``AsrPhraseManager`` /
``VoiceEnrollmentService`` plus the qwen voice-design REST family.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import Client
from dashscope.v2.errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)

BASE_URL = "https://api.test.local/api/v1"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


# ── vocabularies (1.x VocabularyService) ──────────────────────────────


def test_vocabularies_crud_wire():
    calls = []

    def handler(request):
        body = json.loads(request.content.decode())
        calls.append((request.url.path, body))
        action = body["input"]["action"]
        if action == "create_vocabulary":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-v1",
                    "output": {"vocabulary_id": "vocab-1"},
                },
            )
        if action == "list_vocabulary":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "vocabulary_list": [
                            {"vocabulary_id": "vocab-1", "prefix": "med"},
                        ],
                    },
                },
            )
        if action == "query_vocabulary":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "vocabulary_id": "vocab-1",
                        "vocabulary": [{"text": "心肌梗死", "weight": 4}],
                    },
                },
            )
        return httpx.Response(200, json={"output": {}})

    client = _make_client(handler)
    created = client.audio.vocabularies.create(
        target_model="paraformer-v2",
        prefix="med",
        vocabulary=[{"text": "心肌梗死", "weight": 4}],
    )
    assert created.vocabulary_id == "vocab-1"
    assert created.request_id == "req-v1"

    listed = client.audio.vocabularies.list(prefix="med", page_size=5)
    assert listed.vocabularies == [
        {"vocabulary_id": "vocab-1", "prefix": "med"},
    ]

    fetched = client.audio.vocabularies.get("vocab-1")
    assert fetched.vocabulary == [{"text": "心肌梗死", "weight": 4}]

    updated = client.audio.vocabularies.update(
        "vocab-1",
        vocabulary=[{"text": "脑梗死", "weight": 4}],
    )
    assert not updated.is_error
    deleted = client.audio.vocabularies.delete("vocab-1")
    assert not deleted.is_error

    path, body = calls[0]
    assert path == "/api/v1/services/audio/asr/customization"
    assert body == {
        "model": "speech-biasing",
        "input": {
            "action": "create_vocabulary",
            "target_model": "paraformer-v2",
            "prefix": "med",
            "vocabulary": [{"text": "心肌梗死", "weight": 4}],
        },
    }
    assert calls[1][1]["input"] == {
        "action": "list_vocabulary",
        "page_index": 0,
        "page_size": 5,
        "prefix": "med",
    }
    assert calls[2][1]["input"] == {
        "action": "query_vocabulary",
        "vocabulary_id": "vocab-1",
    }
    assert calls[3][1]["input"] == {
        "action": "update_vocabulary",
        "vocabulary_id": "vocab-1",
        "vocabulary": [{"text": "脑梗死", "weight": 4}],
    }
    assert calls[4][1]["input"] == {
        "action": "delete_vocabulary",
        "vocabulary_id": "vocab-1",
    }


# ── voices (1.x VoiceEnrollmentService) ───────────────────────────────


def test_voices_clone_create_and_crud():
    calls = []

    def handler(request):
        body = json.loads(request.content.decode())
        calls.append(body)
        action = body["input"]["action"]
        if action == "create_voice":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-clone",
                    "output": {"voice_id": "voice-1"},
                },
            )
        if action == "list_voice":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "voice_list": [
                            {"voice_id": "voice-1", "status": "OK"},
                        ],
                    },
                },
            )
        if action == "query_voice":
            return httpx.Response(
                200,
                json={
                    "output": {"voice_id": "voice-1", "status": "OK"},
                },
            )
        return httpx.Response(200, json={"output": {}})

    client = _make_client(handler)
    created = client.audio.voices.create(
        target_model="cosyvoice-v3-plus",
        prefix="demo",
        url="https://cdn.test/sample.wav",
        language_hints=["zh"],
        extra_input={"url": "https://evil.test/override.wav", "extra": 1},
    )
    assert created.voice_id == "voice-1"
    assert created.request_id == "req-clone"

    # Named arguments win over the extra_input escape hatch.
    assert calls[0] == {
        "model": "voice-enrollment",
        "input": {
            "action": "create_voice",
            "extra": 1,
            "target_model": "cosyvoice-v3-plus",
            "prefix": "demo",
            "url": "https://cdn.test/sample.wav",
            "language_hints": ["zh"],
        },
    }

    listed = client.audio.voices.list(prefix="demo")
    assert listed.voices == [{"voice_id": "voice-1", "status": "OK"}]
    assert calls[1]["input"] == {
        "action": "list_voice",
        "page_index": 0,
        "page_size": 10,
        "prefix": "demo",
    }

    fetched = client.audio.voices.get("voice-1")
    assert fetched.status == "OK"
    assert calls[2]["input"] == {
        "action": "query_voice",
        "voice_id": "voice-1",
    }

    client.audio.voices.update("voice-1", url="https://cdn.test/new.wav")
    assert calls[3]["input"] == {
        "action": "update_voice",
        "voice_id": "voice-1",
        "url": "https://cdn.test/new.wav",
    }
    client.audio.voices.delete("voice-1")
    assert calls[4]["input"] == {
        "action": "delete_voice",
        "voice_id": "voice-1",
    }
    assert all(call["model"] == "voice-enrollment" for call in calls)


# ── voice design (声音设计, qwen-voice-design) ─────────────────────────


def test_voice_design_crud():
    calls = []

    def handler(request):
        body = json.loads(request.content.decode())
        calls.append(body)
        action = body["input"]["action"]
        if action == "create":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "voice": "announcer",
                        "preview_audio": {
                            "data": "QUJD",
                            "sample_rate": 24000,
                            "response_format": "wav",
                        },
                    },
                    "usage": {"count": 1},
                },
            )
        if action == "list":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "voice_list": [
                            {"voice": "announcer", "language": "zh"},
                        ],
                        "total_count": 1,
                    },
                },
            )
        if action == "query":
            return httpx.Response(
                200,
                json={
                    "output": {"voice": "announcer", "language": "zh"},
                },
            )
        return httpx.Response(
            200,
            json={"output": {"voice": "announcer"}},
        )

    client = _make_client(handler)
    created = client.audio.voice_design.create(
        target_model="qwen3-tts-vd-realtime-2026-01-15",
        voice_prompt="沉稳的中年男性，音色低沉浑厚",
        preview_text="各位听众朋友，大家好",
        preferred_name="announcer",
        language="zh",
        parameters={"sample_rate": 24000, "response_format": "wav"},
    )
    assert created.voice == "announcer"
    # The mock always returns a dict; pylint only sees Optional[Dict].
    # pylint: disable-next=unsubscriptable-object
    assert created.preview_audio["response_format"] == "wav"
    assert calls[0] == {
        "model": "qwen-voice-design",
        "input": {
            "action": "create",
            "target_model": "qwen3-tts-vd-realtime-2026-01-15",
            "voice_prompt": "沉稳的中年男性，音色低沉浑厚",
            "preview_text": "各位听众朋友，大家好",
            "preferred_name": "announcer",
            "language": "zh",
        },
        "parameters": {"sample_rate": 24000, "response_format": "wav"},
    }

    listed = client.audio.voice_design.list()
    assert listed.voices == [{"voice": "announcer", "language": "zh"}]
    assert listed.total_count == 1
    assert calls[1]["input"] == {
        "action": "list",
        "page_index": 0,
        "page_size": 10,
    }

    fetched = client.audio.voice_design.get("announcer")
    assert fetched.language == "zh"
    assert calls[2]["input"] == {"action": "query", "voice": "announcer"}

    client.audio.voice_design.delete("announcer")
    assert calls[3]["input"] == {"action": "delete", "voice": "announcer"}
    assert all(call["model"] == "qwen-voice-design" for call in calls)


# ── phrases (1.x AsrPhraseManager, fine-tunes rider) ──────────────────


def test_phrases_create_and_update_ride_fine_tunes():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path, request.url.params))
        body = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "request_id": "req-p1",
                "output": {
                    "job_id": "phrase-1",
                    "status": "PENDING",
                    "model": body["model"],
                },
            },
        )

    client = _make_client(handler)
    created = client.audio.phrases.create(
        model="paraformer-v2",
        phrases={"下一首": 90, "上一首": 90},
    )
    assert created.phrase_id == "phrase-1"
    assert created.status == "PENDING"
    assert created.request_id == "req-p1"

    method, path, _ = calls[0]
    assert (method, path) == ("POST", "/api/v1/fine-tunes")
    # The create body was captured via the handler's own read; re-assert
    # the wire shape by calling update and inspecting both bodies below.

    updated = client.audio.phrases.update(
        "phrase-1",
        model="paraformer-v2",
        phrases={"暂停": 90},
    )
    assert updated.phrase_id == "phrase-1"


def test_phrases_create_wire_body():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"output": {"job_id": "p-1"}})

    client = _make_client(handler)
    client.audio.phrases.create(
        model="paraformer-v2",
        phrases={"下一首": 90},
    )
    client.audio.phrases.update(
        "p-1",
        model="paraformer-v2",
        phrases={"暂停": 90},
    )
    assert bodies[0] == {
        "model": "paraformer-v2",
        "training_file_ids": [],
        "validation_file_ids": [],
        "training_type": "compile_asr_phrase",
        "hyper_parameters": {"phrase_list": {"下一首": 90}},
    }
    assert bodies[1] == {
        "model": "paraformer-v2",
        "training_file_ids": [],
        "validation_file_ids": [],
        "training_type": "compile_asr_phrase",
        "hyper_parameters": {"phrase_list": {"暂停": 90}},
        "finetuned_output": "p-1",
    }


def test_phrases_get_list_delete_hit_outputs():
    calls = []

    def handler(request):
        calls.append(
            (request.method, request.url.path, dict(request.url.params)),
        )
        if request.method == "GET" and request.url.path.endswith("/outputs"):
            return httpx.Response(
                200,
                json={
                    "output": {
                        "jobs": [{"job_id": "p-1", "status": "SUCCEEDED"}],
                        "total": 1,
                    },
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "job_id": "p-1",
                        "status": "SUCCEEDED",
                        "hyper_parameters": {
                            "phrase_list": {"下一首": 90},
                        },
                    },
                },
            )
        return httpx.Response(200, json={"output": {"deleted": True}})

    client = _make_client(handler)
    listed = client.audio.phrases.list(page_no=2, page_size=20)
    assert listed.phrases[0].phrase_id == "p-1"
    assert listed.phrases[0].status == "SUCCEEDED"
    assert listed.total_count == 1

    fetched = client.audio.phrases.get("p-1")
    assert fetched.phrase_id == "p-1"
    assert fetched.status == "SUCCEEDED"

    deleted = client.audio.phrases.delete("p-1")
    assert not deleted.is_error

    assert calls[0] == (
        "GET",
        "/api/v1/fine-tunes/outputs",
        {"page_no": "2", "page_size": "20"},
    )
    assert calls[1] == ("GET", "/api/v1/fine-tunes/outputs/p-1", {})
    assert calls[2] == ("DELETE", "/api/v1/fine-tunes/outputs/p-1", {})


def test_phrases_local_validation():
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="phrases"):
        client.audio.phrases.create(model="paraformer-v2", phrases={})
    with pytest.raises(ValueError, match="phrase_id"):
        client.audio.phrases.update(
            " ",
            model="paraformer-v2",
            phrases={"暂停": 90},
        )
    with pytest.raises(ValueError, match="phrase_id"):
        client.audio.phrases.get("")
    with pytest.raises(ValueError, match="phrase_id"):
        client.audio.phrases.delete("")


def test_management_error_maps_through_catalog():
    def handler(request):
        return httpx.Response(
            400,
            json={"code": "InvalidParameter", "message": "bad prefix"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError, match="bad prefix"):
        client.audio.vocabularies.create(
            target_model="paraformer-v2",
            prefix="BAD PREFIX",
            vocabulary=[{"text": "x"}],
        )


_MANAGEMENT_RESOURCES = ("vocabularies", "voices", "voice_design", "phrases")


@pytest.mark.parametrize(
    "resource, method",
    [
        (resource, method)
        for resource in _MANAGEMENT_RESOURCES
        for method in ("create", "get", "list", "update", "delete")
        if (resource, method) != ("voice_design", "update")
    ],
)
def test_management_all_actions_keep_real_response_headers(resource, method):
    response = httpx.Response(
        200,
        headers={"x-request-id": "real-id", "x-gateway": "test"},
        json={"output": {}, "headers": {"x-request-id": "body-id"}},
    )
    client = _make_client(lambda request: response)
    create_args = {
        "vocabularies": {
            "target_model": "paraformer-v2",
            "prefix": "test",
            "vocabulary": [{"text": "test", "weight": 1}],
        },
        "voices": {
            "target_model": "cosyvoice-v3-flash",
            "prefix": "test",
            "url": "https://cdn.test/audio.wav",
        },
        "voice_design": {
            "target_model": "qwen-tts-vd",
            "voice_prompt": "warm",
            "preview_text": "hello",
        },
        "phrases": {"model": "paraformer-v2", "phrases": {"test": 1}},
    }
    args = () if method in ("create", "list") else ("resource-1",)
    kwargs = create_args[resource] if method == "create" else {}
    if method == "update":
        kwargs = {
            "vocabularies": {"vocabulary": [{"text": "test", "weight": 1}]},
            "voices": {"url": "https://cdn.test/audio.wav"},
            "phrases": create_args["phrases"],
        }[resource]
    try:
        result = getattr(getattr(client.audio, resource), method)(
            *args,
            **kwargs,
        )
        assert result.headers == dict(response.headers)
        assert result.status_code == 200
        assert not result.is_error
        result.headers["x-gateway"] = "changed"
        assert response.headers["x-gateway"] == "test"
    finally:
        client.close()
        client.transport.http_client.close()


@pytest.mark.parametrize(
    "resource, wire_key, attribute, item",
    [
        ("vocabularies", key, "vocabularies", {"vocabulary_id": "v-1"})
        for key in ("vocabularies", "vocabulary_list")
    ]
    + [
        (resource, key, "voices", {"voice": "voice-1", "enabled": False})
        for resource in ("voices", "voice_design")
        for key in ("voices", "voice_list")
    ]
    + [
        ("phrases", key, "phrases", {"finetuned_output": "phrase-1"})
        for key in ("phrases", "jobs", "outputs")
    ],
)
def test_management_list_aliases_and_envelope(
    resource,
    wire_key,
    attribute,
    item,
):
    body = {
        "request_id": "outer-id",
        "status_code": 200,
        "usage": {"count": 0},
        "output": {
            wire_key: [item],
            "request_id": "inner-id",
            "total": 1,
            "page_no": 2,
            "page_size": 1,
            "enabled": False,
        },
    }
    client = _make_client(lambda request: httpx.Response(200, json=body))
    result = getattr(client.audio, resource).list()
    values = getattr(result, attribute)
    if resource == "phrases":
        assert values[0].phrase_id == "phrase-1"
        assert result.total_count == 1
        assert (result.page_no, result.page_size) == (2, 1)
    else:
        assert values == [item]
    assert result.request_id == "outer-id"
    assert result.status_code == 200
    assert result.usage == {"count": 0}
    assert result.enabled is False
    assert not result.is_error


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
@pytest.mark.parametrize("status", [200, 400, 401, 500])
@pytest.mark.parametrize("raise_on_error", [True, False])
def test_management_error_envelope(resource, status, raise_on_error):
    body = {
        "code": "InvalidApiKey" if status == 401 else "InvalidParameter",
        "message": "management rejected",
        "request_id": "body-error-id",
        "status_code": status,
    }
    client = _make_client(
        lambda request: httpx.Response(
            status,
            json=body,
            headers={"x-request-id": "header-error-id"},
        ),
        raise_on_error=raise_on_error,
        max_retries=0,
    )
    call = getattr(client.audio, resource).list
    if raise_on_error:
        error_type = AuthenticationError if status == 401 else BadRequestError
        with pytest.raises(error_type, match="management rejected") as caught:
            call()
        assert caught.value.code == body["code"]
        assert caught.value.status_code == status
        assert caught.value.request_id == "body-error-id"
        assert caught.value.body == body
        assert caught.value.headers["x-request-id"] == "header-error-id"
    else:
        result = call()
        assert result.is_error
        assert result.code == body["code"]
        assert result.message == body["message"]
        assert result.request_id == "body-error-id"
        assert result.status_code == status
        assert result.headers["x-request-id"] == "header-error-id"


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
def test_management_nonraising_http_failure_without_code_is_error(resource):
    # A non-2xx response must not look like a successful empty resource list.
    client = _make_client(
        lambda request: httpx.Response(503, json={"message": "unavailable"}),
        raise_on_error=False,
        max_retries=0,
    )
    result = getattr(client.audio, resource).list()
    assert result.is_error
    assert result.status_code == 503
    assert result.message == "unavailable"


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
def test_management_redirect_is_not_success(resource):
    client = _make_client(
        lambda request: httpx.Response(
            302,
            json={},
            headers={"location": "https://other.test/"},
        ),
        max_retries=0,
    )
    with pytest.raises(APIStatusError) as caught:
        getattr(client.audio, resource).list()
    assert caught.value.status_code == 302


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
def test_management_non_json_http_error_keeps_header_request_id(resource):
    client = _make_client(
        lambda request: httpx.Response(
            500,
            text="gateway failed",
            headers={"x-request-id": "gateway-id"},
        ),
        max_retries=0,
    )
    with pytest.raises(InternalServerError, match="gateway failed") as caught:
        getattr(client.audio, resource).list()
    assert caught.value.status_code == 500
    assert caught.value.request_id == "gateway-id"


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
@pytest.mark.parametrize("raise_on_error", [True, False])
def test_management_malformed_json_is_not_suppressed(resource, raise_on_error):
    client = _make_client(
        lambda request: httpx.Response(200, content=b'{"output":'),
        raise_on_error=raise_on_error,
    )
    # Compatibility mode suppresses service errors, not parsing failures.
    with pytest.raises(ValueError):
        getattr(client.audio, resource).list()


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
@pytest.mark.parametrize(
    "transport_error, sdk_error",
    [
        (httpx.ConnectError, APIConnectionError),
        (httpx.ReadTimeout, APITimeoutError),
    ],
)
def test_management_transport_error_not_suppressed(
    resource,
    transport_error,
    sdk_error,
):
    def handler(request):
        raise transport_error("offline", request=request)

    client = _make_client(handler, raise_on_error=False, max_retries=0)
    with pytest.raises(sdk_error) as caught:
        getattr(client.audio, resource).list()
    assert isinstance(caught.value.__cause__, transport_error)


@pytest.mark.parametrize("method", ["create", "update"])
def test_phrase_id_prefers_compiled_resource_over_job_id(method):
    # The management guide §3 records this official dual-ID response shape.
    output = {"job_id": "ft-job-1", "finetuned_output": "paraformer-v1-ft-1"}
    client = _make_client(
        lambda request: httpx.Response(200, json={"output": output}),
    )
    args = ("paraformer-v1-ft-1",) if method == "update" else ()
    result = getattr(client.audio.phrases, method)(
        *args,
        model="paraformer-realtime-v1",
        phrases={"下一首": 4},
    )
    assert result.phrase_id == output["finetuned_output"]
    assert result.job_id == output["job_id"]


def test_phrases_list_maps_finetuned_outputs():
    client = _make_client(
        lambda request: httpx.Response(
            200,
            json={
                "output": {
                    "finetuned_outputs": [
                        {"finetuned_output": "paraformer-v1-ft-1"},
                    ],
                    "total": 1,
                },
            },
        ),
    )
    result = client.audio.phrases.list()
    assert [phrase.phrase_id for phrase in result.phrases] == [
        "paraformer-v1-ft-1",
    ]
    assert result.total_count == 1


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
@pytest.mark.parametrize("value", [None, "", "   ", False])
def test_management_empty_id_rejected_without_network(resource, value):
    def handler(request):
        pytest.fail("invalid IDs must be rejected before HTTP")

    client = _make_client(handler)
    for method in ("get", "delete"):
        with pytest.raises(ValueError):
            getattr(getattr(client.audio, resource), method)(value)


@pytest.mark.parametrize("phrases", [None, {}, False])
def test_empty_phrases_rejected_without_network(phrases):
    def handler(request):
        pytest.fail("empty phrases must be rejected before HTTP")

    client = _make_client(handler)
    for method, args in (("create", ()), ("update", ("phrase-1",))):
        with pytest.raises(ValueError, match="phrases"):
            getattr(client.audio.phrases, method)(
                *args,
                model="paraformer-realtime-v1",
                phrases=phrases,
            )


@pytest.mark.parametrize("resource", ["vocabularies", "voices"])
@pytest.mark.parametrize("prefix", [None, "", False])
def test_management_empty_prefix_omitted_and_zero_page_preserved(
    resource,
    prefix,
):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"output": {}})

    client = _make_client(handler)
    getattr(client.audio, resource).list(
        prefix=prefix,
        page_index=0,
        page_size=0,
    )
    assert "prefix" not in bodies[0]["input"]
    assert bodies[0]["input"]["page_index"] == 0
    assert bodies[0]["input"]["page_size"] == 0


@pytest.mark.parametrize("explicit", [False, True])
def test_customization_optional_fields_and_caller_dicts(explicit):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"output": {}})

    client = _make_client(handler)
    voice_extra = {"language_hints": ["en"], "max_prompt_audio_length": 10}
    design_extra = {"language": "en", "preferred_name": "original"}
    original = json.loads(json.dumps([voice_extra, design_extra]))
    parameters = {} if explicit else None
    client.audio.voices.create(
        target_model="cosyvoice-v3-plus",
        prefix="demo",
        url="https://cdn.test/a.wav",
        language_hints=[] if explicit else None,
        max_prompt_audio_length=0 if explicit else None,
        extra_input=voice_extra,
        parameters=parameters,
    )
    client.audio.voice_design.create(
        target_model="qwen3-tts-vd-realtime-2026-01-15",
        voice_prompt="低沉",
        preview_text="你好",
        language="" if explicit else None,
        preferred_name="" if explicit else None,
        extra_input=design_extra,
        parameters=parameters,
    )
    assert bodies[0]["input"]["language_hints"] == ([] if explicit else ["en"])
    assert bodies[0]["input"]["max_prompt_audio_length"] == (
        0 if explicit else 10
    )
    assert bodies[1]["input"]["language"] == ("" if explicit else "en")
    assert bodies[1]["input"]["preferred_name"] == (
        "" if explicit else "original"
    )
    assert [voice_extra, design_extra] == original
    for body in bodies:
        if explicit:
            assert body["parameters"] == {}
        else:
            assert "parameters" not in body


@pytest.mark.parametrize("resource", ["voices", "voice_design"])
def test_customization_extra_input_cannot_change_create_action(resource):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"output": {}})

    client = _make_client(handler)
    if resource == "voices":
        client.audio.voices.create(
            target_model="cosyvoice-v3-plus",
            prefix="demo",
            url="https://cdn.test/a.wav",
            extra_input={"action": "delete_voice"},
        )
        expected_action = "create_voice"
    else:
        client.audio.voice_design.create(
            target_model="qwen3-tts-vd-realtime-2026-01-15",
            voice_prompt="低沉",
            preview_text="你好",
            extra_input={"action": "delete"},
        )
        expected_action = "create"
    assert bodies[0]["input"]["action"] == expected_action


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
@pytest.mark.parametrize("auth_header", ["Authorization", "authorization"])
def test_management_call_auth_is_single_valued_and_does_not_leak(
    resource,
    auth_header,
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"output": {}})

    client = _make_client(handler)
    http_client = client.transport.http_client
    http_client.headers.update(
        {
            "Authorization": "Bearer pool-key",
            "X-DashScope-WorkSpace": "pool-ws",
        },
    )
    workspace_header = (
        "X-DashScope-WorkSpace"
        if auth_header == "Authorization"
        else "x-dashscope-workspace"
    )
    headers = {auth_header: "Bearer header-key", workspace_header: "header-ws"}
    original = dict(headers)
    call = getattr(client.audio, resource).list
    call(api_key="call-key", workspace="call-ws", headers=headers, timeout=0)
    call(api_key=None, workspace=None, timeout=None)
    assert requests[0].headers.get_list("authorization") == ["Bearer call-key"]
    assert requests[0].headers.get_list("x-dashscope-workspace") == ["call-ws"]
    assert requests[0].extensions["timeout"]["read"] == 0
    assert requests[1].headers["authorization"] == "Bearer pool-key"
    assert requests[1].headers["x-dashscope-workspace"] == "pool-ws"
    assert (
        requests[1].extensions["timeout"]["read"] == http_client.timeout.read
    )
    assert all(request.url.host == "api.test.local" for request in requests)
    assert headers == original


@pytest.mark.parametrize("request_id", [None, "outer-id"])
def test_designed_voice_keeps_raw_preview_and_unknown_fields(request_id):
    # Preview audio is explicitly a raw dict, not the decoded SSE bytes API.
    preview = {"data": "%%%", "sample_rate": 24000, "response_format": "wav"}
    body = {
        "request_id": request_id,
        "output": {
            "request_id": "inner-id",
            "voice": "full-voice-id",
            "preview_audio": preview,
            "enabled": False,
            "future_field": [],
        },
    }
    client = _make_client(lambda request: httpx.Response(200, json=body))
    result = client.audio.voice_design.get("full-voice-id")
    assert result.voice == "full-voice-id"
    assert result.preview_audio == preview
    assert result.request_id == (request_id or "inner-id")
    assert result.enabled is False
    assert result.model_dump()["future_field"] == []


@pytest.mark.parametrize("id_key", ["phrase_id", "job_id", "finetuned_output"])
def test_phrase_single_id_alias_preserves_status_and_details(id_key):
    client = _make_client(
        lambda request: httpx.Response(
            200,
            json={
                "output": {
                    id_key: "returned-id",
                    "status": "PENDING",
                    "hyper_parameters": {"phrase_list": {"下一首": 4}},
                },
            },
        ),
    )
    result = client.audio.phrases.get("queried-id")
    assert result.phrase_id == "returned-id"
    assert result.status == "PENDING"
    assert result.hyper_parameters == {"phrase_list": {"下一首": 4}}


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
@pytest.mark.parametrize("status", [302, 503])
@pytest.mark.parametrize("body_kind", ["json", "text"])
def test_management_nonraising_failure_keeps_http_metadata(
    resource,
    status,
    body_kind,
):
    body = {"message": "unavailable", "status_code": 200}
    kwargs = {"json": body} if body_kind == "json" else {"text": "unavailable"}
    client = _make_client(
        lambda request: httpx.Response(
            status,
            headers={"x-request-id": "header-id"},
            **kwargs,
        ),
        raise_on_error=False,
        max_retries=0,
    )
    result = getattr(client.audio, resource).list()
    assert result.status_code == status
    assert result.request_id == "header-id"
    assert result.message == "unavailable"
    assert result.code is None
    assert result.is_error


@pytest.mark.parametrize("resource", _MANAGEMENT_RESOURCES)
@pytest.mark.parametrize("status", [200, 400])
@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize(
    "outer_id, inner_id, expected_id",
    [
        ("outer-id", "inner-id", "outer-id"),
        (None, "inner-id", "inner-id"),
        ("", "inner-id", "inner-id"),
        (None, None, "header-id"),
    ],
)
def test_management_request_id_precedence(
    resource,
    status,
    raise_on_error,
    outer_id,
    inner_id,
    expected_id,
):
    body = {
        "request_id": outer_id,
        "output": {"request_id": inner_id},
    }
    with _make_client(
        lambda request: httpx.Response(
            status,
            json=body,
            headers={"x-request-id": "header-id"},
        ),
        raise_on_error=raise_on_error,
        max_retries=0,
    ) as client:
        if raise_on_error and status == 400:
            with pytest.raises(APIStatusError) as caught:
                getattr(client.audio, resource).list()
            assert caught.value.request_id == expected_id
            assert caught.value.body == body
            assert caught.value.status_code == status
        else:
            result = getattr(client.audio, resource).list()
            assert result.request_id == expected_id
            assert result.status_code == status
            assert result.is_error == (status != 200)


@pytest.mark.parametrize(
    "method",
    ["create", "update", "get", "list", "delete"],
)
def test_phrase_methods_share_nonraising_http_metadata(method):
    client = _make_client(
        lambda request: httpx.Response(
            503,
            text="unavailable",
            headers={"x-request-id": "phrase-error"},
        ),
        raise_on_error=False,
        max_retries=0,
    )
    args = ("phrase-id",) if method in {"update", "get", "delete"} else ()
    kwargs = (
        {"model": "paraformer-v1", "phrases": {"下一首": 4}}
        if method in {"create", "update"}
        else {}
    )
    result = getattr(client.audio.phrases, method)(*args, **kwargs)
    assert result.status_code == 503
    assert result.request_id == "phrase-error"
    assert result.message == "unavailable"
    assert result.is_error


def test_phrase_explicit_resource_id_wins_over_both_aliases():
    client = _make_client(
        lambda request: httpx.Response(
            200,
            json={
                "output": {
                    "phrase_id": "explicit-id",
                    "finetuned_output": "compiled-id",
                    "job_id": "job-id",
                },
            },
        ),
    )
    result = client.audio.phrases.get("queried-id")
    assert result.phrase_id == "explicit-id"
    assert result.finetuned_output == "compiled-id"
    assert result.job_id == "job-id"
