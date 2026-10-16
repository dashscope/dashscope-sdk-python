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
from dashscope.v2.errors import BadRequestError

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
    assert created.preview_audio is not None
    # pylint: disable-next=unsubscriptable-object  # Optional narrowed above
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
