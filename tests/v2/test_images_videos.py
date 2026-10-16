# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.images and client.videos (M3 task 3.3)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client

BASE_URL = "https://api.test.local/api/v1"
IMAGE_PATH = "/api/v1/services/aigc/text2image/image-synthesis"
VIDEO_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
MULTIMODAL_IMAGE_PATH = (
    "/api/v1/services/aigc/multimodal-generation/generation"
)
TASK_IMAGE_PATH = "/api/v1/services/aigc/image-generation/generation"


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


def _task_body(status, output_extra=None):
    output = {"task_id": "task-1", "task_status": status}
    output.update(output_extra or {})
    return {"request_id": "req-1", "output": output}


def _multimodal_image_body():
    """Sync multimodal-generation answer: choices instead of results."""
    return {
        "request_id": "req-1",
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"image": "https://cdn.example.com/1.png"},
                        ],
                    },
                },
            ],
        },
        "usage": {"input_tokens": 10, "image_count": 1},
    }


def test_images_generate_submit_and_wait():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["path"] = request.url.path
            captured["header"] = request.headers.get("x-dashscope-async")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {"results": [{"url": "https://cdn.example.com/1.png"}]},
            ),
        )

    client = _make_client(handler)
    job = client.images.generate(
        model="wanx2.1-t2i",
        prompt="a cat",
        negative_prompt="blurry",
        n=1,
        size="1024*1024",
    )
    assert captured["path"] == IMAGE_PATH
    assert captured["header"] == "enable"
    assert captured["body"] == {
        "model": "wanx2.1-t2i",
        "input": {"prompt": "a cat", "negative_prompt": "blurry"},
        "parameters": {"n": 1, "size": "1024*1024"},
    }
    assert job.succeeded is True
    assert isinstance(job.output, dict)
    assert job.output["results"][0]["url"] == "https://cdn.example.com/1.png"


def test_images_generate_wan26_uses_multimodal_endpoint():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["header"] = request.headers.get("x-dashscope-async")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_multimodal_image_body())

    client = _make_client(handler)
    job = client.images.generate(
        model="wan2.6-t2i",
        prompt="a cat",
        negative_prompt="blurry",
        n=1,
        size="1024x1024",
    )
    assert captured["path"] == MULTIMODAL_IMAGE_PATH
    # One-shot sync call: no async header, no polling.
    assert captured["header"] is None
    assert captured["body"] == {
        "model": "wan2.6-t2i",
        "input": {
            "messages": [
                {"role": "user", "content": [{"text": "a cat"}]},
            ],
        },
        "parameters": {
            "negative_prompt": "blurry",
            "n": 1,
            "size": "1024*1024",
        },
    }
    assert job.succeeded is True
    assert job.image_urls == ["https://cdn.example.com/1.png"]
    assert job.usage.image_count == 1


def test_images_generate_wan26_image_reference_urls():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_multimodal_image_body())

    client = _make_client(handler)
    client.images.generate(
        model="wan2.6-image",
        prompt="combine the references",
        image_urls=[
            "https://example.com/a.png",
            "https://example.com/b.png",
        ],
    )
    content = captured["body"]["input"]["messages"][0]["content"]
    assert content == [
        {"text": "combine the references"},
        {"image": "https://example.com/a.png"},
        {"image": "https://example.com/b.png"},
    ]


def test_images_jobs_wan26_uses_image_generation_task_endpoint():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["path"] = request.url.path
            captured["header"] = request.headers.get("x-dashscope-async")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"image": "https://cdn.example.com/1.png"},
                                ],
                            },
                        },
                    ],
                },
            ),
        )

    client = _make_client(handler)
    pending = client.images.jobs.create(model="wan2.6-t2i", prompt="a cat")
    assert captured["path"] == TASK_IMAGE_PATH
    assert captured["header"] == "enable"
    assert captured["body"]["input"] == {
        "messages": [{"role": "user", "content": [{"text": "a cat"}]}],
    }
    assert pending.status == "PENDING"

    final = client.images.jobs.wait("task-1", poll_interval=0.01)
    assert final.succeeded is True
    # choices-style task output is normalized into results/image_urls.
    assert final.image_urls == ["https://cdn.example.com/1.png"]


def test_images_size_accepts_x_separator():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_task_body("PENDING"))

    client = _make_client(handler)
    client.images.jobs.create(
        model="wanx2.1-t2i",
        prompt="a cat",
        size="1024x1024",
    )
    assert captured["body"]["parameters"] == {"size": "1024*1024"}


def test_images_named_parameters_merge_and_win():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_task_body("PENDING"))

    client = _make_client(handler)
    client.images.jobs.create(
        model="wanx2.1-t2i",
        prompt="a cat",
        seed=42,
        style="<anime>",
        prompt_extend=True,
        watermark=False,
        parameters={"seed": 1, "steps": 30},
    )
    # Named kwargs win over the ``parameters`` escape hatch.
    assert captured["body"]["parameters"] == {
        "seed": 42,
        "style": "<anime>",
        "prompt_extend": True,
        "watermark": False,
        "steps": 30,
    }


def test_images_function_goes_to_input():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_task_body("PENDING"))

    client = _make_client(handler)
    client.images.jobs.create(
        model="wanx2.1-t2i",
        prompt="a cat",
        function="stylization_all",
        extra_input={"function": "stylization_local"},
    )
    # 1.x spells it inputs["function"]; the named kwarg wins over
    # the ``extra_input`` escape hatch.
    assert captured["body"]["input"]["function"] == "stylization_all"


def test_images_function_multimodal_goes_to_input():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_multimodal_image_body())

    client = _make_client(handler)
    client.images.generate(
        model="wan2.6-t2i",
        prompt="a cat",
        function="stylization_all",
        submit_timeout=1,
    )
    assert captured["body"]["input"]["function"] == "stylization_all"


def test_images_jobs_cancel_and_list():
    captured = []

    def handler(request):
        captured.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json=_task_body("CANCELED"))
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "total": 1,
                "page_no": 1,
                "page_size": 10,
                "tasks": [{"task_id": "task-1", "task_status": "CANCELED"}],
            },
        )

    client = _make_client(handler)
    canceled = client.images.jobs.cancel("task-1")
    assert canceled.status == "CANCELED"
    result = client.images.jobs.list()
    assert result.total == 1
    assert result.tasks[0].task_id == "task-1"
    assert captured == [
        ("POST", "/api/v1/tasks/task-1/cancel"),
        ("GET", "/api/v1/tasks"),
    ]


def test_images_jobs_create_fetch_wait_roundtrip():
    statuses = iter(["PENDING", "RUNNING", "RUNNING", "SUCCEEDED"])

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json=_task_body(next(statuses)))
        return httpx.Response(
            200,
            json=_task_body(
                next(statuses),
                {"results": [{"url": "https://cdn.example.com/1.png"}]},
            ),
        )

    client = _make_client(handler)
    pending = client.images.jobs.create(model="wanx2.1-t2i", prompt="a cat")
    assert pending.status == "PENDING"
    assert pending.task_id == "task-1"

    snapshot = client.images.jobs.fetch("task-1")
    assert snapshot.status == "RUNNING"

    delays = []
    final = client.images.jobs.wait(
        "task-1",
        poll_interval=1.0,
        _sleep=delays.append,
    )
    assert final.succeeded is True
    assert delays == [1.0]


def test_videos_generate_with_prompt():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {"video_url": "https://cdn.example.com/v.mp4"},
            ),
        )

    client = _make_client(handler)
    job = client.videos.generate(
        model="wanx2.1-t2v",
        prompt="a running cat",
        parameters={"duration": 5},
    )
    assert captured["path"] == VIDEO_PATH
    assert captured["body"] == {
        "model": "wanx2.1-t2v",
        "input": {"prompt": "a running cat"},
        "parameters": {"duration": 5},
    }
    assert job.succeeded is True
    assert isinstance(job.output, dict)
    assert job.output["video_url"] == "https://cdn.example.com/v.mp4"


def test_videos_named_parameters_merge_and_win():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    client = _make_client(handler)
    client.videos.generate(
        model="wanx2.1-t2v",
        prompt="a running cat",
        seed=7,
        size="1280*720",
        duration=10,
        resolution="1080P",
        ratio="16:9",
        prompt_extend=True,
        watermark=True,
        parameters={"duration": 5},
    )
    # Named kwargs win over the ``parameters`` escape hatch.
    assert captured["body"]["parameters"] == {
        "seed": 7,
        "size": "1280*720",
        "duration": 10,
        "resolution": "1080P",
        "ratio": "16:9",
        "prompt_extend": True,
        "watermark": True,
    }


def test_videos_function_goes_to_input():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_task_body("PENDING"))

    client = _make_client(handler)
    client.videos.jobs.create(
        model="wanx2.1-t2v",
        prompt="a running cat",
        function="video_retalk",
    )
    # 1.x spells it inputs["function"] (video_synthesis.py:219).
    assert captured["body"]["input"]["function"] == "video_retalk"


def test_videos_require_prompt_or_image_url():
    client = _make_client(
        lambda request: httpx.Response(200, json=_task_body("PENDING")),
    )
    with pytest.raises(ValueError, match="prompt or image_url"):
        client.videos.jobs.create(model="wanx2.1-t2v")


def test_videos_legacy_extra_input_only_is_accepted():
    # ``extra_input`` is the escape hatch for fields the typed params do
    # not cover; a legacy model must accept it even without prompt/image_url.
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    client = _make_client(handler)
    client.videos.generate(
        model="wanx2.1-t2v",
        extra_input={"img_url": "https://example.com/cat.png"},
    )
    assert captured["body"]["input"] == {
        "img_url": "https://example.com/cat.png",
    }


def test_videos_image_to_video_input():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    client = _make_client(handler)
    client.videos.generate(
        model="wanx2.1-i2v",
        image_url="https://example.com/cat.png",
    )
    # The service spells this key "img_url"; the SDK does not.
    assert captured["body"]["input"] == {
        "img_url": "https://example.com/cat.png",
    }


def test_videos_generate_wan27_i2v_uses_media_protocol():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["path"] = request.url.path
            captured["header"] = request.headers.get("x-dashscope-async")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {"video_url": "https://cdn.example.com/v.mp4"},
            ),
        )

    media = [
        {"type": "first_frame", "url": "https://example.com/first.png"},
        {"type": "driving_audio", "url": "https://example.com/rap.mp3"},
    ]
    client = _make_client(handler)
    job = client.videos.generate(
        # Dated variants route the same way (prefix match).
        model="wan2.7-i2v-2026-04-25",
        prompt="a rapper comes to life",
        media=media,
        parameters={"resolution": "720P", "duration": 10},
    )
    # Same service and async envelope; only the input shape is new.
    assert captured["path"] == VIDEO_PATH
    assert captured["header"] == "enable"
    assert captured["body"] == {
        "model": "wan2.7-i2v-2026-04-25",
        "input": {"prompt": "a rapper comes to life", "media": media},
        "parameters": {"resolution": "720P", "duration": 10},
    }
    assert job.succeeded is True
    assert job.video_url == "https://cdn.example.com/v.mp4"


def test_videos_wan27_i2v_image_url_becomes_first_frame():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    client = _make_client(handler)
    client.videos.generate(
        model="wan2.7-i2v",
        image_url="https://example.com/cat.png",
        media=[{"type": "driving_audio", "url": "https://example.com/a.mp3"}],
    )
    # image_url is shorthand; it merges into the media list.
    assert captured["body"]["input"] == {
        "media": [
            {"type": "first_frame", "url": "https://example.com/cat.png"},
            {"type": "driving_audio", "url": "https://example.com/a.mp3"},
        ],
    }


def test_videos_wan27_i2v_requires_media():
    client = _make_client(
        lambda request: httpx.Response(200, json=_task_body("PENDING")),
    )
    with pytest.raises(ValueError, match="media or image_url"):
        client.videos.jobs.create(model="wan2.7-i2v", prompt="a cat")


def test_videos_wan27_i2v_image_url_conflicts_with_first_frame():
    client = _make_client(
        lambda request: httpx.Response(200, json=_task_body("PENDING")),
    )
    with pytest.raises(ValueError, match="first_frame"):
        client.videos.jobs.create(
            model="wan2.7-i2v",
            image_url="https://example.com/a.png",
            media=[
                {"type": "first_frame", "url": "https://example.com/b.png"},
            ],
        )


@pytest.mark.asyncio
async def test_async_images_generate():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {"results": [{"url": "https://cdn.example.com/1.png"}]},
            ),
        )

    client = _make_async_client(handler)
    job = await client.images.generate(
        model="wanx2.1-t2i",
        prompt="a cat",
    )
    assert job.succeeded is True


@pytest.mark.asyncio
async def test_async_images_generate_wan26():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["header"] = request.headers.get("x-dashscope-async")
        return httpx.Response(200, json=_multimodal_image_body())

    client = _make_async_client(handler)
    job = await client.images.generate(
        model="wan2.6-t2i",
        prompt="a cat",
    )
    assert captured["path"] == MULTIMODAL_IMAGE_PATH
    assert captured["header"] is None
    assert job.succeeded is True
    assert job.image_urls == ["https://cdn.example.com/1.png"]


@pytest.mark.asyncio
async def test_async_videos_generate():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {"video_url": "https://cdn.example.com/v.mp4"},
            ),
        )

    client = _make_async_client(handler)
    job = await client.videos.generate(
        model="wanx2.1-t2v",
        prompt="a running cat",
    )
    assert job.succeeded is True
    assert isinstance(job.output, dict)
    assert job.output["video_url"] == "https://cdn.example.com/v.mp4"


@pytest.mark.asyncio
async def test_async_videos_generate_wan27_i2v():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {"video_url": "https://cdn.example.com/v.mp4"},
            ),
        )

    client = _make_async_client(handler)
    job = await client.videos.generate(
        model="wan2.7-i2v",
        prompt="a running cat",
        media=[{"type": "first_frame", "url": "https://example.com/cat.png"}],
    )
    assert captured["body"]["input"] == {
        "prompt": "a running cat",
        "media": [
            {"type": "first_frame", "url": "https://example.com/cat.png"},
        ],
    }
    assert job.succeeded is True
    assert job.video_url == "https://cdn.example.com/v.mp4"


def test_image_jobs_create_uploads_with_the_per_call_api_key(monkeypatch):
    seen = {}

    def fake_upload(model, content, api_key, cert, **kwargs):
        seen["api_key"] = api_key
        seen["base_address"] = kwargs.get("base_address")
        return True, "oss://bucket/ref.png", {}

    monkeypatch.setattr(
        "dashscope.v2.resources._oss.check_and_upload_local",
        fake_upload,
    )
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("Authorization")
        captured["oss"] = request.headers.get("X-DashScope-OssResourceResolve")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_task_body("PENDING"))

    client = _make_client(handler)
    client.images.jobs.create(
        model="wan2.6-image",
        prompt="a cat",
        image_urls=["/tmp/ref.png"],
        api_key="sk-override",
    )
    assert seen["api_key"] == "sk-override"
    assert seen["base_address"] == client.config.base_url
    assert captured["auth"] == "Bearer sk-override"
    assert captured["oss"] == "enable"
    content = captured["body"]["input"]["messages"][0]["content"]
    assert {"image": "oss://bucket/ref.png"} in content
