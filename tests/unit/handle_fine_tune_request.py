# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import json

import aiohttp
from aiohttp import web

from tests.unit.mock_sse import sse_response

from .constants import TEST_JOB_ID  # pylint: disable=relative-beyond-top-level


async def create_fine_tune_handler(request: aiohttp.request):
    assert "X-Request-Id" in request.headers
    request_id = request.headers["X-Request-Id"]
    body = await request.json()
    assert body["model"] == "gpt" or body["model"] == "asr"
    if request_id == "111111":
        assert body["training_file_ids"] == "training_001"
        assert body["validation_file_ids"] == "validation_001"
    elif request_id == "empty_file_ids":
        assert len(body["training_file_ids"]) == 0
        assert len(body["validation_file_ids"]) == 0
    else:
        assert len(body["training_file_ids"]) == 2
        assert len(body["validation_file_ids"]) == 2
    if body["model"] == "asr":
        assert "phrase_list" in body["hyper_parameters"]
    else:
        assert body["hyper_parameters"]["epochs"] == 10
    # check body info.
    response = {
        "code": "200",
        "output": {
            "job_id": TEST_JOB_ID,
            "status": "creating",
            "finetuned_output": TEST_JOB_ID,
        },
    }

    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def list_fine_tune_handler(
    request: aiohttp.request,
):  # pylint: disable=unused-argument
    response = {
        "output": {
            "jobs": [
                {
                    "job_id": "xxxxx",
                    "status": "ready",
                    "output_model": "fine-tuned-xxxxxx",
                    "model": "13B",
                    "training_file_ids": ["file-xxxxxx", "file-xxxxxx"],
                    "validation_file_ids": ["file-xxxxxx", "file-xxxxxx"],
                    "hyper_parameters": {
                        "max_epochs": "3",
                    },
                    "message": "Training failed due to xxxxx reason.",
                },
                {
                    "job_id": "xxxxx",
                    "status": "ready",
                    "output_model": "fine-tuned-xxxxxx",
                    "model": "13B",
                    "training_file_ids": ["file-xxxxxx", "file-xxxxxx"],
                    "validation_file_ids": ["file-xxxxxx", "file-xxxxxx"],
                    "hyper_parameters": {
                        "max_epochs": "3",
                    },
                    "message": "Training failed due to xxxxx reason.",
                },
            ],
            "finetuned_outputs": [
                {
                    "finetuned_output": TEST_JOB_ID,
                    "job_id": "xxxxx",
                    "model": "asr",
                },
            ],
        },
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def get_fine_tune_handler(request: aiohttp.request):
    fine_tune_id = request.match_info["id"]
    assert fine_tune_id == TEST_JOB_ID
    response = {
        "code": "200",
        "output": {
            "job_id": TEST_JOB_ID,
            "status": "ready",
            "output_model": "fine-tuned-xxxxxx",
            "model": "13B",
            "training_file_ids": ["file1", "file-xxxxxx", "fiel2"],
            "validation_file_ids": ["file-xxxxxx", "file-xxxxxx"],
            "hyper_parameters": {
                "max_epochs": 3,
            },
            "finetuned_output": TEST_JOB_ID,
        },
        "message": "Training failed due to xxxxx reason.",
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def delete_fine_tune_handler(request: aiohttp.request):
    fine_tune_id = request.match_info["id"]
    assert fine_tune_id == TEST_JOB_ID
    response = {
        "output": {
            "status": "success",
            "finetuned_output": TEST_JOB_ID,
        },
        "message": "fine-tune job has been deleted successfully.",
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def cancel_fine_tune_handler(request: aiohttp.request):
    fine_tune_id = request.match_info["id"]
    assert fine_tune_id == TEST_JOB_ID
    response = {
        "output": {
            "status": "success",
        },
        "message": "fine-tune job has been cancel successfully.",
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def events_fine_tune_handler(request: aiohttp.request):
    fine_tune_id = request.match_info["id"]
    assert fine_tune_id == TEST_JOB_ID
    async with sse_response(request) as resp:
        for idx in range(10):
            log = f"fine-tune logging {idx}"
            print(f"Sending sse data: {log}")
            await resp.send(log, id=idx)
            await asyncio.sleep(1)
        print("logging send completed")
    return await sse_response(request)
