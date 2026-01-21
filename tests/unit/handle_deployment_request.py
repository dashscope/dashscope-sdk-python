# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json

import aiohttp
from aiohttp import web

from tests.unit.constants import TEST_JOB_ID


async def create_deployment_handler(request: aiohttp.request):
    body = await request.json()
    assert body["model_name"] == "gpt"
    assert body["suffix"] == "1"
    assert body["capacity"] == 2
    # check body info.
    response = {
        "code": "200",
        "output": {
            "deployed_model": "deploy123456",
            "status": "PENDING",
            "model_name": "qwen-turbo-ft-202307121513-5dde",
        },
    }

    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def list_deployment_handler(
    request: aiohttp.request,
):  # pylint: disable=unused-argument
    response = {
        "status_code": 200,
        "request_id": "af80b388-b891-43fb-9721-ce5c23d1cafb",
        "code": None,
        "message": "",
        "output": {
            "deployments": [
                {
                    "deployed_model": "chatm6-v1-ft-202305230928-b76b",
                    "status": "PENDING",
                    "model_name": "chatm6-v1-ft-202305230928-b76b",
                },
            ],
        },
        "usage": None,
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def get_deployment_handler(request: aiohttp.request):
    assert request.match_info["id"] == TEST_JOB_ID
    response = {
        "status_code": 200,
        "request_id": "2785283f-10dc-4e4a-80a0-ef4a5fbc6378",
        "code": None,
        "message": "",
        "output": {
            "deployed_model": TEST_JOB_ID,
            "status": "PENDING",
            "model_name": "qwen-turbo-ft-202307121513-5dde",
            "capacity": 2,
        },
        "usage": None,
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def delete_deployment_handler(request: aiohttp.request):
    assert request.match_info["id"] == TEST_JOB_ID
    response = {
        "code": 200,
        "request_id": "test-1223-43043",
        "output": {
            "deployed_model": "qwen-turbo-ft-202307121513-5dde",
        },
        "message": "",
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def events_deployment_handler(request: aiohttp.request):
    deployment_id = request.match_info["id"]
    assert deployment_id == TEST_JOB_ID
    response = {
        "code": 200,
        "message": "",
        "output": {
            "events": "Deployment starting\n Running",
        },
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )
