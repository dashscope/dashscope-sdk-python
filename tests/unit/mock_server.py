# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import json
import multiprocessing
from hashlib import md5
from http import HTTPStatus
from typing import Tuple

import aiohttp
from aiohttp import web

from dashscope.protocol.websocket import ActionType
from tests.unit.constants import (
    TEST_DISABLE_DATA_INSPECTION_REQUEST_ID,
    TEST_ENABLE_DATA_INSPECTION_REQUEST_ID,
    TEST_JOB_ID,
)
from tests.unit.handle_deployment_request import (
    create_deployment_handler,
    delete_deployment_handler,
    events_deployment_handler,
    get_deployment_handler,
    list_deployment_handler,
)
from tests.unit.handle_fine_tune_request import (
    cancel_fine_tune_handler,
    create_fine_tune_handler,
    delete_fine_tune_handler,
    events_fine_tune_handler,
    get_fine_tune_handler,
    list_fine_tune_handler,
)
from tests.unit.mock_sse import sse_response
from tests.legacy.websocket_mock_server_task_handler import (
    WebSocketTaskProcessor,
)


def validate_data_inspection_parameter(request: aiohttp.request):
    if (
        "request_id" in request.headers
        and request.headers["request_id"]
        == TEST_ENABLE_DATA_INSPECTION_REQUEST_ID
    ):
        assert request.headers["X-DashScope-DataInspection"] == "enable"

    if (
        "request_id" in request.headers
        and request.headers["request_id"]
        == TEST_DISABLE_DATA_INSPECTION_REQUEST_ID
    ):
        assert "X-DashScope-DataInspection" not in request.headers


async def post_echo(request: aiohttp.request):
    validate_data_inspection_parameter(request)

    if "X-DashScope-SSE" in request.headers:
        await mock_sse(request)
        return
    body = await request.json()
    print(f"receive request json:\n {body}")
    if "messages" in body["input"]:
        input_text = body["input"]["messages"][0]["content"]
    else:
        input_text = body["input"]["prompt"]
    if (
        "result_format" in body["parameters"]
        and body["parameters"]["request_format"] == "message"
    ):
        response = {
            "output": {
                "choices": [
                    {
                        "finish_reasion": "stop",
                        "message": {
                            "role": "assistant",
                            "content": input_text,
                        },
                    },
                ],
            },
            "usage": {
                "output_tokens": 17,
                "input_tokens": 2,
            },
            "request_id": "d167c38b-bd5d-11ed-981e-00163e0d4788",
        }
    else:
        response = {
            "output": {
                "text": input_text,
            },
            "usage": {
                "output_tokens": 17,
                "input_tokens": 2,
            },
            "request_id": "d167c38b-bd5d-11ed-981e-00163e0d4788",
        }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def response_403(
    request: aiohttp.request,
):  # pylint: disable=unused-argument
    return web.Response(status=403, text='{"message": "Error api key"}')


async def websocket_handler_stream_none(request):
    validate_data_inspection_parameter(request)
    ws = aiohttp.web.WebSocketResponse(heartbeat=100)
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            req = msg.json()
            if req["header"]["action"] == ActionType.START:
                task_id = req["header"]["task_id"]
                streaming_mode = req["header"]["streaming"]
                print(f"receive first payload: {req['payload']}")
                wsc = WebSocketTaskProcessor(
                    ws,
                    task_id,
                    streaming_mode,
                    req["payload"]["model"],
                    req["payload"]["task"],
                    False,
                    False,
                    req,
                )
                await wsc.aio_call()
        await ws.close()
    return ws


async def websocket_handler_stream_in(request):
    ws = aiohttp.web.WebSocketResponse(heartbeat=100)
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            req = msg.json()
            if req["header"]["action"] == ActionType.START:
                task_id = req["header"]["task_id"]
                streaming_mode = req["header"]["streaming"]
                print(f"receive first payload: {req['payload']}")
                wsc = WebSocketTaskProcessor(
                    ws,
                    task_id,
                    streaming_mode,
                    req["payload"]["model"],
                    req["payload"]["task"],
                    True,
                    False,
                    req,
                )
                await wsc.aio_call()
        await ws.close()
    return ws


async def websocket_handler_stream_out(request):
    ws = aiohttp.web.WebSocketResponse(heartbeat=100)
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            req = msg.json()
            if req["header"]["action"] == ActionType.START:
                task_id = req["header"]["task_id"]
                streaming_mode = req["header"]["streaming"]
                print(f"receive first payload: {req['payload']}")
                wsc = WebSocketTaskProcessor(
                    ws,
                    task_id,
                    streaming_mode,
                    req["payload"]["model"],
                    req["payload"]["task"],
                    False,
                    True,
                    req,
                )
                await wsc.aio_call()
        await ws.close()
    return ws


async def websocket_handler_stream_in_out(request):
    ws = aiohttp.web.WebSocketResponse(heartbeat=100)
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            req = msg.json()
            if req["header"]["action"] == ActionType.START:
                task_id = req["header"]["task_id"]
                streaming_mode = req["header"]["streaming"]
                print(f"receive first payload: {req['payload']}")
                wsc = WebSocketTaskProcessor(
                    ws,
                    task_id,
                    streaming_mode,
                    req["payload"]["model"],
                    req["payload"]["task"],
                    True,
                    True,
                    req,
                )
                await wsc.aio_call()
        await ws.close()
    return ws


async def mock_sse(request):
    async with sse_response(request) as resp:
        for idx in range(10):
            data = f"{idx}"
            response = {
                "output": {
                    "text": data,
                },
                "usage": {
                    "output_tokens": 17,
                    "input_tokens": 2,
                },
                "request_id": "d167c38b-bd5d-11ed-981e-00163e0d4788",
            }
            response_str = json.dumps(response)
            print(f"Sending sse data: {response_str}")
            await resp.send(response_str, id=idx)
            await asyncio.sleep(1)
        print("data send completed")


async def handle_send_receive_form_data(request: aiohttp.request):
    data_hash = {
        "dog": "1d5ee55c2453009b14db98e74c453abb",
        "bird": "24c2f9abdb8809982d5bd2e10f1f98d7",
    }
    reader = await request.multipart()
    # dog, and bird,
    async for field in reader:
        print(f"multipart field: {field.name}")
        content = await field.read()
        real_md5 = md5(content).hexdigest()
        if field.name == "files":
            assert real_md5 in data_hash.values()
    # response file to client
    response = {
        "output": {
            "text": "return a text",
        },
        "usage": {
            "output_tokens": 17,
            "input_tokens": 2,
        },
        "request_id": "d167c38b-bd5d-11ed-981e-00163e0d4788",
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def handle_upload_file(request: aiohttp.request):
    dog_file_md5 = "1d5ee55c2453009b14db98e74c453abb"
    reader = await request.multipart()
    # dog, and bird,
    async for field in reader:
        content = await field.read()
        real_md5 = md5(content).hexdigest()
        assert real_md5 == dog_file_md5
    response = {
        "data": {
            "uploaded_files": [
                {
                    "file_id": "xxxx",
                    "name": "test.txt",
                },
            ],
        },
        "request_id": "d167c38b-bd5d-11ed-981e-00163e0d4788",
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def handle_list_file(
    request: aiohttp.request,
):  # pylint: disable=unused-argument
    response = {
        "request_id": "d7bd0668-8bd7-486c-8383-d1582c4b44f0",
        "code": 0,
        "msg": "操作成功",
        "data": {
            "total": 3,
            "page_size": 20,
            "page_no": 1,
            "files": [
                {
                    "id": 11,
                    "file_id": "da55d958-fbb2-4ed9-b979-f29af139d6f3",
                    "name": "fine_tune_example.jsonl",
                    "description": "testfilesfasfdsf",
                    "url": "http://dashscope.oss-cn-beijing.aliyuncs.com/api-fs/1",  # noqa: E501
                },
                {
                    "id": 10,
                    "file_id": "fedffd0c-c247-4442-ae93-cf8525786e6c",
                    "name": "fine_tune_example.jsonl",
                    "description": "testfilesfasfdsf",
                    "url": "http://dashscope.oss-cn-beijing.aliyuncs.com/api-fs/2",  # noqa: E501
                },
                {
                    "id": 9,
                    "file_id": "13ee1928-3ce4-494c-96a8-27219aec298e",
                    "name": "fine_tune_example.jsonl",
                    "description": "testfilesfasfdsf",
                    "url": "http://dashscope.oss-cn-beijing.aliyuncs.com/api-fs/3",  # noqa: E501
                },
            ],
        },
        "success": True,
    }
    return web.json_response(
        text=json.dumps(response, ensure_ascii=True),
        content_type="application/json",
    )


async def handle_get_file(request: aiohttp.request):
    id = request.match_info["id"]  # pylint: disable=redefined-builtin
    response = {
        "request_id": "e2faec4a-1183-47e5-9279-222e0a762c61",
        "code": 0,
        "msg": "操作成功",
        "data": {
            "id": 11,
            "file_id": id,
            "name": "fine_tune_example.jsonl",
            "description": "testfilesfasfdsf",
            "url": "http://dashscope.oss-cn-beijing.aliyuncs.com/api-fs/1",
        },
        "success": True,
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def handle_delete_file(request: aiohttp.request):
    id = request.match_info["id"]  # pylint: disable=redefined-builtin
    if id == "111111":
        response = {"code": "200", "success": True}
        return web.json_response(
            text=json.dumps(response),
            content_type="application/json",
        )
    elif id == "222222":
        rsp = {"code": "404", "success": False}
        return web.json_response(
            status=HTTPStatus.NOT_FOUND,
            text=json.dumps(rsp),
            content_type="application/json",
        )
    elif id == "333333":
        rsp = {"code": "403", "success": False}
        return web.json_response(
            status=HTTPStatus.FORBIDDEN,
            text=json.dumps(rsp),
            content_type="application/json",
        )
    elif id == "333333":
        rsp = {"code": "403", "success": False}
        return web.json_response(
            status=HTTPStatus.FORBIDDEN,
            text=json.dumps(rsp),
            content_type="application/json",
        )
    elif id == "444444":
        assert request.headers["Authorization"] == "Bearer api-key"
        rsp = {"code": "401", "success": False}
        return web.json_response(
            status=HTTPStatus.UNAUTHORIZED,
            text=json.dumps(rsp),
            content_type="application/json",
        )


async def list_models_handler(
    request: aiohttp.request,
):  # pylint: disable=unused-argument
    response = {
        "code": "200",
        "data": {
            "models": [
                {
                    "model_id": "1111",
                    "gmt_create": "2023-03-15 14:25:50",
                },
                {
                    "model_id": "2222",
                    "gmt_create": "2023-03-15 14:25:50",
                },
            ],
        },
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def get_model_handler(request: aiohttp.request):
    model_id = request.match_info["id"]
    assert model_id == TEST_JOB_ID
    response = {
        "code": "200",
        "data": {
            "model_id": TEST_JOB_ID,
            "name": "gpt3",
        },
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


async def handle_text_embedding(request: aiohttp.request):
    body = await request.json()
    assert len(body["input"]["texts"]) >= 1
    response = {
        "output": {
            "embeddings": [
                {
                    "text_index": 0,
                    "embedding": [
                        -0.006929283495992422,
                        -0.005336422007530928,
                    ],
                },
            ],
        },
        "usage": {
            "input_tokens": 12,
        },
        "request_id": "d89c06fb-46a1-47b6-acb9-bfb17f814969",
    }
    return web.json_response(
        text=json.dumps(response),
        content_type="application/json",
    )


def create_app():
    app = web.Application()
    app.router.add_post("/api/v1/services/aigc/generation", post_echo)
    app.router.add_post(
        "/api/v1/services/aigc/text-generation/generation",
        post_echo,
    )
    app.router.add_post(
        "/api/v1/services/embeddings/text-embedding/text-embedding",
        handle_text_embedding,
    )
    app.router.add_post("/api/v1/services/aigc/forbidden", response_403)
    app.router.add_post(
        "/api/v1/services/aigc/image-generation/generation",
        handle_send_receive_form_data,
    )
    app.router.add_route(
        "GET",
        "/ws/aigc/v1/echo",
        websocket_handler_stream_none,
    )
    app.router.add_route("GET", "/ws/aigc/v1/in", websocket_handler_stream_in)
    app.router.add_route(
        "GET",
        "/ws/aigc/v1/out",
        websocket_handler_stream_out,
    )
    app.router.add_route(
        "GET",
        "/ws/aigc/v1/inout",
        websocket_handler_stream_in_out,
    )
    # file upload
    app.router.add_post("/api/v1/files", handle_upload_file)
    app.router.add_get("/api/v1/files", handle_list_file)
    app.router.add_get("/api/v1/files/{id}", handle_get_file)
    app.router.add_delete("/api/v1/files/{id}", handle_delete_file)
    # fine-tune
    app.router.add_post("/api/v1/fine-tunes", create_fine_tune_handler)
    app.router.add_get("/api/v1/fine-tunes", list_fine_tune_handler)
    app.router.add_get("/api/v1/fine-tunes/outputs", list_fine_tune_handler)
    app.router.add_get("/api/v1/fine-tunes/{id}", get_fine_tune_handler)
    app.router.add_get(
        "/api/v1/fine-tunes/outputs/{id}",
        get_fine_tune_handler,
    )
    app.router.add_delete("/api/v1/fine-tunes/{id}", delete_fine_tune_handler)
    app.router.add_delete(
        "/api/v1/fine-tunes/outputs/{id}",
        delete_fine_tune_handler,
    )
    app.router.add_post(
        "/api/v1/fine-tunes/{id}/cancel",
        cancel_fine_tune_handler,
    )
    app.router.add_get(
        "/api/v1/fine-tunes/{id}/stream",
        events_fine_tune_handler,
    )

    app.router.add_post("/api/v1/deployments", create_deployment_handler)
    app.router.add_get("/api/v1/deployments", list_deployment_handler)
    app.router.add_get("/api/v1/deployments/{id}", get_deployment_handler)
    app.router.add_delete(
        "/api/v1/deployments/{id}",
        delete_deployment_handler,
    )
    app.router.add_get(
        "/api/v1/deployments/{id}/events",
        events_deployment_handler,
    )

    app.router.add_get("/api/v1/models", list_models_handler)
    app.router.add_get("/api/v1/models/{id}", get_model_handler)

    runner = web.AppRunner(app)
    return runner


def run_server(runner):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    loop.run_until_complete(site.start())
    loop.run_forever()


class MockServer:
    def __init__(self) -> None:
        self.requests = multiprocessing.Queue()
        self.responses = multiprocessing.Queue()
        app = web.Application()
        app.router.add_post(
            "/api/v1/services/rerank/text-rerank/text-rerank",
            self.handle_post,
        )
        # fine-tune
        app.router.add_post("/api/v1/fine-tunes", self.handle_post)
        app.router.add_get("/api/v1/fine-tunes", self.handle_get)
        app.router.add_get("/api/v1/fine-tunes/outputs", self.handle_get)
        app.router.add_get("/api/v1/fine-tunes/{id}", self.handle_get)
        app.router.add_get("/api/v1/fine-tunes/outputs/{id}", self.handle_get)
        app.router.add_delete("/api/v1/fine-tunes/{id}", self.handle_get)
        app.router.add_delete(
            "/api/v1/fine-tunes/outputs/{id}",
            self.handle_post,
        )
        app.router.add_post(
            "/api/v1/fine-tunes/{id}/cancel",
            self.handle_get,
        )  # no body
        app.router.add_get(
            "/api/v1/fine-tunes/{id}/stream",
            events_fine_tune_handler,
        )
        # end of finetune
        # create assistant file
        app.router.add_post(
            "/api/v1/assistants/{assistant_id}/files",
            self.handle_post,
        )
        # retrieve assistant file
        app.router.add_get(
            "/api/v1/assistants/{assistant_id}/files/{file_id}",
            self.handle_get,
        )
        # delete assistant file
        app.router.add_delete(
            "/api/v1/assistants/{assistant_id}/files/{file_id}",
            self.handle_get,
        )
        # list assistant file
        app.router.add_get(
            "/api/v1/assistants/{assistant_id}/files",
            self.handle_get,
        )

        # create messages
        app.router.add_post(
            "/api/v1/threads/{thread_id}/messages",
            self.handle_create_object,
        )
        # list messages
        app.router.add_get(
            "/api/v1/threads/{thread_id}/messages",
            self.handle_list_object,
        )
        # retrieve message
        app.router.add_get(
            "/api/v1/threads/{thread_id}/messages/{message_id}",
            self.handle_list_object,
        )

        # create run
        app.router.add_post(
            "/api/v1/threads/{thread_id}/runs",
            self.handle_create_object,
        )
        # list runs
        app.router.add_get(
            "/api/v1/threads/{thread_id}/runs",
            self.handle_list_object,
        )
        # retrieve run
        app.router.add_get(
            "/api/v1/threads/{thread_id}/runs/{run_id}",
            self.handle_list_object,
        )
        # cancel run
        app.router.add_post(
            "/api/v1/threads/{thread_id}/runs/{run_id}/cancel",
            self.handle_cancel_object,
        )
        # retrieve run steps
        app.router.add_get(
            "/api/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}",
            self.handle_list_object,
        )
        # list message files
        app.router.add_get(
            "/api/v1/threads/{thread_id}/messages/{message_id}/files",
            self.handle_list_object,
        )
        # retrieve message file
        app.router.add_get(
            "/api/v1/threads/{thread_id}/messages/{message_id}/files/{file_id}",  # noqa: E501
            self.handle_list_object,
        )
        # retrieve message file
        app.router.add_post(
            "/api/v1/threads/{thread_id}/messages/{message_id}",
            self.handle_post,
        )
        # submit tool result
        app.router.add_post(
            "/api/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
            self.handle_create_object,
        )
        app.router.add_get(
            "/api/v1/threads/{thread_id}/runs/{run_id}/steps",
            self.handle_list_object,
        )

        app.router.add_post(
            "/api/v1/services/{group}/{task}/{function}",
            self.handle_mock_request,
        )
        app.router.add_post(
            "/api/v1/{group}/{task}/{function}",
            self.handle_mock_request,
        )
        app.router.add_route(
            "GET",
            "/api-ws/v1/inference",
            self.websocket_handler,
        )
        # create an object
        app.router.add_post("/api/v1/{function}", self.handle_create_object)
        # list objects
        app.router.add_get("/api/v1/{function}", self.handle_list_object)
        # delete object
        app.router.add_delete(
            "/api/v1/{function}/{object_id}",
            self.handle_delete_object,
        )
        # retrieve object
        app.router.add_get(
            "/api/v1/{function}/{object_id}",
            self.handle_retrieve_object,
        )
        # update with post
        app.router.add_post(
            "/api/v1/{function}/{object_id}",
            self.handle_update_object_with_post,
        )

        self.runner = web.AppRunner(app)

    def process_response(self, rsp_str) -> Tuple[int, str]:
        """Input response string, output status_code and new response string

        Args:
            rsp_str (_type_): _description_
        """
        rsp_json = json.loads(rsp_str)
        status_code = 200
        if "status_code" in rsp_json:
            status_code = rsp_json.pop("status_code")
        rsp_str = json.dumps(rsp_json)
        return status_code, rsp_str

    async def handle_get(self, request: aiohttp.web.BaseRequest):
        """Handle get request, put path, headers to requests.

        Args:
            request (aiohttp.request): The request
        """
        headers = {}
        # convert raw bytes to str
        for key, value in request.raw_headers:
            headers[key.decode("utf-8")] = value.decode("utf-8")
        self.requests.put({"path": request.raw_path, "headers": headers})
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        obj = json.loads(rsp)
        return web.json_response(
            text=json.dumps(obj),
            status=status_code,
            content_type="application/json",
        )

    async def handle_post(self, request: aiohttp.web.BaseRequest):
        """Handle post request, put path, body, headers to requests.

        Args:
            request (aiohttp.request): The request

        """
        body = await request.json()
        headers = {}
        # convert raw bytes to str
        for key, value in request.raw_headers:
            headers[key.decode("utf-8")] = value.decode("utf-8")
        self.requests.put(
            {
                "body": body,
                "path": request.raw_path,
                "headers": headers,
            },
        )
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        obj = json.loads(rsp)
        return web.json_response(
            text=json.dumps(obj),
            status=status_code,
            content_type="application/json",
        )

    def handle_cancel_object(self, request: aiohttp.request):
        self.requests.put(request.raw_path)
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        return web.json_response(
            text=rsp,
            status=status_code,
            content_type="application/json",
        )

    def handle_delete_object(self, request: aiohttp.request):
        object_id = request.match_info["object_id"]
        self.requests.put(object_id)
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        return web.json_response(
            text=rsp,
            status=status_code,
            content_type="application/json",
        )

    # response path of request
    def handle_list_object(self, request: aiohttp.request):
        self.requests.put(request.raw_path)
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        return web.json_response(
            text=rsp,
            status=status_code,
            content_type="application/json",
        )

    async def handle_update_object_with_post(self, request: aiohttp.request):
        func = request.match_info["function"]
        object_id = request.match_info["object_id"]
        print(f"function: {func}")
        body = await request.json()
        self.requests.put(body)
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        obj = json.loads(rsp)
        obj["id"] = object_id
        return web.json_response(
            text=json.dumps(obj),
            status=status_code,
            content_type="application/json",
        )

    def handle_retrieve_object(self, request: aiohttp.request):
        func = request.match_info["function"]
        object_id = request.match_info["object_id"]
        print(f"Retrieve {func}, object_id: {object_id}")
        self.requests.put(object_id)
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        return web.json_response(
            text=rsp,
            status=status_code,
            content_type="application/json",
        )

    def add_response(self, response: web.Response):
        self.responses.append(response)

    def get_runner(self):
        return self.runner

    def set_responses(self, responses):
        self.responses = responses

    def set_requests(self, requests):
        self.requests = requests

    async def handle_create_object(self, request: aiohttp.request):
        body = await request.json()
        self.requests.put(body)
        rsp = self.responses.get(block=True)
        status_code, rsp = self.process_response(rsp)
        return web.json_response(
            text=rsp,
            status=status_code,
            content_type="application/json",
        )

    async def handle_mock_request(self, request: aiohttp.request):
        group = request.match_info["group"]
        task = request.match_info["task"]
        func = request.match_info["function"]
        print(f"group: {group}, task: {task}, function: {func}")
        body = await request.json()
        print("handle_mock_request body", str(body))
        self.requests.put(body)
        rsp = self.responses.get(block=True)

        # For testing error response purpose
        status = 200
        try:
            json_resp = json.loads(rsp)
            status = json_resp.get("status_code", 200)
        except Exception:
            print(
                "can not find status code from response, will use default 200",
            )

        return web.json_response(
            status=status,
            text=rsp,
            content_type="application/json",
        )

    async def websocket_handler(self, request):
        ws = aiohttp.web.WebSocketResponse(heartbeat=100)
        await ws.prepare(request)

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                req = msg.json()
                self.requests.put(req)
                if req["header"]["action"] == ActionType.START:
                    task_id = req["header"]["task_id"]
                    streaming_mode = req["header"]["streaming"]
                    print(f"receive first payload: {req['payload']}")
                    wsc = WebSocketTaskProcessor(
                        ws,
                        task_id,
                        streaming_mode,
                        req["payload"]["model"],
                        req["payload"]["task"],
                        False,
                        False,
                        req,
                    )
                    await wsc.aio_call()
            await ws.close()
        return ws


def http_server():
    runner = create_app()
    proc = multiprocessing.Process(  # pylint: disable=redefined-outer-name
        target=run_server,
        args=(runner,),
    )
    proc.start()

    def stop_server():  # pylint: disable=unused-variable
        proc.terminate()
        print("Server stopped")

    return proc


def run_mock_server(requests, responses):
    from signal import signal, SIGPIPE, SIG_DFL

    signal(SIGPIPE, SIG_DFL)
    mock_web_server = MockServer()
    mock_web_server.requests = requests
    mock_web_server.responses = responses
    runner = mock_web_server.get_runner()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", 8089)
    loop.run_until_complete(site.start())
    loop.run_forever()

    print("Server started!!!!!!!!!!!")


def create_mock_server(request):
    mock_web_server = MockServer()
    proc = multiprocessing.Process(  # pylint: disable=redefined-outer-name
        target=run_mock_server,
        args=(
            mock_web_server.requests,
            mock_web_server.responses,
        ),
    )
    proc.start()
    import time

    time.sleep(2)

    def stop_server():
        proc.terminate()
        print("Mock server stopped")

    request.addfinalizer(stop_server)

    return mock_web_server


if __name__ == "__main__":
    proc = http_server()
    proc.join()
