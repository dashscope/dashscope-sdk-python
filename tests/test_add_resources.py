# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from http import HTTPStatus

from dashscope import Generation
from tests.http_task_request import HttpRequest
from tests.mock_request_base import MockServerBase
from tests.mock_server import MockServer

model = Generation.Models.qwen_turbo


class TestAddResources(MockServerBase):
    text_response_obj = {
        "status_code": 200,
        "request_id": "effd2cb1-1a8c-9f18-9a49-a396f673bd40",
        "code": "",
        "message": "",
        "output": {
            "text": "hello",
            "choices": None,
            "finish_reason": "stop",
        },
        "usage": {
            "input_tokens": 27,
            "output_tokens": 110,
        },
    }

    def test_default_no_resources_request(self, mock_server: MockServer):
        response_str = json.dumps(TestAddResources.text_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        response = HttpRequest.call(
            model=model,
            prompt=prompt,
            task="text-generation",
            function="generation",
            max_tokens=1024,
            api_protocol="http",
            result_format="message",
            n=50,
        )
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["parameters"]["max_tokens"] == 1024
        assert req["parameters"]["result_format"] == "message"
        assert "resources" not in req

        assert response.status_code == HTTPStatus.OK
        assert response.output["text"] == "hello"
        assert response.output["choices"] is None
        assert response.output["finish_reason"] == "stop"

    def test_default_with_resources_request(self, mock_server: MockServer):
        response_str = json.dumps(TestAddResources.text_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        response = HttpRequest.call(
            model=model,
            prompt=prompt,
            max_tokens=1024,
            task="text-generation",
            function="generation",
            api_protocol="http",
            result_format="message",
            resources={
                "id1": 1,
                "id2": "2",
                "id3": {
                    "k": "v",
                },
            },
        )
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["parameters"]["max_tokens"] == 1024
        assert req["parameters"]["result_format"] == "message"
        assert req["resources"] == {"id1": 1, "id2": "2", "id3": {"k": "v"}}
        assert response.status_code == HTTPStatus.OK
        assert response.output["text"] == "hello"
        assert response.output["choices"] is None
        assert response.output["finish_reason"] == "stop"

    def test_default_websocket_no_resources_request(
        self,
        mock_server: MockServer,
    ):
        response_str = json.dumps(TestAddResources.text_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        HttpRequest.call(
            model=model,
            prompt=prompt,
            task="text-generation",
            function="generation",
            max_tokens=1024,
            ws_stream_mode="none",
            api_protocol="websocket",
            result_format="message",
            n=50,
        )
        req = mock_server.requests.get(block=True)
        assert req["payload"]["model"] == model
        assert req["payload"]["parameters"]["max_tokens"] == 1024
        assert req["payload"]["parameters"]["result_format"] == "message"
        assert "resources" not in req["payload"]

    def test_websocket_with_resources_request(self, mock_server: MockServer):
        response_str = json.dumps(TestAddResources.text_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        HttpRequest.call(
            model=model,
            prompt=prompt,
            task="text-generation",
            function="generation",
            max_tokens=1024,
            ws_stream_mode="none",
            api_protocol="websocket",
            result_format="message",
            n=50,
            resources={
                "id1": 1,
                "id2": "2",
                "id3": {
                    "k": "v",
                },
            },
        )
        req = mock_server.requests.get(block=True)
        assert req["payload"]["model"] == model
        assert req["payload"]["parameters"]["max_tokens"] == 1024
        assert req["payload"]["parameters"]["result_format"] == "message"
        assert req["payload"]["resources"] == {
            "id1": 1,
            "id2": "2",
            "id3": {
                "k": "v",
            },
        }
