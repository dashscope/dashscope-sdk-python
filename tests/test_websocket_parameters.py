# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import dashscope
from dashscope.protocol.websocket import WebsocketStreamingMode
from tests.base_test import BaseTestEnvironment
from tests.constants import (
    TEST_DISABLE_DATA_INSPECTION_REQUEST_ID,
    TEST_ENABLE_DATA_INSPECTION_REQUEST_ID,
    TestTasks,
)
from tests.websocket_task_request import WebSocketRequest


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append([x[1] for x in items])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


batch = ("batch", {"stream": False})
stream = ("stream", {"stream": True})


def request_generator():
    yield "hello"


# set mock server url.
base_websocket_api_url = "ws://localhost:8080/ws/aigc/v1"


# text output: text binary out put: image
# text input: prompt binary input: video
class TestWebSocketSyncRequest(BaseTestEnvironment):
    scenarios = [batch, stream]

    # test streaming none
    def test_default_disable_data_inspection(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            prompt="hello",
            task=TestTasks.streaming_none_text_to_text,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
            headers={"request_id": TEST_DISABLE_DATA_INSPECTION_REQUEST_ID},
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "hello"
        else:
            assert responses.output["text"] == "hello"

    def test_disable_data_inspection(
        self,
        stream,
        http_server,
        mock_disable_data_inspection_env,
    ):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            prompt="hello",
            task=TestTasks.streaming_none_text_to_text,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
            headers={"request_id": TEST_DISABLE_DATA_INSPECTION_REQUEST_ID},
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "hello"
        else:
            assert responses.output["text"] == "hello"

    def test_enable_data_inspection_by_env(
        self,
        stream,
        http_server,
        mock_enable_data_inspection_env,
    ):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            prompt="hello",
            task=TestTasks.streaming_none_text_to_text,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
            headers={"request_id": TEST_ENABLE_DATA_INSPECTION_REQUEST_ID},
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "hello"
        else:
            assert responses.output["text"] == "hello"
