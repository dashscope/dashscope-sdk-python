# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import dashscope
from dashscope.protocol.websocket import WebsocketStreamingMode
from tests.base_test import BaseTestEnvironment
from tests.constants import TestTasks
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
    def test_streaming_none_text_to_text(self, stream, http_server):
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
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "hello"
        else:
            assert responses.output["text"] == "hello"

    def test_streaming_none_text_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_none_text_to_binary,
            prompt="hello",
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    def test_streaming_none_binary_to_text(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/in" % base_websocket_api_url
        video = bytes([0x01] * 100)
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_none_binary_to_text,
            prompt=video,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )

        if stream:
            for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    def test_streaming_none_binary_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url
        video = bytes([0x01] * 100)
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_none_binary_to_binary,
            prompt=video,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    # test string in
    def test_streaming_in_text_to_text(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_text_to_text,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert len(resp.output) == 1
        else:
            assert len(responses.output) == 1  # echo the input out.

    def test_streaming_in_text_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_text_to_binary,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    def test_streaming_in_text_to_text_with_file(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        text_file = open("tests/data/multi_line.txt", encoding="utf-8")

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_text_to_text,
            prompt=text_file,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert len(resp.output) == 1
        else:
            assert len(responses.output) == 1  # echo the input out.

    def test_streaming_in_text_to_binary_generator(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_text_to_binary,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    def test_streaming_in_binary_to_text(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/in" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_binary_to_text,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )

        if stream:
            for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    def test_streaming_in_binary_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_binary_to_binary,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    # streaming out
    def test_streaming_out_text_to_text(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_text_to_text,
            prompt="hello",
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "world"
        else:
            responses.output["text"] == "world"

    def test_streaming_out_text_to_text_stream(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_text_to_text,
            prompt="hello",
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "hello"

    def test_streaming_out_text_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_text_to_binary,
            prompt="hello",
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    def test_streaming_out_binary_to_text(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/in" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_binary_to_text,
            prompt=bytes([0x01] * 100),
            max_tokens=1024,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=True,
            n=50,
        )

        if stream:
            for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    def test_streaming_out_binary_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_binary_to_binary,
            prompt=bytes([0x01] * 100),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    #  streaming duplex
    def test_streaming_duplex_text_to_text(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_text_to_text,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    def test_streaming_duplex_text_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_text_to_binary,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=False,
            max_tokens=1024,
            n=50,
        )

        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    def test_streaming_duplex_binary_to_text(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/in" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_binary_to_text,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    def test_streaming_duplex_binary_to_binary(self, stream, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_binary_to_binary,
            prompt=make_input(),
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    def test_streaming_duplex_binary_to_binary_with_input_file(
        self,
        stream,
        http_server,
    ):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url
        binary_file = open(
            "tests/data/action_recognition_test_video.mp4",
            "rb",
        )  # TODO no rb
        responses = WebSocketRequest.call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_binary_to_binary,
            prompt=binary_file,
            stream=stream,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=True,
            max_tokens=1024,
            n=50,
        )
        if stream:
            for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)
