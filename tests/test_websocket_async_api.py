# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import pytest

import dashscope
from dashscope.protocol.websocket import WebsocketStreamingMode
from tests.base_test import BaseTestEnvironment
from tests.constants import TestTasks
from tests.websocket_task_request import WebSocketRequest

# set mock server url.
base_websocket_api_url = "ws://localhost:8080/ws/aigc/v1"


# text output: text binary out put: image
# text input: prompt binary input: video
# @pytest.mark.skip(reason='Not support async interface')
class TestWebSocketAsyncRequest(BaseTestEnvironment):
    stream = True

    #  test streaming none
    @pytest.mark.asyncio
    async def test_streaming_none_text_to_text(self, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_none_text_to_text,
            prompt="hello",
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "hello"
        else:
            assert responses.output["text"] == "hello"

    @pytest.mark.asyncio
    async def test_streaming_none_text_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_none_text_to_binary,
            prompt="hello",
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    @pytest.mark.asyncio
    async def test_streaming_none_binary_to_text(self, http_server):
        dashscope.base_websocket_api_url = (
            "%s/in" % base_websocket_api_url
        )  # noqa E501
        video = bytes([0x01] * 100)
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_none_binary_to_text,
            prompt=video,
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=True,
            n=50,
        )

        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    @pytest.mark.asyncio
    async def test_streaming_none_binary_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url
        video = bytes([0x01] * 100)
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_none_binary_to_binary,
            prompt=video,
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.NONE,
            is_binary_input=True,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    #  test string in
    @pytest.mark.asyncio
    async def test_streaming_in_text_to_text(self, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_text_to_text,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert (
                responses.output["text"] == "world"
            )  # echo the input out(input 10)

    @pytest.mark.asyncio
    async def test_streaming_in_text_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_text_to_binary,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    @pytest.mark.asyncio
    async def test_streaming_in_binary_to_text(self, http_server):
        dashscope.base_websocket_api_url = (
            "%s/in" % base_websocket_api_url
        )  # noqa E501

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_binary_to_text,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=True,
            n=50,
        )

        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    @pytest.mark.asyncio
    async def test_streaming_in_binary_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_in_binary_to_binary,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.IN,
            is_binary_input=True,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    #  streaming out
    @pytest.mark.asyncio
    async def test_streaming_out_text_to_text(self, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_text_to_text,
            prompt="hello",
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    @pytest.mark.asyncio
    async def test_streaming_out_text_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_text_to_binary,
            prompt="hello",
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert len(responses.output) and responses.output == bytes(
                [0x01] * 100,
            )

    @pytest.mark.asyncio
    async def test_streaming_out_binary_to_text(self, http_server):
        dashscope.base_websocket_api_url = (
            "%s/in" % base_websocket_api_url
        )  # noqa E501
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_binary_to_text,
            prompt=bytes([0x01] * 100),
            max_tokens=1024,
            stream=self.stream,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=True,
            n=50,
        )

        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    @pytest.mark.asyncio
    async def test_streaming_out_binary_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url
        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_out_binary_to_binary,
            prompt=bytes([0x01] * 100),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.OUT,
            is_binary_input=True,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert len(responses.output) == 100 and responses.output == bytes(
                [0x01] * 100,
            )

    #  streaming duplex
    @pytest.mark.asyncio
    async def test_streaming_duplex_text_to_text(self, http_server):
        dashscope.base_websocket_api_url = "%s/echo" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_text_to_text,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    @pytest.mark.asyncio
    async def test_streaming_duplex_text_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/out" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield "input message %s" % i

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_text_to_binary,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=False,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)

    @pytest.mark.asyncio
    async def test_streaming_duplex_binary_to_text(self, http_server):
        dashscope.base_websocket_api_url = (
            "%s/in" % base_websocket_api_url
        )  # noqa E501

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_binary_to_text,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=True,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output["text"] == "world"
        else:
            assert responses.output["text"] == "world"

    @pytest.mark.asyncio
    async def test_streaming_duplex_binary_to_binary(self, http_server):
        dashscope.base_websocket_api_url = "%s/inout" % base_websocket_api_url

        def make_input():
            for i in range(10):
                yield bytes([0x01] * 100)

        responses = await WebSocketRequest.aio_call(
            model="qwen-turbo",
            task=TestTasks.streaming_duplex_binary_to_binary,
            prompt=make_input(),
            stream=self.stream,
            max_tokens=1024,
            ws_stream_mode=WebsocketStreamingMode.DUPLEX,
            is_binary_input=True,
            n=50,
        )
        if self.stream:
            async for resp in responses:
                assert resp.output == bytes([0x01] * 100)
        else:
            assert responses.output == bytes([0x01] * 100)
