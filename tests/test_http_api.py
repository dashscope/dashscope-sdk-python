# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from dashscope import Generation
from tests.constants import (
    TEST_DISABLE_DATA_INSPECTION_REQUEST_ID,
    TEST_ENABLE_DATA_INSPECTION_REQUEST_ID,
)
from tests.http_task_request import HttpRequest
from tests.mock_request_base import MockRequestBase


def request_generator():
    yield "hello"


class TestHttpRequest(MockRequestBase):
    def test_independent_model_sync_batch_request(self, http_server):
        resp = Generation.call(
            model=Generation.Models.qwen_turbo,
            prompt="hello",
            max_tokens=1024,
            api_protocol="http",
            n=50,
            headers={"request_id": TEST_DISABLE_DATA_INSPECTION_REQUEST_ID},
        )
        assert resp.output.text == "hello"
        assert resp.output["text"] == "hello"

    def test_disable_data_inspection(
        self,
        http_server,
        mock_disable_data_inspection_env,
    ):
        resp = Generation.call(
            model=Generation.Models.qwen_turbo,
            prompt="hello",
            max_tokens=1024,
            api_protocol="http",
            n=50,
            headers={"request_id": TEST_DISABLE_DATA_INSPECTION_REQUEST_ID},
        )
        assert resp.output.text == "hello"
        assert resp.output["text"] == "hello"

    def test_enable_data_inspection(
        self,
        http_server,
        mock_enable_data_inspection_env,
    ):
        resp = Generation.call(
            model=Generation.Models.qwen_turbo,
            prompt="hello",
            max_tokens=1024,
            api_protocol="http",
            n=50,
            headers={"request_id": TEST_ENABLE_DATA_INSPECTION_REQUEST_ID},
        )
        assert resp.output.text == "hello"
        assert resp.output["text"] == "hello"

    def test_independent_model_sync_stream_request(self, http_server):
        resp = Generation.call(
            model=Generation.Models.qwen_turbo,
            prompt="hello",
            max_tokens=1024,
            stream=True,
            api_protocol="http",
            n=50,
        )
        for idx, rsp in enumerate(resp):
            assert rsp.output.text == str(idx)
            print(rsp.output["text"])

    def test_echo_request_with_file_object(self, http_server):
        with open("tests/data/request_file.bin") as f:
            resp = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=f,
                max_tokens=1024,
                api_protocol="http",
                n=50,
            )
            assert resp.output.text[0] == "hello"

    def test_echo_request_with_generator(self, http_server):
        resp = Generation.call(
            model=Generation.Models.qwen_turbo,
            prompt=request_generator(),
            max_tokens=1024,
            api_protocol="http",
            n=50,
        )
        assert resp.output.text == "hello"

    def test_send_receive_files(self, http_server):
        bird_file = open("tests/data/bird.JPEG", "rb")
        dogs_file = open("tests/data/dogs.jpg", "rb")
        resp = HttpRequest.call(
            model=Generation.Models.qwen_turbo,
            prompt="hello",
            task_group="aigc",
            task="image-generation",
            function="generation",
            api_protocol="http",
            request_id="1111111111",
            form={
                "bird": bird_file,
                "dog": dogs_file,
            },
        )
        assert resp.status_code == HTTPStatus.OK
