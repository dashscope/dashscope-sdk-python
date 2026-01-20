# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from http import HTTPStatus

from dashscope import ImageSynthesis
from tests.mock_request_base import MockServerBase
from tests.mock_server import MockServer


class TestSketchImageSynthesis(MockServerBase):
    text_response_obj = {
        "status_code": 200,
        "request_id": "effd2cb1-1a8c-9f18-9a49-a396f673bd40",
        "code": "",
        "message": "",
        "output": {
            "task_id": "hello",
            "task_status": "SUCCEEDED",
            "results": [
                {
                    "url": "url1",
                },
            ],
        },
        "usage": {
            "image_count": 4,
        },
    }

    def test_with_all_parameters(self, mock_server: MockServer):
        response_str = json.dumps(TestSketchImageSynthesis.text_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        response = ImageSynthesis.async_call(
            model=ImageSynthesis.Models.wanx_sketch_to_image_v1,
            prompt=prompt,
            sketch_image_url="http://sketch_url",
            n=4,
            size="1024*1024",
            sketch_weight=8,
            realisticness=9,
        )
        req = mock_server.requests.get(block=True)
        expect_req_str = '{"model": "wanx-sketch-to-image-v1", "parameters": {"n": 4, "size": "1024*1024", "sketch_weight": 8, "realisticness": 9}, "input": {"prompt": "hello", "sketch_image_url": "http://sketch_url"}}'  # noqa E501
        expect_req = json.loads(expect_req_str)
        assert expect_req == req

        assert response.status_code == HTTPStatus.OK
        assert response.output.results[0]["url"] == "url1"

    def test_with_not_all_parameters(self, mock_server: MockServer):
        response_str = json.dumps(TestSketchImageSynthesis.text_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        response = ImageSynthesis.async_call(
            model=ImageSynthesis.Models.wanx_sketch_to_image_v1,
            prompt=prompt,
            sketch_image_url="http://sketch_url",
            n=4,
            size="1024*1024",
            realisticness=9,
        )
        req = mock_server.requests.get(block=True)
        expect_req_str = '{"model": "wanx-sketch-to-image-v1", "parameters": {"n": 4, "size": "1024*1024", "realisticness": 9}, "input": {"prompt": "hello", "sketch_image_url": "http://sketch_url"}}'  # noqa E501
        expect_req = json.loads(expect_req_str)
        assert expect_req == req

        assert response.status_code == HTTPStatus.OK
        assert response.output.results[0]["url"] == "url1"
