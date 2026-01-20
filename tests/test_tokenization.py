# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from http import HTTPStatus

from dashscope import Tokenization
from tests.mock_request_base import MockServerBase
from tests.mock_server import MockServer


class TestTokenization(MockServerBase):
    text_response_obj = {
        "output": {
            "token_ids": [115798, 198],
            "tokens": ["<|im_start|>", "\n"],
            "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n如何做土豆炖猪脚?<|im_end|>\n<|im_start|>assistant\n",  # noqa E501
        },
        "usage": {
            "input_tokens": 28,
        },
        "request_id": "c25e14cf-f986-900b-853c-4644a3196f39",
    }

    def test_default_no_resources_request(self, mock_server: MockServer):
        response_str = json.dumps(TestTokenization.text_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        model = Tokenization.Models.qwen_turbo
        response = Tokenization.call(model=model, prompt=prompt)
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["parameters"] == {}
        assert req["input"] == {"prompt": prompt}

        assert response.status_code == HTTPStatus.OK
        assert len(response.output["token_ids"]) == 2
        assert len(response.output["tokens"]) == 2
        assert response.usage["input_tokens"] == 28
