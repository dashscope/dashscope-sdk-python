# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from http import HTTPStatus

from dashscope import Understanding
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer

model = Understanding.Models.opennlu_v1


class TestUnderstandingRequest(MockServerBase):
    text_response_obj = {
        "status_code": 200,
        "request_id": "addddde9-1b8b-9707-979a-09b61f91302d",
        "code": "",
        "message": "",
        "output": {
            "rt": 0.06531630805693567,
            "text": "积极;",
        },
        "usage": {
            "total_tokens": 22,
            "output_tokens": 2,
            "input_tokens": 20,
        },
    }

    def test_http_text_call(self, mock_server: MockServer):
        response_str = json.dumps(TestUnderstandingRequest.text_response_obj)
        mock_server.responses.put(response_str)
        response = Understanding.call(
            model=Understanding.Models.opennlu_v1,
            sentence="老师今天表扬我了",
            labels="积极，消极",
            task="classification",
        )
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["input"]["sentence"] == "老师今天表扬我了"
        assert req["input"]["labels"] == "积极，消极"
        assert req["input"]["task"] == "classification"
        assert req["parameters"] == {}

        assert response.status_code == HTTPStatus.OK
        assert response.request_id == "addddde9-1b8b-9707-979a-09b61f91302d"
        assert response.output["rt"] == 0.06531630805693567
        assert response.output["text"] == "积极;"
        assert response.usage["total_tokens"] == 22
        assert response.usage["output_tokens"] == 2
        assert response.usage["input_tokens"] == 20
