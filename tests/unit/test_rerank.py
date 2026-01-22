# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import uuid

from dashscope import TextReRank
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer


class TestReRank(MockServerBase):
    def test_call(self, mock_server: MockServer):
        response_body = {
            "output": {
                "results": [
                    {
                        "index": 1,
                        "relevance_score": 0.987654,
                        "document": {  # 如果return_documents=true
                            "text": "哈尔滨是中国黑龙江省的省会，位于中国东北",
                        },
                    },
                    {
                        "index": 0,
                        "relevance_score": 0.876543,
                        "document": {  # 如果return_documents=true
                            "text": "黑龙江离俄罗斯很近",
                        },
                    },
                ],
            },
            "usage": {
                "input_tokens": 1279,
            },
            "request_id": "b042e72d-7994-97dd-b3d2-7ee7e0140525",
        }
        mock_server.responses.put(json.dumps(response_body))
        model = str(uuid.uuid4())
        query = str(uuid.uuid4())
        documents = [
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]
        response = TextReRank.call(
            model=model,
            query=query,
            documents=documents,
            return_documents=False,
            top_n=10,
        )
        req = mock_server.requests.get(block=True)
        assert req["path"] == "/api/v1/services/rerank/text-rerank/text-rerank"
        assert req["body"]["parameters"] == {
            "return_documents": False,
            "top_n": 10,
        }
        assert req["body"]["input"] == {"query": query, "documents": documents}
        assert response.usage["input_tokens"] == 1279
        assert len(response.output["results"]) == 2
        assert response.output["results"][0]["index"] == 1
        assert response.output["results"][1]["document"]["text"] == "黑龙江离俄罗斯很近"
