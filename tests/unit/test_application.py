# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
@File    :   test_completion.py
@Date    :   2024-02-26
@Desc    :   Application call test cases
"""
import json
import uuid
from typing import Dict

from dashscope import Application
from dashscope.app.application_response import ApplicationResponse
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer


class TestCompletion(MockServerBase):
    def test_rag_call(self, mock_server: MockServer):
        test_response = {
            "status_code": 200,
            "request_id": str(uuid.uuid4()),
            "output": {
                "text": "API接口说明中，通过parameters的topP属性设置，取值范围在(0,1.0)。",
                "finish_reason": "stop",
                "session_id": str(uuid.uuid4()),
                "doc_references": [
                    {
                        "index_id": "1",
                        "doc_id": "1234",
                        "doc_name": "API接口说明.pdf",
                        "doc_url": "https://127.0.0.1/dl/API接口说明.pdf",
                        "title": "API接口说明",
                        "text": "topP取值范围在(0,1.0),取值越大,生成的随机性越高",
                        "biz_id": "2345",
                        "images": [
                            "http://127.0.0.1:8080/qqq.png",
                            "http://127.0.0.1:8080/www.png",
                        ],
                    },
                ],
                "thoughts": [
                    {
                        "thought": "开启了文档增强，优先检索文档内容",
                        "action_type": "api",
                        "action_name": "文档检索",
                        "action": "searchDocument",
                        "action_input_stream": (
                            '{"query":"API接口说明中, ' 'TopP参数改如何传递?"}'
                        ),
                        "action_input": {
                            "query": ("API接口说明中, TopP参数改如何传递?"),
                        },
                        "observation": (
                            '{"data": [{"docId": "1234", '
                            '"docName": "API接口说明", '
                            '"docUrl": "https://127.0.0.1/dl/'
                            'API接口说明.pdf", "indexId": "1", '
                            '"score": 0.11992252, "text": "填(0,1.0),'
                            "取值越大,生成的随机性越高;启用文档检索后,"
                            '文档引用类型,取值包括:simple|indexed。", '
                            '"title": "API接口说明", "titlePath": '
                            '"API接口说明>>>接口说明>>>是否必   说明>>>填"}], '
                            '"status": "SUCCESS"}'
                        ),
                        "response": (
                            "API接口说明中, TopP参数是一个float类型的"
                            "参数,取值范围为0到1.0,默认为1.0。取值越大,"
                            "生成的随机性越高。[5]"
                        ),
                    },
                ],
            },
            "usage": {
                "models": [
                    {
                        "model_id": "123",
                        "input_tokens": 27,
                        "output_tokens": 110,
                    },
                ],
            },
        }

        mock_server.responses.put(json.dumps(test_response))
        resp = Application.call(
            app_id="1234",
            workspace="ws_1234",
            prompt="API接口说明中, TopP参数改如何传递?",
            top_p=0.2,
            temperature=1.0,
            doc_tag_codes=["t1234", "t2345"],
            doc_reference_type=(Application.DocReferenceType.simple),
            has_thoughts=True,
        )

        # Test mock response type
        self.check_result(resp, test_response)  # type: ignore[arg-type]

    def test_flow_call(self, mock_server: MockServer):
        test_response = {
            "status_code": 200,
            "request_id": str(uuid.uuid4()),
            "output": {
                "text": "当月的居民用电量为102千瓦。",
                "finish_reason": "stop",
                "thoughts": [
                    {
                        "thought": "开启了插件增强",
                        "action_type": "api",
                        "action_name": "plugin",
                        "action": "api",
                        "action_input_stream": (
                            '{"userId": "123", "date": "202402", '
                            '"city": "hangzhou"}'
                        ),
                        "action_input": {
                            "userId": "123",
                            "date": "202402",
                            "city": "hangzhou",
                        },
                        "observation": (
                            '{"quantity": 102, "type": "resident", '
                            '"date": "202402", "unit": "千瓦"}'
                        ),
                        "response": "当月的居民用电量为102千瓦。",
                    },
                ],
            },
            "usage": {
                "models": [
                    {
                        "model_id": "123",
                        "input_tokens": 50,
                        "output_tokens": 33,
                    },
                ],
            },
        }

        mock_server.responses.put(json.dumps(test_response))

        biz_params = {"userId": "123"}

        resp = Application.call(
            app_id="1234",
            prompt="本月的用电量是多少?",
            workspace="ws_1234",
            top_p=0.2,
            biz_params=biz_params,
            has_thoughts=True,
        )

        # Test mock response type
        self.check_result(resp, test_response)  # type: ignore[arg-type]

    def test_call_with_error(self, mock_server: MockServer):
        test_response = {
            "status_code": 400,
            "request_id": str(uuid.uuid4()),
            "code": "InvalidAppId",
            "message": "App id is invalid",
        }

        mock_server.responses.put(json.dumps(test_response))
        resp = Application.call(
            app_id="1234",
            workspace="ws_1234",
            prompt="API接口说明中, TopP参数改如何传递?",
            top_p=0.2,
            temperature=1.0,
            doc_reference_type=Application.DocReferenceType.simple,
            has_thoughts=True,
        )

        assert resp.status_code == test_response.get("status_code")
        assert resp.request_id == test_response.get("request_id")
        assert resp.code == test_response.get("code")
        assert resp.message == test_response.get("message")

    @staticmethod
    def check_result(resp: ApplicationResponse, test_response: Dict):
        assert resp.status_code == 200
        assert resp.request_id == test_response.get("request_id")

        # output
        assert resp.output is not None
        assert resp.output.text == test_response.get("output", {}).get("text")
        assert resp.output.finish_reason == test_response.get(
            "output",
            {},
        ).get("finish_reason")
        assert resp.output.session_id == test_response.get(
            "output",
            {},
        ).get("session_id")

        # usage
        assert resp.usage.models is not None and len(resp.usage.models) > 0
        model_usage = resp.usage.models[0]
        expected_model_usage = test_response.get(
            "usage",
            {},
        ).get(
            "models",
            [],
        )[0]
        assert model_usage.model_id == expected_model_usage.get("model_id")
        assert model_usage.input_tokens == expected_model_usage.get(
            "input_tokens",
        )
        assert model_usage.output_tokens == expected_model_usage.get(
            "output_tokens",
        )

        # doc reference
        expected_doc_refs = test_response.get(
            "output",
            {},
        ).get("doc_references")
        if expected_doc_refs is not None and len(expected_doc_refs) > 0:
            doc_refs = resp.output.doc_references
            assert doc_refs is not None and len(doc_refs) == len(
                expected_doc_refs,
            )

            for i, doc_ref in enumerate(doc_refs):
                assert doc_ref.index_id == expected_doc_refs[i].get(
                    "index_id",
                )
                assert doc_ref.doc_id == expected_doc_refs[i].get("doc_id")
                assert doc_ref.doc_name == expected_doc_refs[i].get(
                    "doc_name",
                )
                assert doc_ref.doc_url == expected_doc_refs[i].get(
                    "doc_url",
                )
                assert doc_ref.title == expected_doc_refs[i].get("title")
                assert doc_ref.text == expected_doc_refs[i].get("text")
                assert doc_ref.biz_id == expected_doc_refs[i].get("biz_id")
                assert json.dumps(doc_ref.images) == json.dumps(
                    expected_doc_refs[i].get("images"),
                )

        # thoughts
        expected_thoughts = test_response.get("output", {}).get("thoughts")
        if expected_thoughts is not None and len(expected_thoughts) > 0:
            thoughts = resp.output.thoughts
            assert thoughts is not None and len(thoughts) == len(
                expected_thoughts,
            )

            for i, thought in enumerate(thoughts):
                assert thought.thought == expected_thoughts[i].get(
                    "thought",
                )
                assert thought.action == expected_thoughts[i].get("action")
                assert thought.action_name == expected_thoughts[i].get(
                    "action_name",
                )
                assert thought.action_type == expected_thoughts[i].get(
                    "action_type",
                )
                assert json.dumps(thought.action_input) == json.dumps(
                    expected_thoughts[i].get("action_input"),
                )
                assert thought.action_input_stream == (
                    expected_thoughts[i].get("action_input_stream")
                )
                assert thought.observation == expected_thoughts[i].get(
                    "observation",
                )
                assert thought.response == expected_thoughts[i].get(
                    "response",
                )
