# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from http import HTTPStatus

from dashscope import Generation
from dashscope.aigc.conversation import Conversation, History, HistoryItem
from dashscope.api_entities.dashscope_response import Choice, Message, Role
from dashscope.common.message_manager import MessageManager
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer


def request_generator():
    yield "hello"


model = Generation.Models.qwen_turbo


class TestConversationRequest(MockServerBase):
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
    message_response_obj = {
        "status_code": 200,
        "request_id": "1",
        "code": "",
        "message": "",
        "output": {
            "text": None,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hello world",
                    },
                },
            ],
        },
        "usage": {
            "input_tokens": 27,
            "output_tokens": 110,
        },
    }

    def test_message_request(self, mock_server: MockServer):
        response_str = json.dumps(TestConversationRequest.message_response_obj)
        mock_server.responses.put(response_str)
        prompt = "hello"
        messages = [{"role": "user", "content": prompt}]
        resp = Generation.call(
            model=model,
            messages=messages,
            max_tokens=1024,
            api_protocol="http",
            result_format="message",
            n=50,
        )
        req = mock_server.requests.get()
        assert req["model"] == model
        assert req["parameters"]["result_format"] == "message"
        assert req["input"]["messages"][0] == Message(
            role=Role.USER,
            content=prompt,
        )
        assert resp.output.text is None
        assert resp.output.choices[0] == Choice(
            finish_reason="stop",
            message={
                "role": "assistant",
                "content": "hello world",
            },
        )

    def test_conversation_with_history(self, mock_server: MockServer):
        history = History()
        item = HistoryItem("user", text="今天天气好吗")
        history.append(item)
        item = HistoryItem("bot", text="今天天气不错，要出去玩玩嘛？")
        history.append(item)

        item = HistoryItem("user", text="那你有什么地方推荐?")
        history.append(item)
        item = HistoryItem("bot", text="我建议你去公园，春天来了，花朵开了，很美丽。")
        history.append(item)
        mock_server.responses.put(
            json.dumps(TestConversationRequest.text_response_obj),
        )
        chat = Conversation(history)
        response = chat.call(
            model,
            prompt="推荐一个附近的公园",
            auto_history=True,
        )

        assert response.status_code == HTTPStatus.OK
        assert response.output.text == "hello"
        assert response.output.choices is None
        assert response.output.finish_reason == "stop"
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["parameters"] == {}
        assert len(req["input"]["history"]) == 2
        mock_server.responses.put(
            json.dumps(TestConversationRequest.text_response_obj),
        )
        response = chat.call(
            model,
            prompt="这个公园去过很多次了，远一点的呢",
            auto_history=True,
        )
        assert response.status_code == HTTPStatus.OK
        assert response.output.text == "hello"
        assert response.output.choices is None
        assert response.output.finish_reason == "stop"
        req = mock_server.requests.get(block=True)
        print(req)
        assert req["model"] == model
        assert req["parameters"] == {}
        assert len(req["input"]["history"]) == 3

    def test_conversation_with_message_and_prompt(
        self,
        mock_server: MockServer,
    ):
        messageManager = MessageManager(10)
        messageManager.add(Message(role="system", content="你是达摩院的生活助手机器人。"))
        mock_server.responses.put(
            json.dumps(TestConversationRequest.message_response_obj),
        )
        conv = Conversation()
        response = conv.call(
            model,
            prompt="推荐一个附近的公园",
            messages=messageManager.get(),
            result_format="message",
        )
        assert response.status_code == HTTPStatus.OK
        assert response.output.text is None
        choices = TestConversationRequest.message_response_obj["output"][
            "choices"
        ]
        assert response.output.choices == choices
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["parameters"] == {"result_format": "message"}
        assert len(req["input"]["messages"]) == 2

    def test_conversation_with_messages(self, mock_server: MockServer):
        messageManager = MessageManager(10)
        messageManager.add(Message(role="system", content="你是达摩院的生活助手机器人。"))
        messageManager.add(Message(role=Role.USER, content="推荐一个附近的公园"))
        mock_server.responses.put(
            json.dumps(TestConversationRequest.message_response_obj),
        )
        conv = Conversation()
        response = conv.call(
            model,
            messages=messageManager.get(),
            result_format="message",
        )
        assert response.status_code == HTTPStatus.OK
        assert response.output.text is None
        choices = TestConversationRequest.message_response_obj["output"][
            "choices"
        ]
        assert response.output.choices == choices
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["parameters"] == {"result_format": "message"}
        assert len(req["input"]["messages"]) == 2

    def test_conversation_call_with_messages(self, mock_server: MockServer):
        messageManager = MessageManager(10)
        messageManager.add(Message(role="system", content="你是达摩院的生活助手机器人。"))
        messageManager.add(Message(role=Role.USER, content="推荐一个附近的公园"))
        mock_server.responses.put(
            json.dumps(TestConversationRequest.message_response_obj),
        )
        conv = Conversation()
        response = conv.call(
            model,
            messages=messageManager.get(),
            result_format="message",
        )
        assert response.status_code == HTTPStatus.OK
        assert response.output.text is None
        choices = TestConversationRequest.message_response_obj["output"][
            "choices"
        ]
        assert response.output.choices == choices
        req = mock_server.requests.get(block=True)
        assert req["model"] == model
        assert req["parameters"] == {"result_format": "message"}
        assert len(req["input"]["messages"]) == 2

    def test_not_qwen(self, mock_server: MockServer):
        prompt = "介绍下杭州"
        mock_server.responses.put(
            json.dumps(TestConversationRequest.text_response_obj),
        )
        response = Generation.call(
            model=Generation.Models.dolly_12b_v2,
            prompt=prompt,
        )

        assert response.status_code == HTTPStatus.OK
        assert response.output.text == "hello"
        assert response.output.choices is None
        assert response.output.finish_reason == "stop"
        req = mock_server.requests.get(block=True)
        assert req["model"] == Generation.Models.dolly_12b_v2
        assert req["parameters"] == {}
        assert req["input"] == {"prompt": prompt}
