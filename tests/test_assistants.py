# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import uuid

from dashscope import Assistants
from tests.mock_request_base import MockServerBase
from tests.mock_server import MockServer


class TestAssistants(MockServerBase):
    TEST_MODEL_NAME = "test_model"
    ASSISTANT_ID = "asst_42bff274-6d44-45b8-90b1-11dd14534499"
    case_data = None

    @classmethod
    def setup_class(cls):
        cls.case_data = json.load(
            open("tests/data/assistant.json", "r", encoding="utf-8"),
        )
        super().setup_class()

    def test_create_assistant_only_model(self, mock_server: MockServer):
        response_obj = {
            "id": "asst_42bff274-6d44-45b8-90b1-11dd14534499",
            "object": "assistant",
            "created_at": 1709633914432,
            "model": self.TEST_MODEL_NAME,
            "name": "",
            "description": "",
            "instructions": "",
            "tools": [],
            "file_ids": [],
            "metadata": {},
            "account_id": "id0",
            "gmt_crete": "2024-03-05 18:18:34",
            "gmt_update": "2024-03-05 18:18:34",
            "is_deleted": False,
            "request_id": "e547a1ea-ddc9-9ced-a620-23cf36e57359",
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = Assistants.create(model=self.TEST_MODEL_NAME)
        req = mock_server.requests.get(block=True)
        assert response.model == req["model"]
        assert req["model"] == self.TEST_MODEL_NAME

    def test_create_assistant(self, mock_server: MockServer):
        response_obj = {
            "id": "asst",
            "object": "assistant",
            "created_at": 1709634513039,
            "model": self.TEST_MODEL_NAME,
            "name": "hello",
            "description": "desc",
            "instructions": "Your a helpful assistant.",
            "tools": [
                {
                    "type": "search",
                },
                {
                    "type": "wanx",
                },
            ],
            "file_ids": [],
            "account_id": "xxxx",
            "gmt_crete": "2024-03-05 18:28:33",
            "gmt_update": "2024-03-05 18:28:33",
            "is_deleted": False,
            "metadata": {
                "key": "value",
            },
            "request_id": "request_id",
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = Assistants.create(
            model=self.TEST_MODEL_NAME,
            name="hello",
            description="desc",
            instructions="Your a helpful assistant.",
            tools=[
                {
                    "type": "search",
                },
                {
                    "type": "wanx",
                },
            ],
            metadata={"key": "value"},
        )
        req = mock_server.requests.get(block=True)
        assert response.model == req["model"]
        assert req["model"] == self.TEST_MODEL_NAME
        assert req["tools"] == [{"type": "search"}, {"type": "wanx"}]
        assert req["instructions"] == "Your a helpful assistant."
        assert req["name"] == "hello"
        assert response.file_ids == []
        assert response.instructions == req["instructions"]
        assert response.metadata == req["metadata"]

    def test_create_assistant_function_call(self, mock_server: MockServer):
        request_body = self.case_data["test_function_call_request"]
        response_body = json.dumps(
            self.case_data["test_function_call_response"],
        )
        mock_server.responses.put(response_body)
        response = Assistants.create(**request_body)
        req = mock_server.requests.get(block=True)
        assert response.model == req["model"]
        assert response.tools[2].function.name == "big_add"
        assert response.file_ids == []
        assert response.instructions == req["instructions"]

    def test_retrieve_assistant(self, mock_server: MockServer):
        response_obj = {
            "id": self.ASSISTANT_ID,
            "object": "assistant",
            "created_at": 1709635413785,
            "model": self.TEST_MODEL_NAME,
            "name": "hello",
            "description": "desc",
            "instructions": "Your a helpful assistant.",
            "tools": [
                {
                    "type": "search",
                },
                {
                    "type": "wanx",
                },
            ],
            "file_ids": [],
            "metadata": {},
            "account_id": "sk-xxx",
            "gmt_crete": "2024-03-05 18:43:33",
            "gmt_update": "2024-03-05 18:43:33",
            "is_deleted": False,
            "request_id": "dc2c8195-14df-997a-9d03-ee14887b7e1d",
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = Assistants.retrieve(self.ASSISTANT_ID)
        # get assistant id we send.
        req_assistant_id = mock_server.requests.get(block=True)
        assert response.model == self.TEST_MODEL_NAME
        assert req_assistant_id == self.ASSISTANT_ID
        assert response.file_ids == []
        assert response.instructions == response_obj["instructions"]
        assert response.metadata == response_obj["metadata"]

    def test_list_assistant(self, mock_server: MockServer):
        response_obj = self.case_data["test_list"]
        mock_server.responses.put(json.dumps(response_obj))
        response = Assistants.list(
            limit=10,
            order="inc",
            after="after",
            before="before",
            api_key="123",
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        assert (
            req
            == "/api/v1/assistants?limit=10&order=inc&after=after&before=before"
        )
        assert len(response.data) == 2
        assert response.data[0].id == "asst_1"
        assert response.data[1].id == "asst_2"

    def test_update_assistant(self, mock_server: MockServer):
        updated_desc = str(uuid.uuid4())
        response_obj = {
            "id": self.ASSISTANT_ID,
            "model": self.TEST_MODEL_NAME,
            "name": "hello",
            "created_at": 1709635413785,
            "description": updated_desc,
            "file_ids": [],
            "instructions": "Your a helpful assistant.",
            "metadata": {},
            "tools": [],
            "object": "assistant",
            "account_id": "ff",
            "gmt_crete": "2024-03-05 18:43:33",
            "gmt_update": "2024-03-06 16:12:52",
            "is_deleted": False,
            "request_id": "00300fca-2b54-9cc6-8973-5c88df51d194",
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = Assistants.update(
            self.ASSISTANT_ID,
            description=updated_desc,
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        assert req is not None
        assert response.model == self.TEST_MODEL_NAME
        assert response.id == self.ASSISTANT_ID
        assert response.file_ids == []
        assert response.instructions == response_obj["instructions"]
        assert response.tools == response_obj["tools"]
        assert response.metadata == response_obj["metadata"]
        assert response.description == updated_desc

    def test_delete_assistant(self, mock_server: MockServer):
        response_obj = {
            "id": self.ASSISTANT_ID,
            "object": "assistant.deleted",
            "deleted": True,
        }
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        response = Assistants.delete(self.ASSISTANT_ID)
        req = mock_server.requests.get(block=True)

        assert req == self.ASSISTANT_ID
        assert response.id == self.ASSISTANT_ID
        assert response.object == "assistant.deleted"
        assert response.deleted is True
