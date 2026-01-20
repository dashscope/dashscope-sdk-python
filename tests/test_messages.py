# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import uuid

from dashscope import Messages
from dashscope.threads.messages.files import Files
from tests.mock_request_base import MockServerBase
from tests.mock_server import MockServer


class TestMessages(MockServerBase):
    TEST_MODEL_NAME = "test_model"
    ASSISTANT_ID = "asst_42bff274-6d44-45b8-90b1-11dd14534499"
    case_data = None

    @classmethod
    def setup_class(cls):
        cls.case_data = json.load(
            open("tests/data/messages.json", "r", encoding="utf-8"),
        )
        super().setup_class()

    def test_create(self, mock_server: MockServer):
        request_body = self.case_data["create_message_request"]
        response_body = self.case_data["create_message_response"]
        mock_server.responses.put(json.dumps(response_body))
        response = Messages.create(**request_body)
        req = mock_server.requests.get(block=True)
        assert req["role"] == "user"
        assert req["content"] == response.content[0].text.value
        assert response.thread_id == response_body["thread_id"]
        assert len(response.content) == 1

    def test_update(self, mock_server: MockServer):
        response_body = self.case_data["create_message_response"]
        mock_server.responses.put(json.dumps(response_body))
        thread_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        metadata = {"key": "value"}
        response = Messages.update(
            message_id,
            thread_id=thread_id,
            metadata=metadata,
            workspace="111",
        )
        req = mock_server.requests.get(block=True)
        assert req["body"]["metadata"] == metadata
        assert (
            req["path"] == f"/api/v1/threads/{thread_id}/messages/{message_id}"
        )
        assert req["headers"]["X-DashScope-WorkSpace"] == "111"
        assert response.thread_id == response_body["thread_id"]
        assert len(response.content) == 1

    def test_retrieve(self, mock_server: MockServer):
        response_obj = self.case_data["create_message_response"]
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        thread_id = "tid"
        message_id = "mid"
        response = Messages.retrieve(message_id, thread_id=thread_id)
        # get assistant id we send.
        path = mock_server.requests.get(block=True)
        assert path == f"/api/v1/threads/{thread_id}/messages/{message_id}"
        assert response.thread_id == response_obj["thread_id"]
        assert len(response.content) == 1

    def test_list(self, mock_server: MockServer):
        response_obj = self.case_data["list_message_response"]
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = "test_thread_id"
        response = Messages.list(
            thread_id,
            limit=10,
            order="inc",
            after="after",
            before="before",
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        assert (
            req
            == f"/api/v1/threads/{thread_id}/messages?limit=10&order=inc&after=after&before=before"
        )
        assert len(response.data) == 2
        assert response.data[0].id == "msg_1"
        assert response.data[1].id == "msg_0"

    def test_list_message_files(self, mock_server: MockServer):
        response_obj = self.case_data["list_message_files_response"]
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = "test_thread_id"
        message_id = "test_message_id"
        response = Files.list(
            message_id,
            thread_id=thread_id,
            limit=10,
            order="inc",
            after="after",
            before="before",
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        assert (
            req
            == f"/api/v1/threads/{thread_id}/messages/{message_id}/files?limit=10&order=inc&after=after&before=before"
        )  # noqa E501
        assert len(response.data) == 2
        assert response.data[0].id == "file-1"
        assert response.data[1].id == "file-2"

    def test_retrieve_message_file(self, mock_server: MockServer):
        file_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        response_obj = {
            "id": file_id,
            "object": "thread.message.file",
            "created_at": 11111111,
            "message_id": message_id,
        }
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = "test_thread_id"
        response = Files.retrieve(
            file_id,
            thread_id=thread_id,
            message_id=message_id,
            limit=10,
            order="inc",
            after="after",
            before="before",
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        assert (
            req
            == f"/api/v1/threads/{thread_id}/messages/{message_id}/files/{file_id}"
        )
        assert response.id == file_id
        assert response.message_id == message_id
