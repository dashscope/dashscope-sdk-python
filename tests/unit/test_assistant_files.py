# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import uuid

from dashscope.assistants.files import Files
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer


class TestAssistantFiles(MockServerBase):
    def test_create(self, mock_server: MockServer):
        file_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        response_body = {
            "id": file_id,
            "object": "assistant.file",
            "created_at": 111111,
            "assistant_id": assistant_id,
        }
        mock_server.responses.put(json.dumps(response_body))
        response = Files.create(assistant_id, file_id=file_id)
        req = mock_server.requests.get(block=True)
        assert req["body"]["file_id"] == file_id
        assert req["path"] == f"/api/v1/assistants/{assistant_id}/files"
        assert response.id == file_id
        assert response.assistant_id == assistant_id

    def test_retrieve(self, mock_server: MockServer):
        file_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        response_body = {
            "id": file_id,
            "object": "assistant.file",
            "created_at": 111111,
            "assistant_id": assistant_id,
        }
        mock_server.responses.put(json.dumps(response_body))
        response = Files.retrieve(file_id, assistant_id=assistant_id)
        req = mock_server.requests.get(block=True)
        assert (
            req["path"] == f"/api/v1/assistants/{assistant_id}/files/{file_id}"
        )
        assert response.id == file_id
        assert response.assistant_id == assistant_id

    def test_list(self, mock_server: MockServer):
        file_id_1 = str(uuid.uuid4())
        file_id_2 = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        response_body = {
            "first_id": file_id_1,
            "last_id": file_id_2,
            "has_more": False,
            "data": [
                {
                    "id": file_id_1,
                    "object": "assistant.file",
                    "created_at": 111111,
                    "assistant_id": assistant_id,
                },
                {
                    "id": file_id_2,
                    "object": "assistant.file",
                    "created_at": 111111,
                    "assistant_id": assistant_id,
                },
            ],
        }
        mock_server.responses.put(json.dumps(response_body))
        response = Files.list(assistant_id, limit=10, order="asc")
        req = mock_server.requests.get(block=True)
        assert (
            req["path"]
            == f"/api/v1/assistants/{assistant_id}/files?limit=10&order=asc"
        )
        assert response.first_id == file_id_1
        assert response.last_id == file_id_2
        assert response.data[0].assistant_id == assistant_id

    def test_delete(self, mock_server: MockServer):
        file_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        response_body = {
            "id": file_id,
            "object": "assistant.file",
            "created_at": 111111,
            "deleted": True,
        }
        mock_server.responses.put(json.dumps(response_body))
        response = Files.delete(file_id, assistant_id=assistant_id)
        req = mock_server.requests.get(block=True)
        assert (
            req["path"] == f"/api/v1/assistants/{assistant_id}/files/{file_id}"
        )
        assert response.id == file_id
        assert response.deleted is True
