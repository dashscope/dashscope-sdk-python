# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import uuid

from dashscope import Threads
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer


class TestThreads(MockServerBase):
    def test_create_with_no_messages(self, mock_server: MockServer):
        thread_id = str(uuid.uuid4())
        metadata = {"key": "value"}
        response_obj = {
            "id": thread_id,
            "object": "thread",
            "created_at": 1699012949,
            "metadata": metadata,
        }
        response_body = json.dumps(response_obj)
        mock_server.responses.put(response_body)
        response = Threads.create(metadata=metadata)
        req = mock_server.requests.get(block=True)
        assert response.id == thread_id
        assert response.metadata == metadata
        assert req["metadata"] == metadata

    def test_create_with_messages(self, mock_server: MockServer):
        thread_id = str(uuid.uuid4())
        metadata = {"key": "value"}
        response_obj = {
            "id": thread_id,
            "object": "thread",
            "created_at": 1699012949,
            "metadata": metadata,
        }
        response_body = json.dumps(response_obj)
        mock_server.responses.put(response_body)
        messages = [
            {
                "role": "user",
                "content": "How does AI work? Explain it in simple terms.",
                "file_ids": ["123"],
            },
            {
                "role": "user",
                "content": "画幅画",
            },
        ]
        thread = Threads.create(messages=messages)  # type: ignore[arg-type]

        assert thread.id == thread_id
        assert thread.metadata == metadata
        req = mock_server.requests.get(block=True)
        assert req["messages"] == messages

    def test_retrieve(self, mock_server: MockServer):
        thread_id = str(uuid.uuid4())
        metadata = {"key": "value"}
        response_obj = {
            "id": thread_id,
            "object": "thread",
            "created_at": 1699012949,
            "metadata": metadata,
        }
        response_body = json.dumps(response_obj)
        mock_server.responses.put(response_body)
        response = Threads.retrieve(thread_id)
        # get thread id we send.
        req_thread_id = mock_server.requests.get(block=True)
        assert req_thread_id == thread_id
        assert response.id == thread_id
        assert response.metadata == metadata

    def test_update(self, mock_server: MockServer):
        thread_id = str(uuid.uuid4())
        metadata = {"key": "value"}
        response_obj = {
            "id": thread_id,
            "object": "thread",
            "created_at": 1699012949,
            "metadata": metadata,
        }
        response_body = json.dumps(response_obj)
        mock_server.responses.put(response_body)
        response = Threads.update(thread_id, metadata=metadata)
        # get thread id we send.
        req = mock_server.requests.get(block=True)
        assert req["metadata"] == metadata
        assert response.id == thread_id
        assert response.metadata == metadata

    def test_delete(self, mock_server: MockServer):
        thread_id = str(uuid.uuid4())
        response_obj = {
            "id": thread_id,
            "object": "thread",
            "created_at": 1699012949,
            "deleted": True,
        }
        mock_server.responses.put(json.dumps(response_obj))
        response = Threads.delete(thread_id)
        req = mock_server.requests.get(block=True)

        assert req == thread_id
        assert response.id == thread_id
        assert response.deleted is True
