# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import uuid

from dashscope import Runs, Steps
from tests.unit.mock_request_base import MockServerBase
from tests.unit.mock_server import MockServer


class TestRuns(MockServerBase):
    TEST_MODEL_NAME = "test_model"
    ASSISTANT_ID = "asst_42bff274-6d44-45b8-90b1-11dd14534499"
    case_data = None

    @classmethod
    def setup_class(cls):
        # pylint: disable=consider-using-with
        cls.case_data = json.load(
            open("tests/data/runs.json", "r", encoding="utf-8"),
        )
        super().setup_class()

    def test_create_simple(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_body = self.case_data["create_run_response"]  # type: ignore
        mock_server.responses.put(json.dumps(response_body))
        thread_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        response = Runs.create(thread_id, assistant_id=assistant_id)
        req = mock_server.requests.get(block=True)
        assert req["assistant_id"] == assistant_id
        assert response.thread_id == response_body["thread_id"]
        assert response.metadata == {"key": "value"}

    def test_create_complicated(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_body = self.case_data["create_run_response"]  # type: ignore
        mock_server.responses.put(json.dumps(response_body))
        thread_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        model_name = str(uuid.uuid4())
        instructions = "Your a tool."
        additional_instructions = "additional_instructions"
        tools = [
            {
                "type": "code_interpreter",
            },
            {
                "type": "search",
            },
            {
                "type": "function",
                "function": {
                    "name": "big_add",
                    "description": "Add to number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "left": {
                                "type": "integer",
                                "description": "The left operator",
                            },
                            "right": {
                                "type": "integer",
                                "description": "The right operator.",
                            },
                        },
                        "required": ["left", "right"],
                    },
                },
            },
        ]
        metadata = {"key": "meta"}
        response = Runs.create(
            thread_id,
            assistant_id=assistant_id,
            model=model_name,
            instructions=instructions,
            additional_instructions=additional_instructions,
            tools=tools,  # type: ignore[arg-type]
            metadata=metadata,
        )
        req = mock_server.requests.get(block=True)
        assert req["assistant_id"] == assistant_id
        assert req["model"] == model_name
        assert req["instructions"] == instructions
        assert req["additional_instructions"] == additional_instructions
        assert req["metadata"] == metadata
        assert req["tools"] == tools
        assert response.thread_id == response_body["thread_id"]
        assert len(response.tools) == 3
        assert response.tools[0].type == "code_interpreter"

    def test_retrieve(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_obj = self.case_data["create_run_response"]  # type: ignore
        response_str = json.dumps(response_obj)
        mock_server.responses.put(response_str)
        thread_id = "tid"
        run_id = str(uuid.uuid4())
        response = Runs.retrieve(run_id, thread_id=thread_id)
        # get assistant id we send.
        path = mock_server.requests.get(block=True)
        assert path == f"/api/v1/threads/{thread_id}/runs/{run_id}"
        assert len(response.tools) == 3
        assert response.tools[0].type == "code_interpreter"

    def test_list(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_obj = self.case_data["list_run_response"]  # type: ignore
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = "test_thread_id"
        response = Runs.list(
            thread_id,
            limit=10,
            order="inc",
            after="after",
            before="before",
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        expected_path = (
            f"/api/v1/threads/{thread_id}/runs?"
            "limit=10&order=inc&after=after&before=before"
        )
        assert req == expected_path
        assert len(response.data) == 1
        assert response.data[0].id == "1"
        assert response.data[0].tools[2].type == "function"

    def test_create_thread_and_run(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_body = self.case_data["create_run_response"]  # type: ignore
        mock_server.responses.put(json.dumps(response_body))
        assistant_id = str(uuid.uuid4())
        model_name = str(uuid.uuid4())
        instructions = "Your a tool."
        additional_instructions = "additional_instructions"
        tools = [
            {
                "type": "code_interpreter",
            },
            {
                "type": "search",
            },
            {
                "type": "function",
                "function": {
                    "name": "big_add",
                    "description": "Add to number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "left": {
                                "type": "integer",
                                "description": "The left operator",
                            },
                            "right": {
                                "type": "integer",
                                "description": "The right operator.",
                            },
                        },
                        "required": ["left", "right"],
                    },
                },
            },
        ]
        metadata = {"key": "meta"}
        thread = {
            "messages": [
                {
                    "role": "user",
                    "content": "Test content",
                },
            ],
            "metadata": {
                "key": "meta",
            },
        }
        # process by handle_update_object_with_post,
        response = Runs.create_thread_and_run(
            assistant_id=assistant_id,
            thread=thread,
            model=model_name,
            instructions=instructions,
            additional_instructions=additional_instructions,
            tools=tools,  # type: ignore[arg-type]
            metadata=metadata,
        )
        req = mock_server.requests.get(block=True)
        assert req["assistant_id"] == assistant_id
        assert req["model"] == model_name
        assert req["instructions"] == instructions
        assert req["additional_instructions"] == additional_instructions
        assert req["metadata"] == metadata
        assert req["tools"] == tools
        assert req["thread"] == thread
        assert response.thread_id == response_body["thread_id"]
        assert len(response.tools) == 3
        assert response.tools[0].type == "code_interpreter"

    def test_submit_tool_outputs(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_body = self.case_data[  # type: ignore
            "submit_function_call_result"
        ]
        mock_server.responses.put(json.dumps(response_body))
        thread_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        tool_outputs = [
            {
                "output": "789076524",
                "tool_call_id": "call_DqGuSZ1NtWimgQcj8tGph6So",
            },
        ]
        response = Runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
        )
        req = mock_server.requests.get(block=True)
        assert req["tool_outputs"] == tool_outputs
        assert response.thread_id == response_body["thread_id"]
        assert len(response.tools) == 3
        assert response.tools[0].type == "code_interpreter"

    def test_run_required_function_call(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_obj = self.case_data[  # type: ignore
            "required_action_function_call_response"
        ]
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        response = Runs.create(thread_id, assistant_id=assistant_id)
        # how to dump response to json.
        s = json.dumps(
            response,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4,
        )
        print(s)
        req = mock_server.requests.get(block=True)
        assert req["assistant_id"] == assistant_id
        assert (
            response.required_action.submit_tool_outputs.tool_calls[0].id
            == "call_1"
        )

    def test_list_run_steps(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_obj = self.case_data[  # type: ignore
            "list_run_steps_response"
        ]
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = "test_thread_id"
        run_id = str(uuid.uuid4())
        response = Steps.list(
            run_id,
            thread_id=thread_id,
            limit=10,
            order="inc",
            after="after",
            before="before",
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        expected_path = (
            f"/api/v1/threads/{thread_id}/runs/{run_id}/steps?"
            "limit=10&order=inc&after=after&before=before"
        )
        assert req == expected_path
        assert len(response.data) == 2
        assert response.data[0].id == "step_1"
        assert response.data[0].step_details.type == "message_creation"
        assert (
            response.data[0].step_details.message_creation.message_id
            == "msg_1"
        )
        assert response.data[0].usage.completion_tokens == 25
        assert response.data[0].usage.prompt_tokens == 809
        assert response.data[0].usage.total_tokens == 834
        assert response.data[1].id == "step_2"
        assert response.data[1].step_details.type == "tool_calls"
        assert response.data[1].step_details.tool_calls[0].type == "function"
        assert response.data[1].step_details.tool_calls[0].id == "call_1"
        assert (
            response.data[1].step_details.tool_calls[0].function.arguments
            == '{"left":87787,"right":788988737}'
        )
        assert (
            response.data[1].step_details.tool_calls[0].function.output
            == "789076524"
        )
        assert (
            response.data[1].step_details.tool_calls[0].function.name
            == "big_add"
        )

    def test_retrieve_run_steps(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_obj = self.case_data["retrieve_run_step"]  # type: ignore
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        step_id = str(uuid.uuid4())
        response = Steps.retrieve(
            step_id,
            thread_id=thread_id,
            run_id=run_id,
            limit=10,
            order="inc",
            after="after",
            before="before",
        )
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        expected_path = (
            f"/api/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}"
        )
        assert req == expected_path
        assert response.id == "step_1"

        assert response.step_details.type == "tool_calls"
        assert response.step_details.tool_calls[0].id == "call_1"
        assert response.step_details.tool_calls[0].function.name == "big_add"
        assert (
            response.step_details.tool_calls[0].function.output == "789076524"
        )
        assert response.usage.total_tokens == 798
        assert response.usage.prompt_tokens == 776
        assert response.usage.completion_tokens == 22

    def test_cancel(self, mock_server: MockServer):
        # type: ignore[index]
        # pylint: disable=unsubscriptable-object
        response_obj = self.case_data["retrieve_run_step"]  # type: ignore
        mock_server.responses.put(json.dumps(response_obj))
        thread_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        response = Runs.cancel(run_id, thread_id=thread_id)
        # get assistant id we send.
        req = mock_server.requests.get(block=True)
        assert req == f"/api/v1/threads/{thread_id}/runs/{run_id}/cancel"
        assert response.id == "step_1"
