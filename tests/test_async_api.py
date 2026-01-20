# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus
from typing import Union

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.client.base_api import BaseAsyncApi
from dashscope.common.constants import ApiProtocol, HTTPMethod, TaskStatus
from tests.base_test import BaseTestEnvironment


class AsyncRequest(BaseAsyncApi):
    """API for File Transcriber models."""

    @classmethod
    def fetch(
        cls,
        task: Union[str, DashScopeAPIResponse],
        api_key: str = None,
        workspace: str = None,
    ) -> DashScopeAPIResponse:
        """Query the task status.

        Args:
            task (Union[str, AsyncTaskResponse]): The async task, can be
                task_id or async_call response.

        Returns:
            DashScopeAPIResponse: The task status information.
        """

        return super().fetch(task, api_key=api_key, workspace=workspace)

    @classmethod
    def wait(
        cls,
        task: Union[str, DashScopeAPIResponse],
        api_key: str = None,
        workspace: str = None,
    ) -> DashScopeAPIResponse:
        """Wait for the task to complete and return the result.

        Args:
            task (Union[str, AsyncTaskResponse]): The async task.

        Returns:
            DashScopeAPIResponse: The async task result.
        """
        return super().wait(task, api_key=api_key, workspace=workspace)

    @classmethod
    def call(
        cls,
        model: str,
        url: str,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """Call the async interface and return the result

        Args:
            model (str): The requested model, such as paraformer-16k-1
            url (str): The URL of the request file.

        Returns:
            DashScopeAPIResponse: The response body.

        Raises:
            InputRequired: The file cannot be empty.
        """
        return super().call(
            model,
            url,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )

    @classmethod
    def async_call(
        cls,
        model: str,
        url: str,
        api_key: str = None,
        workspace: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """Call the async interface and return task information

        Args:
            model (str): The requested model, such as paraformer-16k-1
            url (str): The request url.

        Returns:
            AsyncTaskResponse: The response body.

        Raises:
            InputRequired: The file cannot be empty.
        """

        response = super().async_call(
            model=model,
            task_group="audio",
            task="asr",
            function="transcription",
            input={"file_urls": [url]},
            api_protocol=ApiProtocol.HTTP,
            http_method=HTTPMethod.POST,
            channel_id=[0],
            workspace=workspace,
            **kwargs,
        )
        return response


class TestAsyncRequest(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = "paraformer-8k-v1"

    def test_start_async_request(self):
        url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav"  # noqa: E501
        resp = AsyncRequest.async_call(model="paraformer-8k-1", url=url)
        import json

        js = json.dumps(resp, ensure_ascii=False)
        print(js)
        print(resp.output)
        if resp.status_code == HTTPStatus.OK:
            assert resp.output["task_id"] is not None
        else:
            print(
                "Failed id: %s code: %s, message: %s"
                % (resp.request_id, resp.status_code, resp.message),
            )

    def test_status_async_request(self):
        url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav"  # noqa: E501
        resp = AsyncRequest.async_call(model=self.model, url=url)
        assert resp.status_code == HTTPStatus.OK
        resp = AsyncRequest.fetch(resp)
        assert resp.status_code == HTTPStatus.OK
        if resp.status_code == HTTPStatus.OK:
            print(resp.output)
            assert resp.output["task_id"] is not None
        else:
            print(
                "Failed id: %s code: %s, message: %s"
                % (resp.request_id, resp.status_code, resp.message),
            )

    def test_wait_async_request(self):
        url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav"  # noqa: E501
        resp = AsyncRequest.async_call(model=self.model, url=url)
        assert resp.status_code == HTTPStatus.OK
        rsp = AsyncRequest.wait(resp)
        assert rsp.status_code == HTTPStatus.OK
        print(rsp.output)
        import json

        js = json.dumps(rsp, ensure_ascii=False)
        print(js)

    def test_sync_request(self):
        url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav"  # noqa: E501
        resp = AsyncRequest.call(model=self.model, url=url)
        assert resp.status_code == HTTPStatus.OK
        print(resp.output)
        assert resp.output["task_id"] is not None

    def test_wait(self):
        resp = AsyncRequest.wait("dfjkdasfjadsfasd")
        assert resp.status_code == HTTPStatus.OK
        assert resp.output["task_status"] != TaskStatus.SUCCEEDED
        print(resp)
