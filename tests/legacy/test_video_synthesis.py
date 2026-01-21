# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from dashscope import VideoSynthesis
from dashscope.common.constants import TaskStatus
from tests.unit.base_test import BaseTestEnvironment


class TestAsyncVideoSynthesisRequest(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()

    def test_create_task(self):
        rsp = VideoSynthesis.call(
            model=VideoSynthesis.Models.wanx_kf2v,
            first_frame_url="https://static.dingtalk.com/media/lQLPD2ob9dfKPBvNBADNAkCwOcPjjaFVcEcHqrO8n1BLAA_576_1024.png_620x10000q90.png",
            last_frame_url="https://static.dingtalk.com/media/lQLPD2jl0mg85BvNBADNAkCwNJvjWJXBVMwHqrO0OvZlAA_576_1024.png_620x10000q90.png",
        )
        assert rsp.status_code == HTTPStatus.OK

        assert rsp.output["task_status"] == "SUCCEEDED"

    def test_fetch_status(self):
        rsp = VideoSynthesis.call(
            model=VideoSynthesis.Models.wanx_kf2v,
            first_frame_url="https://static.dingtalk.com/media/lQLPD2ob9dfKPBvNBADNAkCwOcPjjaFVcEcHqrO8n1BLAA_576_1024.png_620x10000q90.png",
            last_frame_url="https://static.dingtalk.com/media/lQLPD2jl0mg85BvNBADNAkCwNJvjWJXBVMwHqrO0OvZlAA_576_1024.png_620x10000q90.png",
        )
        assert rsp.status_code == HTTPStatus.OK

        rsp = VideoSynthesis.fetch(rsp)
        assert rsp.status_code == HTTPStatus.OK

    def test_wait(self):
        rsp = VideoSynthesis.async_call(
            model=VideoSynthesis.Models.wanx_kf2v,
            first_frame_url="https://static.dingtalk.com/media/lQLPD2ob9dfKPBvNBADNAkCwOcPjjaFVcEcHqrO8n1BLAA_576_1024.png_620x10000q90.png",
            last_frame_url="https://static.dingtalk.com/media/lQLPD2jl0mg85BvNBADNAkCwNJvjWJXBVMwHqrO0OvZlAA_576_1024.png_620x10000q90.png",
        )
        assert rsp.status_code == HTTPStatus.OK

        rsp = VideoSynthesis.wait(rsp)
        assert rsp.status_code == HTTPStatus.OK
        assert rsp.output.task_id != ""  # verify access by properties.
        assert rsp.output.task_status == TaskStatus.SUCCEEDED
        assert rsp.output.video_url != ""

        assert rsp.output["task_id"] != ""
        assert rsp.output["task_status"] == TaskStatus.SUCCEEDED
        assert rsp.output["video_url"] != ""

    def test_list_cancel_task(self):
        rsp = VideoSynthesis.list(status="CANCELED")
        assert rsp.status_code == HTTPStatus.OK

    def test_list_all(self):
        rsp = VideoSynthesis.list()
        assert rsp.status_code == HTTPStatus.OK
        print(rsp)
