# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import sys
import time

import pytest
from dashscope.multimodal.dialog_state import DialogState
from dashscope.multimodal.multimodal_dialog import (
    MultiModalDialog,
    MultiModalCallback,
)
from dashscope.multimodal.multimodal_request_params import (
    Upstream,
    Downstream,
    ClientInfo,
    RequestParameters,
    Device,
)
from tests.unit.base_test import BaseTestEnvironment

logger = logging.getLogger("dashscope")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()

# Global variable for dialog ID
g_dialog_id = None
# Global variable for conversation instance
conver_instance = None


# 定义voice chat服务回调
class TestCallback(MultiModalCallback):
    def on_connected(self):
        logger.debug("connected with server.")

    def on_started(self, dialog_id):
        global g_dialog_id
        g_dialog_id = dialog_id

    def on_stopped(self):
        logger.info("stopped with server.")

    def on_state_changed(self, state: DialogState):
        if state == DialogState.LISTENING:
            # app.update_text("开始收音。。。请提问")
            return
        if state == DialogState.THINKING:
            # app.update_text("思考中。。。请耐心等待")
            return
        if state == DialogState.RESPONDING:
            # app.update_text("正在回答。。。")
            return

    def on_speech_audio_data(
        self,
        data: bytes,
    ):  # pylint: disable=unused-argument
        # pcm_play.play(data)
        return

    def on_error(self, error):
        logger.error(error)
        sys.exit(0)

    def on_responding_started(self):
        # 开始端侧播放
        # pcm_play.start_play()
        global conver_instance  # pylint: disable=global-variable-not-assigned
        if conver_instance is not None:
            conver_instance.send_local_responding_started()

    def on_responding_ended(self, payload):  # pylint: disable=unused-argument
        logger.debug("on responding ended")
        global conver_instance  # pylint: disable=global-variable-not-assigned
        if conver_instance is not None:
            conver_instance.send_local_responding_ended()
        # pcm_play.stop_play()

    def on_speech_content(self, payload):
        if payload is not None:
            logger.debug(payload)

    def on_responding_content(self, payload):
        if payload is not None:
            logger.debug(payload)

    def on_request_accepted(self):
        # 服务端接受打断当前播报
        # pcm_play.cancel_play()
        return

    def on_close(self, close_status_code, close_msg):
        logger.info(
            "close with status code: %d, msg: %s",
            close_status_code,
            close_msg,
        )


class TestMultiModalDialog(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = "multimodal-dialog"
        cls.voice = "longxiaochun_v2"

    @pytest.mark.skip
    def test_multimodal_dialog_one_turn(self):
        # 对话状态 Listening->Thinking->Responding->Listening
        up_stream = Upstream(
            type="AudioOnly",
            mode="push2talk",
            audio_format="pcm",
        )
        down_stream = Downstream(voice=self.voice, sample_rate=16000)

        client_info = ClientInfo(
            user_id="aabb",
            device=Device(uuid="1234567890"),
        )
        request_params = RequestParameters(
            upstream=up_stream,
            downstream=down_stream,
            client_info=client_info,
        )

        self.callback = TestCallback()
        self.conversation = MultiModalDialog(
            app_id="",
            workspace_id="llm-xxxx",
            url="wss://poc-dashscope.aliyuncs.com/api-ws/v1/inference",
            request_params=request_params,
            multimodal_callback=self.callback,
            model=self.model,
        )

        self.conversation.start("")
        # 首轮进入Listening
        while (
            DialogState.LISTENING is not self.conversation.get_dialog_state()
        ):
            time.sleep(0.1)
        self.conversation.request_to_respond(
            "prompt",
            "今天天气不错",
            parameters=None,
        )
        # 等待第二轮Listening
        while (
            DialogState.LISTENING is not self.conversation.get_dialog_state()
        ):
            time.sleep(0.1)
        self.conversation.stop()
