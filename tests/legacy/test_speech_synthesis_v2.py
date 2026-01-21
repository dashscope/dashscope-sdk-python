# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import pytest

from dashscope.audio.tts_v2 import ResultCallback, SpeechSynthesizer
from tests.unit.base_test import BaseTestEnvironment


class TestCallback(ResultCallback):
    def on_open(self):
        print("websocket is open.")

    def on_complete(self):
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print("websocket is closed.")
        self.file.close()

    def on_event(self, message):
        print(f"recv speech synthsis message {message}")

    def on_data(self, data: bytes) -> None:
        # save audio to file
        print("recv speech audio {}".format(len(data)))


class TestSynthesis(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = "pre-cosyvoice-test"
        cls.voice = "longxiaochun"
        cls.text_array = [
            "流式文本语音合成SDK，",
            "可以将输入的文本",
            "合成为语音二进制数据，",
            "相比于非流式语音合成，",
            "流式合成的优势在于实时性",
            "更强。用户在输入文本的同时",
            "可以听到接近同步的语音输出，",
            "极大地提升了交互体验，",
            "减少了用户等待时间。",
            "适用于调用大规模",
            "语言模型（LLM），以",
            "流式输入文本的方式",
            "进行语音合成的场景。",
        ]

    @pytest.mark.skip
    def test_sync_call_with_multi_formats(self):
        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            url=self.url,
        )
        audio = synthesizer.call(self.text_array[0])
        print("recv audio length {}".format(len(audio)))

    @pytest.mark.skip
    def test_sync_streaming_call_with_multi_formats(self):
        test_callback = TestCallback()

        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            callback=test_callback,
        )
        for text in self.text_array:
            synthesizer.streaming_call(text)
        synthesizer.streaming_complete()

    @pytest.mark.skip
    def test_sync_streaming_call_cancel_with_multi_formats(self):
        test_callback = TestCallback()

        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            callback=test_callback,
        )
        for text in self.text_array:
            synthesizer.streaming_call(text)
        synthesizer.streaming_cancel()
