# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time

import pytest

from dashscope.audio.asr import TranslationRecognizerCallback
from dashscope.audio.asr.translation_recognizer import (
    TranscriptionResult,
    TranslationRecognizerRealtime,
    TranslationResult,
)
from tests.unit.base_test import BaseTestEnvironment


class Callback(TranslationRecognizerCallback):
    def __init__(self, tag, file_path) -> None:
        super().__init__()
        self.tag = tag
        self.file_path = file_path
        self.text = ""
        self.translate_text = ""

    def on_open(self) -> None:
        print(f"[{self.tag}] TranslationRecognizerCallback open.")

    def on_close(self) -> None:
        print(f"[{self.tag}] TranslationRecognizerCallback close.")

    # pylint: disable=unused-argument
    def on_event(
        self,
        request_id,
        transcription_result: TranscriptionResult,
        translation_result: TranslationResult,
        usage,
    ) -> None:
        if translation_result is not None:
            translation = translation_result.get_translation("en")
            # partial recognition result
            if translation.is_sentence_end:
                self.translate_text = self.translate_text + translation.text
        if transcription_result is not None:
            if transcription_result.is_sentence_end:
                self.text = self.text + transcription_result.text

    def on_error(self, message) -> None:
        print(f"error: {message}")

    def on_complete(self) -> None:
        print(f"[{self.tag}] Transcript ==> ", self.text)
        print(f"[{self.tag}] Translate ==> ", self.translate_text)
        print(f"[{self.tag}] Translation completed")


class TestSynthesis(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = "gummy-realtime-v1"
        cls.format = "pcm"
        cls.sample_rate = 16000
        cls.file = "./tests/data/asr_example.wav"

    @pytest.mark.skip
    def test_translate_from_file(self):
        callback = Callback(f"process {os.getpid()}", self.file)

        # Call translation service by async mode, you can customize the
        # translation parameters, like model, format, sample_rate
        # For more information, please refer to
        # https://help.aliyun.com/document_detail/2712536.html
        translator = TranslationRecognizerRealtime(
            model=self.model,
            format=self.format,
            sample_rate=self.sample_rate,
            transcription_enabled=True,
            translation_enabled=True,
            translation_target_languages=["en"],
            callback=callback,
        )

        # Start translation
        translator.start()

        with open(self.file, "rb") as f:
            if os.path.getsize(self.file):
                while True:
                    audio_data = f.read(3200)
                    if not audio_data:
                        break
                    translator.send_audio_frame(audio_data)
                    time.sleep(0.01)
            # pylint: disable=broad-exception-raised
            raise Exception(
                "The supplied file was empty (zero bytes long)",
            )

        translator.stop()
        request_id = translator.get_last_request_id()
        first_delay = translator.get_first_package_delay()
        last_delay = translator.get_last_package_delay()
        print(
            f"[Metric] requestId: {request_id}, "
            f"first package delay ms: {first_delay}, "
            f"last package delay ms: {last_delay}",
        )
