# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import sys
import time
from http import HTTPStatus

import pytest

import dashscope
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts import (
    ResultCallback,
    SpeechSynthesisResult,
    SpeechSynthesizer,
)
from tests.unit.base_test import BaseTestEnvironment

logger = logging.getLogger("dashscope")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# add formatter to ch
console_handler.setFormatter(formatter)

# add ch to logger
logger.addHandler(console_handler)

first_data_flag: bool = True
call_time: int = 0
first_data_time: int = 0


class TestCallback(ResultCallback):
    def on_error(self, response: SpeechSynthesisResponse):
        assert response.status_code == HTTPStatus.OK

    def on_event(self, result: SpeechSynthesisResult):
        audio_frame = result.get_audio_frame()
        timestamp = result.get_timestamp()
        assert audio_frame is not None or timestamp is not None

        if audio_frame is not None:
            assert sys.getsizeof(audio_frame) > 0

        if timestamp is not None:
            assert "begin_time" in timestamp
            assert "end_time" in timestamp
            assert len(timestamp["words"]) > 0


class TestSynthesis(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = "sambert-zhichu-v1"
        cls.text = "今天天气真不错，我想去操场踢足球。"

    def check_result(self, result):
        assert result.get_response().status_code == HTTPStatus.OK
        assert (
            result.get_response().code is None
            or len(
                result.get_response().code,
            )
            == 0
        )
        assert (
            result.get_response().message is None
            or len(
                result.get_response().message,
            )
            == 0
        )
        assert sys.getsizeof(result.get_audio_data()) > 0

    def test_sync_call_with_multi_formats(self):
        test_callback = TestCallback()

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            format=SpeechSynthesizer.AudioFormat.format_mp3,
        )
        self.check_result(result)

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            format=SpeechSynthesizer.AudioFormat.format_pcm,
        )
        self.check_result(result)

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            format=SpeechSynthesizer.AudioFormat.format_wav,
        )
        self.check_result(result)

    def test_sync_call_with_sample_rate(self):
        test_callback = TestCallback()

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            sample_rate=16000,
        )
        self.check_result(result)

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            sample_rate=24000,
        )
        self.check_result(result)

    def test_sync_call_with_volume(self):
        test_callback = TestCallback()

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            volume=1,
        )
        self.check_result(result)

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            volume=100,
        )
        self.check_result(result)

    def test_sync_call_with_rate(self):
        test_callback = TestCallback()

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            rate=-500,
        )
        self.check_result(result)

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            rate=500,
        )
        self.check_result(result)

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            pitch=-500,
        )
        self.check_result(result)

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            pitch=500,
        )
        self.check_result(result)

    def test_sync_call_with_timestamp(self):
        test_callback = TestCallback()

        result = SpeechSynthesizer.call(
            model=self.model,
            text=self.text,
            callback=test_callback,
            word_timestamp_enabled=True,
            phoneme_timestamp_enabled=True,
        )
        self.check_result(result)


class Callback(ResultCallback):
    def on_open(self):
        print("Synthesis is opened.")

    def on_complete(self):
        print("Synthesis is completed.")

    def on_error(self, response: SpeechSynthesisResponse):
        print("Synthesis failed, response is %s" % (str(response)))

    def on_close(self):
        print("Synthesis is closed.")

    def on_event(self, result: SpeechSynthesisResult):
        global first_data_flag
        global call_time
        global first_data_time
        audio_frame = result.get_audio_frame()
        timestamp = result.get_timestamp()

        if audio_frame is not None and sys.getsizeof(audio_frame) > 0:
            if first_data_flag:
                cur_time = time.time()
                first_data_time = cur_time - call_time
                first_data_flag = False

            print("get binary data: ", sys.getsizeof(audio_frame))

        if timestamp is not None:
            if "begin_time" in timestamp and "end_time" in timestamp:
                print(
                    " time: %d - %d"
                    % (timestamp["begin_time"], timestamp["end_time"]),
                )
                words_list = timestamp["words"]
                print(" words: ")
                for word in words_list:
                    print(
                        "   %d - %d : %s"
                        % (word["begin_time"], word["end_time"], word["text"]),
                    )
                    if "phonemes" in word:
                        for phoneme in word["phonemes"]:
                            print(
                                "     %d - %d :  text: %s  tone: %s"
                                % (
                                    phoneme["begin_time"],
                                    phoneme["end_time"],
                                    phoneme["text"],
                                    phoneme["tone"],
                                ),
                            )


def str2bool(str):
    return True if str.lower() == "true" else False


@pytest.mark.skip
def test_by_user():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sambert-zhichu-v1")
    parser.add_argument("--text", type=str, default="今天天气真不错，我想去操场踢足球。")
    parser.add_argument(
        "--callback",
        type=str2bool,
        default="False",
        help="run with callback or not.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=SpeechSynthesizer.AudioFormat.format_wav,
    )
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--volume", type=int, default=50)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--pitch", type=float, default=1.0)
    parser.add_argument(
        "--word_timestamp",
        type=str2bool,
        default="False",
        help="run with word_timestamp or not.",
    )
    parser.add_argument(
        "--phoneme_timestamp",
        type=str2bool,
        default="False",
        help="run with phoneme_timestamp or not.",
    )
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str)
    args = parser.parse_args()

    if args.api_key is not None:
        dashscope.api_key = args.api_key
    if args.base_url is not None:
        dashscope.base_websocket_api_url = args.base_url

    global call_time
    global first_data_time
    call_time = time.time()

    callback = None
    if args.callback is True:
        callback = Callback()

    result = SpeechSynthesizer.call(
        model=args.model,
        text=args.text,
        callback=callback,
        format=args.format,
        sample_rate=args.sample_rate,
        volume=args.volume,
        rate=args.rate,
        pitch=args.pitch,
        word_timestamp_enabled=args.word_timestamp,
        phoneme_timestamp_enabled=args.phoneme_timestamp,
    )
    print("Speech synthesis finish:")
    if result.get_audio_data() is not None:
        print(
            "  get audio data: %dbytes"
            % (sys.getsizeof(result.get_audio_data())),
        )
    print("  get sentences size: %d" % (len(result.get_timestamps())))
    print("  get response: %s" % (result.get_response()))

    if first_data_time > 0:
        print(
            "The cost time of first audio data: %6dms"
            % (first_data_time * 1000),
        )


if __name__ == "__main__":
    test_by_user()
