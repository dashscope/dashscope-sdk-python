# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import time
from http import HTTPStatus
from typing import Any, List
from urllib.parse import urlparse

import pytest

import dashscope
from dashscope.audio.asr import (
    Recognition,
    RecognitionCallback,
    RecognitionResult,
)
from tests.base_test import BaseTestEnvironment

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


class TestCallback(RecognitionCallback):
    def on_error(self, result: RecognitionResult) -> None:
        raise result.message

    def on_event(self, result: RecognitionResult) -> None:
        assert result.get_sentence() is not None


class TestSpeechRecognition(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = "paraformer-realtime-v1"
        cls.format = "pcm"
        cls.sample_rate = 16000
        cls.file = "./tests/data/asr_example.wav"

    def test_sync_call_with_file(self):
        recognition = Recognition(
            model=self.model,
            format=self.format,
            sample_rate=self.sample_rate,
            callback=None,
        )
        result = recognition.call(self.file)
        assert result is not None
        assert result.get_sentence() is not None
        assert len(result.get_sentence()) > 0

    def test_async_start_with_stream(self):
        callback = TestCallback()
        recognition = Recognition(
            model=self.model,
            format=self.format,
            sample_rate=self.sample_rate,
            callback=callback,
        )
        recognition.start()
        f = open(self.file, "rb")
        while True:
            chunk = f.read(3200)
            if not chunk:
                break
            else:
                recognition.send_audio_frame(chunk)
        f.close()
        recognition.stop()


class Callback(RecognitionCallback):
    def on_open(self) -> None:
        print("RecognitionCallback open.")

    def on_complete(self) -> None:
        print("RecognitionCallback complete.")

    def on_error(self, result: RecognitionResult) -> None:
        print("RecognitionCallback task_id: ", result.request_id)
        print("RecognitionCallback error: ", result.message)

    def on_close(self) -> None:
        print("RecognitionCallback close.")

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if "text" in sentence:
            print("RecognitionCallback text: ", sentence["text"])
            if RecognitionResult.is_sentence_end(sentence):
                print(
                    "RecognitionCallback sentence end, request_id:%s, usage:%s"
                    % (result.get_request_id(), result.get_usage(sentence)),
                )


def str2bool(str):
    return True if str.lower() == "true" else False


def complete_url(url: str) -> str:
    parsed = urlparse(url)
    base_url = "".join([parsed.scheme, "://", parsed.netloc])
    dashscope.base_websocket_api_url = "/".join(
        [base_url, "api-ws", dashscope.common.env.api_version, "inference"],
    )
    dashscope.base_http_api_url = url = "/".join(
        [base_url, "api", dashscope.common.env.api_version],
    )
    print("Set base_websocket_api_url: ", dashscope.base_websocket_api_url)
    print("Set base_http_api_url: ", dashscope.base_http_api_url)


@pytest.mark.skip
def test_by_user():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="paraformer-realtime-v1")
    parser.add_argument("--format", type=str, default="pcm")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--file",
        type=str,
        default="./tests/data/asr_example.wav",
    )
    parser.add_argument("--sync", type=str2bool, default="False")
    parser.add_argument("--phrase_id", type=str, default=None)
    parser.add_argument(
        "--disfluency_removal_enabled",
        type=str2bool,
        default="False",
    )
    parser.add_argument(
        "--diarization_enabled",
        type=str2bool,
        default="False",
    )
    parser.add_argument("--speaker_count", type=int, default=None)
    parser.add_argument(
        "--timestamp_alignment_enabled",
        type=str2bool,
        default="False",
    )
    parser.add_argument("--special_word_filter", type=str, default=None)
    parser.add_argument(
        "--audio_event_detection_enabled",
        type=str2bool,
        default="False",
    )
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--base_url", type=str)
    args = parser.parse_args()

    if args.api_key is not None:
        dashscope.api_key = args.api_key
    if args.base_url is not None:
        complete_url(args.base_url)

    callback = None
    if args.sync is False:
        callback = Callback()

    recognition = Recognition(
        model=args.model,
        format=args.format,
        sample_rate=args.sample_rate,
        disfluency_removal_enabled=args.disfluency_removal_enabled,
        diarization_enabled=args.diarization_enabled,
        speaker_count=args.speaker_count,
        timestamp_alignment_enabled=args.timestamp_alignment_enabled,
        special_word_filter=args.special_word_filter,
        audio_event_detection_enabled=args.audio_event_detection_enabled,
        callback=callback,
    )

    phrase_id = args.phrase_id

    if args.sync:
        result = recognition.call(args.file, phrase_id=phrase_id)
        if result.status_code == HTTPStatus.OK:
            sentences: List[Any] = result.get_sentence()
            if sentences and len(sentences) > 0:
                for sentence in sentences:
                    print(
                        "Recognizing: %s, usage: %s"
                        % (sentence, result.get_usage(sentence)),
                    )
            else:
                print("Warn: get an empty recognition result: ", result)
        else:
            print("Error: ", result.message)
    else:
        recognition.start(phrase_id=phrase_id)

        try:
            f = open(args.file, "rb")
            while True:
                chunk = f.read(3200)
                if not chunk:
                    break
                else:
                    recognition.send_audio_frame(chunk)
                    time.sleep(0.1)
            f.close()
        except Exception as e:
            print("Open file or send audio failed:", e)

        recognition.stop()


if __name__ == "__main__":
    test_by_user()
