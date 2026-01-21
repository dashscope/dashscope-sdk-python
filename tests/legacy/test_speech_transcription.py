# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import logging
import time
from http import HTTPStatus
from urllib.parse import urlparse

import pytest

import dashscope
from dashscope.audio.asr import Transcription
from dashscope.common.constants import TaskStatus
from tests.unit.base_test import BaseTestEnvironment

HTTPS_16K_CH1_WAV = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example.wav"  # noqa: *
HTTPS_16K_CH2_WAV = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_2ch.wav"  # noqa: *

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


class TestSpeechTranscribe(BaseTestEnvironment):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = Transcription.Models.paraformer_8k_v1

    def test_async_call_and_wait(self):
        task = Transcription.async_call(
            model=self.model,
            file_urls=[HTTPS_16K_CH1_WAV],
        )

        # wait results with task_id.
        results = task
        if task.status_code == HTTPStatus.OK:
            results = Transcription.wait(task.output.task_id)

        assert results.status_code == HTTPStatus.OK
        assert results.output is not None
        assert results.output.task_status == TaskStatus.SUCCEEDED

        # wait results with task obj.
        results = Transcription.wait(task)
        assert results.status_code == HTTPStatus.OK
        assert results.output is not None
        assert results.output.task_status == TaskStatus.SUCCEEDED

    def test_async_call_and_fetch(self):
        task = Transcription.async_call(
            model=self.model,
            file_urls=[HTTPS_16K_CH1_WAV],
        )

        # poll results with task_id.
        results = task
        if task.status_code == HTTPStatus.OK:
            while True:
                results = Transcription.fetch(task.output.task_id)
                if results.status_code == HTTPStatus.OK:
                    if (
                        results.output is not None
                        and results.output.task_status
                        in [TaskStatus.PENDING, TaskStatus.RUNNING]
                    ):
                        time.sleep(2)
                        continue

                break
        assert results.status_code == HTTPStatus.OK
        assert results.output is not None
        assert results.output.task_status == TaskStatus.SUCCEEDED

        # poll results with task obj.
        while True:
            results = Transcription.fetch(task)
            if results.status_code == HTTPStatus.OK:
                if (
                    results.output is not None
                    and results.output.task_status
                    in [TaskStatus.PENDING, TaskStatus.RUNNING]
                ):
                    time.sleep(2)
                    continue

            break
        assert results.status_code == HTTPStatus.OK
        assert results.output is not None
        assert results.output.task_status == TaskStatus.SUCCEEDED

    def test_sync_call(self):
        results = Transcription.call(
            model=self.model,
            file_urls=[HTTPS_16K_CH1_WAV],
        )
        assert results.status_code == HTTPStatus.OK
        assert results.output is not None
        assert results.output.task_status == TaskStatus.SUCCEEDED

    def test_sync_call_with_2ch(self):
        results = Transcription.call(
            model=self.model,
            file_urls=[HTTPS_16K_CH2_WAV],
            channel_id=[0, 1],
        )
        assert results.status_code == HTTPStatus.OK
        assert results.output is not None
        assert results.output.task_status == TaskStatus.SUCCEEDED


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
    parser.add_argument(
        "--model",
        type=str,
        default=Transcription.Models.paraformer_v1,
    )
    parser.add_argument("--files", type=str, default=HTTPS_16K_CH1_WAV)
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

    phrase_id = args.phrase_id

    if args.sync:
        results = Transcription.call(
            model=args.model,
            file_urls=[args.files],
            phrase_id=phrase_id,
            disfluency_removal_enabled=args.disfluency_removal_enabled,
            diarization_enabled=args.diarization_enabled,
            speaker_count=args.speaker_count,
            timestamp_alignment_enabled=args.timestamp_alignment_enabled,
            special_word_filter=args.special_word_filter,
            audio_event_detection_enabled=args.audio_event_detection_enabled,
        )
        print("sync output: ", results.output)
    else:
        task = Transcription.async_call(
            model=args.model,
            file_urls=[args.files],
            phrase_id=phrase_id,
            disfluency_removal_enabled=args.disfluency_removal_enabled,
            diarization_enabled=args.diarization_enabled,
            speaker_count=args.speaker_count,
            timestamp_alignment_enabled=args.timestamp_alignment_enabled,
            special_word_filter=args.special_word_filter,
            audio_event_detection_enabled=args.audio_event_detection_enabled,
        )
        print("async task code: ", task.status_code)
        print("async task output: ", task.output)

        results = None
        if task.status_code == HTTPStatus.OK:
            while True:
                results = Transcription.fetch(task)
                if results.status_code == HTTPStatus.OK:
                    if (
                        results.output is not None
                        and results.output.task_status
                        in [TaskStatus.PENDING, TaskStatus.RUNNING]
                    ):
                        time.sleep(2)
                        continue

                break

            print("async output: ", results.output)
            print("async task_status of output: ", results.output.task_status)
            print("async results of output: ", results.output.results)

            results = Transcription.wait(task)
            print("async output with wait: ", results.output)

        else:
            print("async failed")


if __name__ == "__main__":
    test_by_user()
