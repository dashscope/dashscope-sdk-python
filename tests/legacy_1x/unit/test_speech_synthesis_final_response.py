# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# pylint: disable=protected-access

from http import HTTPStatus

from dashscope.legacy_1x.api_entities.dashscope_response import (
    DashScopeAPIResponse,
)
from dashscope.legacy_1x.audio.tts.speech_synthesizer import SpeechSynthesizer
from dashscope.legacy_1x.client.base_api import BaseApi


def _part(output=None, usage=None, request_id="task-1"):
    return DashScopeAPIResponse(
        status_code=HTTPStatus.OK,
        request_id=request_id,
        output=output,
        usage=usage,
    )


def _mock_stream(monkeypatch, parts):
    def mock_call(cls, **kwargs):  # pylint: disable=unused-argument
        return iter(parts)

    monkeypatch.setattr(BaseApi, "call", classmethod(mock_call))


class TestFinalResponseCapture:
    def test_usage_only_task_finished_preserves_final_response(
        self,
        monkeypatch,
    ):
        # Server task-finished payload carries usage but no "output" key,
        # which the websocket layer converts to output={} (not None).
        _mock_stream(
            monkeypatch,
            [
                _part(output=b"fake-audio"),
                _part(
                    output={
                        "sentence": {
                            "begin_time": 0,
                            "end_time": 100,
                            "text": "你好世界",
                        },
                    },
                ),
                _part(output={}, usage={"characters": 17}),
            ],
        )

        result = SpeechSynthesizer.call(
            model="sambert-zhichu-v1",
            text="你好世界",
            word_timestamp_enabled=True,
            phoneme_timestamp_enabled=True,
        )

        assert result.get_audio_data() == b"fake-audio"
        assert len(result.get_timestamps()) == 1
        response = result.get_response()
        assert response is not None
        assert response.status_code == HTTPStatus.OK
        assert response.usage is not None
        assert response.usage.characters == 17

    def test_non_sentence_output_with_usage_preserves_final_response(
        self,
        monkeypatch,
    ):
        # Task-finished output is a non-empty dict without "sentence".
        _mock_stream(
            monkeypatch,
            [
                _part(output=b"fake-audio"),
                _part(
                    output={"sentences": []},
                    usage={"characters": 8},
                ),
            ],
        )

        result = SpeechSynthesizer.call(
            model="sambert-zhichu-v1",
            text="你好世界",
        )

        response = result.get_response()
        assert response is not None
        assert response.usage is not None
        assert response.usage.characters == 8

    def test_output_none_task_finished_still_captured(self, monkeypatch):
        # Regression: the classic output=None final message keeps working.
        _mock_stream(
            monkeypatch,
            [
                _part(output=b"fake-audio"),
                _part(output=None, usage={"characters": 5}),
            ],
        )

        result = SpeechSynthesizer.call(
            model="sambert-zhichu-v1",
            text="你好世界",
        )

        response = result.get_response()
        assert response is not None
        assert response.usage.characters == 5

    def test_extra_message_does_not_mask_real_final_response(
        self,
        monkeypatch,
    ):
        # An extra non-sentence message mid-stream must not overwrite the
        # real final response arriving later with output=None.
        _mock_stream(
            monkeypatch,
            [
                _part(output=b"fake-audio"),
                _part(output={"extra": "bgm-info"}),
                _part(
                    output={
                        "sentence": {
                            "begin_time": 0,
                            "end_time": 100,
                            "text": "你好世界",
                        },
                    },
                ),
                _part(output=None, usage={"characters": 5}),
            ],
        )

        result = SpeechSynthesizer.call(
            model="sambert-zhichu-v1",
            text="你好世界",
        )

        assert len(result.get_timestamps()) == 1
        response = result.get_response()
        assert response is not None
        assert response.usage.characters == 5
