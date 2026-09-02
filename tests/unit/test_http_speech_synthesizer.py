# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# pylint: disable=protected-access

import base64
from http import HTTPStatus

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.audio.http_tts.http_speech_synthesizer import (
    HttpSpeechSynthesizer,
)


def _ok_response(output, request_id="req-1"):
    return DashScopeAPIResponse(
        status_code=HTTPStatus.OK,
        request_id=request_id,
        output=output,
    )


class TestHandleNonStreamingResponse:
    def test_success_proxies_status_code(self):
        response = _ok_response(
            {
                "audio": {
                    "url": "https://example.com/a.wav",
                    "id": "audio-1",
                    "expires_at": 1893456000,
                },
            },
        )

        result = HttpSpeechSynthesizer._handle_non_streaming_response(
            response,
        )

        assert result.status_code == HTTPStatus.OK
        assert result.request_id == "req-1"
        assert result.audio_url == "https://example.com/a.wav"
        assert result.audio_id == "audio-1"
        assert result.expires_at == 1893456000

    def test_failed_response_is_surfaced_not_raised(self):
        response = DashScopeAPIResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            request_id="req-err",
            code="InvalidParameter",
            message="bad voice",
        )

        result = HttpSpeechSynthesizer._handle_non_streaming_response(
            response,
        )

        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert result.request_id == "req-err"
        assert result.code == "InvalidParameter"
        assert result.message == "bad voice"
        assert result.audio_url is None


class TestHandleStreamingResponse:
    def test_chunks_and_final_result_proxy_status_code(self):
        audio_bytes = b"fake-audio"
        chunks = [
            _ok_response(
                {
                    "type": "sentence-begin",
                    "sentence": {"index": 1},
                    "audio": {"data": base64.b64encode(audio_bytes)},
                },
                request_id="req-s1",
            ),
            _ok_response(
                {
                    "finish_reason": "stop",
                    "audio": {
                        "url": "https://example.com/a.wav",
                        "id": "audio-2",
                        "expires_at": 1893456000,
                    },
                },
                request_id="req-s2",
            ),
        ]

        results = list(
            HttpSpeechSynthesizer._handle_streaming_response(iter(chunks)),
        )

        assert len(results) == 2
        assert results[0].audio_data == audio_bytes
        assert results[0].status_code == HTTPStatus.OK
        assert results[0].request_id == "req-s1"
        assert results[1].audio_data == audio_bytes
        assert results[1].audio_url == "https://example.com/a.wav"
        assert results[1].request_id == "req-s2"


class TestCall:
    def test_call_returns_result_with_real_status(self, monkeypatch):
        captured = {}

        def mock_http_call(cls, **kwargs):  # pylint: disable=unused-argument
            captured.update(kwargs)
            return _ok_response(
                {"audio": {"url": "https://example.com/a.wav"}},
            )

        monkeypatch.setattr(
            HttpSpeechSynthesizer,
            "_http_call",
            classmethod(mock_http_call),
        )

        result = HttpSpeechSynthesizer.call(
            model="cosyvoice-v2",
            text="你好世界",
            voice="longxiaochun_v2",
            volume=None,
            rate=None,
            pitch=None,
        )

        assert captured["body"]["input"] == {
            "text": "你好世界",
            "voice": "longxiaochun_v2",
            "format": "wav",
            "sample_rate": 24000,
        }
        assert result.status_code == HTTPStatus.OK
        assert result.audio_url == "https://example.com/a.wav"
