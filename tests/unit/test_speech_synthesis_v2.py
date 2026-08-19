# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json

import pytest

import dashscope
from dashscope.audio.tts_v2 import ResultCallback, SpeechSynthesizer
from dashscope.common.error import RequestFailure
from dashscope.protocol.websocket import ActionType, EventType
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
        print(f"recv speech audio {len(data)}")


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
        print(f"recv audio length {len(audio)}")

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


_TASK_ID = "test-task-id"


def _task_event(event, **header):
    return json.dumps(
        {"header": {"task_id": _TASK_ID, "event": event, **header}},
    )


def _task_failed_event(error_code="InternalError"):
    return _task_event(
        EventType.FAILED,
        error_code=error_code,
        error_message="synthesis interrupted",
    )


class _FakeSocket:
    def __init__(self):
        self.connected = True


class _FakeWebSocket:
    """Minimal websocket-client stand-in, replaying frames on each send."""

    def __init__(self, on_send):
        self.sock = _FakeSocket()
        self.closed = False
        self._on_send = on_send

    def send(self, data):
        self._on_send(self, json.loads(data)["header"]["action"])

    def close(self):
        self.closed = True
        self.sock.connected = False


class TestTaskFailedPropagation(BaseTestEnvironment):
    """A task-failed event must reach the caller, see github issue #167."""

    @classmethod
    def setup_class(cls):
        super().setup_class()
        dashscope.api_key = "test-api-key"
        cls.model = "cosyvoice-v2"
        cls.voice = "longxiaochun"

    def test_call_raises_when_task_failed_after_partial_audio(self):
        synthesizer = SpeechSynthesizer(model=self.model, voice=self.voice)

        def on_send(ws, action):
            if action == ActionType.START:
                synthesizer.on_message(ws, _task_event(EventType.STARTED))
            elif action == ActionType.FINISHED:
                synthesizer.on_message(ws, b"\x00" * 320)
                synthesizer.on_message(ws, _task_failed_event())

        synthesizer.ws = _FakeWebSocket(on_send)
        with pytest.raises(RequestFailure) as error:
            synthesizer.call("hello world")
        assert error.value.name == "InternalError"
        assert error.value.request_id == _TASK_ID
        # resources are released before the error is raised
        assert synthesizer.ws.closed

    def test_call_raises_when_task_failed_at_start(self):
        synthesizer = SpeechSynthesizer(model=self.model, voice=self.voice)

        def on_send(ws, action):
            if action == ActionType.START:
                synthesizer.on_message(
                    ws,
                    _task_failed_event(error_code="InvalidParameter"),
                )

        synthesizer.ws = _FakeWebSocket(on_send)
        with pytest.raises(RequestFailure) as error:
            synthesizer.call("hello world")
        assert error.value.name == "InvalidParameter"
        assert synthesizer.ws.closed

    def test_next_task_is_not_affected_by_previous_error(self):
        synthesizer = SpeechSynthesizer(model=self.model, voice=self.voice)

        def fail_on_send(ws, action):
            if action == ActionType.START:
                synthesizer.on_message(ws, _task_event(EventType.STARTED))
            elif action == ActionType.FINISHED:
                synthesizer.on_message(ws, _task_failed_event())

        synthesizer.ws = _FakeWebSocket(fail_on_send)
        with pytest.raises(RequestFailure):
            synthesizer.call("hello world")

        def succeed_on_send(ws, action):
            if action == ActionType.START:
                synthesizer.on_message(ws, _task_event(EventType.STARTED))
            elif action == ActionType.FINISHED:
                synthesizer.on_message(ws, b"\x01" * 320)
                synthesizer.on_message(ws, _task_event(EventType.FINISHED))

        synthesizer.ws = _FakeWebSocket(succeed_on_send)
        assert synthesizer.call("hello again") == b"\x01" * 320

    def test_callback_mode_reports_error_without_raising(self):
        errors = []

        class _RecordingCallback(ResultCallback):
            def on_error(self, message) -> None:
                errors.append(message)

        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            callback=_RecordingCallback(),
        )

        def on_send(ws, action):
            if action == ActionType.START:
                synthesizer.on_message(ws, _task_event(EventType.STARTED))
            elif action == ActionType.FINISHED:
                synthesizer.on_message(ws, _task_failed_event())

        synthesizer.ws = _FakeWebSocket(on_send)
        synthesizer.streaming_call("hello world")
        synthesizer.streaming_complete()
        assert len(errors) == 1
        assert synthesizer.ws.closed
