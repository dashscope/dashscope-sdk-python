# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Test explicit parameters for Recognition class."""

# pylint: disable=protected-access
import threading
from contextlib import contextmanager
from unittest.mock import patch

from dashscope.audio.asr.recognition import Recognition, RecognitionCallback


class MockRecognitionCallback(RecognitionCallback):
    """Mock callback for testing."""

    def on_event(self, result):
        pass

    def on_complete(self):
        pass

    def on_error(self, result):
        pass

    def on_close(self):
        pass


@contextmanager
def _stubbed_worker(recognition):
    """Let ``start()`` run without its worker opening a real WebSocket.

    The worker has to stay alive until released: ``start()`` checks
    ``is_alive()`` immediately after spawning it and raises ``InvalidTask``
    if the thread has already finished.
    """
    release = threading.Event()
    try:
        with patch.object(
            Recognition,
            "_Recognition__receive_worker",
            lambda self: release.wait(10),
        ):
            yield recognition
    finally:
        recognition._running = False
        if recognition._silence_timer is not None:
            recognition._silence_timer.cancel()
        release.set()
        if recognition._worker is not None:
            recognition._worker.join(timeout=5)


class TestRecognitionExplicitParams:
    """Test explicit parameters in Recognition class."""

    def test_init_with_all_explicit_params(self):
        """Test __init__ with all explicit parameters."""
        recognition = Recognition(
            model="test-model",
            callback=MockRecognitionCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=True,
            diarization_enabled=True,
            speaker_count=2,
            timestamp_alignment_enabled=True,
            special_word_filter="test_filter",
            audio_event_detection_enabled=True,
        )

        # Verify parameters are stored in _kwargs
        assert recognition._kwargs["disfluency_removal_enabled"] is True
        assert recognition._kwargs["diarization_enabled"] is True
        assert recognition._kwargs["speaker_count"] == 2
        assert recognition._kwargs["timestamp_alignment_enabled"] is True
        assert recognition._kwargs["special_word_filter"] == "test_filter"
        assert recognition._kwargs["audio_event_detection_enabled"] is True

    def test_init_with_none_params(self):
        """Test __init__ with None parameters should not add to _kwargs."""
        recognition = Recognition(
            model="test-model",
            callback=MockRecognitionCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=None,
            diarization_enabled=None,
            speaker_count=None,
        )

        # Verify None parameters are not in _kwargs
        assert "disfluency_removal_enabled" not in recognition._kwargs
        assert "diarization_enabled" not in recognition._kwargs
        assert "speaker_count" not in recognition._kwargs

    def test_init_with_partial_params(self):
        """Test __init__ with some explicit parameters."""
        recognition = Recognition(
            model="test-model",
            callback=MockRecognitionCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=True,
            speaker_count=3,
        )

        # Verify only specified parameters are in _kwargs
        assert recognition._kwargs["disfluency_removal_enabled"] is True
        assert recognition._kwargs["speaker_count"] == 3
        assert "diarization_enabled" not in recognition._kwargs
        assert "timestamp_alignment_enabled" not in recognition._kwargs

    def test_init_with_extra_kwargs(self):
        """Test __init__ with extra kwargs should still work."""
        recognition = Recognition(
            model="test-model",
            callback=MockRecognitionCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=True,
            custom_param="custom_value",
        )

        # Verify both explicit and extra kwargs are stored
        assert recognition._kwargs["disfluency_removal_enabled"] is True
        assert recognition._kwargs["custom_param"] == "custom_value"

    def test_start_with_explicit_params(self):
        """Test start() method with explicit parameters."""
        recognition = Recognition(
            model="test-model",
            callback=MockRecognitionCallback(),
            format="pcm",
            sample_rate=16000,
        )

        recognition._running = False
        recognition._callback = MockRecognitionCallback()

        with _stubbed_worker(recognition):
            recognition.start(
                phrase_id="test_phrase",
                disfluency_removal_enabled=True,
                diarization_enabled=True,
                speaker_count=2,
            )

        # Verify parameters are updated in _kwargs
        assert recognition._kwargs.get("disfluency_removal_enabled") is True
        assert recognition._kwargs.get("diarization_enabled") is True
        assert recognition._kwargs.get("speaker_count") == 2

    def test_param_override_in_start(self):
        """Test that start() can override __init__ parameters."""
        recognition = Recognition(
            model="test-model",
            callback=MockRecognitionCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=False,
            speaker_count=1,
        )

        # Verify initial values
        assert recognition._kwargs["disfluency_removal_enabled"] is False
        assert recognition._kwargs["speaker_count"] == 1

        recognition._running = False

        # Override parameters in start
        with _stubbed_worker(recognition):
            recognition.start(
                disfluency_removal_enabled=True,
                speaker_count=3,
            )

        # Verify parameters are overridden
        assert recognition._kwargs.get("disfluency_removal_enabled") is True
        assert recognition._kwargs.get("speaker_count") == 3
