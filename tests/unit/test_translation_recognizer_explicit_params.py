# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Test explicit parameters for TranslationRecognizer classes."""

# pylint: disable=protected-access
from dashscope.audio.asr.translation_recognizer import (
    TranslationRecognizerRealtime,
    TranslationRecognizerChat,
    TranslationRecognizerCallback,
)


class MockTranslationCallback(TranslationRecognizerCallback):
    """Mock callback for testing."""

    def on_open(self):
        pass

    def on_event(self, request_id, transcription, translations, usage):
        pass

    def on_error(self, result):
        pass

    def on_close(self):
        pass


class TestTranslationRecognizerRealtimeExplicitParams:
    """Test explicit parameters in TranslationRecognizerRealtime class."""

    def test_init_with_all_explicit_params(self):
        """Test __init__ with all explicit parameters."""
        recognizer = TranslationRecognizerRealtime(
            model="test-model",
            callback=MockTranslationCallback(),
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
        assert recognizer._kwargs["disfluency_removal_enabled"] is True
        assert recognizer._kwargs["diarization_enabled"] is True
        assert recognizer._kwargs["speaker_count"] == 2
        assert recognizer._kwargs["timestamp_alignment_enabled"] is True
        assert recognizer._kwargs["special_word_filter"] == "test_filter"
        assert recognizer._kwargs["audio_event_detection_enabled"] is True

        # Clean up
        recognizer._running = False

    def test_init_with_none_params(self):
        """Test __init__ with None parameters should not add to _kwargs."""
        recognizer = TranslationRecognizerRealtime(
            model="test-model",
            callback=MockTranslationCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=None,
            diarization_enabled=None,
            speaker_count=None,
        )

        # Verify None parameters are not in _kwargs
        assert "disfluency_removal_enabled" not in recognizer._kwargs
        assert "diarization_enabled" not in recognizer._kwargs
        assert "speaker_count" not in recognizer._kwargs

        # Clean up
        recognizer._running = False

    def test_init_with_partial_params(self):
        """Test __init__ with some explicit parameters."""
        recognizer = TranslationRecognizerRealtime(
            model="test-model",
            callback=MockTranslationCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=True,
            speaker_count=3,
        )

        # Verify only specified parameters are in _kwargs
        assert recognizer._kwargs["disfluency_removal_enabled"] is True
        assert recognizer._kwargs["speaker_count"] == 3
        assert "diarization_enabled" not in recognizer._kwargs

        # Clean up
        recognizer._running = False

    def test_init_with_translation_params(self):
        """Test __init__ with translation-specific parameters."""
        recognizer = TranslationRecognizerRealtime(
            model="test-model",
            callback=MockTranslationCallback(),
            format="pcm",
            sample_rate=16000,
            transcription_enabled=True,
            translation_enabled=True,
            source_language="zh",
            disfluency_removal_enabled=True,
        )

        # Verify translation parameters
        assert recognizer.transcription_enabled is True
        assert recognizer.translation_enabled is True
        assert recognizer.source_language == "zh"
        # Verify recognition parameters
        assert recognizer._kwargs["disfluency_removal_enabled"] is True

        # Clean up
        recognizer._running = False


class TestTranslationRecognizerChatExplicitParams:
    """Test explicit parameters in TranslationRecognizerChat class."""

    def test_init_with_all_explicit_params(self):
        """Test __init__ with all explicit parameters."""
        recognizer = TranslationRecognizerChat(
            model="test-model",
            callback=MockTranslationCallback(),
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
        assert recognizer._kwargs["disfluency_removal_enabled"] is True
        assert recognizer._kwargs["diarization_enabled"] is True
        assert recognizer._kwargs["speaker_count"] == 2
        assert recognizer._kwargs["timestamp_alignment_enabled"] is True
        assert recognizer._kwargs["special_word_filter"] == "test_filter"
        assert recognizer._kwargs["audio_event_detection_enabled"] is True

        # Clean up
        recognizer._running = False

    def test_init_with_none_params(self):
        """Test __init__ with None parameters should not add to _kwargs."""
        recognizer = TranslationRecognizerChat(
            model="test-model",
            callback=MockTranslationCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=None,
            diarization_enabled=None,
        )

        # Verify None parameters are not in _kwargs
        assert "disfluency_removal_enabled" not in recognizer._kwargs
        assert "diarization_enabled" not in recognizer._kwargs

        # Clean up
        recognizer._running = False

    def test_init_with_partial_params(self):
        """Test __init__ with some explicit parameters."""
        recognizer = TranslationRecognizerChat(
            model="test-model",
            callback=MockTranslationCallback(),
            format="pcm",
            sample_rate=16000,
            diarization_enabled=True,
            audio_event_detection_enabled=True,
        )

        # Verify only specified parameters are in _kwargs
        assert recognizer._kwargs["diarization_enabled"] is True
        assert recognizer._kwargs["audio_event_detection_enabled"] is True
        assert "disfluency_removal_enabled" not in recognizer._kwargs
        assert "speaker_count" not in recognizer._kwargs

        # Clean up
        recognizer._running = False

    def test_init_with_extra_kwargs(self):
        """Test __init__ with extra kwargs should still work."""
        recognizer = TranslationRecognizerChat(
            model="test-model",
            callback=MockTranslationCallback(),
            format="pcm",
            sample_rate=16000,
            disfluency_removal_enabled=True,
            custom_param="custom_value",
        )

        # Verify both explicit and extra kwargs are stored
        assert recognizer._kwargs["disfluency_removal_enabled"] is True
        assert recognizer._kwargs["custom_param"] == "custom_value"

        # Clean up
        recognizer._running = False
