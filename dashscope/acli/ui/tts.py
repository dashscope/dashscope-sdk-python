# -*- coding: utf-8 -*-
"""Voice output: Text-to-Speech using DashScope CosyVoice API.

Supports both streaming playback (low-latency, audio starts before synthesis
finishes) and one-shot synthesis.  Uses
``dashscope.audio.tts_v2.SpeechSynthesizer``.
"""
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements

from __future__ import annotations

import collections
import io
import os
import re
import threading
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Available TTS models (CosyVoice 2 is the recommended default)
TTS_MODELS = [
    "cosyvoice-v2",
    "cosyvoice-v1",
]

# Available voices per model — curated list, not exhaustive
TTS_VOICES: dict[str, list[str]] = {
    "cosyvoice-v2": [
        "longxiaochun_v2",
        "longxiaoxia_v2",
        "longxiaochen_v2",
        "longxiaobai_v2",
        "longlaotie_v2",
        "longshu_v2",
        "longwan_v2",
        "longxiang_v2",
        "longjing_v2",
        "longfei_v2",
        "longze_v2",
        "longtong_v2",
        "loongstella_v2",
        "loongbella_v2",
    ],
    "cosyvoice-v1": [
        "longxiaochun",
        "longxiaoxia",
        "longxiaochen",
        "longxiaobai",
        "longlaotie",
        "longshu",
        "longwan",
        "longxiang",
        "longjing",
        "longfei",
    ],
}

# Default voice per model
DEFAULT_VOICE: dict[str, str] = {
    "cosyvoice-v2": "longxiaochun_v2",
    "cosyvoice-v1": "longxiaochun",
}

# Voice display names
VOICE_DISPLAY: dict[str, str] = {
    "longxiaochun_v2": "Intellectual female",
    "longxiaoxia_v2": "Sweet female",
    "longxiaochen_v2": "Young male",
    "longxiaobai_v2": "Fresh male",
    "longlaotie_v2": "Northeastern buddy",
    "longshu_v2": "Warm male",
    "longwan_v2": "Energetic male",
    "longxiang_v2": "Magnetic male",
    "longjing_v2": "Broadcast female",
    "longfei_v2": "Passionate narrator",
    "longze_v2": "Dialect male",
    "longtong_v2": "Cute child",
    "loongstella_v2": "Intellectual female 2",
    "loongbella_v2": "Sweet female 2",
    "longxiaochun": "Intellectual female (v1)",
    "longxiaoxia": "Sweet female (v1)",
    "longxiaochen": "Young male (v1)",
    "longxiaobai": "Fresh male (v1)",
    "longlaotie": "Northeastern buddy (v1)",
    "longshu": "Warm male (v1)",
    "longwan": "Energetic male (v1)",
    "longxiang": "Magnetic male (v1)",
    "longjing": "Broadcast female (v1)",
    "longfei": "Passionate narrator (v1)",
}

# Max text length per TTS call (DashScope limit ~2000 chars)
MAX_TEXT_LENGTH = 1800


def is_available() -> tuple[bool, str]:
    """Check if TTS dependencies are installed."""
    try:
        import dashscope  # noqa: F401  # pylint: disable=unused-import
        import sounddevice  # noqa: F401  # pylint: disable=unused-import

        return True, ""
    except ImportError:
        return False, (
            "Voice output requires dependencies: pip install acli[voice]"
        )


def _strip_markdown(text: str) -> str:
    """Strip common markdown formatting for cleaner speech."""
    # Remove code blocks entirely
    text = re.sub(r"```[\s\S]*?```", "(code block omitted)", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove bold/italic markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Remove image syntax (before links — the link rule would otherwise
    # half-consume ![alt](url) and leave a stray '!alt')
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove headers markers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bullet points
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text(text: str, max_len: int = MAX_TEXT_LENGTH) -> list[str]:
    """Split long text into chunks suitable for TTS.

    Splits on sentence boundaries (。！？.!?\\n) when possible.
    """
    if len(text) <= max_len:
        return [text] if text else []

    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break

        # Find the last sentence boundary before max_len
        split_pos = -1
        for sep in [
            "。",
            "！",
            "？",
            ".\n",
            "!\n",
            "?\n",
            "\n",
            "。",
            ".",
            "！",
            "!",
            "？",
            "?",
            "；",
            ";",
        ]:
            pos = text.rfind(sep, 0, max_len)
            if pos > split_pos:
                split_pos = pos + len(sep)

        if split_pos <= 0:
            # No sentence boundary found; force split at max_len
            split_pos = max_len

        chunks.append(text[:split_pos])
        text = text[split_pos:]

    return [c for c in chunks if c.strip()]


class TTSPlayer:
    """Manages TTS synthesis and audio playback with streaming support."""

    def __init__(
        self,
        api_key: str,
        model: str = "cosyvoice-v2",
        voice: Optional[str] = None,
        speech_rate: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice or DEFAULT_VOICE.get(model, "longxiaochun_v2")
        self.speech_rate = speech_rate
        self._playing = False
        self._cancel = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None

    def set_model(self, model: str):
        """Switch TTS model."""
        if model in TTS_MODELS:
            self.model = model
            if self.voice not in TTS_VOICES.get(model, []):
                self.voice = DEFAULT_VOICE.get(model, self.voice)

    def set_voice(self, voice: str):
        """Switch TTS voice."""
        self.voice = voice

    def set_speed(self, rate: float):
        """Set speech rate (0.5 to 2.0, 1.0 = normal)."""
        self.speech_rate = max(0.5, min(2.0, rate))

    def cancel(self):
        """Cancel ongoing synthesis/playback."""
        self._cancel.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    @property
    def is_playing(self) -> bool:
        return self._playing

    def speak(self, text: str, *, strip_md: bool = True) -> Optional[str]:
        """Synthesize and play text synchronously.

        Returns error message string, or None on success.
        """
        ok, err = is_available()
        if not ok:
            return err

        if strip_md:
            text = _strip_markdown(text)

        if not text.strip():
            return None

        chunks = _split_text(text)
        if not chunks:
            return None

        self._cancel.clear()
        self._playing = True
        errors: list[str] = []

        try:
            for chunk in chunks:
                if self._cancel.is_set():
                    break
                err = self._synthesize_and_play(chunk)
                if err:
                    errors.append(err)
                    break
        finally:
            self._playing = False

        return errors[0] if errors else None

    def _synthesize_and_play(self, text: str) -> Optional[str]:
        """Call DashScope TTS and play the result.

        Streams audio to the speaker as soon as the WAV header is received,
        so playback starts well before synthesis finishes.
        """
        # Ensure the API key is available to the dashscope SDK.
        # The SDK reads dashscope.api_key (module-level), not the env var
        # directly in all code paths, so we must set both.
        import dashscope
        import numpy as np
        import sounddevice as sd

        os.environ.setdefault("DASHSCOPE_API_KEY", self.api_key)
        if not dashscope.api_key and self.api_key:
            dashscope.api_key = self.api_key

        try:
            from dashscope.audio.tts_v2 import (
                AudioFormat,
                ResultCallback,
                SpeechSynthesizer,
            )
        except ImportError:
            return (
                "DashScope TTS v2 unavailable; please upgrade dashscope: "
                "pip install -U dashscope"
            )

        audio_deque: collections.deque[bytes] = collections.deque()
        audio_complete = threading.Event()
        audio_error: list[str] = []

        class _Callback(ResultCallback):
            def on_open(self):
                pass

            def on_complete(self):
                audio_complete.set()

            def on_error(self, message: str):
                audio_error.append(message)
                audio_complete.set()

            def on_close(self):
                audio_complete.set()

            def on_data(self, data: bytes) -> None:
                audio_deque.append(data)

        callback = _Callback()

        # DashScope CosyVoice speech_rate is a float in [0.5, 2.0] where 1.0
        # is normal speed. Our tts_speed already uses the same scale.
        ds_speech_rate = max(0.5, min(2.0, self.speech_rate))

        try:
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.WAV_16000HZ_MONO_16BIT,
                speech_rate=ds_speech_rate,
                callback=callback,
            )
        except TypeError:
            # Older SDK without speech_rate kwarg
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.WAV_16000HZ_MONO_16BIT,
                callback=callback,
            )

        try:
            # Streaming synthesis — call in a thread to not block playback
            synth_done = threading.Event()
            synth_error: list[str] = []

            def _run_synth():
                try:
                    synthesizer.streaming_call(text)
                    synthesizer.streaming_complete()
                except Exception as e:
                    synth_error.append(str(e))
                    audio_complete.set()
                finally:
                    synth_done.set()

            synth_thread = threading.Thread(target=_run_synth, daemon=True)
            synth_thread.start()

            # Wait for first chunk (or completion) with timeout
            first_chunk_timeout = 10.0
            deadline = time.time() + first_chunk_timeout
            while not audio_deque and not audio_complete.is_set():
                if time.time() > deadline:
                    break
                audio_complete.wait(timeout=0.1)

            if not audio_deque:
                if audio_error:
                    return f"TTS synthesis failed: {audio_error[0]}"
                if synth_error:
                    return f"TTS synthesis failed: {synth_error[0]}"
                # Wait a bit more
                audio_complete.wait(timeout=5.0)
                if not audio_deque:
                    return "TTS synthesis failed: no audio data"

            if self._cancel.is_set():
                return None

            # Accumulate enough bytes to parse the WAV header, then start
            # streaming playback immediately.
            import wave

            header_buf = b""
            header_parsed = False
            sample_rate = 16000
            channels = 1
            sample_width = 2
            header_size = 44

            header_deadline = time.time() + 5.0
            while not header_parsed:
                while audio_deque and len(header_buf) < 8192:
                    header_buf += audio_deque.popleft()
                try:
                    with io.BytesIO(header_buf) as buf:
                        with wave.open(buf, "rb") as wf:
                            sample_rate = wf.getframerate()
                            channels = wf.getnchannels()
                            sample_width = wf.getsampwidth()
                            header_size = buf.tell()
                            header_parsed = True
                except (wave.Error, EOFError):
                    if (
                        len(header_buf) >= 8192
                        or time.time() > header_deadline
                    ):
                        break
                    audio_complete.wait(timeout=0.05)

            if not header_parsed:
                # Fallback: wait for all audio data and play the complete WAV
                # file the old-fashioned way.
                audio_complete.wait(timeout=60)
                all_audio = header_buf + b"".join(audio_deque)
                if self._cancel.is_set():
                    return None
                if not all_audio:
                    return "TTS playback failed: no audio data"
                try:
                    with io.BytesIO(all_audio) as buf:
                        with wave.open(buf, "rb") as wf:
                            sample_rate = wf.getframerate()
                            frames = wf.readframes(wf.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    sd.play(audio_array, samplerate=sample_rate)
                    sd.wait()
                    return None
                except Exception as e:
                    return f"TTS playback failed: {e}"

            # Any bytes after the WAV header are already-received audio data.
            # Put them at the front of the deque so playback order is
            # preserved.
            if header_size < len(header_buf):
                audio_deque.appendleft(header_buf[header_size:])

            playback_complete = threading.Event()
            playback_error: list[str] = []

            def _audio_callback(outdata, frames, _time, status):
                try:
                    if status:
                        pass
                    needed = frames * channels * sample_width
                    data = bytearray()
                    while len(data) < needed:
                        if not audio_deque:
                            break
                        chunk = audio_deque[0]
                        take = needed - len(data)
                        if len(chunk) <= take:
                            data += audio_deque.popleft()
                        else:
                            data += chunk[:take]
                            audio_deque[0] = chunk[take:]
                    if len(data) < needed:
                        if audio_complete.is_set() and not audio_deque:
                            playback_complete.set()
                        data += b"\x00" * (needed - len(data))
                    # Trim to exactly what we need and reshape.
                    samples = np.frombuffer(data[:needed], dtype=np.int16)
                    if channels > 1:
                        samples = samples.reshape(-1, channels)
                    else:
                        samples = samples.reshape(-1, 1)
                    outdata[:] = samples
                except Exception as e:
                    playback_error.append(str(e))
                    playback_complete.set()

            with sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=np.int16,
                callback=_audio_callback,
                blocksize=1024,
            ) as stream:
                # Wait for synthesis to finish and queue to drain.
                audio_complete.wait(timeout=60)
                # Wait until the callback has drained the queue.
                drain_deadline = time.time() + 5.0
                while (
                    not playback_complete.is_set()
                    and time.time() < drain_deadline
                ):
                    if not audio_deque:
                        playback_complete.set()
                    audio_complete.wait(timeout=0.1)
                # Give the sound device buffer time to finish playing.
                time.sleep(stream.latency + 0.2)

            if playback_error:
                return f"TTS playback failed: {playback_error[0]}"
            return None

        except Exception as e:
            return f"TTS playback failed: {e}"


def speak_text(
    api_key: str,
    text: str,
    model: str = "cosyvoice-v2",
    voice: Optional[str] = None,
    speech_rate: float = 1.0,
) -> Optional[str]:
    """One-shot TTS function. Returns error message or None on success."""
    player = TTSPlayer(api_key, model, voice, speech_rate)
    return player.speak(text)


# ---------------------------------------------------------------------------
# Streaming TTS Feeder - for real-time text-to-speech during streaming output
# ---------------------------------------------------------------------------


class StreamingTTSFeeder:
    """Feeds streaming text chunks to TTS, buffering by sentence boundaries.

    Usage:
        feeder = StreamingTTSFeeder(api_key, model, voice, speech_rate)
        feeder.start()
        for chunk in text_stream:
            feeder.feed(chunk)
        feeder.finish()  # flush remaining text and wait for completion
    """

    def __init__(
        self,
        api_key: str,
        model: str = "cosyvoice-v2",
        voice: Optional[str] = None,
        speech_rate: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice or DEFAULT_VOICE.get(model, "longxiaochun_v2")
        self.speech_rate = speech_rate

        self._buffer = ""
        self._code_block_depth = 0  # Track ``` blocks to skip
        self._queue: collections.deque[str] = collections.deque()
        self._queue_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._error: list[str] = []
        self._worker_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background TTS worker thread."""
        self._stop_event.clear()
        self._done_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )
        self._worker_thread.start()

    def feed(self, text: str):
        """Feed a text chunk. Call this for each streaming chunk."""
        # Track code blocks
        self._buffer += text

        # Count ``` to track code block depth
        backtick_count = self._buffer.count("```")
        in_code_block = (backtick_count % 2) == 1

        if in_code_block:
            # Still accumulating code block, don't send to TTS
            return

        # Not in code block, process buffer
        self._process_buffer()

    def _process_buffer(self):
        """Extract complete sentences from buffer and queue them."""
        # Sentence boundaries
        sentence_endings = re.compile(r"[。！？.!?\n；;](?=\s|$)")

        while True:
            match = sentence_endings.search(self._buffer)
            if not match:
                break

            sentence = self._buffer[: match.end()]
            self._buffer = self._buffer[match.end() :]

            # Strip markdown and queue
            sentence = _strip_markdown(sentence)
            if sentence.strip():
                with self._queue_lock:
                    self._queue.append(sentence)

    def finish(self) -> Optional[str]:
        """Flush remaining text and wait for TTS completion.

        Returns error message or None on success.
        """
        # Flush any remaining buffer
        if self._buffer.strip():
            text = _strip_markdown(self._buffer)
            if text.strip():
                with self._queue_lock:
                    self._queue.append(text)
            self._buffer = ""

        # Signal worker to stop after queue is empty
        with self._queue_lock:
            self._queue.append(None)  # Sentinel

        # Wait for completion
        self._done_event.wait(timeout=120)
        return self._error[0] if self._error else None

    def cancel(self):
        """Cancel TTS immediately."""
        self._stop_event.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def _worker_loop(self):
        """Background worker that processes queued sentences."""
        try:
            player = TTSPlayer(
                self.api_key,
                self.model,
                self.voice,
                self.speech_rate,
            )

            while not self._stop_event.is_set():
                # Get next sentence
                with self._queue_lock:
                    sentence = self._queue.popleft() if self._queue else ...
                if sentence is ...:
                    # Sleep outside the lock so feed()/finish() are never
                    # blocked by the idle poll.
                    time.sleep(0.05)
                    continue

                if sentence is None:  # Sentinel - done
                    break

                # Synthesize and play
                err = player.speak(
                    sentence,
                    strip_md=False,
                )  # Already stripped
                if err:
                    self._error.append(err)
                    break

        except Exception as e:
            self._error.append(str(e))
        finally:
            self._done_event.set()
