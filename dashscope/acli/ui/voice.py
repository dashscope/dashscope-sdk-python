# -*- coding: utf-8 -*-
"""Voice input: record from microphone + realtime DashScope ASR streaming
with VAD."""
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements,protected-access,unused-argument

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional


def is_available() -> tuple[bool, str]:
    """Check if voice input dependencies are installed."""
    try:
        import dashscope  # noqa: F401  # pylint: disable=unused-import
        import numpy  # noqa: F401  # pylint: disable=unused-import
        import sounddevice  # noqa: F401  # pylint: disable=unused-import

        return True, ""
    except ImportError:
        return False, (
            "Voice input requires dependencies: pip install acli[voice]"
        )


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
DEFAULT_SILENCE_THRESHOLD = (
    500  # RMS below this = silence (speech typically 800+)
)
DEFAULT_SILENCE_DURATION = 2.0  # seconds of silence to auto-stop
DEFAULT_MAX_RECORDING_SECONDS = 60  # maximum recording duration
MAX_ASR_RESTARTS = 3  # max session restarts before giving up

# Shared cancellation event used by /voice off across CLI and TUI.
_voice_cancel_event: Optional[threading.Event] = None


def try_cancel_voice_input() -> bool:
    """Signal an ongoing voice_input call to cancel. Returns True if
    cancelled."""
    if _voice_cancel_event is not None and not _voice_cancel_event.is_set():
        _voice_cancel_event.set()
        return True
    return False


class _RealtimeASR:
    """Wraps DashScope Recognition in streaming mode with live partial display.
    Supports automatic session restart when the server silently drops the
    connection.
    """

    def __init__(self, api_key: str, model: str):
        os.environ.setdefault("DASHSCOPE_API_KEY", api_key)
        from dashscope.audio.asr import Recognition, RecognitionCallback

        self._model = model
        self._Recognition = Recognition
        self._RecognitionCallback = RecognitionCallback

        self._final_sentences: list[str] = []
        self._partial_text: str = ""
        self._error: Optional[str] = None
        self._complete = threading.Event()
        self._lock = threading.Lock()
        self._event_count: int = 0
        self._send_errors: list[str] = []
        self._callback_log: list[str] = []
        self._restart_count: int = 0
        self._recognition = None
        self._stopped = False

    def _make_recognition(self):
        """Create a fresh Recognition instance with callbacks."""
        parent = self
        self._complete.clear()

        class _Callback(self._RecognitionCallback):
            def on_open(self):
                with parent._lock:
                    parent._callback_log.append("on_open")

            def on_event(self, result) -> None:
                with parent._lock:
                    parent._event_count += 1
                output = getattr(result, "output", None) or {}
                sentence = output.get("sentence", {})
                if isinstance(sentence, dict):
                    text = sentence.get("text", "")
                    is_final = sentence.get("end_time") is not None
                    with parent._lock:
                        if is_final:
                            if text:
                                parent._final_sentences.append(text)
                            parent._partial_text = ""
                        else:
                            parent._partial_text = text

            def on_close(self):
                with parent._lock:
                    parent._callback_log.append("on_close")

            def on_complete(self) -> None:
                with parent._lock:
                    parent._callback_log.append("on_complete")
                parent._complete.set()

            def on_error(self, result) -> None:
                msg = getattr(result, "message", None) or str(result)
                with parent._lock:
                    parent._callback_log.append(f"on_error: {msg}")
                parent._error = msg
                parent._complete.set()

        callback = _Callback()
        return self._Recognition(
            model=self._model,
            callback=callback,
            format="pcm",
            sample_rate=SAMPLE_RATE,
            language_hints=["zh", "en"],
        )

    def start(self):
        with self._lock:
            self._stopped = False
        self._recognition = self._make_recognition()
        self._recognition.start()
        with self._lock:
            self._callback_log.append("start() called")

    def restart(self):
        """Restart the ASR session. Called when server silently drops
        connection."""
        with self._lock:
            if self._stopped:
                return
            if self._restart_count >= MAX_ASR_RESTARTS:
                self._error = (
                    f"ASR connection dropped; max restarts reached "
                    f"({MAX_ASR_RESTARTS})"
                )
                return
            self._restart_count += 1
            self._partial_text = ""
        try:
            if self._recognition:
                try:
                    self._recognition.stop()
                except Exception:
                    pass
        except Exception:
            pass
        self._recognition = self._make_recognition()
        self._recognition.start()
        with self._lock:
            self._callback_log.append(f"restart #{self._restart_count}")

    def send(self, audio_bytes: bytes):
        try:
            self._recognition.send_audio_frame(audio_bytes)
        except Exception as e:
            with self._lock:
                err_msg = f"{type(e).__name__}: {e}"
                if not self._send_errors or self._send_errors[-1] != err_msg:
                    self._send_errors.append(err_msg)
            # Auto-restart on "has stopped" — server dropped the session
            if "stopped" in str(e).lower():
                self.restart()
                # Re-send the frame on the new session
                try:
                    self._recognition.send_audio_frame(audio_bytes)
                except Exception:
                    pass

    def stop(self):
        with self._lock:
            self._stopped = True
        try:
            self._recognition.stop()
        except Exception:
            pass
        self._complete.wait(timeout=10)

    @property
    def current_display(self) -> str:
        with self._lock:
            parts = self._final_sentences[:]
            if self._partial_text:
                parts.append(self._partial_text)
            return "".join(parts)

    @property
    def final_text(self) -> str:
        with self._lock:
            parts = self._final_sentences[:]
            if self._partial_text:
                parts.append(self._partial_text)
            return "".join(parts)

    @property
    def has_content(self) -> bool:
        with self._lock:
            return bool(self._final_sentences or self._partial_text)

    @property
    def error(self) -> Optional[str]:
        return self._error


async def voice_input(
    api_key: str,
    model: str = "paraformer-realtime-v2",
    display_callback=None,
    cancel_event: Optional[threading.Event] = None,
    silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
    silence_duration: float = DEFAULT_SILENCE_DURATION,
    max_recording_seconds: int = DEFAULT_MAX_RECORDING_SECONDS,
) -> Optional[str]:
    """Record from mic with realtime ASR + VAD auto-stop on silence.

    Args:
        api_key: DashScope API key
        model: ASR model name
        display_callback: Optional callable(text: str) for TUI mode. If
                         provided, real-time ASR output is sent here
                         instead of rich.console.
        cancel_event: Optional threading.Event for TUI mode. If provided,
                     set it to cancel recording instead of pressing Enter.
        silence_threshold: RMS below this is treated as silence.
        silence_duration: Seconds of silence before auto-stop.
        max_recording_seconds: Maximum recording duration before auto-stop.

    Returns:
        Recognized text or None if cancelled/error
    """
    global _voice_cancel_event
    _voice_cancel_event = (
        cancel_event if cancel_event is not None else threading.Event()
    )
    try:
        return await _voice_input_impl(
            api_key,
            model,
            display_callback,
            cancel_event,
            silence_threshold,
            silence_duration,
            max_recording_seconds,
        )
    finally:
        # Clear the global on every exit path — otherwise a finished
        # session leaves an unset event behind and a later /voice off
        # falsely reports having cancelled a recording.
        _voice_cancel_event = None


async def _voice_input_impl(
    api_key: str,
    model: str = "paraformer-realtime-v2",
    display_callback=None,
    cancel_event: Optional[threading.Event] = None,
    silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
    silence_duration: float = DEFAULT_SILENCE_DURATION,
    max_recording_seconds: int = DEFAULT_MAX_RECORDING_SECONDS,
) -> Optional[str]:
    import asyncio

    import numpy as np

    # Only import rich if no display_callback (non-TUI mode)
    if display_callback is None:
        from rich.console import Console
        from rich.live import Live
        from rich.text import Text

        console = Console()
    else:
        console = None

    ok, err = is_available()
    if not ok:
        if console:
            console.print(f"[red]{err}[/red]")
        elif display_callback:
            display_callback(f"[Error] {err}")
        return None

    import sounddevice as sd

    stop_event = threading.Event()
    asr = _RealtimeASR(api_key, model)

    try:
        asr.start()
    except Exception as e:
        if console:
            console.print(f"[red]ASR connection failed: {e}[/red]")
        elif display_callback:
            display_callback(f"[Error] ASR connection failed: {e}")
        return None

    block_size = 3200  # 200ms of 16kHz 16-bit mono
    silence_start: Optional[float] = None
    has_speech = False
    recording_start = time.time()

    def _audio_callback(indata, frames, time_info, status):
        nonlocal silence_start, has_speech
        if stop_event.is_set():
            return
        audio_bytes = indata.tobytes()
        asr.send(audio_bytes)

        # VAD: check RMS energy
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

        if rms > silence_threshold:
            has_speech = True
            silence_start = None
        else:
            # Start silence timer even without prior speech (fallback after 5s)
            if silence_start is None:
                silence_start = time.time()
            elif (
                has_speech and time.time() - silence_start >= silence_duration
            ):
                # Speech detected, then silence for silence_duration
                stop_event.set()
            elif (
                not has_speech
                and time.time() - recording_start > 5
                and time.time() - silence_start >= silence_duration
            ):
                # No speech detected after 5 seconds, and still silent
                stop_event.set()

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=block_size,
            callback=_audio_callback,
        )
    except Exception as e:
        if console:
            console.print(f"[red]Failed to open microphone: {e}[/red]")
        elif display_callback:
            display_callback(f"[Error] Failed to open microphone: {e}")
        asr.stop()
        return None

    if console:
        console.print(
            "[bold green]🎤 Recording...[/bold green] "
            "[dim](auto-stops on silence, or press Enter)[/dim]",
        )
    elif display_callback:
        display_callback("🎤 Recording... (auto-stops on silence)")

    loop = asyncio.get_event_loop()
    last_display = ""

    with stream:
        # Non-TUI mode: use rich.live.Live and wait for Enter
        if console:
            with Live(
                "",
                console=console,
                refresh_per_second=8,
                transient=True,
            ) as live:
                enter_future = loop.run_in_executor(
                    None,
                    _wait_for_enter,
                    stop_event,
                )

                while not stop_event.is_set():
                    current = asr.current_display
                    if current != last_display:
                        live.update(Text(f"  {current}", style="cyan"))
                        last_display = current
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(enter_future),
                            timeout=0.15,
                        )
                        break
                    except asyncio.TimeoutError:
                        continue
                    except (KeyboardInterrupt, EOFError):
                        stop_event.set()
                        asr.stop()
                        console.print("[dim]Cancelled[/dim]")
                        return None
        # TUI mode: use display_callback and cancel_event
        else:
            while not stop_event.is_set():
                # Check max recording duration
                elapsed = time.time() - recording_start
                if elapsed >= max_recording_seconds:
                    if display_callback:
                        display_callback(
                            f"[Info] Max recording duration reached "
                            f"({max_recording_seconds}s)",
                        )
                    stop_event.set()
                    break

                # Check if cancelled from outside
                if cancel_event and cancel_event.is_set():
                    if display_callback:
                        display_callback("[Info] Recording cancelled")
                    stop_event.set()
                    break

                current = asr.current_display
                if current != last_display and display_callback:
                    display_callback(current)
                    last_display = current

                try:
                    await asyncio.sleep(0.125)  # 8 FPS equivalent
                except (KeyboardInterrupt, EOFError):
                    stop_event.set()
                    asr.stop()
                    return None

    stop_event.set()
    asr.stop()

    if asr.error:
        if console:
            console.print(f"[red]ASR error: {asr.error}[/red]")
        elif display_callback:
            display_callback(f"[Error] ASR error: {asr.error}")
        return None

    text = asr.final_text.strip()
    if not text:
        if console:
            console.print("[yellow]Nothing recognized[/yellow]")
        elif display_callback:
            display_callback("[Info] Nothing recognized")
        return None

    if console:
        console.print(f"[green]✓[/green] {text}")
    return text


def _wait_for_enter(stop_event: threading.Event) -> None:
    """Block until user presses Enter or stop_event is set."""
    if os.name == "nt":
        # Windows: use msvcrt for non-blocking keyboard input
        import msvcrt

        while not stop_event.is_set():
            if msvcrt.kbhit():
                msvcrt.getch()
                stop_event.set()
                return
            time.sleep(0.2)
    else:
        # POSIX: use select for non-blocking stdin check
        import select

        while not stop_event.is_set():
            # Check if stdin has data (non-blocking)
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                sys.stdin.readline()
                stop_event.set()
                return
