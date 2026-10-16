# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Optional adapter letting 1.x-style callback objects consume v2 streams.

Both 1.x callback shapes are duck-typed — asr ``RecognitionCallback``
(on_open/on_event/on_complete/on_error/on_close, recognition.py:140) and
tts_v2 ``ResultCallback`` (which adds ``on_data`` for binary audio,
speech_synthesizer.py:69-90) — so one helper serves both: only the
methods the object defines are invoked.
"""

from __future__ import annotations

from typing import Any, Iterable

from dashscope.v2.transport.ws import EVENT_FINISHED

__all__ = [
    "run_with_callback",
]


def run_with_callback(frames: Iterable[Any], callback: Any) -> None:
    """Consume a v2 frame stream, driving a 1.x-style callback object.

    ``frames`` is any v2 stream: ``Recognition.events()`` (raw dicts) or
    ``Recognition.results()`` (typed ``RecognitionResult``, matching what
    1.x hands ``on_event``), ``Tts.stream()``/
    ``SynthesisSession.audio_chunks()``.
    Bytes go to ``on_data``; dict events go to ``on_event``.

    v2 sessions expose no open/close hooks, so the adapter fires
    ``on_open`` itself before the first frame and ``on_close`` last —
    the points where 1.x raises them around the websocket lifecycle. A
    terminal ``task-finished`` event maps to ``on_complete`` (1.x never
    routes it to ``on_event``). A stream error is delivered through
    ``on_error`` as the raised exception and is not re-raised, matching
    1.x callback mode.
    """
    _invoke(callback, "on_open")
    try:
        for frame in frames:
            if isinstance(frame, (bytes, bytearray)):
                _invoke(callback, "on_data", bytes(frame))
            elif _is_task_finished(frame):
                _invoke(callback, "on_complete")
                break
            else:
                _invoke(callback, "on_event", frame)
        else:
            # Streams without a terminal event (e.g. TTS audio chunks)
            # still report completion, as 1.x does on task-finished.
            _invoke(callback, "on_complete")
    except Exception as exc:  # pylint: disable=broad-except
        _invoke(callback, "on_error", exc)
    finally:
        _invoke(callback, "on_close")


def _is_task_finished(frame: Any) -> bool:
    return (
        isinstance(frame, dict)
        and (frame.get("header") or {}).get("event") == EVENT_FINISHED
    )


def _invoke(callback: Any, name: str, *args: Any) -> None:
    method = getattr(callback, name, None)
    if callable(method):
        method(*args)
