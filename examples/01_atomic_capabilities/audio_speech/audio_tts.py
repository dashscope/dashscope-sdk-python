# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 speech-synthesis (TTS) example: one-shot URL
result and streaming audio chunks."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "sambert-zhichu-v1")
VOICE = os.environ.get("DASHSCOPE_TTS_VOICE", "zhichu")

TEXT = "你好，这是 dashscope 2.0 的语音合成示例。"


if __name__ == "__main__":
    client = dashscope.Client()

    # One-shot synthesis: the rendered audio comes back as a
    # temporary URL
    print("----- synthesize request -----")
    result = client.audio.tts.http.synthesize(
        TEXT,
        model=MODEL,
        voice=VOICE,
        format="wav",
        sample_rate=16000,
    )
    print(f"audio url: {result.audio_url}")

    # Streaming synthesis: audio arrives as binary chunks that can be
    # written straight to a file
    print("----- streaming synthesis request -----")
    stream = client.audio.tts.http.stream(
        TEXT,
        model=MODEL,
        voice=VOICE,
    )
    with open("tts_output.wav", "wb") as f:
        for chunk in stream:
            if chunk.audio:
                f.write(chunk.audio)
    print("audio saved to tts_output.wav")


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   No dedicated harness for this capability yet; saving the
#   stream to disk plus the generic patterns above is enough.
# -------------------------------------------------------------
