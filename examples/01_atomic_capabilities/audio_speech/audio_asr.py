# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 speech-recognition (ASR) example: batch
transcription over audio file URLs."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "paraformer-v2")

# Replace with your own audio file URLs (must be publicly accessible)
FILE_URLS = [
    "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3",
]


if __name__ == "__main__":
    client = dashscope.Client()

    # Batch transcription: submits the task and waits for the result
    print("----- transcribe request -----")
    job = client.audio.asr.transcribe(
        FILE_URLS,
        model=MODEL,
        parameters={"language_hints": ["zh"]},
    )
    if job.succeeded:
        assert job.output is not None
        for item in job.output["results"]:
            print(f"file: {item['file_url']}")
            print(f"transcription url: {item['transcription_url']}")
    else:
        print(f"job failed: {job.output}")

    # Realtime recognition of a local file (websocket) is available
    # via client.audio.asr.recognize_file(path, model=...), which
    # yields recognition events as an iterator.


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Realtime hosting: 04_real_world_scenarios/realtime_audio_meeting_harness.py
# -------------------------------------------------------------
