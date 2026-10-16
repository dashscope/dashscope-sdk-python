# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 TingWu example: offline task call and realtime
session.

Before running, replace the appid / fileUrl placeholders with your
own TingWu app ID and a publicly accessible audio/video URL.
"""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get(
    "DASHSCOPE_MODEL",
    "tingwu-automotive-service-inspection",
)
APP_ID = os.environ.get("DASHSCOPE_TINGWU_APP_ID", "your-app-id")

# Audio file streamed to the realtime session (PCM, 16 kHz); it can
# be overridden via the DASHSCOPE_DEMO_AUDIO environment variable
AUDIO_FILE = os.environ.get(
    "DASHSCOPE_DEMO_AUDIO",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "data",
        "asr_example.wav",
    ),
)


if __name__ == "__main__":
    client = dashscope.Client()

    # Offline task call: input carries the business-defined fields;
    # fileUrl is the audio/video file to analyze and appid is the
    # TingWu app ID
    print("----- offline call -----")
    result = client.tingwu.call(
        model=MODEL,
        input={
            "fileUrl": "http://demo.com/test.mp3",
            "appid": "123456",
        },
    )
    if result.is_error:
        print(f"error: {result.code} {result.message}")
    else:
        print(result.output)

    # Realtime session (websocket): stream a local audio file and
    # print the recognition events
    print("----- realtime session -----")
    with client.tingwu.realtime(
        model="tingwu-turbo",
        app_id=APP_ID,
        format="pcm",
        sample_rate=16000,
    ) as session:
        with open(AUDIO_FILE, "rb") as f:
            while True:
                data = f.read(3200)
                if not data:
                    break
                session.send_audio_frame(data)
        session.stop()
        for event in session.events():
            print(event)


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Meeting harness: 04_real_world_scenarios/realtime_audio_meeting_harness.py
#   (concurrency hosting, reconnects and per-room fault isolation)
# -------------------------------------------------------------
