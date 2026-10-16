# -*- coding: utf-8 -*-
"""Qwen TTS speech-synthesis example (via MultiModalConversation)."""

import os

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-tts-flash")

TEXT = "Today is a wonderful day to build something people love!"


if __name__ == "__main__":
    # Non-streaming: returns the URL of the synthesized audio
    print("----- non-streaming request -----")
    response = dashscope.MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        text=TEXT,
        voice="Cherry",
        stream=False,
        language_type="English",
    )
    if response.output is None:
        print(f"error: {response}")
    else:
        print(f"synthesized audio url: {response.output.audio.url}")
        print(f"expires at: {response.output.audio.expires_at}")

    # Streaming: returns base64-encoded audio data chunks
    print("----- streaming request -----")
    responses = dashscope.MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        text=TEXT,
        voice="Cherry",
        stream=True,
        language_type="English",
    )
    for chunk in responses:
        if chunk.output is None:
            print(f"error: {chunk}")
            break
        print(f"base64 audio data: {chunk.output.audio.data}")
        if chunk.output.finish_reason == "stop":
            print(f"finish at: {chunk.output.audio.expires_at}")
