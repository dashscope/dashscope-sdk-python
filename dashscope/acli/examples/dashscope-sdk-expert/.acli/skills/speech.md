---
name: speech
description: Quick reference for Speech (SpeechSynthesizer synthesis / Transcription) models, API parameters, inputs/outputs, and error codes
---
# Speech

For Python SDK 2.0.3, start with `dashscope.Client` or `dashscope.AsyncClient`; the 2.0 API is at the package root (`dashscope.v2` does not exist). Verify parameters against `dashscope/resources/audio/`, `dashscope/resources/tingwu.py`, and `dashscope/types/` rather than answering from memory. Use a context manager or call `close()`/`aclose()` so HTTP and WebSocket resources are released.

## Choose the Current API

| Scenario | Current entry point | Contract |
| --- | --- | --- |
| WebSocket TTS, complete audio | `client.audio.tts.synthesize(...)` | Returns audio `bytes` |
| WebSocket TTS, output streaming | `client.audio.tts.stream(...)` | Iterates audio `bytes`; close the iterator if exiting early |
| WebSocket TTS, incremental text | `client.audio.tts.connect(...)` | Synchronous duplex `SynthesisSession` |
| HTTP/SSE TTS | `client.audio.tts.http.synthesize/stream(...)` | URL result or SSE audio chunks, not a WebSocket task |
| Recorded-file transcription | `client.audio.asr.transcribe(...)` or `client.audio.asr.jobs` | Asynchronous server task represented by a flat `Job` |
| Streaming speech recognition | `client.audio.asr.recognize(...)` | Synchronous duplex WebSocket `Recognition` session |
| Omni/Qwen-Audio/Livetranslate | `client.audio.realtime.connect(...)` | Synchronous realtime WebSocket JSON events |
| TingWu offline meeting processing | `client.tingwu.call(...)` | `TingWuResult`; an async variant also exists |
| TingWu streaming recognition | `client.tingwu.realtime(...)` | Synchronous WebSocket session |

**AsyncAudio limitation:** `AsyncClient.audio` currently exposes only `audio.asr` for file transcription (`transcribe` and `jobs`). It does not expose TTS, real-time recognition, or `audio.realtime`; use synchronous `Client` for those WebSocket APIs. TingWu real-time sessions are also synchronous. Do not wrap a synchronous session method in `await`.

## WebSocket TTS (`client.audio.tts`)

`synthesize`, `stream`, and `connect` default to `model="cosyvoice-v1"`, `voice="longxiaochun"`, `format="mp3"`, and `sample_rate=22050`. Named options include `volume`, `rate`, `pitch`, `language_hints`, `enable_ssml`, `parameters`, `timeout`, `api_key`, and `workspace`.

```python
from pathlib import Path
import dashscope

with dashscope.Client() as client:  # reads DASHSCOPE_API_KEY by default
    audio = client.audio.tts.synthesize(
        "Hello, Bailian.",
        model="cosyvoice-v1",
        voice="longxiaochun",
        format="mp3",
        sample_rate=22050,
    )
    Path("out.mp3").write_bytes(audio)
```

- `stream(text, ...)` yields encoded audio chunks. Consume it fully, or call its `close()` when abandoning iteration; do not mistake it for an HTTP response.
- `connect(...)` is a context manager. Use `send_text()` for fragments, `flush()` to request buffered synthesis without ending input, and `stop()` to end input. `stop()` does not wait for output: keep one receiver draining `audio_chunks()` or `events()` through completion. Use a sender thread and a single receiver thread for concurrent duplex traffic; `cancel()` requests cancellation.
- A caller-owned `client.ws_connection()` can carry compatible ASR/TTS/TingWu tasks. Close that connection separately; realtime Omni sessions cannot use it.

### HTTP/SSE TTS (`client.audio.tts.http`)

Use this sibling when the desired model/protocol returns an audio URL instead of WebSocket bytes:

```python
import dashscope

with dashscope.Client() as client:
    result = client.audio.tts.http.synthesize(
        "Hello, Bailian.",
        model="cosyvoice-v1",
        voice="longxiaochun",
    )
    print(result.audio_url, result.audio_id, result.expires_at)
```

`http.stream(...)` yields chunks with flat `audio` bytes, `sentence`, `finish_reason`, and `is_final`; the final chunk can carry whole-audio URL fields. Close an SSE stream if not consuming it fully. These fields are not WebSocket `bytes` frames.

## File Transcription (`client.audio.asr`)

`transcribe(file_urls, ...)` submits and waits. Its current default model is `qwen-audio-3.1-asr-flash-filetrans`; pass `model=` explicitly when selecting another model. `parameters` carries service-specific options and `phrase_id` selects a hotword resource. Waiting defaults to 86400 seconds and uses bounded exponential polling.

For separate submission and polling, use `jobs.create(*, model, file_urls, parameters=None, phrase_id=None, ...)`, `jobs.fetch(task_id)`, and `jobs.wait(job_or_task_id)`. A `Job` exposes `task_id`, `status`, `results`, `finished`, `succeeded`, and common flat metadata. On success, each result's `transcription_url` is a URL to result JSON that the application must download and parse. Terminal states include `SUCCEEDED`, `FAILED`, `CANCELED`, and `UNKNOWN`.

```python
import dashscope

with dashscope.Client() as client:
    submitted = client.audio.asr.jobs.create(
        model="qwen-audio-3.1-asr-flash-filetrans",
        file_urls=["https://example.com/audio.wav"],
    )
    job = client.audio.asr.jobs.wait(submitted)
    print(job.task_id, job.status)
    for result in job.results or []:
        print(result.transcription_url)
```

The same file-transaction surface is asynchronous on `AsyncClient`:

```python
import asyncio
import dashscope

async def main():
    async with dashscope.AsyncClient() as client:
        job = await client.audio.asr.transcribe(
            file_urls=["https://example.com/audio.wav"],
        )
        print(job.status)

asyncio.run(main())
```

`Job.output`, when present, is the retained raw service payload, not the normal 2.0 access pattern; use the flat fields above.

## Real-Time Recognition (`client.audio.asr.recognize`)

`recognize(*, model, format="pcm", sample_rate=16000, vocabulary_id=None, disfluency_removal_enabled=None, diarization_enabled=None, parameters=None, ...)` creates a synchronous session. It supports `start()`, `send_audio_frame(bytes)`, `update_context()`, `stop()`, `events()`, `results()`, and `close()`, and can be used as a context manager.

```python
import dashscope

pcm_frames = [b"..."]  # replace with actual PCM frames

with dashscope.Client() as client:
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2", format="pcm", sample_rate=16000,
    ) as session:
        for frame in pcm_frames:
            session.send_audio_frame(frame)
        session.stop()  # ends input only
        for event in session.events():  # drain through task-finished
            print(event)
```

For incremental sending while results arrive, run the sender and one receiver concurrently. `recognize_file(path, *, model, ...)` drives a local file through the same session and yields event dictionaries. Neither real-time method exists on `AsyncAudio`.

## Realtime Omni / Qwen-Audio / Livetranslate

`client.audio.realtime.connect(model=...)` is a synchronous context manager. Important methods include `update_session`, `append_audio`, `append_video`, `commit`, `clear_appended_audio`, `create_item`, `retrieve_item`, `delete_item`, `create_response`, `cancel_response`, `send_event`, `recv_event`, `events`, `end_session`, and `close`.

```python
import dashscope

with dashscope.Client() as client:
    with client.audio.realtime.connect(
        model="qwen-audio-3.0-realtime-plus",
    ) as session:
        session.update_session(["text"], enable_turn_detection=False)
        session.create_item({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello"}],
        })
        session.create_response()
        for event in session.events():
            print(event.get("type"))
            if event.get("type") == "response.done":
                break  # Qwen-Audio: close after the response; no end_session()
```

Events are JSON dictionaries; output audio is Base64 inside events, not an iterator of `bytes`. `events()` normally ends at `session.finished` or normal close, not automatically at every `response.done`, so a single-turn caller may break as above. Protocol capabilities differ by model: Qwen-Audio does not support `end_session()`, while Omni/Livetranslate lifecycle and session options must follow their own protocol. Realtime uses one socket per session and does not support `WsConnection` task reuse.

## TingWu

Offline processing uses `client.tingwu.call(*, model, input: dict, parameters=None, ...)`; with `AsyncClient`, await the same method. The input dictionary contains service fields such as `fileUrl` and `appid`.

```python
import dashscope

with dashscope.Client() as client:
    result = client.tingwu.call(
        model="tingwu-turbo",
        input={
            "fileUrl": "https://example.com/meeting.mp3",
            "appid": "YOUR_APP_ID",
        },
    )
    print(result.request_id, result.status_code)
```

`TingWuResult` intentionally retains service `output` and `usage` fields; this is a resource-specific result model, not a reason to use legacy `.output` envelopes on other 2.0 resources.

Real-time TingWu uses synchronous `client.tingwu.realtime(*, model, app_id, format="pcm", sample_rate=16000, terminology=None, max_end_silence=None, parameters=None, ...)`. In its context manager, send PCM with `send_audio_frame()`, call `stop()`, and drain `events()`; iteration ends at `speech-end`.

## 1.x Compatibility — Maintenance Only

Keep the following contracts only when repairing existing 1.x code. Do not combine them with `Client` resource calls or flat 2.0 models.

- `dashscope.audio.tts.SpeechSynthesizer`: Sambert synthesis; class method `call(model, text, callback=None, workspace=None, **kwargs)`. Important kwargs include `format`, `sample_rate`, `volume`, `rate`, `pitch`, timestamps, and `ResultCallback`. `SpeechSynthesisResult` uses `get_audio_data()`, `get_audio_frame()`, timestamp getters, and `get_response()`.
- `dashscope.audio.tts_v2.SpeechSynthesizer`: constructor-based CosyVoice synthesis. `call(text)` returns full `bytes` without a callback; callbacks receive `on_data(data)`. Incremental input uses `streaming_call()` / `streaming_complete()` / `streaming_cancel()`. Diagnostics include `get_last_request_id()`, `get_first_package_delay()`, and `get_response()`.
- `dashscope.audio.asr.Transcription`: recorded-file tasks. `call()` waits, `async_call()` submits, and `fetch()` / `wait()` query the task. Its legacy response uses `output.task_id`, `output.task_status`, and `output.results[].transcription_url`; `phrase_id` becomes an `asr_phrase` resource. Preserve this envelope only in old code.
- Legacy local exceptions include `ModelRequired`, `InputRequired`, and `InvalidTask`; legacy WebSocket callbacks can receive `task-failed`. Diagnose task failure from the legacy response/event's code and message.

For new 2.0 code, errors come from `dashscope.errors`; normally `raise_on_error=True` raises them. With `raise_on_error=False`, inspect flat `is_error`, `status_code`, `code`, `message`, and `request_id`. See the dedicated errors guide for the hierarchy and retry policy.

## Java SDK

Java SDK v2.22.23 entry points:

- `SpeechSynthesizer`: `ByteBuffer call(SpeechSynthesisParam param)` / `call(param, ResultCallback)` / `Flowable<SpeechSynthesisResult> streamCall(param)`; diagnostics: `getLastRequestId()`, `getFirstPackageDelay()`
- `Transcription` (async file transcription): `asyncCall(TranscriptionParam)` → `wait(TranscriptionQueryParam)` (optional `timeoutSeconds`) / `fetch(queryParam)`

```java
import com.alibaba.dashscope.audio.tts.*;

SpeechSynthesizer synthesizer = new SpeechSynthesizer();
SpeechSynthesisParam param = SpeechSynthesisParam.builder()
        .model("cosyvoice-v1").text("hello").voice("longxiaochun").build();
java.nio.ByteBuffer audio = synthesizer.call(param);  // write audio.array() to file
```

Samples: `HttpSpeechSynthesizerUsage.java`, `AudioRecognitionUsage.java`, `Qwen3AsrRealtimeUsage.java`.
