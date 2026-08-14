---
name: speech
description: Quick reference for Speech (SpeechSynthesizer synthesis / Transcription) models, API parameters, inputs/outputs, and error codes
---
# Speech

Before answering, locate the right path: synthesis goes through `SpeechSynthesizer` (legacy `dashscope.audio.tts` / new `dashscope.audio.tts_v2`); recorded-file transcription goes through `dashscope.audio.asr.Transcription` (async task). Parameter names must be verified against the source in `dashscope/audio/tts/`, `dashscope/audio/tts_v2/`, and `dashscope/audio/asr/transcription.py`; never answer from memory.

## Applicable Models and Scenarios

- Speech synthesis (TTS):
  - `dashscope.audio.tts.SpeechSynthesizer`: Sambert series (e.g. `sambert-zhichu-v1`), WebSocket streaming, class method `call` for one-shot synthesis.
  - `dashscope.audio.tts_v2.SpeechSynthesizer`: CosyVoice series (e.g. `cosyvoice-v1`, `cosyvoice-v2`), requires a `voice` (e.g. `longxiaochun`, `longxiaochun_v2`), supports streaming text input and voice-clone timbres.
- Recorded-file transcription (ASR): `dashscope.audio.asr.Transcription`, models listed in `Transcription.Models`: `paraformer-v1`, `paraformer-8k-v1` (8kHz telephony), `paraformer-mtl-v1` (multilingual); input is a list of publicly accessible audio URLs, async task.

## SDK APIs

`tts.SpeechSynthesizer.call(model, text, callback=None, workspace=None, **kwargs) -> SpeechSynthesisResult`, key kwargs:

| Parameter | Description |
|---|---|
| format | `SpeechSynthesizer.AudioFormat`: `format_wav` (default) / `format_pcm` / `format_mp3` |
| sample_rate | sample rate; defaults to the model's sample rate |
| volume | volume 0~100, default 50 |
| rate / pitch | speech rate / pitch, 0.5~2.0, default 1.0 |
| word_timestamp_enabled / phoneme_timestamp_enabled | word-level / phoneme-level timestamps, default False |
| callback | subclass of `ResultCallback`: `on_open/on_event/on_error/on_complete/on_close` |

`tts_v2.SpeechSynthesizer(model, voice, format=AudioFormat.DEFAULT, volume=50, speech_rate=1.0, pitch_rate=1.0, seed=0, instruction=None, language_hints=None, callback=None, workspace=None, ...)` is constructor-based:
- `call(text, timeout_millis=None)`: without callback, blocks and returns the full audio `bytes`; with callback, audio is delivered in real time via `on_data(data: bytes)`.
- Streaming input: `streaming_call(text)` (may be called multiple times) -> `streaming_complete()` to wait for completion; `streaming_cancel()` to cancel. Empty `model` raises `ModelRequired`; empty apikey raises `InputRequired`.
- `AudioFormat` enum values look like `MP3_22050HZ_MONO_256KBPS`, `WAV_16000HZ_MONO_16BIT`, `PCM_*`, `OGG_OPUS_*`; `DEFAULT` is treated as mp3/22050Hz.
- Helpers: `get_last_request_id()`, `get_first_package_delay()`, `get_response()`.

`Transcription` (async task):

| Method | Description |
|---|---|
| `call(model, file_urls, phrase_id=None, api_key=None, workspace=None, **kwargs)` | synchronous wrapper that waits for the result |
| `async_call(...)` (same signature) | submits the task only; returns a response containing `output.task_id` |
| `fetch(task, ...)` | single query; `task` accepts a task_id string or the response object from async_call |
| `wait(task, wait_timeout=-1, ...)` | polls until a terminal state; `wait_timeout=-1` means no time limit |

Transcription kwargs: `channel_id: List[int]`, `disfluency_removal_enabled` (remove filler words), `diarization_enabled` (speaker diarization), `speaker_count`, `timestamp_alignment_enabled`, `special_word_filter`, `audio_event_detection_enabled`; `phrase_id` (hotwords) is converted to `resources=[{"resource_id": phrase_id, "resource_type": "asr_phrase"}]`. Task flow: `async_call` -> `output.task_id` (`task_status=PENDING`) -> `wait`/`fetch` -> `SUCCEEDED` to get results.

## Input/Output

- Synthesis input: `text` (utf-8 text). Output `SpeechSynthesisResult`: `get_audio_data()` for the full audio `bytes`, `get_audio_frame()` for streaming frames, `get_timestamp()/get_timestamps()` for sentence timestamps (`sentence` contains `begin_time`/`end_time`), `get_response()` for the `SpeechSynthesisResponse` (`status_code/code/message`, `usage.characters` billed character count).
- Transcription input: `file_urls: List[str]` (audio file URLs).
- Transcription output `TranscriptionResponse`: `status_code`, `request_id`, `code`, `message`, `output.task_id`, `output.task_status` (`PENDING/RUNNING/SUCCEEDED/FAILED/CANCELED/UNKNOWN`), `usage`. On success, each item in `output.results[]` contains `transcription_url` — the download URL of the recognition-result JSON, which you must GET and parse yourself (contains per-sentence text and timestamps).

## Minimal Examples

Synthesis (Sambert):

```python
from dashscope.audio.tts import SpeechSynthesizer

result = SpeechSynthesizer.call(model="sambert-zhichu-v1", text="Hello, Bailian.")
with open("out.wav", "wb") as f:
    f.write(result.get_audio_data())
```

Transcription (async task):

```python
from http import HTTPStatus
from dashscope.audio.asr import Transcription

task = Transcription.async_call(
    model=Transcription.Models.paraformer_v1,
    file_urls=["https://example.com/audio.wav"],
)
if task.status_code == HTTPStatus.OK:
    rsp = Transcription.wait(task)   # or Transcription.wait(task.output.task_id)
    if rsp.output.task_status == "SUCCEEDED":
        print(rsp.output.results)
```

## Common Error Codes

| Error code | HTTP status | Meaning | Handling |
|---|---|---|---|
| InvalidApiKey | 401 | API Key missing/invalid | Check `dashscope.api_key` or the `DASHSCOPE_API_KEY` environment variable |
| InvalidParameter | 400 | Invalid parameters (file_urls unreachable, unsupported format/sample_rate, etc.) | Check each item against the parameter tables in this file |
| InvalidModel | 400 | Wrong model name | Use `Transcription.Models` constants or official model names |
| Throttling | 429 | Rate limited | Retry with exponential backoff, reduce concurrency (`fetch` already retries timeout/connection errors 3 times internally) |
| InternalError | 500 | Server-side error | Retry with `request_id` or file a ticket |
| Task FAILED | 200 | Transcription subtask failed | Read `output.task_status` and failure details in `output`; check audio URL reachability, format, and duration |
| task-failed (WebSocket) | - | Synthesis task failed; tts_v2 without callback raises `Exception("TaskFailed: ...")` | Read the code/message in the event message to locate the cause |

Local validation exceptions (no request sent, see `dashscope/common/error.py`): `ModelRequired` (missing model), `InputRequired` (missing apikey/format/callback), `InvalidTask` (started twice or submitted before starting).

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
