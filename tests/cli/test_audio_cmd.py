# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope audio`` -- tts and asr."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources._jobs import Job
from dashscope.resources.audio.http_tts import HttpSynthesis

_AUDIO = b"\xff\xfb\x90\x00audio-bytes"


def _synthesis(**overrides: Any) -> HttpSynthesis:
    data: Dict[str, Any] = {
        "output": {
            "audio": {
                "url": "https://example.com/hello.mp3",
                "expires_at": 1767225600,
            },
        },
        "request_id": "req-1",
    }
    data.update(overrides)
    return HttpSynthesis(**data)


def _job(**overrides: Any) -> Job:
    data: Dict[str, Any] = {
        "task_id": "task-abc123",
        "status": "SUCCEEDED",
        "output": {
            "results": [
                {
                    "file_url": "https://example.com/a.wav",
                    "transcription_url": "https://example.com/a.json",
                },
            ],
        },
    }
    data.update(overrides)
    return Job(**data)


def _event(
    text: str,
    *,
    end_time: Any = None,
    name: str = "result-generated",
    heartbeat: bool = False,
) -> Dict[str, Any]:
    sentence: Dict[str, Any] = {"text": text}
    if end_time is not None:
        sentence["end_time"] = end_time
    if heartbeat:
        sentence["heartbeat"] = True
    return {
        "header": {"event": name},
        "payload": {"output": {"sentence": sentence}},
    }


class _Tts:
    """Stand-in for ``client.audio.tts``."""

    def __init__(self, http_result: Any = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.http = _HttpTts(
            http_result if http_result is not None else _synthesis(),
        )

    def synthesize(self, text, **kwargs):
        self.calls.append(
            {"method": "synthesize", "text": text, "kwargs": kwargs},
        )
        return _AUDIO

    def stream(self, text, **kwargs):
        self.calls.append(
            {"method": "stream", "text": text, "kwargs": kwargs},
        )
        return iter((_AUDIO[:2], _AUDIO[2:]))


class _HttpTts:
    def __init__(self, result: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._result = result

    def synthesize(self, text, **kwargs):
        self.calls.append({"text": text, "kwargs": kwargs})
        return self._result


class _AsrJobs:
    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def create(self, **kwargs):
        self.calls.append({"method": "create", "args": (), "kwargs": kwargs})
        return self._results.get("create", _job(status="PENDING", output=None))

    def fetch(self, task_id, **kwargs):
        self.calls.append(
            {"method": "fetch", "args": (task_id,), "kwargs": kwargs},
        )
        return self._results.get("fetch", _job())

    def wait(self, job, **kwargs):
        self.calls.append({"method": "wait", "args": (job,), "kwargs": kwargs})
        return self._results.get("wait", job)


class _Asr:
    """Stand-in for ``client.audio.asr``."""

    def __init__(self, *, transcribe=None, events=None, **jobs: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.jobs = _AsrJobs(**jobs)
        self._transcribe = transcribe if transcribe is not None else _job()
        self._events = events if events is not None else []

    def transcribe(self, file_urls, **kwargs):
        self.calls.append(
            {"method": "transcribe", "args": (file_urls,), "kwargs": kwargs},
        )
        return self._transcribe

    def recognize_file(self, path, **kwargs):
        self.calls.append(
            {"method": "recognize_file", "args": (path,), "kwargs": kwargs},
        )
        return iter(self._events)


class _RealtimeSession:
    """Stand-in for one ``client.audio.realtime.connect()`` session."""

    def __init__(self, events: List[Dict[str, Any]]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fed = bytearray()
        self.chunks: List[int] = []
        self.closed = False
        self._events = events

    def __enter__(self) -> "_RealtimeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.closed = True

    def update_session(self, **kwargs):
        self.calls.append({"method": "update_session", "kwargs": kwargs})

    def append_audio(self, chunk: bytes):
        self.chunks.append(len(chunk))
        self.fed.extend(chunk)

    def commit(self):
        self.calls.append({"method": "commit"})

    def create_response(self, **kwargs):
        self.calls.append({"method": "create_response", "kwargs": kwargs})

    def events(self):
        return iter(self._events)


class _Realtime:
    """Stand-in for ``client.audio.realtime``."""

    def __init__(self, session: _RealtimeSession) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._session = session

    def connect(self, **kwargs):
        self.calls.append(kwargs)
        return self._session


def _rt_event(kind: str, **fields: Any) -> Dict[str, Any]:
    return {"type": kind, **fields}


def _turn(text: str = "hi there") -> List[Dict[str, Any]]:
    """One complete realtime turn, ending with the event that closes it."""
    return [
        _rt_event(
            "conversation.item.input_audio_transcription.completed",
            transcript="what time is it",
        ),
        _rt_event("response.audio.delta", delta="QUJD"),
        _rt_event("response.text.delta", delta=text),
        _rt_event(
            "response.done",
            response={"id": "resp-1", "usage": {"input_tokens": 20}},
        ),
        # Never reached: the turn is over at response.done.
        _rt_event("session.finished"),
    ]


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def audio(make_client):
    def install(**asr_kwargs: Any) -> SimpleNamespace:
        tts = _Tts(asr_kwargs.pop("http_result", None))
        asr = _Asr(**asr_kwargs)
        make_client(audio=SimpleNamespace(tts=tts, asr=asr))
        return SimpleNamespace(tts=tts, asr=asr)

    return install


@pytest.fixture
def realtime(make_client):
    def install(events: List[Dict[str, Any]] = None) -> SimpleNamespace:
        session = _RealtimeSession(events if events is not None else _turn())
        resource = _Realtime(session)
        make_client(audio=SimpleNamespace(realtime=resource))
        return SimpleNamespace(connect=resource, session=session)

    return install


class TestTtsSynthesize:
    def test_writes_the_audio_to_stdout(self, invoke, audio):
        resource = audio()
        result = invoke(["audio", "tts", "synthesize", "hello"])
        assert result.exit_code == 0
        assert result.stdout_bytes == _AUDIO
        assert resource.tts.calls[0]["text"] == "hello"

    def test_writes_the_audio_to_a_file(self, invoke, audio, tmp_path):
        audio()
        out = tmp_path / "hello.mp3"
        result = invoke(
            ["audio", "tts", "synthesize", "hello", "--out", str(out)],
        )
        assert result.exit_code == 0
        assert out.read_bytes() == _AUDIO
        # Nothing on stdout: a binary payload must not be echoed twice.
        assert result.stdout_bytes == b""
        assert "wrote 15 bytes" in result.stderr

    def test_passes_the_voice_options(self, invoke, audio):
        resource = audio()
        invoke(
            [
                "audio",
                "tts",
                "synthesize",
                "hello",
                "-m",
                "cosyvoice-v2",
                "--voice",
                "longcheng",
                "--format",
                "wav",
                "--sample-rate",
                "48000",
                "--volume",
                "80",
                "--rate",
                "1.2",
                "--pitch",
                "0.9",
                "--language-hint",
                "zh",
                "--language-hint",
                "en",
                "--enable-ssml",
                "--parameters",
                '{"a": 1}',
            ],
        )
        kwargs = resource.tts.calls[0]["kwargs"]
        assert kwargs["model"] == "cosyvoice-v2"
        assert kwargs["voice"] == "longcheng"
        assert kwargs["format"] == "wav"
        assert kwargs["sample_rate"] == 48000
        assert kwargs["volume"] == 80
        assert kwargs["rate"] == 1.2
        assert kwargs["pitch"] == 0.9
        assert kwargs["language_hints"] == ["zh", "en"]
        assert kwargs["enable_ssml"] is True
        assert kwargs["parameters"] == {"a": 1}

    def test_reads_the_text_from_a_file(self, invoke, audio, tmp_path):
        payload = tmp_path / "script.txt"
        payload.write_text("a long script\n", encoding="utf-8")
        resource = audio()
        invoke(["audio", "tts", "synthesize", f"@{payload}"])
        assert resource.tts.calls[0]["text"] == "a long script\n"

    def test_reads_the_text_from_stdin(self, invoke, audio):
        resource = audio()
        result = invoke(
            ["audio", "tts", "synthesize", "-"],
            input="from stdin\n",
        )
        assert result.exit_code == 0
        assert resource.tts.calls[0]["text"] == "from stdin\n"

    def test_refuses_blank_text(self, invoke, audio):
        resource = audio()
        result = invoke(["audio", "tts", "synthesize", "   "])
        assert result.exit_code == 2
        assert "no text given" in result.stderr
        assert resource.tts.calls == []

    def test_a_missing_output_file_is_a_usage_error(
        self,
        invoke,
        audio,
        tmp_path,
    ):
        audio()
        result = invoke(
            [
                "audio",
                "tts",
                "synthesize",
                "hello",
                "--out",
                str(tmp_path / "nope" / "hello.mp3"),
            ],
        )
        assert result.exit_code == 2
        assert "cannot write" in result.stderr


class TestTtsStream:
    def test_streams_the_audio_to_stdout(self, invoke, audio):
        resource = audio()
        result = invoke(["audio", "tts", "stream", "hello"])
        assert result.exit_code == 0
        assert result.stdout_bytes == _AUDIO
        assert resource.tts.calls[0]["method"] == "stream"
        assert resource.tts.calls[0]["text"] == "hello"

    def test_streams_the_audio_to_a_file(self, invoke, audio, tmp_path):
        resource = audio()
        out = tmp_path / "hello.mp3"
        result = invoke(
            ["audio", "tts", "stream", "hello", "--out", str(out)],
        )
        assert result.exit_code == 0
        assert out.read_bytes() == _AUDIO
        assert result.stdout_bytes == b""
        assert resource.tts.calls[0]["method"] == "stream"
        assert "wrote 15 bytes" in result.stderr

    def test_shares_the_synthesis_options(self, invoke, audio):
        resource = audio()
        result = invoke(
            [
                "audio",
                "tts",
                "stream",
                "hello",
                "-m",
                "cosyvoice-v2",
                "--voice",
                "longcheng",
                "--parameters",
                '{"a": 1}',
            ],
        )
        assert result.exit_code == 0
        kwargs = resource.tts.calls[0]["kwargs"]
        assert kwargs["model"] == "cosyvoice-v2"
        assert kwargs["voice"] == "longcheng"
        assert kwargs["parameters"] == {"a": 1}


class TestTtsHttp:
    def test_prints_the_audio_url(self, invoke, audio):
        resource = audio()
        result = invoke(
            [
                "audio",
                "tts",
                "http",
                "hello",
                "-m",
                "qwen3-tts-flash",
                "--voice",
                "Cherry",
            ],
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "https://example.com/hello.mp3"
        call = resource.tts.http.calls[0]
        assert call["text"] == "hello"
        assert call["kwargs"]["model"] == "qwen3-tts-flash"
        assert call["kwargs"]["voice"] == "Cherry"

    def test_model_and_voice_are_required(self, invoke, audio):
        audio()
        assert invoke(["audio", "tts", "http", "hello"]).exit_code == 2

    def test_json_output_is_the_whole_response(self, invoke, audio):
        audio()
        result = invoke(
            [
                "audio",
                "tts",
                "http",
                "hello",
                "-m",
                "qwen3-tts-flash",
                "--voice",
                "Cherry",
            ],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["output"]["audio"]["url"] == "https://example.com/hello.mp3"

    def test_passes_the_extra_options(self, invoke, audio):
        resource = audio()
        invoke(
            [
                "audio",
                "tts",
                "http",
                "hello",
                "-m",
                "qwen3-tts-flash",
                "--voice",
                "Cherry",
                "--format",
                "wav",
                "--sample-rate",
                "24000",
                "--language-type",
                "zh",
                "--extra-input",
                '{"text_type": "ssml"}',
                "--parameters",
                '{"a": 1}',
            ],
        )
        kwargs = resource.tts.http.calls[0]["kwargs"]
        assert kwargs["format"] == "wav"
        assert kwargs["sample_rate"] == 24000
        assert kwargs["language_type"] == "zh"
        assert kwargs["extra_input"] == {"text_type": "ssml"}
        assert kwargs["parameters"] == {"a": 1}


class TestAsrTranscribe:
    def test_waits_by_default_and_prints_the_transcripts(self, invoke, audio):
        resource = audio()
        result = invoke(
            ["audio", "asr", "transcribe", "https://example.com/a.wav"],
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "https://example.com/a.json"
        call = resource.asr.calls[0]
        assert call["method"] == "transcribe"
        assert call["args"] == (["https://example.com/a.wav"],)
        assert call["kwargs"]["timeout"] == 3600.0

    def test_sends_every_url(self, invoke, audio):
        resource = audio()
        invoke(
            [
                "audio",
                "asr",
                "transcribe",
                "https://example.com/a.wav",
                "https://example.com/b.wav",
                "-m",
                "paraformer-v2",
                "--phrase-id",
                "phrase-1",
                "--parameters",
                '{"channel_id": [0]}',
            ],
        )
        kwargs = resource.asr.calls[0]["kwargs"]
        assert kwargs["model"] == "paraformer-v2"
        assert kwargs["phrase_id"] == "phrase-1"
        assert kwargs["parameters"] == {"channel_id": [0]}

    def test_no_wait_only_submits(self, invoke, audio):
        resource = audio()
        result = invoke(
            [
                "audio",
                "asr",
                "transcribe",
                "https://example.com/a.wav",
                "--no-wait",
            ],
        )
        assert result.exit_code == 0
        assert "task-abc123" in result.stdout
        assert resource.asr.calls == []
        assert resource.asr.jobs.calls[0]["method"] == "create"
        assert resource.asr.jobs.calls[0]["kwargs"]["file_urls"] == [
            "https://example.com/a.wav",
        ]

    def test_requires_a_url(self, invoke, audio):
        audio()
        assert invoke(["audio", "asr", "transcribe"]).exit_code == 2

    def test_a_running_task_shows_its_state(self, invoke, audio):
        audio(transcribe=_job(status="RUNNING", output=None))
        result = invoke(
            ["audio", "asr", "transcribe", "https://example.com/a.wav"],
        )
        assert "RUNNING" in result.stdout

    def test_json_output_is_the_whole_job(self, invoke, audio):
        audio()
        result = invoke(
            ["audio", "asr", "transcribe", "https://example.com/a.wav"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["output"]["results"][0]["transcription_url"] == (
            "https://example.com/a.json"
        )


class TestAsrJobs:
    def test_get(self, invoke, audio):
        resource = audio()
        invoke(["audio", "asr", "get", "task-abc123"])
        assert resource.asr.jobs.calls[0]["args"] == ("task-abc123",)

    def test_wait_passes_the_timeout(self, invoke, audio):
        # ``wait`` takes an id and answers with the finished job.
        resource = audio(wait=_job())
        result = invoke(
            ["audio", "asr", "wait", "task-abc123", "--timeout", "30"],
        )
        assert result.exit_code == 0
        call = resource.asr.jobs.calls[0]
        assert call["method"] == "wait"
        assert call["args"] == ("task-abc123",)
        assert call["kwargs"]["timeout"] == 30.0
        assert result.stdout.strip() == "https://example.com/a.json"

    @pytest.mark.parametrize("timeout", ["0", "nan"])
    def test_wait_rejects_a_timeout_it_could_not_honour(
        self,
        invoke,
        audio,
        timeout,
    ):
        """``nan`` is false against every comparison, so the poll never ends."""
        resource = audio(wait=_job())
        result = invoke(
            ["audio", "asr", "wait", "task-abc123", "--timeout", timeout],
        )
        assert result.exit_code == 2
        assert "--timeout" in result.stderr
        assert resource.asr.jobs.calls == []


class TestAsrStream:
    def test_prints_only_the_final_sentences(self, invoke, audio, tmp_path):
        source = tmp_path / "a.wav"
        source.write_bytes(b"RIFF")
        resource = audio(
            events=[
                _event("hel"),
                _event("hello", end_time=900),
                _event("", heartbeat=True),
                _event("ignored", name="task-finished"),
            ],
        )
        result = invoke(
            [
                "audio",
                "asr",
                "stream",
                str(source),
                "-m",
                "paraformer-realtime-v2",
            ],
        )
        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        call = resource.asr.calls[0]
        assert call["method"] == "recognize_file"
        assert call["args"] == (str(source),)

    def test_passes_the_recognition_options(self, invoke, audio, tmp_path):
        source = tmp_path / "a.wav"
        source.write_bytes(b"RIFF")
        resource = audio()
        invoke(
            [
                "audio",
                "asr",
                "stream",
                str(source),
                "-m",
                "paraformer-realtime-v2",
                "--format",
                "wav",
                "--sample-rate",
                "8000",
                "--vocabulary-id",
                "vocab-1",
                "--disfluency-removal",
                "--no-diarization",
                "--parameters",
                '{"a": 1}',
            ],
        )
        kwargs = resource.asr.calls[0]["kwargs"]
        assert kwargs["model"] == "paraformer-realtime-v2"
        assert kwargs["format"] == "wav"
        assert kwargs["sample_rate"] == 8000
        assert kwargs["vocabulary_id"] == "vocab-1"
        assert kwargs["disfluency_removal_enabled"] is True
        assert kwargs["diarization_enabled"] is False
        assert kwargs["parameters"] == {"a": 1}

    def test_json_output_is_one_event_per_line(self, invoke, audio, tmp_path):
        source = tmp_path / "a.wav"
        source.write_bytes(b"RIFF")
        audio(events=[_event("hel"), _event("hello", end_time=900)])
        result = invoke(
            [
                "audio",
                "asr",
                "stream",
                str(source),
                "-m",
                "paraformer-realtime-v2",
            ],
            global_args=["-o", "json"],
        )
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(lines) == 2
        assert lines[1]["payload"]["output"]["sentence"]["end_time"] == 900

    def test_a_missing_file_is_a_usage_error(self, invoke, audio, tmp_path):
        resource = audio()
        result = invoke(
            [
                "audio",
                "asr",
                "stream",
                str(tmp_path / "nope.wav"),
                "-m",
                "paraformer-realtime-v2",
            ],
        )
        assert result.exit_code == 2
        assert "not found" in result.stderr
        assert resource.asr.calls == []


class TestRealtimeRun:
    def test_feeds_the_file_and_prints_the_answer(
        self,
        invoke,
        realtime,
        tmp_path,
    ):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00\x01" * 100)
        resource = realtime()
        result = invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
            ],
        )
        assert result.exit_code == 0
        assert result.stdout == "hi there\n"
        assert resource.connect.calls == [
            {"model": "qwen3-omni-flash-realtime", "timeout": None},
        ]
        session = resource.session
        assert bytes(session.fed) == source.read_bytes()
        assert [call["method"] for call in session.calls] == [
            "update_session",
            "commit",
            "create_response",
        ]
        # The socket is hung up even though the turn ended cleanly.
        assert session.closed is True

    def test_long_audio_is_sent_in_chunks(self, invoke, realtime, tmp_path):
        from dashscope.cli.commands.audio import AUDIO_CHUNK_SIZE

        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00" * (AUDIO_CHUNK_SIZE * 2 + 7))
        resource = realtime()
        invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
            ],
        )
        assert resource.session.chunks == [
            AUDIO_CHUNK_SIZE,
            AUDIO_CHUNK_SIZE,
            7,
        ]

    def test_the_input_transcript_goes_to_stderr(
        self,
        invoke,
        realtime,
        tmp_path,
    ):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        realtime()
        result = invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
            ],
        )
        assert "input: what time is it" in result.stderr
        assert "what time is it" not in result.stdout

    def test_the_reply_audio_lands_in_out(self, invoke, realtime, tmp_path):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        reply = tmp_path / "reply.wav"
        realtime()
        result = invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
                "--out",
                str(reply),
            ],
        )
        assert result.exit_code == 0
        assert reply.read_bytes() == b"ABC"
        # Text stays on stdout, so the two payloads never share a redirect.
        assert result.stdout == "hi there\n"

    def test_without_out_the_reply_audio_is_dropped(
        self,
        invoke,
        realtime,
        tmp_path,
    ):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        realtime()
        result = invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
            ],
        )
        assert result.stdout_bytes == b"hi there\n"

    def test_instructions_drive_the_response(self, invoke, realtime, tmp_path):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        resource = realtime()
        invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
                "--instructions",
                "answer in one word",
            ],
        )
        response = [
            call
            for call in resource.session.calls
            if call["method"] == "create_response"
        ][0]
        assert response["kwargs"] == {"instructions": "answer in one word"}

    def test_session_options_are_passed(self, invoke, realtime, tmp_path):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        resource = realtime()
        invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
                "--input-format",
                "wav",
                "--output-format",
                "wav",
                "--voice",
                "Cherry",
                "--modality",
                "text",
                "--modality",
                "audio",
                "--transcribe-input",
                "--timeout",
                "30",
            ],
        )
        assert resource.connect.calls[0]["timeout"] == 30.0
        settings = [
            call
            for call in resource.session.calls
            if call["method"] == "update_session"
        ][0]["kwargs"]
        assert settings == {
            "input_audio_format": "wav",
            "voice": "Cherry",
            "output_audio_format": "wav",
            "output_modalities": ["text", "audio"],
            "enable_input_audio_transcription": True,
        }

    def test_unset_options_are_left_out_entirely(
        self,
        invoke,
        realtime,
        tmp_path,
    ):
        """``update_session`` is sentinel-defaulted: None would unset them."""
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        resource = realtime()
        invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
            ],
        )
        settings = resource.session.calls[0]["kwargs"]
        assert settings == {"input_audio_format": "pcm"}

    def test_json_output_is_every_event_of_the_turn(
        self,
        invoke,
        realtime,
        tmp_path,
    ):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        realtime()
        result = invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
            ],
            global_args=["-o", "json"],
        )
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        assert [line["type"] for line in lines] == [
            "conversation.item.input_audio_transcription.completed",
            "response.audio.delta",
            "response.text.delta",
            "response.done",
        ]

    def test_the_usage_note_goes_to_stderr(self, invoke, realtime, tmp_path):
        source = tmp_path / "in.pcm"
        source.write_bytes(b"\x00")
        realtime()
        result = invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(source),
            ],
        )
        assert "input_tokens" in result.stderr
        assert "usage" not in result.stdout

    def test_a_missing_audio_file_is_a_usage_error(
        self,
        invoke,
        realtime,
        tmp_path,
    ):
        resource = realtime()
        result = invoke(
            [
                "audio",
                "realtime",
                "run",
                "-m",
                "qwen3-omni-flash-realtime",
                "--audio",
                str(tmp_path / "nope.pcm"),
            ],
        )
        assert result.exit_code == 2
        assert "not found" in result.stderr
        assert resource.connect.calls == []

    def test_no_audio_at_all_is_a_usage_error(self, invoke, realtime):
        """The runner's stdin is an empty pipe, so there is nothing to feed."""
        resource = realtime()
        result = invoke(["audio", "realtime", "run", "-m", "qwen3-omni"])
        assert result.exit_code == 2
        assert "no audio given" in result.stderr
        assert resource.connect.calls == []

    def test_the_model_is_required(self, invoke, realtime):
        realtime()
        assert invoke(["audio", "realtime", "run"]).exit_code == 2
