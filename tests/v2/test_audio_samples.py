# -*- coding: utf-8 -*-
"""Offline wire and output checks for the protocol-specific audio examples."""

import base64
import importlib
import json
import wave
from contextlib import contextmanager, nullcontext
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from dashscope import Client
from dashscope.v2.errors import DashScopeError


def _sample(name):
    # ``01_atomic_capabilities`` is not a valid identifier in source-level
    # imports, so the samples are loaded through importlib instead.
    return importlib.import_module(
        f"examples.01_atomic_capabilities.audio.{name}",
    )


asr_file = _sample("asr_file")
asr_http_stream = _sample("asr_http_stream")
asr_microphone = _sample("asr_microphone")
connection_reuse = _sample("connection_reuse")
realtime_omni = _sample("realtime_omni")
realtime_qwen_audio = _sample("realtime_qwen_audio")
realtime_translation = _sample("realtime_translation")
tts_http = _sample("tts_http")
tts_http_stream = _sample("tts_http_stream")
tts_websocket = _sample("tts_websocket")
voice_clone = _sample("voice_clone")
asr_file_direct_call = _sample("direct_call.asr_file")
asr_http_stream_direct_call = _sample("direct_call.asr_http_stream")
connection_reuse_direct_call = _sample("direct_call.connection_reuse")
realtime_omni_direct_call = _sample("direct_call.realtime_omni")
realtime_qwen_audio_direct_call = _sample("direct_call.realtime_qwen_audio")
realtime_translation_direct_call = _sample("direct_call.realtime_translation")
tts_http_direct_call = _sample("direct_call.tts_http")
tts_http_stream_direct_call = _sample("direct_call.tts_http_stream")
tts_websocket_direct_call = _sample("direct_call.tts_websocket")
voice_clone_direct_call = _sample("direct_call.voice_clone")


BASE_URL = "https://api.test.local/api/v1"
HTTP_CLIENT = httpx.Client
PCM = b"\x01\x00\xff\x7f" * 64


def attach_client(monkeypatch, module, handler):
    responses = []

    def respond(request):
        response = handler(request)
        responses.append(response)
        return response

    transport = httpx.MockTransport(respond)
    http_client = HTTP_CLIENT(transport=transport, base_url=BASE_URL)
    client = Client(
        api_key="sk-test",
        workspace="ws-test",
        base_url=BASE_URL,
        http_client=http_client,
        max_retries=0,
    )
    monkeypatch.setattr(module, "Client", lambda **kwargs: client)
    return responses


def assert_wav(path, pcm=PCM):
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 24000
        assert audio.readframes(audio.getnframes()) == pcm


def sse(parts):
    return "".join("data: " + json.dumps(part) + "\n\n" for part in parts)


def audio_part(pcm):
    return {"output": {"audio": {"data": base64.b64encode(pcm).decode()}}}


def assert_tts_request(request, streaming):
    assert request.method == "POST"
    assert request.url.path == "/api/v1/services/audio/tts/SpeechSynthesizer"
    assert request.headers.get("x-dashscope-sse") == (
        "enable" if streaming else None
    )
    body = json.loads(request.content)
    assert body == {
        "model": "qwen-audio-3.1-tts-flash",
        "input": {
            "text": "你好",
            "voice": "longanhuan_v3.1",
            "format": "pcm",
            "sample_rate": 24000,
        },
    }


@pytest.mark.parametrize("module", [tts_http, tts_http_direct_call])
def test_http_tts_download_has_no_sdk_credentials(
    monkeypatch,
    tmp_path,
    module,
):
    def handler(request):
        assert_tts_request(request, False)
        return httpx.Response(
            200,
            json={
                "request_id": "tts-http-test",
                "output": {
                    "audio": {"url": "http://cdn.test/a.pcm?signature=test"},
                },
            },
        )

    responses = attach_client(monkeypatch, module, handler)
    downloads = []

    def download(request):
        assert str(request.url) == "https://cdn.test/a.pcm?signature=test"
        assert "authorization" not in request.headers
        assert "x-dashscope-workspace" not in request.headers
        downloads.append(request)
        return httpx.Response(200, content=PCM)

    downloader = HTTP_CLIENT(transport=httpx.MockTransport(download))
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: downloader)
    path = tmp_path / "http.wav"
    module.main(["--text", "你好", "--output", str(path), "--no-play"])
    assert_wav(path)
    assert len(downloads) == 1
    assert downloader.is_closed
    assert all(response.is_closed for response in responses)


@pytest.mark.parametrize(
    "module",
    [tts_http_stream, tts_http_stream_direct_call],
)
def test_http_tts_sse_appends_pcm_and_closes(monkeypatch, tmp_path, module):
    def handler(request):
        assert_tts_request(request, True)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                [
                    audio_part(PCM[:1]),
                    audio_part(PCM[1:]),
                    {"output": {"finish_reason": "stop"}},
                ],
            ),
        )

    responses = attach_client(monkeypatch, module, handler)
    path = tmp_path / "sse.wav"
    module.main(["--text", "你好", "--output", str(path), "--no-play"])
    assert_wav(path)
    assert len(responses) == 1
    assert responses[0].is_closed


@pytest.mark.parametrize(
    "module",
    [tts_http_stream, tts_http_stream_direct_call],
)
@pytest.mark.parametrize(
    "parts",
    [
        [audio_part(PCM)],
        [{"output": {"finish_reason": "stop"}}],
        [audio_part(b"\x01"), {"output": {"finish_reason": "stop"}}],
        [
            audio_part(b"RIFF\x00\x00\x00\x00WAVEdata"),
            {"output": {"finish_reason": "stop"}},
        ],
    ],
)
def test_http_tts_sse_rejects_incomplete_audio(
    monkeypatch,
    tmp_path,
    parts,
    module,
):
    responses = attach_client(
        monkeypatch,
        module,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(parts),
        ),
    )
    with pytest.raises(module.SampleError):
        module.main(
            [
                "--text",
                "你好",
                "--output",
                str(tmp_path / "bad.wav"),
                "--no-play",
            ],
        )
    assert responses[0].is_closed


@pytest.mark.parametrize(
    "module",
    [
        tts_http,
        tts_http_direct_call,
        tts_http_stream,
        tts_http_stream_direct_call,
    ],
)
def test_http_tts_output_exists_before_request(monkeypatch, tmp_path, module):
    def forbidden(**kwargs):
        pytest.fail("Client created before output preflight")

    monkeypatch.setattr(module, "Client", forbidden)
    path = tmp_path / "existing.wav"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        module.main(["--text", "你好", "--output", str(path), "--no-play"])
    assert path.read_bytes() == b"keep"


@pytest.mark.parametrize(
    "module",
    [
        tts_http,
        tts_http_direct_call,
        tts_http_stream,
        tts_http_stream_direct_call,
    ],
)
def test_http_tts_empty_input_before_request(monkeypatch, tmp_path, module):
    def forbidden(**kwargs):
        pytest.fail("Client created with empty text")

    monkeypatch.setattr(module, "Client", forbidden)
    path = tmp_path / "unused.wav"
    with pytest.raises(SystemExit) as caught:
        module.main(["--text", " \n", "--output", str(path), "--no-play"])
    assert caught.value.code == 2
    assert not path.exists()


@pytest.mark.parametrize("module", [asr_file, asr_file_direct_call])
@pytest.mark.parametrize("resume", [False, True])
def test_asr_file_http_submit_or_resume(monkeypatch, tmp_path, resume, module):
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            assert (
                request.url.path == "/api/v1/services/audio/asr/transcription"
            )
            assert request.headers["x-dashscope-async"] == "enable"
            assert json.loads(request.content) == {
                "model": "qwen-audio-3.1-asr-flash-filetrans",
                "input": {"file_urls": ["https://cdn.test/input.wav"]},
                "parameters": {},
            }
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": "file-task",
                        "task_status": "PENDING",
                    },
                },
            )
        assert request.method == "GET"
        assert request.url.path == "/api/v1/tasks/file-task"
        return httpx.Response(
            200,
            json={
                "request_id": "file-request",
                "output": {
                    "task_id": "file-task",
                    "task_status": "SUCCEEDED",
                    "results": [
                        {
                            "subtask_status": "SUCCEEDED",
                            "transcription_url": "https://cdn.test/transcript.json",
                        },
                    ],
                },
            },
        )

    responses = attach_client(monkeypatch, module, handler)
    client = module.Client()
    close = Mock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)

    def download(request):
        assert "authorization" not in request.headers
        assert "x-dashscope-workspace" not in request.headers
        return httpx.Response(200, json={"transcripts": [{"text": "识别结果。"}]})

    downloader = HTTP_CLIENT(transport=httpx.MockTransport(download))
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: downloader)
    folder = tmp_path / "result"
    source = (
        ["--task-id", "file-task"]
        if resume
        else [
            "--url",
            "https://cdn.test/input.wav",
        ]
    )
    assert module.main([*source, "--output-dir", str(folder)]) == 0
    metadata = json.loads((folder / "task.json").read_text())
    assert metadata["task_id"] == "file-task"
    assert metadata["status"] == "SUCCEEDED"
    assert (folder / "001.txt").read_text() == "识别结果。\n"
    assert sum(request.method == "POST" for request in calls) == (
        0 if resume else 1
    )
    assert all(response.is_closed for response in responses)
    assert downloader.is_closed
    close.assert_called_once_with()


@pytest.mark.parametrize("module", [asr_file, asr_file_direct_call])
@pytest.mark.parametrize("error_type", [DashScopeError, KeyboardInterrupt])
def test_asr_file_wait_failure_preserves_id_and_closes(
    monkeypatch,
    tmp_path,
    error_type,
    module,
):
    def handler(request):
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "output": {"task_id": "saved-task", "task_status": "PENDING"},
            },
        )

    responses = attach_client(monkeypatch, module, handler)
    client = module.Client()
    close = Mock(wraps=client.close)
    monkeypatch.setattr(client, "close", close)
    outputs = []
    checkpoint = module.checkpoint

    def save(output, metadata):
        outputs.append(output)
        checkpoint(output, metadata)

    monkeypatch.setattr(module, "checkpoint", save)
    folder = tmp_path / "interrupted"

    def wait(job, *, timeout):
        assert job.task_id == "saved-task"
        assert timeout == 600.0
        assert not outputs[0].closed
        metadata = json.loads((folder / "task.json").read_text())
        assert metadata["task_id"] == "saved-task"
        assert metadata["status"] == "PENDING"
        raise error_type("wait interrupted")

    monkeypatch.setattr(client.audio.asr.jobs, "wait", wait)
    with pytest.raises(error_type):
        module.main(
            [
                "--url",
                "https://cdn.test/input.wav",
                "--output-dir",
                str(folder),
            ],
        )
    assert len(responses) == 1
    assert responses[0].is_closed
    assert len(outputs) == 1
    assert outputs[0].closed
    assert (
        json.loads((folder / "task.json").read_text())["task_id"]
        == "saved-task"
    )
    close.assert_called_once_with()


@pytest.mark.parametrize("module", [asr_file, asr_file_direct_call])
def test_asr_file_output_exists_before_request(monkeypatch, tmp_path, module):
    def forbidden(**kwargs):
        pytest.fail("Client created before output preflight")

    monkeypatch.setattr(module, "Client", forbidden)
    folder = tmp_path / "existing"
    folder.mkdir()
    checkpoint = folder / "task.json"
    checkpoint.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.main(
            [
                "--task-id",
                "saved-task",
                "--output-dir",
                str(folder),
            ],
        )
    assert checkpoint.read_text() == "keep"


@pytest.mark.parametrize("module", [asr_file, asr_file_direct_call])
def test_asr_file_single_url_before_request(monkeypatch, tmp_path, module):
    def forbidden(**kwargs):
        pytest.fail("Client created before single-URL validation")

    monkeypatch.setattr(module, "Client", forbidden)
    folder = tmp_path / "unused"
    with pytest.raises(SystemExit) as caught:
        module.main(
            [
                "--url",
                "https://cdn.test/one.wav",
                "--url",
                "https://cdn.test/two.wav",
                "--output-dir",
                str(folder),
            ],
        )
    assert caught.value.code == 2
    assert not folder.exists()


@pytest.mark.parametrize(
    "module",
    [asr_http_stream, asr_http_stream_direct_call],
)
@pytest.mark.parametrize("local_file", [False, True])
def test_rest_asr_wire_and_cumulative_revisions(
    monkeypatch,
    tmp_path,
    local_file,
    module,
):
    fixture = tmp_path / "input.wav"
    with wave.open(str(fixture), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(PCM)

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == (
            "/api/v1/services/aigc/multimodal-generation/generation"
        )
        assert request.headers["x-dashscope-sse"] == "enable"
        body = json.loads(request.content)
        assert body["model"] == "qwen-audio-3.1-asr-flash"
        assert body["parameters"] == {"format": "wav", "sample_rate": "16000"}
        part = body["input"]["messages"][0]["content"][0]
        assert part["type"] == "input_audio"
        data = part["input_audio"]["data"]
        if local_file:
            assert data.startswith("data:audio/wav;base64,")
            assert (
                base64.b64decode(data.split(",", 1)[1]) == fixture.read_bytes()
            )
        else:
            assert data == "https://cdn.test/input.wav"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(
                [
                    {"output": {"text": "可能被修订的长文本"}},
                    {
                        "output": {
                            "text": "你好",
                            "sentence": {
                                "sentence_end": True,
                            },
                        },
                    },
                    {
                        "output": {
                            "text": "你好。再见。",
                            "sentence": {
                                "sentence_end": True,
                            },
                        },
                    },
                ],
            ),
        )

    responses = attach_client(monkeypatch, module, handler)
    transcript = tmp_path / "rest.txt"
    events = tmp_path / "rest.jsonl"
    source = (
        ["--file", str(fixture)]
        if local_file
        else [
            "--url",
            "https://cdn.test/input.wav",
        ]
    )
    assert (
        module.main(
            [
                *source,
                "--transcript",
                str(transcript),
                "--events",
                str(events),
            ],
        )
        == 0
    )
    assert transcript.read_text() == "你好。再见。"
    assert len(events.read_text().splitlines()) == 3
    assert responses[0].is_closed


@pytest.mark.parametrize(
    "module",
    [asr_http_stream, asr_http_stream_direct_call],
)
@pytest.mark.parametrize(
    "parts,error",
    [
        ([{"code": "InvalidParameter", "message": "bad"}], DashScopeError),
        ([{"output": {}}], RuntimeError),
        ([{"output": {"text": 42}}], ValueError),
        ([], RuntimeError),
    ],
)
def test_rest_asr_errors_close_response(
    monkeypatch,
    tmp_path,
    parts,
    error,
    module,
):
    responses = attach_client(
        monkeypatch,
        module,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(parts),
        ),
    )
    with pytest.raises(error):
        module.main(
            [
                "--url",
                "https://cdn.test/input.wav",
                "--transcript",
                str(tmp_path / "rest.txt"),
            ],
        )
    assert responses[0].is_closed


@pytest.mark.parametrize("save_events", [False, True])
@pytest.mark.parametrize("source", ["text", "file", "prompt"])
def test_websocket_tts_always_uses_duplex(
    monkeypatch,
    tmp_path,
    source,
    save_events,
):
    parts = ["你好", " ", "世界"]
    stopped = Event()
    session = Mock(
        session=SimpleNamespace(request_id="duplex-request"),
        first_package_delay=1.0,
        usage={"characters": 5},
    )
    final = {"header": {"event": "task-finished"}}

    def frames():
        yield PCM[:128]
        assert stopped.wait(2)
        yield PCM[128:]
        yield final

    session.events.side_effect = frames
    session.stop.side_effect = stopped.set
    session.close.side_effect = stopped.set
    connect = Mock(return_value=nullcontext(session))
    llm_stream = SimpleNamespace(text_stream=iter(parts), close=Mock())
    create = Mock(return_value=llm_stream)
    client = SimpleNamespace(
        audio=SimpleNamespace(tts=SimpleNamespace(connect=connect)),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        tts_websocket,
        "Client",
        lambda **kwargs: nullcontext(client),
    )
    if source == "text":
        source_args = [
            "--text",
            parts[0],
            "--text",
            parts[1],
            "--text",
            parts[2],
        ]
    elif source == "file":
        parts = ["你好\n", " \n", "世界\n"]
        text_file = tmp_path / "source.txt"
        text_file.write_text("".join(parts), encoding="utf-8")
        source_args = ["--text-file", str(text_file)]
    else:
        source_args = ["--prompt", "请打个招呼"]
    path = tmp_path / "duplex.wav"
    events = tmp_path / "events.jsonl"
    event_args = ["--events", str(events)] if save_events else []
    tts_websocket.main(
        [
            *source_args,
            "--output",
            str(path),
            *event_args,
            "--no-play",
        ],
    )
    connect.assert_called_once_with(
        model="qwen-audio-3.1-tts-flash",
        voice="longanhuan_v3.1",
        format="pcm",
        sample_rate=24000,
        timeout=30.0,
        enable_ssml=False,
    )
    assert [call.args[0] for call in session.send_text.call_args_list] == parts
    session.stop.assert_called_once()
    session.close.assert_called_once()
    assert_wav(path)
    if save_events:
        assert json.loads(events.read_text()) == final
    else:
        assert not events.exists()
    if source == "prompt":
        create.assert_called_once()
        llm_stream.close.assert_called_once()
    else:
        create.assert_not_called()


def test_websocket_tts_send_error_closes_before_join():
    closed = Event()
    reader_saw_close = []
    session = Mock()

    def frames():
        reader_saw_close.append(closed.wait(2))
        yield from ()

    session.events.side_effect = frames
    session.send_text.side_effect = RuntimeError("send failed")
    session.close.side_effect = closed.set
    client = SimpleNamespace(
        audio=SimpleNamespace(
            tts=SimpleNamespace(
                connect=Mock(return_value=nullcontext(session)),
            ),
        ),
    )
    args = tts_websocket.make_parser().parse_args(
        [
            "--text",
            "你好",
            "--output",
            "unused.wav",
            "--no-play",
        ],
    )
    with pytest.raises(RuntimeError, match="send failed"):
        tts_websocket.synthesize(client, args, ["你好"], Mock(), None)
    assert reader_saw_close == [True]
    session.close.assert_called_once()


@pytest.mark.parametrize("module", [tts_websocket, tts_websocket_direct_call])
def test_websocket_tts_has_no_mode_option(module):
    parser = module.make_parser()
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(
            [
                "--mode",
                "duplex",
                "--text",
                "你好",
                "--output",
                "unused.wav",
            ],
        )
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "module",
    [asr_http_stream, asr_http_stream_direct_call],
)
def test_rest_asr_existing_output_prevents_request(
    monkeypatch,
    tmp_path,
    module,
):
    def forbidden(**kwargs):
        pytest.fail("Client created before output preflight")

    monkeypatch.setattr(module, "Client", forbidden)
    transcript = tmp_path / "existing.txt"
    transcript.write_text("keep")
    with pytest.raises(FileExistsError):
        module.main(
            [
                "--url",
                "https://cdn.test/input.wav",
                "--transcript",
                str(transcript),
            ],
        )
    assert transcript.read_text() == "keep"


@pytest.fixture
def opened_files(monkeypatch):
    files = []
    open_file = Path.open

    def tracked_open(path, *args, **kwargs):
        file = open_file(path, *args, **kwargs)
        files.append(file)
        return file

    monkeypatch.setattr(Path, "open", tracked_open)
    yield files
    assert files
    assert all(file.closed for file in files)


@pytest.mark.parametrize(
    "module,arguments",
    [
        (module, arguments)
        for modules, arguments in [
            (
                (asr_http_stream, asr_http_stream_direct_call),
                [
                    "--url",
                    "https://cdn.test/input.wav",
                    "--transcript",
                    "rest.txt",
                    "--events",
                    "events.jsonl",
                ],
            ),
            (
                (tts_websocket, tts_websocket_direct_call),
                [
                    "--text-file",
                    "input.txt",
                    "--output",
                    "tts.wav",
                    "--events",
                    "events.jsonl",
                    "--no-play",
                ],
            ),
            (
                (realtime_omni, realtime_omni_direct_call),
                [
                    "--wav",
                    "input.wav",
                    "--output",
                    "reply.wav",
                    "--no-play",
                ],
            ),
            (
                (realtime_qwen_audio, realtime_qwen_audio_direct_call),
                [
                    "--wav",
                    "input.wav",
                    "--output",
                    "reply.wav",
                    "--no-play",
                ],
            ),
            (
                (realtime_translation, realtime_translation_direct_call),
                [
                    "--wav",
                    "input.wav",
                    "--output",
                    "reply.wav",
                    "--no-play",
                ],
            ),
            (
                (connection_reuse, connection_reuse_direct_call),
                [
                    "asr",
                    "input.wav",
                    "--output-dir",
                    "reuse",
                ],
            ),
            (
                (connection_reuse, connection_reuse_direct_call),
                [
                    "tts",
                    "--output-dir",
                    "reuse",
                ],
            ),
            (
                (voice_clone, voice_clone_direct_call),
                ["list", "--output", "voices.json"],
            ),
        ]
        for module in modules
    ],
)
def test_audio_client_failure_closes_files(
    monkeypatch,
    tmp_path,
    opened_files,
    module,
    arguments,
):
    with wave.open(str(tmp_path / "input.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(PCM * 16)
    (tmp_path / "input.txt").write_text("你好\n", encoding="utf-8")
    source_open = wave.open
    readers = []

    def track_source(path, mode=None):
        audio = source_open(path, mode)
        if mode == "rb":
            readers.append(audio)
        return audio

    monkeypatch.setattr(wave, "open", track_source)
    client = Mock(side_effect=RuntimeError("client failed"))
    monkeypatch.setattr(module, "Client", client)
    filenames = {
        "input.wav",
        "input.txt",
        "rest.txt",
        "events.jsonl",
        "tts.wav",
        "reply.wav",
        "reuse",
        "voices.json",
    }
    argv = [
        str(tmp_path / arg) if arg in filenames else arg for arg in arguments
    ]
    with pytest.raises(RuntimeError, match="client failed"):
        module.main(argv)
    client.assert_called_once()
    assert all(reader._file is None for reader in readers)
    assert all(file.closed for file in opened_files)


@pytest.mark.parametrize(
    "module",
    [
        asr_http_stream,
        asr_http_stream_direct_call,
        tts_websocket,
        tts_websocket_direct_call,
    ],
)
def test_audio_existing_events_prevents_request(
    monkeypatch,
    tmp_path,
    opened_files,
    module,
):
    events = tmp_path / "events.jsonl"
    events.write_text("keep", encoding="utf-8")
    client = Mock(side_effect=AssertionError("unexpected client"))
    monkeypatch.setattr(module, "Client", client)
    if module in (asr_http_stream, asr_http_stream_direct_call):
        arguments = [
            "--url",
            "https://cdn.test/input.wav",
            "--transcript",
            str(tmp_path / "rest.txt"),
        ]
    else:
        arguments = [
            "--text",
            "你好",
            "--output",
            str(tmp_path / "tts.wav"),
            "--no-play",
        ]
    with pytest.raises(FileExistsError):
        module.main([*arguments, "--events", str(events)])
    client.assert_not_called()
    assert events.read_text() == "keep"


def test_websocket_llm_failure_closes_stream(
    monkeypatch,
    tmp_path,
    opened_files,
):
    stream = SimpleNamespace(text_stream=iter(["你好"]), close=Mock())
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(return_value=stream),
            ),
        ),
    )
    closed = []

    @contextmanager
    def client_context(**kwargs):
        try:
            yield client
        finally:
            closed.append(True)

    monkeypatch.setattr(tts_websocket, "Client", client_context)
    monkeypatch.setattr(
        tts_websocket,
        "synthesize",
        Mock(
            side_effect=RuntimeError("synthesis failed"),
        ),
    )
    with pytest.raises(RuntimeError, match="synthesis failed"):
        tts_websocket.main(
            [
                "--prompt",
                "你好",
                "--output",
                str(tmp_path / "tts.wav"),
                "--events",
                str(tmp_path / "events.jsonl"),
                "--no-play",
            ],
        )
    stream.close.assert_called_once()
    assert closed == [True]


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_microphone_optional_recording_cleanup(
    monkeypatch,
    tmp_path,
    opened_files,
    record,
    fail,
):
    pcm = b"\x01\x00" * 1600
    stream = Mock()
    stream.read.return_value = (pcm, False)
    closed = []

    @contextmanager
    def microphone(*args, **kwargs):
        try:
            yield stream
        finally:
            closed.append(True)

    monkeypatch.setattr(asr_microphone, "load_sounddevice", Mock())
    monkeypatch.setattr(asr_microphone, "microphone", microphone)
    session = Mock()
    receiver = Mock()
    receiver.done.return_value = False

    def recognize(args, sample_rate, transcript, send_audio):
        assert sample_rate == 16000
        if fail:
            raise RuntimeError("recognition failed")
        send_audio(session, receiver)
        transcript.write("你好\n")

    monkeypatch.setattr(asr_microphone, "run_recognition", recognize)
    recording = tmp_path / "recording.wav"
    transcript = tmp_path / "microphone.txt"
    record_args = (
        ["--recording", str(recording)] if record else ["--no-recording"]
    )
    result = asr_microphone.main(
        [
            "--seconds",
            "0.1",
            "--transcript",
            str(transcript),
            *record_args,
        ],
    )
    assert result == (1 if fail else 0)
    assert closed == [True]
    assert recording.exists() == record
    if not fail:
        session.send_audio_frame.assert_called_once_with(pcm)
        assert transcript.read_text() == "你好\n"
    if record:
        with wave.open(str(recording), "rb") as audio:
            assert audio.readframes(audio.getnframes()) == (
                b"" if fail else pcm
            )


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("status", ["OK", "DEPLOYING"])
def test_voice_get_optional_output_cleanup(
    monkeypatch,
    tmp_path,
    save,
    status,
):
    voice = SimpleNamespace(
        status=status,
        model_dump=Mock(
            return_value={"voice_id": "voice-test", "status": status},
        ),
    )
    get = Mock(return_value=voice)
    client = SimpleNamespace(
        config=SimpleNamespace(region="cn-beijing", workspace="ws-test"),
        audio=SimpleNamespace(voices=SimpleNamespace(get=get)),
    )
    closed = []

    @contextmanager
    def client_context(**kwargs):
        try:
            yield client
        finally:
            closed.append(True)

    monkeypatch.setattr(voice_clone, "Client", client_context)
    path = tmp_path / "voice.json"
    opened = []
    checkpoint = voice_clone.checkpoint

    def save_result(output, data, response):
        opened.append(output)
        checkpoint(output, data, response)

    monkeypatch.setattr(voice_clone, "checkpoint", save_result)
    output_args = ["--output", str(path)] if save else []
    result = voice_clone.main(
        ["get", "--voice-id", "voice-test", *output_args],
    )
    assert result == (0 if status == "OK" else 1)
    get.assert_called_once_with("voice-test")
    assert closed == [True]
    assert path.exists() == save
    if save:
        assert opened[0].closed
        assert json.loads(path.read_text())["result"]["status"] == status
    else:
        assert opened == [None]


@pytest.mark.parametrize(
    "module,model",
    [
        (realtime_omni, "qwen3.8-omni-flash-realtime"),
        (realtime_qwen_audio, "qwen-audio-3.0-realtime-plus"),
        (realtime_translation, "qwen3.8-livetranslate-flash-realtime"),
    ],
)
@pytest.mark.parametrize("use_wav", [False, True])
def test_realtime_source_and_context_cleanup(
    monkeypatch,
    tmp_path,
    opened_files,
    use_wav,
    module,
    model,
):
    session = Mock()
    session.recv_event.side_effect = [
        {"type": "session.updated"},
        {
            "type": "response.audio.delta",
            "delta": base64.b64encode(PCM).decode(),
        },
        {"type": "response.done", "response": {"status": "completed"}},
        {"type": "session.finished"},
    ]
    closed = []

    @contextmanager
    def connect(**kwargs):
        assert kwargs["model"] == model
        try:
            yield session
        finally:
            closed.append("session")

    @contextmanager
    def client_context(**kwargs):
        try:
            yield SimpleNamespace(
                audio=SimpleNamespace(
                    realtime=SimpleNamespace(
                        connect=connect,
                    ),
                ),
            )
        finally:
            closed.append("client")

    monkeypatch.setattr(module, "Client", client_context)
    sounddevice = Mock()
    monkeypatch.setattr(module, "load_sounddevice", lambda: sounddevice)
    send = Mock()
    monkeypatch.setattr(module, "send_input", send)
    path = tmp_path / "reply.wav"
    if use_wav:
        source = tmp_path / "input.wav"
        with wave.open(str(source), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(PCM * 16)
        source_args = ["--wav", str(source)]
    else:
        source_args = ["--microphone"]
    module.main([*source_args, "--output", str(path), "--no-play"])
    assert_wav(path)
    if module is realtime_translation:
        session.commit.assert_not_called()
        session.create_response.assert_not_called()
        session.end_session.assert_called_once()
        session.update_session.assert_called_once_with(
            output_modalities=["text", "audio"],
            translation_params={"language": "en"},
        )
    else:
        session.commit.assert_called_once()
        session.create_response.assert_called_once()
        session.end_session.assert_not_called()
    send.assert_called_once()
    input_audio = send.call_args.args[3]
    if use_wav:
        assert input_audio._file is None
        sounddevice.check_input_settings.assert_not_called()
    else:
        assert input_audio is None
        sounddevice.check_input_settings.assert_called_once()
    assert closed == ["session", "client"]
    session.close.assert_called_once()


@pytest.mark.parametrize("kind", ["asr", "tts"])
@pytest.mark.parametrize("fail", [False, True])
def test_reuse_source_and_context_cleanup(
    monkeypatch,
    tmp_path,
    opened_files,
    kind,
    fail,
):
    connection = SimpleNamespace(closed=False)
    connection.close = Mock(
        side_effect=lambda: setattr(connection, "closed", True),
    )
    connect = Mock(return_value=connection)
    closed = []

    @contextmanager
    def client_context(**kwargs):
        try:
            yield SimpleNamespace(
                ws_connection=connect,
                config=SimpleNamespace(
                    region="cn-beijing",
                    workspace="ws-test",
                ),
            )
        finally:
            closed.append("client")

    monkeypatch.setattr(connection_reuse, "Client", client_context)
    sources = []

    def run_one(client, shared, args, *extra):
        assert shared is connection
        assert not shared.closed
        index = extra[-1]
        if kind == "asr":
            sources.append(extra[0])
            assert extra[0].getframerate() == 16000
        if fail and index == 2:
            raise RuntimeError("task failed")
        return {"task_id": f"task-{index}"}

    monkeypatch.setattr(connection_reuse, "recognize_one", run_one)
    monkeypatch.setattr(connection_reuse, "synthesize_one", run_one)
    source_args = []
    if kind == "asr":
        source = tmp_path / "input.wav"
        with wave.open(str(source), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(PCM)
        source_args = [str(source)]
    folder = tmp_path / "reuse"
    argv = [kind, *source_args, "--output-dir", str(folder)]
    if fail:
        with pytest.raises(RuntimeError, match="task failed"):
            connection_reuse.main(argv)
    else:
        assert connection_reuse.main(argv) == 0
    result = json.loads((folder / "summary.json").read_text())
    assert result["status"] == ("failed" if fail else "completed")
    assert len(result["runs"]) == (1 if fail else 5)
    assert result["connection_closed"] is True
    assert result["connection_count"] == 1
    connect.assert_called_once()
    assert connection.closed
    assert closed == ["client"]
    if kind == "asr":
        assert all(source is sources[0] for source in sources)
        assert sources[0]._file is None


@pytest.mark.parametrize(
    "module,model,settings",
    [
        (module, model, settings)
        for modules, model, settings in [
            (
                (realtime_omni, realtime_omni_direct_call),
                "qwen3.8-omni-flash-realtime",
                {
                    "modalities": ["text", "audio"],
                    "audio": {
                        "input": {
                            "format": {"type": "pcm", "sample_rate": 16000},
                        },
                        "output": {
                            "format": {"type": "pcm", "sample_rate": 24000},
                            "voice": "longanlingxin",
                        },
                    },
                    "input_audio_transcription": None,
                    "turn_detection": None,
                    "instructions": "请用简洁自然的中文回答。",
                },
            ),
            (
                (realtime_qwen_audio, realtime_qwen_audio_direct_call),
                "qwen-audio-3.0-realtime-plus",
                {
                    "modalities": ["text", "audio"],
                    "voice": "longanqian",
                    "input_audio_format": "pcm",
                    "output_audio_format": "pcm",
                    "turn_detection": None,
                    "instructions": "请用简洁自然的中文回答。",
                },
            ),
            (
                (realtime_translation, realtime_translation_direct_call),
                "qwen3.8-livetranslate-flash-realtime",
                {
                    "output_modalities": ["text", "audio"],
                    "translation": {"language": "en"},
                },
            ),
        ]
        for module in modules
    ],
)
def test_realtime_examples_sdk_wire(
    monkeypatch,
    tmp_path,
    module,
    model,
    settings,
):
    translating = module in (
        realtime_translation,
        realtime_translation_direct_call,
    )
    inbox = Queue()
    inbox.put({"type": "session.created", "session": {"id": "test-session"}})
    sent = []
    urls = []

    def response(identifier):
        inbox.put({"type": "response.created", "response": {"id": identifier}})
        inbox.put(
            {"type": "response.audio_transcript.delta", "delta": "Hello"},
        )
        inbox.put(
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(PCM).decode(),
            },
        )
        inbox.put(
            {"type": "response.audio_transcript.done", "transcript": "Hello"},
        )
        inbox.put(
            {
                "type": "response.done",
                "response": {
                    "id": identifier,
                    "status": "completed",
                    "usage": {},
                },
            },
        )

    def send(payload):
        event = json.loads(payload)
        sent.append(event)
        if event["type"] == "session.update":
            inbox.put(
                {"type": "session.updated", "session": {"id": "test-session"}},
            )
        elif event["type"] == "input_audio_buffer.append":
            if (
                translating
                and sum(
                    item["type"] == "input_audio_buffer.append"
                    for item in sent
                )
                == 1
            ):
                response("translation-1")
        elif event["type"] == "response.create":
            response("reply-1")
        elif event["type"] == "session.finish":
            response("translation-2")
            inbox.put(
                {
                    "type": "session.finished",
                    "session": {"id": "test-session"},
                },
            )

    def recv():
        event = inbox.get(timeout=2)
        return json.dumps(event) if event is not None else None

    ws = SimpleNamespace(
        send=send,
        recv=recv,
        settimeout=Mock(),
        close=Mock(side_effect=lambda: inbox.put(None)),
    )
    client = Client(api_key="sk-test", workspace="ws-test", max_retries=0)
    connect = client.audio.realtime.connect

    def connector(url, headers):
        urls.append(url)
        return ws

    monkeypatch.setattr(
        client.audio.realtime,
        "connect",
        lambda **kwargs: connect(
            **kwargs,
            connector=connector,
        ),
    )
    monkeypatch.setattr(module, "Client", lambda **kwargs: client)
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(PCM * 16)
    output = tmp_path / "output.wav"
    module.main(["--wav", str(source), "--output", str(output), "--no-play"])
    assert len(urls) == 1
    assert f"model={model}" in urls[0]
    assert sent[0]["type"] == "session.update"
    assert sent[0]["session"] == settings
    audio_events = [
        event for event in sent if event["type"] == "input_audio_buffer.append"
    ]
    sent_audio = b"".join(
        base64.b64decode(event["audio"]) for event in audio_events
    )
    assert sent_audio == PCM * 16
    kinds = [event["type"] for event in sent]
    if translating:
        assert kinds[-1] == "session.finish"
        assert "response.create" not in kinds
        assert "input_audio_buffer.commit" not in kinds
    else:
        assert kinds[-2:] == ["input_audio_buffer.commit", "response.create"]
        assert "session.finish" not in kinds
        if module in (realtime_omni, realtime_omni_direct_call):
            assert "response" not in sent[-1]
        else:
            assert sent[-1]["response"] == {}
    assert_wav(output, PCM * (2 if translating else 1))
    ws.close.assert_called_once()


@pytest.mark.parametrize(
    "module",
    [
        realtime_omni,
        realtime_qwen_audio,
        realtime_translation,
    ],
)
def test_realtime_send_failure_closes_before_join(
    monkeypatch,
    tmp_path,
    module,
):
    closed = Event()
    receiving = Event()
    session = Mock()

    def recv():
        if not receiving.is_set():
            receiving.set()
            return {"type": "session.updated"}
        assert closed.wait(2)
        return None

    session.recv_event.side_effect = recv
    session.close.side_effect = closed.set
    client = SimpleNamespace(
        audio=SimpleNamespace(
            realtime=SimpleNamespace(
                connect=Mock(return_value=nullcontext(session)),
            ),
        ),
    )
    monkeypatch.setattr(module, "Client", lambda **kwargs: nullcontext(client))
    monkeypatch.setattr(
        module,
        "send_input",
        Mock(
            side_effect=RuntimeError("send failed"),
        ),
    )
    args = module.make_parser().parse_args(
        [
            "--microphone",
            "--output",
            str(tmp_path / "failed.wav"),
            "--no-play",
        ],
    )
    with pytest.raises(RuntimeError, match="send failed"):
        module.run_turn(args)
    session.close.assert_called_once()
    session.commit.assert_not_called()
    session.create_response.assert_not_called()
    session.end_session.assert_not_called()


@pytest.mark.parametrize(
    "module",
    [
        realtime_translation,
        realtime_translation_direct_call,
    ],
)
@pytest.mark.parametrize(
    "events,error",
    [
        (
            [
                {"type": "response.audio.delta", "delta": "AAE="},
                {"type": "response.done", "response": {"status": "completed"}},
                None,
            ],
            RuntimeError,
        ),
        (
            [{"type": "response.done", "response": {"status": "failed"}}],
            RuntimeError,
        ),
        ([{"type": "response.audio.delta", "delta": "!invalid!"}], ValueError),
        ([{"type": "session.finished"}], RuntimeError),
    ],
)
def test_translation_rejects_incomplete_response(events, error, module):
    session = Mock()
    session.recv_event.side_effect = events
    with pytest.raises(error):
        module.receive_until(
            session,
            "session.finished",
            module.time.monotonic() + 2,
            Mock(),
        )


@pytest.mark.parametrize(
    "module",
    [
        realtime_omni,
        realtime_omni_direct_call,
        realtime_qwen_audio,
        realtime_qwen_audio_direct_call,
        realtime_translation,
        realtime_translation_direct_call,
    ],
)
def test_realtime_existing_output_prevents_connection(
    monkeypatch,
    tmp_path,
    module,
):
    client = Mock(side_effect=AssertionError("unexpected client"))
    monkeypatch.setattr(module, "Client", client)
    path = tmp_path / "keep.wav"
    path.write_bytes(b"keep")
    args = module.make_parser().parse_args(
        [
            "--microphone",
            "--output",
            str(path),
            "--no-play",
        ],
    )
    with pytest.raises(FileExistsError):
        module.run_turn(args)
    assert path.read_bytes() == b"keep"
    client.assert_not_called()


@pytest.mark.parametrize(
    "module",
    [
        realtime_omni,
        realtime_omni_direct_call,
        realtime_qwen_audio,
        realtime_qwen_audio_direct_call,
        realtime_translation,
        realtime_translation_direct_call,
    ],
)
def test_realtime_examples_do_not_switch_model_families(module):
    with pytest.raises(SystemExit) as caught:
        module.make_parser().parse_args(
            [
                "--microphone",
                "--output",
                "unused.wav",
                "--model",
                "other-family",
            ],
        )
    assert caught.value.code == 2
