# -*- coding: utf-8 -*-
"""Explicit audio sample startup, ownership, and shutdown contracts."""

import ast
import importlib
import json
import wave
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import dashscope


def _direct(name):
    # ``01_atomic_capabilities`` is not a valid identifier in source-level
    # imports, so the samples are loaded through importlib instead.
    return importlib.import_module(
        f"examples.01_atomic_capabilities.audio.direct_call.{name}",
    )


_common = _direct("_common")
asr_microphone = _direct("asr_microphone")
asr_websocket = _direct("asr_websocket")
connection_reuse = _direct("connection_reuse")
realtime_omni = _direct("realtime_omni")
realtime_qwen_audio = _direct("realtime_qwen_audio")
realtime_translation = _direct("realtime_translation")
tts_websocket = _direct("tts_websocket")
voice_clone = _direct("voice_clone")


SAMPLES = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "01_atomic_capabilities"
    / "audio"
)
NAMES = sorted(
    path.stem
    for path in SAMPLES.glob("*.py")
    # tingwu is a standalone example without a direct_call copy.
    if not path.name.startswith("_") and path.stem != "tingwu"
)
REALTIME = [realtime_omni, realtime_qwen_audio, realtime_translation]
PCM = b"\x01\x00" * 1600


@pytest.fixture
def wav_input(tmp_path):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(PCM)
    return path


@pytest.mark.parametrize("name", ["_common", *NAMES])
def test_copies_do_not_use_context_managers_or_original_helpers(name):
    path = SAMPLES / "direct_call" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.With, ast.AsyncWith))
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("__enter__", "__exit__")
        if isinstance(node, ast.ImportFrom):
            assert node.module != "contextlib"
            # Samples import their own package relatively; a level-2 import
            # (``from ..``) would reach the original context-manager helpers.
            assert node.level <= 1
    assert len(NAMES) == 12


@pytest.mark.parametrize("name", NAMES)
def test_direct_cli_help_has_no_side_effects(monkeypatch, capsys, name):
    module = _direct(name)
    forbidden = Mock(side_effect=AssertionError("help opened a resource"))
    monkeypatch.setattr(dashscope, "Client", forbidden)
    if hasattr(module, "Client"):
        monkeypatch.setattr(module, "Client", forbidden)
    monkeypatch.setattr(_common, "load_sounddevice", forbidden)
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    assert (
        "examples.01_atomic_capabilities.audio.direct_call."
        in capsys.readouterr().out
    )
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "name,arguments",
    [
        (
            "asr_file",
            ["--url", "https://cdn.test/input.wav", "--output-dir", "result"],
        ),
        (
            "asr_http_stream",
            ["--file", "input.wav", "--transcript", "asr.txt"],
        ),
        ("asr_websocket", ["input.wav", "--transcript", "asr.txt"]),
        (
            "asr_microphone",
            [
                "--seconds",
                "0.1",
                "--transcript",
                "asr.txt",
                "--recording",
                "recording.wav",
            ],
        ),
        ("tts_http", ["--text", "你好", "--output", "output.wav", "--no-play"]),
        (
            "tts_http_stream",
            ["--text", "你好", "--output", "output.wav", "--no-play"],
        ),
        (
            "tts_websocket",
            ["--text", "你好", "--output", "output.wav", "--no-play"],
        ),
        ("voice_clone", ["list", "--output", "voices.json"]),
        ("connection_reuse", ["asr", "input.wav", "--output-dir", "result"]),
        ("connection_reuse", ["tts", "--output-dir", "result"]),
        *[
            (
                module.__name__.rsplit(".", 1)[1],
                [
                    "--wav",
                    "input.wav",
                    "--output",
                    "output.wav",
                    "--no-play",
                ],
            )
            for module in REALTIME
        ],
    ],
)
def test_client_creation_failure_closes_earlier_files(
    monkeypatch,
    tmp_path,
    wav_input,
    name,
    arguments,
):
    module = _direct(name)
    files, readers = [], []
    original_open, original_wave = Path.open, wave.open

    def open_file(path, *args, **kwargs):
        file = original_open(path, *args, **kwargs)
        files.append(file)
        return file

    def open_wave(path, mode=None):
        audio = original_wave(path, mode)
        if mode == "rb":
            readers.append(audio)
        return audio

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(wave, "open", open_wave)
    factory = Mock(side_effect=RuntimeError("client failed"))
    monkeypatch.setattr(dashscope, "Client", factory)
    if hasattr(module, "Client"):
        monkeypatch.setattr(module, "Client", factory)
    stream = Mock()
    if module is asr_microphone:
        monkeypatch.setattr(module, "load_sounddevice", Mock())
        monkeypatch.setattr(module, "microphone", Mock(return_value=stream))
    filenames = {
        "input.wav",
        "asr.txt",
        "recording.wav",
        "output.wav",
        "voices.json",
        "result",
    }
    argv = [
        str(tmp_path / arg) if arg in filenames else arg for arg in arguments
    ]
    if module in (asr_microphone, asr_websocket):
        assert module.main(argv) == 1
    else:
        with pytest.raises(RuntimeError, match="client failed"):
            module.main(argv)
    factory.assert_called_once()
    assert files and all(file.closed for file in files)
    assert all(reader._file is None for reader in readers)
    if module is asr_microphone:
        stream.close.assert_called_once()
        stream.start.assert_not_called()


@pytest.mark.parametrize("module", REALTIME)
@pytest.mark.parametrize("failure", ["connect", "executor", "send", "close"])
def test_realtime_cleanup_on_partial_start_and_send_failure(
    monkeypatch,
    tmp_path,
    module,
    failure,
):
    session = Mock()
    closed = Event()
    order = []
    session.recv_event.side_effect = [{"type": "session.updated"}]

    def connect():
        if failure == "connect":
            raise RuntimeError("connect failed")

    def close():
        order.append("session.close")
        closed.set()
        if failure == "close":
            raise RuntimeError("close failed")

    session.connect.side_effect = connect
    session.close.side_effect = close
    client = SimpleNamespace(
        audio=SimpleNamespace(
            realtime=SimpleNamespace(
                connect=Mock(return_value=session),
            ),
        ),
        close=Mock(side_effect=lambda: order.append("client.close")),
    )
    monkeypatch.setattr(module, "Client", Mock(return_value=client))
    executor = Mock()
    executor.submit.return_value = Mock()

    def shutdown(*, wait):
        assert wait and closed.is_set()
        order.append("executor.shutdown")

    executor.shutdown.side_effect = shutdown
    factory = Mock(return_value=executor)
    if failure == "executor":
        factory.side_effect = RuntimeError("executor failed")
    monkeypatch.setattr(module, "ThreadPoolExecutor", factory)
    monkeypatch.setattr(
        module,
        "send_input",
        Mock(
            side_effect=RuntimeError("send failed"),
        ),
    )
    raw, output = Mock(), Mock()
    monkeypatch.setattr(module, "wav_output", Mock(return_value=(raw, output)))
    args = module.make_parser().parse_args(
        [
            "--microphone",
            "--output",
            str(tmp_path / "reply.wav"),
            "--no-play",
        ],
    )
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        module.run_turn(args)
    session.connect.assert_called_once()
    session.close.assert_called_once()
    client.close.assert_called_once()
    output.close.assert_called_once()
    raw.close.assert_called_once()
    if failure in ("send", "close"):
        assert order == ["session.close", "executor.shutdown", "client.close"]
    else:
        executor.shutdown.assert_not_called()


@pytest.mark.parametrize("module", REALTIME)
def test_realtime_close_unblocks_real_receiver(monkeypatch, tmp_path, module):
    closed = Event()
    calls = 0

    def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "session.updated"}
        assert closed.wait(2)
        return None

    session = Mock()
    session.recv_event.side_effect = receive
    session.close.side_effect = closed.set
    client = SimpleNamespace(
        audio=SimpleNamespace(
            realtime=SimpleNamespace(
                connect=Mock(return_value=session),
            ),
        ),
        close=Mock(),
    )
    monkeypatch.setattr(module, "Client", Mock(return_value=client))
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
            str(tmp_path / "reply.wav"),
            "--no-play",
        ],
    )
    with pytest.raises(RuntimeError, match="send failed"):
        module.run_turn(args)
    session.connect.assert_called_once()
    session.close.assert_called_once()
    client.close.assert_called_once()


@pytest.mark.parametrize("module", REALTIME)
@pytest.mark.parametrize("overflow", [False, True])
def test_realtime_microphone_explicit_lifecycle(monkeypatch, module, overflow):
    stream = Mock()
    stream.read.return_value = (PCM, overflow)
    monkeypatch.setattr(module, "microphone", Mock(return_value=stream))
    receiver = Mock()
    receiver.done.return_value = False
    session = Mock()
    args = SimpleNamespace(device=None, seconds=0.1)
    if overflow:
        with pytest.raises(RuntimeError, match="溢出"):
            module.send_input(
                session,
                receiver,
                args,
                None,
                module.time.monotonic() + 2,
            )
        session.append_audio.assert_not_called()
    else:
        module.send_input(
            session,
            receiver,
            args,
            None,
            module.time.monotonic() + 2,
        )
        session.append_audio.assert_called_once_with(PCM)
    stream.start.assert_called_once()
    stream.stop.assert_called_once()
    stream.close.assert_called_once()


class TaskSession:
    def __init__(self, identifier):
        self.session = SimpleNamespace(
            task_id=identifier,
            request_id=identifier,
        )
        self.request_id = identifier
        self.first_package_delay = 1
        self.last_package_delay = 2
        self.usage = {}
        self.last_response = None
        self.stopped = Event()
        self.started = False
        self.closed = False
        self.sent = []

    def start(self):
        assert not self.started
        self.started = True

    def send_audio_frame(self, data):
        assert self.started and not self.closed
        self.sent.append(data)

    def send_text(self, text):
        assert self.started and not self.closed
        self.sent.append(text)

    def stop(self):
        assert self.started and self.sent
        self.stopped.set()

    def finish(self):
        assert self.stopped.wait(2)
        self.last_response = {
            "header": {
                "event": "task-finished",
                "task_id": self.session.task_id,
            },
        }

    def results(self):
        self.finish()
        yield SimpleNamespace(
            get_sentence=lambda: {"text": "你好"},
            is_sentence_end=lambda: True,
        )

    def events(self):
        self.finish()
        yield PCM
        yield self.last_response

    def close(self):
        self.closed = True
        self.stopped.set()


@pytest.mark.parametrize("kind", ["asr", "tts"])
def test_websocket_samples_start_and_drain(
    monkeypatch,
    tmp_path,
    wav_input,
    kind,
):
    session = TaskSession("single-task")
    client = SimpleNamespace(
        audio=SimpleNamespace(
            asr=SimpleNamespace(recognize=Mock(return_value=session)),
            tts=SimpleNamespace(connect=Mock(return_value=session)),
        ),
        close=Mock(),
    )
    monkeypatch.setattr(dashscope, "Client", Mock(return_value=client))
    if kind == "asr":
        transcript = tmp_path / "transcript.txt"
        assert (
            asr_websocket.main(
                [
                    str(wav_input),
                    "--transcript",
                    str(transcript),
                ],
            )
            == 0
        )
        assert transcript.read_text() == "你好\n"
        assert b"".join(session.sent) == PCM
    else:
        monkeypatch.setattr(tts_websocket, "Client", Mock(return_value=client))
        output = tmp_path / "output.wav"
        tts_websocket.main(
            [
                "--text",
                "你好",
                "--output",
                str(output),
                "--no-play",
            ],
        )
        with wave.open(str(output), "rb") as audio:
            assert audio.readframes(audio.getnframes()) == PCM
        assert session.sent == ["你好"]
    assert session.started and session.closed and session.last_response
    client.close.assert_called_once()


@pytest.mark.parametrize("kind", ["asr", "tts"])
@pytest.mark.parametrize("failure", ["start", "executor"])
def test_websocket_partial_start_cleanup(monkeypatch, kind, failure):
    session = Mock()
    if failure == "start":
        session.start.side_effect = RuntimeError("start failed")
    client = SimpleNamespace(
        audio=SimpleNamespace(
            asr=SimpleNamespace(recognize=Mock(return_value=session)),
            tts=SimpleNamespace(connect=Mock(return_value=session)),
        ),
        close=Mock(),
    )
    executor = Mock()
    factory = Mock(return_value=executor)
    if failure == "executor":
        factory.side_effect = RuntimeError("executor failed")
    module = asr_websocket if kind == "asr" else tts_websocket
    monkeypatch.setattr(module, "ThreadPoolExecutor", factory)
    if kind == "asr":
        monkeypatch.setattr(dashscope, "Client", Mock(return_value=client))
        args = SimpleNamespace(
            timeout=1,
            region=None,
            workspace=None,
            model="qwen-audio-3.1-asr-flash-streaming",
            vocabulary_id=None,
            partials=False,
        )
        with pytest.raises(asr_websocket.SampleError):
            module.run_recognition(args, 16000, Mock(), Mock())
        client.close.assert_called_once()
    else:
        args = module.make_parser().parse_args(
            [
                "--text",
                "你好",
                "--output",
                "unused.wav",
                "--no-play",
            ],
        )
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            module.synthesize(client, args, ["你好"], Mock(), None)
    session.close.assert_called_once()
    session.stop.assert_not_called()
    if failure == "start" and kind == "asr":
        executor.shutdown.assert_called_once_with(wait=True)


def test_llm_stream_closes_when_tts_fails(monkeypatch, tmp_path):
    stream = SimpleNamespace(text_stream=iter(["你好"]), close=Mock())
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=Mock(return_value=stream),
            ),
        ),
        close=Mock(),
    )
    monkeypatch.setattr(tts_websocket, "Client", Mock(return_value=client))
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
                str(tmp_path / "reply.wav"),
                "--no-play",
            ],
        )
    stream.close.assert_called_once()
    client.close.assert_called_once()


@pytest.mark.parametrize("kind", ["asr", "tts"])
def test_five_tasks_keep_shared_connection_open(
    monkeypatch,
    tmp_path,
    wav_input,
    kind,
):
    sessions = []
    connection = SimpleNamespace(closed=False)
    connection.close = Mock(
        side_effect=lambda: setattr(connection, "closed", True),
    )

    def task(**kwargs):
        assert kwargs["connection"] is connection and not connection.closed
        assert all(session.closed for session in sessions)
        session = TaskSession(f"task-{len(sessions) + 1}")
        sessions.append(session)
        return session

    client = SimpleNamespace(
        config=SimpleNamespace(region="cn-beijing", workspace="ws-test"),
        audio=SimpleNamespace(
            asr=SimpleNamespace(recognize=task),
            tts=SimpleNamespace(connect=task),
        ),
        ws_connection=Mock(return_value=connection),
        close=Mock(),
    )
    monkeypatch.setattr(connection_reuse, "Client", Mock(return_value=client))
    folder = tmp_path / "reuse"
    source_args = [str(wav_input)] if kind == "asr" else []
    assert (
        connection_reuse.main(
            [
                kind,
                *source_args,
                "--output-dir",
                str(folder),
            ],
        )
        == 0
    )
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert len(summary["runs"]) == len(sessions) == 5
    assert all(session.started and session.closed for session in sessions)
    assert summary["connection_count"] == 1 and summary["connection_closed"]
    connection.close.assert_called_once()
    client.ws_connection.assert_called_once()
    client.close.assert_called_once()


@pytest.mark.parametrize("record", [False, True])
def test_asr_microphone_direct_start_stop(monkeypatch, tmp_path, record):
    stream = Mock()
    stream.read.return_value = (PCM, False)
    monkeypatch.setattr(asr_microphone, "load_sounddevice", Mock())
    monkeypatch.setattr(
        asr_microphone,
        "microphone",
        Mock(return_value=stream),
    )
    session = TaskSession("microphone-task")
    client = SimpleNamespace(
        audio=SimpleNamespace(
            asr=SimpleNamespace(
                recognize=Mock(return_value=session),
            ),
        ),
        close=Mock(),
    )
    monkeypatch.setattr(dashscope, "Client", Mock(return_value=client))
    recording = tmp_path / "recording.wav"
    options = ["--recording", str(recording)] if record else ["--no-recording"]
    assert (
        asr_microphone.main(
            [
                "--seconds",
                "0.1",
                "--transcript",
                str(tmp_path / "asr.txt"),
                *options,
            ],
        )
        == 0
    )
    stream.start.assert_called_once()
    stream.stop.assert_called_once()
    stream.close.assert_called_once()
    assert session.started and session.closed and session.sent == [PCM]
    assert recording.exists() == record
    if record:
        with wave.open(str(recording), "rb") as audio:
            assert audio.readframes(audio.getnframes()) == PCM


@pytest.mark.parametrize("save", [False, True])
def test_voice_query_closes_client_and_optional_output(
    monkeypatch,
    tmp_path,
    save,
):
    voice = SimpleNamespace(
        status="OK",
        model_dump=Mock(return_value={"voice_id": "v-test"}),
    )
    client = SimpleNamespace(
        config=SimpleNamespace(region="cn-beijing", workspace="ws-test"),
        audio=SimpleNamespace(
            voices=SimpleNamespace(get=Mock(return_value=voice)),
        ),
        close=Mock(),
    )
    monkeypatch.setattr(voice_clone, "Client", Mock(return_value=client))
    output = tmp_path / "voice.json"
    options = ["--output", str(output)] if save else []
    assert voice_clone.main(["get", "--voice-id", "v-test", *options]) == 0
    client.close.assert_called_once()
    assert output.exists() == save


def test_wav_helpers_own_resources_explicitly(monkeypatch, tmp_path):
    raw = Mock()
    writer = Mock()
    monkeypatch.setattr(Path, "open", Mock(return_value=raw))
    monkeypatch.setattr(wave, "open", Mock(return_value=writer))
    writer.setframerate.side_effect = RuntimeError("invalid format")
    with pytest.raises(RuntimeError, match="invalid format"):
        _common.wav_output(tmp_path / "invalid.wav", 24000)
    writer.close.assert_called_once()
    raw.close.assert_called_once()
    reader = Mock()
    reader.getnchannels.return_value = 2
    monkeypatch.setattr(wave, "open", Mock(return_value=reader))
    with pytest.raises(ValueError):
        _common.open_wav(tmp_path / "stereo.wav")
    reader.close.assert_called_once()
