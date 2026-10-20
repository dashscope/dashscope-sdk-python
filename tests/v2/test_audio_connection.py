# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resource sessions running as tasks on a shared WsConnection."""

# pylint: disable=protected-access

import json
import queue

import pytest

from dashscope.v2.client import Client
from dashscope.v2.errors import DashScopeError

BASE_URL = "https://api.test.local/api/v1"
WS_URL = "wss://ws.test.local/api-ws/v1/inference"


class AutoWebSocket:
    """Blocking stub that answers run-task/finish-task automatically."""

    def __init__(self):
        self.sent = []
        self.inbox = queue.SimpleQueue()
        self.closed = False

    def send(self, payload, opcode=1):
        assert opcode == (2 if isinstance(payload, bytes) else 1)
        self.sent.append(payload)
        if isinstance(payload, str):
            header = json.loads(payload)["header"]
            action = header.get("action")
            if action == "run-task":
                self.inbox.put(self._event("task-started", header["task_id"]))
            elif action == "finish-task":
                self.inbox.put(self._event("task-finished", header["task_id"]))

    @staticmethod
    def _event(name, task_id, payload=None):
        return json.dumps(
            {
                "header": {
                    "event": name,
                    "task_id": task_id,
                    "request_id": "r",
                },
                "payload": payload or {},
            },
        )

    def recv(self):
        item = self.inbox.get(timeout=5)
        if isinstance(item, BaseException):
            raise item
        return item

    def settimeout(self, seconds):
        pass

    def close(self):
        self.closed = True
        self.inbox.put(None)


def _make_client_and_connection():
    connects = []

    def connector(url, headers):
        ws = AutoWebSocket()
        connects.append({"url": url, "headers": headers, "ws": ws})
        return ws

    client = Client(api_key="sk-test", base_url=BASE_URL)
    conn = client.ws_connection(
        url=WS_URL,
        connector=connector,
        timeout=5,
    )
    return client, conn, connects


def _run_task_frames(ws):
    return [
        json.loads(item)
        for item in ws.sent
        if isinstance(item, str)
        and json.loads(item)["header"].get("action") == "run-task"
    ]


def test_ws_connection_factory_uses_client_config():
    client, conn, connects = _make_client_and_connection()
    assert connects[0]["url"] == WS_URL
    assert connects[0]["headers"]["Authorization"] == "Bearer sk-test"
    conn.close()
    client.close()


def test_recognition_sequential_reuse_on_shared_connection():
    client, conn, connects = _make_client_and_connection()
    ws = connects[0]["ws"]
    task_ids = []
    audio_frames = [bytes([index, index + 1]) for index in range(5)]
    try:
        for index, audio in enumerate(audio_frames):
            rec = client.audio.asr.recognize(
                model="qwen-audio-3.1-asr-flash-streaming",
                connection=conn,
                timeout=5,
            )
            try:
                rec.start()
                task_id = rec.session.task_id
                task_ids.append(task_id)
                rec.send_audio_frame(audio)
                assert conn._binary_uplink_owner is rec.session
                result = AutoWebSocket._event(
                    "result-generated",
                    task_id,
                    {
                        "output": {
                            "sentence": {"text": f"recognized task {index}"},
                        },
                    },
                )
                ws.inbox.put(result)
                rec.stop()
                assert list(rec.events()) == [
                    json.loads(result),
                    json.loads(AutoWebSocket._event("task-finished", task_id)),
                ]
                assert rec.session.finished
                assert conn._binary_uplink_owner is None
            finally:
                rec.close()
            assert not conn._tasks
            assert not conn.closed and not ws.closed
            assert conn._reader.is_alive()

        assert len(connects) == 1
        run_tasks = _run_task_frames(ws)
        assert [frame["header"]["task_id"] for frame in run_tasks] == task_ids
        assert len(set(task_ids)) == 5
        assert all(
            frame["payload"]["model"] == "qwen-audio-3.1-asr-flash-streaming"
            for frame in run_tasks
        )
        assert [
            item for item in ws.sent if isinstance(item, bytes)
        ] == audio_frames
    finally:
        try:
            conn.close()
        finally:
            client.close()
    assert conn.closed and ws.closed
    assert not conn._reader.is_alive()


def test_synthesis_sequential_reuse_on_shared_connection():
    client, conn, connects = _make_client_and_connection()
    ws = connects[0]["ws"]
    task_ids = []
    texts = [f"Synthesize task {index}." for index in range(5)]
    try:
        for index, text in enumerate(texts):
            tts = client.audio.tts.connect(
                model="qwen-audio-3.1-tts-flash",
                voice="longanhuan_v3.1",
                connection=conn,
                timeout=5,
            )
            try:
                tts.start()
                task_id = tts.session.task_id
                task_ids.append(task_id)
                assert conn._binary_downlink_owner is tts.session
                tts.send_text(text)
                audio = f"synthesized-audio-{index}".encode("utf-8")
                ws.inbox.put(audio)
                tts.stop()
                assert list(tts.audio_chunks()) == [audio]
                assert tts.last_response == json.loads(
                    AutoWebSocket._event("task-finished", task_id),
                )
                assert tts.session.finished
                assert conn._binary_downlink_owner is None
            finally:
                tts.close()
            assert not conn._tasks
            assert conn._binary_uplink_owner is None
            assert conn._binary_downlink_owner is None
            assert not conn.closed and not ws.closed
            assert conn._reader.is_alive()

        assert len(connects) == 1
        run_tasks = _run_task_frames(ws)
        assert [frame["header"]["task_id"] for frame in run_tasks] == task_ids
        assert len(set(task_ids)) == 5
        assert all(
            frame["payload"]["model"] == "qwen-audio-3.1-tts-flash"
            and frame["payload"]["parameters"]["voice"] == "longanhuan_v3.1"
            for frame in run_tasks
        )
        text_frames = [
            json.loads(item)
            for item in ws.sent
            if isinstance(item, str)
            and json.loads(item)["header"].get("action") == "continue-task"
        ]
        assert [
            frame["header"]["task_id"] for frame in text_frames
        ] == task_ids
        assert [frame["payload"]["input"] for frame in text_frames] == [
            {"text": text} for text in texts
        ]
    finally:
        try:
            conn.close()
        finally:
            client.close()
    assert conn.closed and ws.closed
    assert not conn._reader.is_alive()


def test_recognition_and_synthesis_concurrently_on_one_connection():
    client, conn, connects = _make_client_and_connection()
    ws = connects[0]["ws"]

    rec = client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connection=conn,
    )
    rec.start()
    tts = client.audio.tts.connect(model="cosyvoice-v1", connection=conn)
    tts.start()

    # ASR owns the binary uplink, TTS the binary downlink.
    rec.send_audio_frame(b"\x02\x03")
    tts.send_text("你好")
    ws.inbox.put(b"synthesized-audio")
    ws.inbox.put(
        AutoWebSocket._event(
            "result-generated",
            rec.session.task_id,
            {"output": {"sentence": {"text": "hello"}}},
        ),
    )
    chunks = []
    for frame in tts.events():
        if isinstance(frame, bytes):
            chunks.append(frame)
        # task-finished arrives from stop(); stop after the first chunk.
        tts.stop()
    assert chunks == [b"synthesized-audio"]
    event = rec.session.recv_event()
    assert event["payload"]["output"]["sentence"]["text"] == "hello"

    tts.close()
    rec.close()
    assert not conn.closed
    conn.close()
    assert len(connects) == 1
    client.close()


def test_two_syntheses_on_one_connection_fail_fast():
    client, conn, _ = _make_client_and_connection()
    tts1 = client.audio.tts.connect(model="cosyvoice-v1", connection=conn)
    tts1.start()
    tts2 = client.audio.tts.connect(model="cosyvoice-v1", connection=conn)
    with pytest.raises(RuntimeError, match="binary downlink"):
        tts2.start()
    # The first session is unaffected.
    tts1.send_text("继续")
    tts1.close()
    conn.close()
    client.close()


def test_tingwu_realtime_on_shared_connection():
    client, conn, connects = _make_client_and_connection()
    ws = connects[0]["ws"]

    with client.tingwu.realtime(
        model="tingwu-turbo",
        app_id="app-1",
        connection=conn,
    ) as session:
        session.send_audio_frame(b"\x04\x05")
    assert not conn.closed

    run_tasks = _run_task_frames(ws)
    assert len(run_tasks) == 1
    assert run_tasks[0]["payload"]["task_group"] == "aigc"
    conn.close()
    assert len(connects) == 1
    client.close()


def test_recognition_close_detaches_but_connection_survives_errors():
    client, conn, connects = _make_client_and_connection()
    ws = connects[0]["ws"]
    rec1 = client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connection=conn,
    )
    rec1.start()
    rec1.close()

    # A second recognition still opens on the same socket afterwards.
    rec2 = client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connection=conn,
    )
    rec2.start()
    rec2.stop()
    list(rec2.events())
    rec2.close()
    assert len(connects) == 1
    assert not ws.closed
    conn.close()
    assert ws.closed
    client.close()


class _RealtimeWebSocket:
    """Minimal realtime stub: one socket per session, no task framing."""

    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        return None

    def settimeout(self, seconds):
        pass

    def close(self):
        self.closed = True


def test_realtime_opens_one_socket_per_session():
    """The realtime protocol cannot share a socket (no task id; the
    connection closes with the session), so sequential sessions each
    perform their own handshake -- reuse only exists via WsConnection
    on the run-task family."""
    client = Client(api_key="sk-test", base_url=BASE_URL)
    sockets = []

    def connector(url, headers):
        ws = _RealtimeWebSocket()
        sockets.append({"url": url, "ws": ws})
        return ws

    for _ in range(2):
        with client.audio.realtime.connect(
            model="qwen-omni-turbo-realtime",
            ws_url="wss://ws.test.local/api-ws/v1/realtime",
            connector=connector,
            timeout=5,
        ) as session:
            session.send_event({"type": "session.update", "session": {}})
    assert len(sockets) == 2
    assert sockets[0]["ws"].closed
    assert sockets[1]["ws"].closed
    client.close()


@pytest.fixture
def bounded_audio_connection():
    resources = []

    def make(ws=None):
        ws = AutoWebSocket() if ws is None else ws
        client = Client(api_key="sk-test", base_url=BASE_URL, timeout=0.5)
        try:
            conn = client.ws_connection(
                url=WS_URL,
                connector=lambda url, headers: ws,
                timeout=0.5,
            )
        except BaseException:
            client.close()
            raise
        resources.append((client, conn))
        return client, conn, ws

    yield make
    for client, conn in resources:
        try:
            conn.close()
            conn._reader.join(timeout=1)
            assert not conn._reader.is_alive(), "websocket reader leaked"
        finally:
            client.close()


@pytest.mark.parametrize(
    "reply",
    ["task-failed", "result-generated", "timeout"],
)
def test_shared_recognition_start_failure_detaches_and_allows_retry(
    bounded_audio_connection,
    reply,
):
    class FailOnceSocket(AutoWebSocket):
        failed_once = False

        def send(self, payload, opcode=1):
            if (
                opcode == 1
                and json.loads(payload)["header"]["action"] == "run-task"
            ):
                if not self.failed_once:
                    self.failed_once = True
                    self.sent.append(payload)
                    if reply != "timeout":
                        task_id = json.loads(payload)["header"]["task_id"]
                        self.inbox.put(self._event(reply, task_id))
                    return
            super().send(payload, opcode=opcode)

    client, conn, ws = bounded_audio_connection(FailOnceSocket())
    rec = client.audio.asr.recognize(
        model="fun-asr-realtime",
        connection=conn,
        timeout=0.05,
    )
    try:
        expected = TimeoutError if reply == "timeout" else DashScopeError
        with pytest.raises(expected):
            rec.start()
        assert not conn._tasks
        assert not conn.closed and not ws.closed
        with pytest.raises(RuntimeError, match="not open"):
            _ = rec.session
        with rec:
            rec.send_audio_frame(b"retry")
            rec.stop()
            assert list(rec.events())[-1]["header"]["event"] == "task-finished"
        assert not conn._tasks
        assert len(_run_task_frames(ws)) == 2
    finally:
        rec.close()


def test_shared_recognition_same_object_restarts_and_releases_owners(
    bounded_audio_connection,
):
    client, conn, ws = bounded_audio_connection()
    rec = client.audio.asr.recognize(model="fun-asr-realtime", connection=conn)
    ids = []
    try:
        for _ in range(2):
            with rec:
                ids.append(rec.session.task_id)
                assert rec.usage is None and rec.first_package_delay is None
                assert rec.last_package_delay is None
                rec.send_audio_frame(b"audio")
                ws.inbox.put(
                    ws._event(
                        "result-generated",
                        rec.session.task_id,
                        {
                            "output": {
                                "sentence": {"text": "hello", "end_time": 0},
                            },
                            "usage": {"duration": 1},
                        },
                    ),
                )
                rec.stop()
                assert len(list(rec.results())) == 1
                assert rec.usage == {"duration": 1}
            assert not conn._tasks
            assert conn._binary_uplink_owner is None
            assert not conn.closed
        assert ids[0] != ids[1]
        assert len(_run_task_frames(ws)) == 2
    finally:
        rec.close()


def test_shared_recognition_close_after_uplink_conflict_does_not_break_owner(
    bounded_audio_connection,
):
    client, conn, ws = bounded_audio_connection()
    first = client.audio.asr.recognize(
        model="fun-asr-realtime",
        connection=conn,
    )
    second = client.audio.asr.recognize(
        model="fun-asr-realtime",
        connection=conn,
    )
    try:
        first.start()
        second.start()
        first.send_audio_frame(b"first")
        with pytest.raises(RuntimeError, match="binary uplink"):
            second.send_audio_frame(b"blocked")
        second.close()
        first.send_audio_frame(b"still first")
        first.stop()
        assert list(first.events())[-1]["header"]["event"] == "task-finished"
        with second:
            second.send_audio_frame(b"now second")
            second.stop()
            list(second.events())
        assert [frame for frame in ws.sent if isinstance(frame, bytes)] == [
            b"first",
            b"still first",
            b"now second",
        ]
        assert not conn.closed
    finally:
        first.close()
        second.close()


def test_shared_recognition_task_failure_keeps_other_task_usable(
    bounded_audio_connection,
):
    client, conn, ws = bounded_audio_connection()
    first = client.audio.asr.recognize(
        model="fun-asr-realtime",
        connection=conn,
    )
    second = client.audio.asr.recognize(
        model="fun-asr-realtime",
        connection=conn,
    )
    try:
        first.start()
        second.start()
        first.send_audio_frame(b"first")
        ws.inbox.put(ws._event("task-failed", first.session.task_id))
        with pytest.raises(DashScopeError):
            next(first.events())
        second.send_audio_frame(b"second")
        second.stop()
        assert list(second.events())[-1]["header"]["event"] == "task-finished"
        assert not conn.closed
    finally:
        first.close()
        second.close()
