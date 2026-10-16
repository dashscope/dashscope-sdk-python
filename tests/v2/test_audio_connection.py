# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resource sessions running as tasks on a shared WsConnection."""

# pylint: disable=protected-access

import json
import queue

import pytest

from dashscope.v2.client import Client

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

    for _ in range(2):
        rec = client.audio.asr.recognize(
            model="paraformer-realtime-v2",
            connection=conn,
        )
        rec.start()
        rec.send_audio_frame(b"\x00\x01")
        rec.stop()
        list(rec.events())
        rec.close()
        assert not conn.closed

    assert len(connects) == 1
    run_tasks = _run_task_frames(ws)
    assert len(run_tasks) == 2
    assert (
        run_tasks[0]["header"]["task_id"] != run_tasks[1]["header"]["task_id"]
    )
    # Exactly one binary frame per recognition, both on the shared socket.
    assert len([item for item in ws.sent if isinstance(item, bytes)]) == 2
    conn.close()
    client.close()


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
