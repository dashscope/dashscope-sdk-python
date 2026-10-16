# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""WsConnection: multi-task demux over one shared duplex socket."""

# pylint: disable=protected-access

import json
import queue

import pytest

from dashscope.v2.errors import APIConnectionError, DashScopeError
from dashscope.v2.transport.ws import WsEventError
from dashscope.v2.transport.ws_connection import WsConnection

URL = "wss://ws.test.local/api-ws/v1/inference"


class QueueWebSocket:
    """Blocking websocket stub: records sends, replays a queued inbox."""

    def __init__(self):
        self.sent = []
        self.inbox = queue.SimpleQueue()
        self.closed = False
        self.timeouts = []

    def send(self, payload, opcode=1):
        assert opcode == (2 if isinstance(payload, bytes) else 1)
        self.sent.append(payload)

    def recv(self):
        item = self.inbox.get(timeout=5)
        if isinstance(item, BaseException):
            raise item
        return item

    def settimeout(self, seconds):
        self.timeouts.append(seconds)

    def close(self):
        self.closed = True
        # Wake the reader thread so connection.close() can join it.
        self.inbox.put(None)


def _make_connection(**kwargs):
    connects = []

    def connector(url, headers):
        ws = QueueWebSocket()
        connects.append({"url": url, "headers": headers, "ws": ws})
        return ws

    kwargs.setdefault("url", URL)
    conn = WsConnection(connector=connector, **kwargs)
    return conn, connects


def _event(name, task_id, payload=None, request_id="r"):
    return json.dumps(
        {
            "header": {
                "event": name,
                "task_id": task_id,
                "request_id": request_id,
            },
            "payload": payload or {},
        },
    )


def _started(conn, task, **run_kwargs):
    """Feed task-started, then run the task (reader routes async)."""
    task_id = task.task_id
    conn._ws.inbox.put(_event("task-started", task_id))
    kwargs = {
        "task_group": "audio",
        "task": "asr",
        "function": "recognition",
        "model": "paraformer-realtime-v2",
    }
    kwargs.update(run_kwargs)
    return task.run_task(**kwargs)


def _sent_frames(ws):
    return [json.loads(item) for item in ws.sent if isinstance(item, str)]


def test_concurrent_json_tasks_demux_by_task_id():
    conn, connects = _make_connection(api_key="sk-test", timeout=5)
    assert connects[0]["url"] == URL
    assert connects[0]["headers"]["Authorization"] == "Bearer sk-test"
    # The reader thread owns socket reads; the socket must block forever.
    assert connects[0]["ws"].timeouts[-1] is None

    t1 = conn.open_task()
    t2 = conn.open_task()
    _started(conn, t1)
    _started(conn, t2)

    # Events arrive out of order; each task gets only its own.
    conn._ws.inbox.put(_event("result-generated", t2.task_id, {"n": 2}))
    conn._ws.inbox.put(_event("result-generated", t1.task_id, {"n": 1}))
    assert t2.recv_event()["payload"] == {"n": 2}
    assert t1.recv_event()["payload"] == {"n": 1}

    conn._ws.inbox.put(_event("task-finished", t1.task_id))
    conn._ws.inbox.put(_event("task-finished", t2.task_id))
    t1.recv_event()
    t2.recv_event()
    assert t1.finished and t2.finished

    run_tasks = [
        f
        for f in _sent_frames(conn._ws)
        if f["header"]["action"] == "run-task"
    ]
    assert {f["header"]["task_id"] for f in run_tasks} == {
        t1.task_id,
        t2.task_id,
    }
    conn.close()
    assert len(connects) == 1


def test_sequential_reuse_single_handshake():
    conn, connects = _make_connection(timeout=5)
    t1 = conn.open_task()
    _started(conn, t1)
    t1.finish_task()
    conn._ws.inbox.put(_event("task-finished", t1.task_id))
    t1.recv_event()
    t1.close()

    t2 = conn.open_task()
    assert t2.task_id != t1.task_id
    _started(conn, t2)
    conn._ws.inbox.put(_event("task-finished", t2.task_id))
    t2.recv_event()
    assert t2.finished
    conn.close()
    assert len(connects) == 1


def test_binary_downlink_single_active_task_implicit():
    conn, _ = _make_connection(timeout=5)
    task = conn.open_task()
    _started(conn, task)
    conn._ws.inbox.put(b"\x00\x01audio")
    assert task.recv() == b"\x00\x01audio"
    conn.close()


def test_binary_downlink_two_active_tasks_requires_claim():
    conn, _ = _make_connection(timeout=5)
    t1 = conn.open_task()
    t2 = conn.open_task()
    _started(conn, t1)
    _started(conn, t2)
    t2.claim_binary_downlink()
    with pytest.raises(RuntimeError, match="binary downlink"):
        t1.claim_binary_downlink()
    conn._ws.inbox.put(b"audio-for-t2")
    assert t2.recv() == b"audio-for-t2"
    # t1 still receives its own JSON events.
    conn._ws.inbox.put(_event("result-generated", t1.task_id))
    assert t1.recv_event()["header"]["event"] == "result-generated"
    conn.close()


def test_binary_downlink_ambiguous_fails_connection():
    conn, _ = _make_connection(timeout=5)
    t1 = conn.open_task()
    t2 = conn.open_task()
    _started(conn, t1)
    _started(conn, t2)
    conn._ws.inbox.put(b"undemuxable")
    with pytest.raises(WsEventError, match="claim_binary_downlink"):
        t1.recv()
    with pytest.raises(WsEventError, match="claim_binary_downlink"):
        t2.recv()
    assert conn.closed
    with pytest.raises(RuntimeError, match="closed"):
        conn.open_task()


def test_binary_uplink_single_owner_and_release_on_finish():
    conn, _ = _make_connection(timeout=5)
    t1 = conn.open_task()
    t2 = conn.open_task()
    _started(conn, t1)
    _started(conn, t2)
    t1.send_binary(b"frame-1")
    with pytest.raises(RuntimeError, match="binary uplink"):
        t2.send_binary(b"frame-2")
    assert conn._ws.sent[-1] == b"frame-1"

    # Finishing the uplink owner releases it for the next task.
    conn._ws.inbox.put(_event("task-finished", t1.task_id))
    t1.recv_event()
    t2.send_binary(b"frame-2")
    assert conn._ws.sent[-1] == b"frame-2"
    conn.close()


def test_socket_error_broadcasts_to_all_tasks():
    conn, _ = _make_connection(timeout=5)
    t1 = conn.open_task()
    t2 = conn.open_task()
    _started(conn, t1)
    _started(conn, t2)
    conn._ws.inbox.put(OSError("boom"))
    with pytest.raises(OSError, match="boom"):
        t1.recv()
    with pytest.raises(OSError, match="boom"):
        t2.recv()
    assert conn.closed


def test_peer_close_normalizes_to_api_connection_error():
    """A library close on the shared reader surfaces as APIConnectionError,
    not a raw websocket exception -- matching WsSession.recv / realtime."""
    from websocket import WebSocketConnectionClosedException

    conn, _ = _make_connection(timeout=5)
    t1 = conn.open_task()
    t2 = conn.open_task()
    _started(conn, t1)
    _started(conn, t2)
    conn._ws.inbox.put(WebSocketConnectionClosedException("closed"))
    with pytest.raises(APIConnectionError, match="closed by peer"):
        t1.recv()
    with pytest.raises(APIConnectionError, match="closed by peer"):
        t2.recv()
    assert conn.closed


def test_task_close_detaches_and_late_events_drop():
    conn, _ = _make_connection(timeout=5)
    t1 = conn.open_task()
    t2 = conn.open_task()
    _started(conn, t1)
    _started(conn, t2)
    t1.close()
    # A late event for the detached task is dropped, not fatal.
    conn._ws.inbox.put(_event("result-generated", t1.task_id))
    conn._ws.inbox.put(_event("result-generated", t2.task_id, {"ok": 1}))
    assert t2.recv_event()["payload"] == {"ok": 1}
    assert not conn.closed
    with pytest.raises(RuntimeError, match="closed"):
        t1.recv()
    conn.close()


def test_task_failed_maps_error_and_finishes_task():
    conn, _ = _make_connection(timeout=5)
    task = conn.open_task()
    _started(conn, task)
    conn._ws.inbox.put(
        json.dumps(
            {
                "header": {
                    "event": "task-failed",
                    "task_id": task.task_id,
                    "request_id": "req-x",
                    "error_code": "InvalidParameter",
                    "error_message": "bad parameter",
                },
                "payload": {},
            },
        ),
    )
    with pytest.raises(DashScopeError, match="bad parameter"):
        task.recv()
    assert task.finished
    assert task.request_id == "req-x"
    conn.close()


def test_finish_task_and_cancel_semantics():
    conn, _ = _make_connection(timeout=5)
    task = conn.open_task()
    _started(conn, task)
    task.finish_task()
    task.finish_task()  # duplicate finish is a no-op
    task.finish_task({"directive": "cancel"})
    task.finish_task({"directive": "cancel"})  # duplicate cancel no-op
    finishes = [
        f
        for f in _sent_frames(conn._ws)
        if f["header"]["action"] == "finish-task"
    ]
    assert len(finishes) == 2
    assert finishes[0]["payload"]["input"] == {}
    assert finishes[1]["payload"]["input"] == {"directive": "cancel"}
    conn.close()


def test_recv_timeout_raises():
    conn, _ = _make_connection()
    task = conn.open_task(timeout=0.05)
    with pytest.raises(TimeoutError):
        task.recv()
    conn.close()


def test_close_broadcasts_and_closes_socket():
    conn, connects = _make_connection(timeout=5)
    task = conn.open_task()
    _started(conn, task)
    conn.close()
    assert conn.closed
    assert connects[0]["ws"].closed
    with pytest.raises(APIConnectionError, match="connection closed"):
        task.recv()
    conn.close()  # idempotent
