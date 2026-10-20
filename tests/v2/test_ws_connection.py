# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""WsConnection: multi-task demux over one shared duplex socket."""

# pylint: disable=protected-access

import json
import queue
import threading

import pytest

from dashscope.v2.errors import APIConnectionError, DashScopeError
from dashscope.v2.transport.ws import WsEventError, WsSession
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


@pytest.fixture
def bounded_connection():
    """Always stop and join readers, including when a contract assertion fails."""
    connections = []

    def make(ws=None, timeout=0.5):
        ws = QueueWebSocket() if ws is None else ws
        conn = WsConnection(
            url=URL,
            connector=lambda url, headers: ws,
            timeout=timeout,
        )
        connections.append(conn)
        return conn

    yield make
    for conn in connections:
        conn.close()
        conn._reader.join(timeout=1)
        assert not conn._reader.is_alive(), "websocket reader leaked"


@pytest.mark.parametrize("transport", [WsSession, WsConnection])
def test_initial_timeout_setup_failure_closes_socket(transport):
    class BrokenTimeoutSocket(QueueWebSocket):
        def settimeout(self, seconds):
            raise OSError("timeout setup failed")

    ws = BrokenTimeoutSocket()
    try:
        with pytest.raises(OSError, match="timeout setup failed"):
            transport(url=URL, connector=lambda url, headers: ws, timeout=0.5)
        assert ws.closed
    finally:
        ws.close()


@pytest.mark.parametrize(
    "error_type",
    [OSError, KeyboardInterrupt, SystemExit],
)
def test_shared_blocking_mode_setup_failure_closes_socket(error_type):
    error = error_type("blocking mode failed")

    class BrokenBlockingSocket(QueueWebSocket):
        def settimeout(self, seconds):
            super().settimeout(seconds)
            if seconds is None:
                raise error

    ws = BrokenBlockingSocket()
    try:
        with pytest.raises(error_type) as caught:
            WsConnection(
                url=URL,
                connector=lambda url, headers: ws,
                timeout=0.5,
            )
        assert caught.value is error
        assert (
            ws.closed
        ), "post-handshake initialization must release the socket"
    finally:
        ws.close()


@pytest.mark.parametrize("transport", [WsSession, WsConnection])
def test_connector_failure_creates_no_reader(transport):
    before = set(threading.enumerate())

    def connector(url, headers):
        raise OSError("handshake failed")

    with pytest.raises(OSError, match="handshake failed"):
        transport(url=URL, connector=connector, timeout=0.5)
    assert not [
        thread
        for thread in threading.enumerate()
        if thread not in before and thread.name == "dashscope-ws-reader"
    ]


@pytest.mark.parametrize("action", ["run", "binary", "continue", "finish"])
def test_shared_send_failure_closes_and_wakes_other_tasks(
    bounded_connection,
    action,
):
    class BrokenSendSocket(QueueWebSocket):
        fail_send = False

        def send(self, payload, opcode=1):
            if self.fail_send:
                raise OSError("send failed")
            super().send(payload, opcode=opcode)

    ws = BrokenSendSocket()
    conn = bounded_connection(ws)
    task, other = conn.open_task(), conn.open_task()
    _started(conn, other)
    if action != "run":
        _started(conn, task)
    ws.fail_send = True
    with pytest.raises(OSError, match="send failed"):
        if action == "run":
            _started(conn, task)
        elif action == "binary":
            task.send_binary(b"audio")
        elif action == "continue":
            task.continue_task({"context": "next"})
        else:
            task.finish_task()
    assert conn.closed and ws.closed
    with pytest.raises(APIConnectionError):
        other.recv()
    with pytest.raises(RuntimeError, match="closed"):
        conn.open_task()
    task.close()
    other.close()


@pytest.mark.parametrize("raw", [None, "", OSError("receive failed")])
def test_shared_premature_eof_and_recv_failure_stop_reader(
    bounded_connection,
    raw,
):
    conn = bounded_connection()
    tasks = [conn.open_task(), conn.open_task()]
    for task in tasks:
        _started(conn, task)
    conn._ws.inbox.put(raw)
    for task in tasks:
        with pytest.raises(
            OSError if isinstance(raw, OSError) else APIConnectionError,
        ):
            task.recv()
    assert conn.closed and conn._ws.closed


@pytest.mark.parametrize(
    "raw, error",
    [
        pytest.param("{", json.JSONDecodeError, id="invalid-json"),
        pytest.param("[]", WsEventError, id="non-object"),
        pytest.param("null", WsEventError, id="null-event"),
        pytest.param("{}", WsEventError, id="missing-header"),
        pytest.param('{"header": []}', WsEventError, id="invalid-header"),
        pytest.param('{"header": null}', WsEventError, id="null-header"),
        pytest.param(
            '{"header": {"event": "result-generated"}}',
            WsEventError,
            id="missing-task-id",
        ),
        *[
            pytest.param(
                _event("result-generated", task_id),
                WsEventError,
                id=name,
            )
            for task_id, name in [
                ([], "list-task-id"),
                ({}, "object-task-id"),
                (None, "null-task-id"),
                (1, "numeric-task-id"),
                (True, "boolean-task-id"),
                ("", "empty-task-id"),
            ]
        ],
    ],
)
def test_shared_invalid_control_frame_fails_all_tasks(
    bounded_connection,
    raw,
    error,
):
    conn = bounded_connection()
    tasks = [conn.open_task(), conn.open_task()]
    for task in tasks:
        _started(conn, task)
    conn._ws.inbox.put(raw)
    for task in tasks:
        for _ in range(2):
            with pytest.raises(error):
                task.recv()
    assert conn.closed and conn._ws.closed


@pytest.mark.parametrize("terminal", ["task-finished", "task-failed", "close"])
def test_both_binary_owners_release_for_next_task(
    bounded_connection,
    terminal,
):
    conn = bounded_connection()
    first, second = conn.open_task(), conn.open_task()
    _started(conn, first)
    _started(conn, second)
    first.send_binary(b"first")
    first.claim_binary_downlink()
    if terminal == "close":
        first.close()
    else:
        conn._ws.inbox.put(_event(terminal, first.task_id))
        if terminal == "task-failed":
            with pytest.raises(DashScopeError):
                first.recv()
        else:
            first.recv_event()
    second.send_binary(b"second")
    second.claim_binary_downlink()
    conn._ws.inbox.put(b"second-output")
    assert second.recv() == b"second-output"
    assert not conn.closed


@pytest.mark.parametrize("terminal", ["task-finished", "close"])
def test_inactive_task_cannot_reclaim_binary_downlink(
    bounded_connection,
    terminal,
):
    conn = bounded_connection()
    task, next_task = conn.open_task(), conn.open_task()
    _started(conn, task)
    _started(conn, next_task)
    task.claim_binary_downlink()
    if terminal == "close":
        task.close()
    else:
        conn._ws.inbox.put(_event(terminal, task.task_id))
        task.recv_event()
    with pytest.raises(RuntimeError):
        task.claim_binary_downlink()
    next_task.claim_binary_downlink()
    conn._ws.inbox.put(b"next-output")
    assert next_task.recv() == b"next-output"


@pytest.mark.parametrize("failure", ["close", "recv"])
def test_connection_shutdown_releases_binary_owners(
    bounded_connection,
    failure,
):
    conn = bounded_connection()
    task = conn.open_task()
    _started(conn, task)
    task.send_binary(b"audio")
    task.claim_binary_downlink()
    if failure == "close":
        conn.close()
    else:
        conn._ws.inbox.put(OSError("receive failed"))
    with pytest.raises(APIConnectionError if failure == "close" else OSError):
        task.recv()
    assert conn._binary_uplink_owner is None
    assert conn._binary_downlink_owner is None
    with pytest.raises(RuntimeError, match="closed"):
        task.claim_binary_downlink()
    assert conn._binary_downlink_owner is None


@pytest.mark.parametrize("terminal", ["task-finished", "task-failed"])
@pytest.mark.parametrize("claim", [False, True])
def test_reader_releases_terminal_owners_before_consumer_finishes(
    bounded_connection,
    terminal,
    claim,
):
    conn = bounded_connection()
    first, second = conn.open_task(), conn.open_task()
    _started(conn, first)
    _started(conn, second)
    first.send_binary(b"first-input")
    first.claim_binary_downlink()
    conn._ws.inbox.put(_event("result-generated", first.task_id, {"n": 1}))
    conn._ws.inbox.put(b"first-output")
    conn._ws.inbox.put(_event(terminal, first.task_id))
    conn._ws.inbox.put(
        _event("task-started", first.task_id),
    )  # Late duplicate.
    conn._ws.inbox.put(_event("result-generated", second.task_id))
    second.recv_event()  # The reader has handled first's terminal event.
    assert not first.finished  # The consumer must still drain its frames.
    assert conn._binary_uplink_owner is None
    assert conn._binary_downlink_owner is None
    with pytest.raises(RuntimeError, match="not active"):
        first.claim_binary_downlink()
    with pytest.raises(RuntimeError, match="not active"):
        first.send_binary(b"too late")
    second.send_binary(b"second-input")
    if claim:
        second.claim_binary_downlink()
    conn._ws.inbox.put(b"second-output")
    assert second.recv() == b"second-output"
    assert first.recv_event()["payload"] == {"n": 1}
    assert first.recv() == b"first-output"
    if terminal == "task-failed":
        with pytest.raises(DashScopeError):
            first.recv()
    else:
        assert first.recv_event()["header"]["event"] == terminal
    assert first.finished and first.recv() is None
    # Consuming the old terminal event must not affect the new owners.
    assert conn._binary_uplink_owner is second
    assert conn._binary_downlink_owner is second
    assert not conn.closed


@pytest.mark.parametrize("direction", ["uplink", "downlink"])
def test_binary_owner_must_be_attached_to_this_connection(
    bounded_connection,
    direction,
):
    conn, other = bounded_connection(), bounded_connection()
    task, foreign = conn.open_task(), other.open_task()
    foreign._task_id = (
        task.task_id
    )  # Matching ids alone do not imply attachment.
    with pytest.raises(RuntimeError, match="not active"):
        getattr(conn, "_claim_binary_" + direction)(foreign)
    assert getattr(conn, "_binary_" + direction + "_owner") is None
    task.claim_binary_downlink()


@pytest.mark.parametrize("close_target", ["task", "connection"])
def test_close_wakes_already_blocked_receiver(
    bounded_connection,
    close_target,
):
    entered = threading.Event()

    class ObservedQueue(queue.Queue):
        def get(self, *args, **kwargs):
            entered.set()
            return super().get(*args, **kwargs)

    conn = bounded_connection(timeout=None)
    task = conn.open_task()
    _started(conn, task)
    task._queue = ObservedQueue()
    errors = []

    def receive():
        try:
            task.recv()
        except BaseException as exc:
            errors.append(exc)

    receiver = threading.Thread(
        target=receive,
        name="test-ws-receiver",
        daemon=True,
    )
    receiver.start()
    try:
        assert entered.wait(timeout=1)
        (task if close_target == "task" else conn).close()
        receiver.join(timeout=1)
        assert not receiver.is_alive(), "close did not wake the receiver"
        expected = (
            RuntimeError if close_target == "task" else APIConnectionError
        )
        assert len(errors) == 1 and isinstance(errors[0], expected), errors
    finally:
        # Test-only fallback also bounds cleanup if production close is broken.
        task._queue.put(("error", RuntimeError("test cleanup")))
        receiver.join(timeout=1)
        task.close()
        assert not receiver.is_alive()


def test_close_uses_bounded_join_if_socket_does_not_wake_reader(
    bounded_connection,
    monkeypatch,
):
    entered, release = threading.Event(), threading.Event()

    class SlowClosingSocket(QueueWebSocket):
        def recv(self):
            entered.set()
            assert release.wait(timeout=5)
            return None

    conn = bounded_connection(SlowClosingSocket())
    task = conn.open_task()
    original_join = conn._reader.join
    timeouts = []

    def observe_join(timeout=None):
        timeouts.append(timeout)
        assert timeout is not None and 0 < timeout <= 2
        original_join(timeout=timeout)

    monkeypatch.setattr(conn._reader, "join", observe_join)
    try:
        assert entered.wait(timeout=1)
        conn.close()
        assert timeouts == [2]
        assert (
            conn._reader.is_alive()
        )  # close() returned without an unbounded wait.
        with pytest.raises(APIConnectionError):
            task.recv()
    finally:
        release.set()
        original_join(timeout=1)
        assert not conn._reader.is_alive()


@pytest.mark.parametrize(
    "shutdown",
    ["task", "connection", "recv", "invalid-json"],
)
def test_error_is_persistent_but_preserves_earlier_frames(
    bounded_connection,
    shutdown,
):
    conn = bounded_connection()
    task, barrier = conn.open_task(), conn.open_task()
    _started(conn, task)
    conn._ws.inbox.put(_event("result-generated", task.task_id, {"n": 1}))
    conn._ws.inbox.put(b"queued-audio")
    conn._ws.inbox.put(_event("result-generated", barrier.task_id))
    barrier.recv_event()  # Earlier frames have reached the task's queue.
    if shutdown in ("task", "connection"):
        (task if shutdown == "task" else conn).close()
        expected = RuntimeError if shutdown == "task" else APIConnectionError
    else:
        conn._ws.inbox.put(
            OSError("receive failed") if shutdown == "recv" else "{",
        )
        conn._reader.join(timeout=1)
        assert not conn._reader.is_alive()
        expected = OSError if shutdown == "recv" else json.JSONDecodeError
    assert task.recv_event()["payload"] == {"n": 1}
    assert task.recv() == b"queued-audio"
    with pytest.raises(expected) as first:
        task.recv()
    task.close()  # Cleanup must not overwrite an earlier connection error.
    with pytest.raises(expected) as again:
        task.recv()
    assert again.value is first.value


def test_task_handshake_timeout_detaches_without_killing_connection(
    bounded_connection,
):
    conn = bounded_connection()
    task = conn.open_task(timeout=0.02)
    with pytest.raises(TimeoutError):
        task.run_task(
            task_group="audio",
            task="asr",
            function="recognition",
            model="test",
        )
    assert task.task_id not in conn._tasks
    assert not conn.closed
    next_task = conn.open_task()
    _started(conn, next_task)
    next_task.send_binary(b"still usable")


def test_task_timeout_is_isolated_and_explicit_close_releases_owners(
    bounded_connection,
):
    conn = bounded_connection()
    first, second = conn.open_task(timeout=0.02), conn.open_task()
    _started(conn, first)
    _started(conn, second)
    first.send_binary(b"first")
    first.claim_binary_downlink()
    with pytest.raises(TimeoutError):
        first.recv()
    first.close()
    second.send_binary(b"second")
    second.claim_binary_downlink()
    conn._ws.inbox.put(_event("task-finished", second.task_id))
    assert second.recv_event()["header"]["event"] == "task-finished"
    assert not conn.closed


@pytest.mark.parametrize("claim", [False, True])
def test_binary_immediately_after_started_is_not_dropped(
    bounded_connection,
    claim,
):
    """The reader may read audio before run_task consumes task-started."""
    routed = threading.Event()

    class FastReplySocket(QueueWebSocket):
        def send(self, payload, opcode=1):
            super().send(payload, opcode=opcode)
            task_id = json.loads(payload)["header"]["task_id"]
            self.inbox.put(_event("task-started", task_id))
            self.inbox.put(b"first-output")
            self.inbox.put(_event("result-generated", "reader-barrier"))
            assert routed.wait(timeout=1)

    conn = bounded_connection(FastReplySocket())
    task = conn.open_task()
    if claim:
        task.claim_binary_downlink()  # TTS claims before run_task().
    original_route = conn._route_event

    def observe_route(event):
        original_route(event)
        if event["header"].get("task_id") == "reader-barrier":
            routed.set()

    conn._route_event = observe_route
    task.run_task(
        task_group="audio",
        task="tts",
        function="SpeechSynthesizer",
        model="test",
    )
    assert task.recv() == b"first-output"


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt, SystemExit],
)
@pytest.mark.parametrize("after_start", [False, True])
def test_reader_start_failure_closes_connected_socket(
    monkeypatch,
    error_type,
    after_start,
):
    ws = QueueWebSocket()
    error = error_type("cannot start reader")
    original_start = threading.Thread.start
    readers = []

    def fail_start(self):
        readers.append(self)
        if after_start:
            original_start(self)
        raise error

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        with pytest.raises(error_type) as caught:
            WsConnection(
                url=URL,
                connector=lambda url, headers: ws,
                timeout=0.5,
            )
        assert caught.value is error
        assert ws.closed
        assert all(not reader.is_alive() for reader in readers)
    finally:
        ws.close()
        for reader in readers:
            if reader.is_alive():
                reader.join(timeout=1)


def test_independent_session_close_wakes_blocked_receiver():
    entered = threading.Event()

    class ObservedSocket(QueueWebSocket):
        def recv(self):
            entered.set()
            return super().recv()

    ws = ObservedSocket()
    session = WsSession(
        url=URL,
        connector=lambda url, headers: ws,
        timeout=0.5,
    )
    errors = []

    def receive():
        try:
            session.recv()
        except BaseException as exc:
            errors.append(exc)

    receiver = threading.Thread(
        target=receive,
        name="test-session-receiver",
        daemon=True,
    )
    receiver.start()
    try:
        assert entered.wait(timeout=1)
        session.close()
        receiver.join(timeout=1)
        assert not receiver.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], APIConnectionError)
        assert ws.closed
    finally:
        session.close()
        ws.close()
        receiver.join(timeout=1)
        assert not receiver.is_alive()
