# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared realtime transport and independent Live/Omni contracts."""

from __future__ import annotations

import json
import socket
import sys
import threading
import traceback

import pytest
from websocket import WebSocket, WebSocketException, WebSocketTimeoutException

from dashscope.v2 import Client
from dashscope.v2.errors import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
)
from dashscope.v2.resources.audio._ws_session import WsSessionBase


class FrameSocket:
    def __init__(self, inbox=()):
        self.inbox = list(inbox)
        self.sent = []
        self.close_calls = 0

    def settimeout(self, timeout):
        self.timeout = timeout

    def send(self, raw):
        self.sent.append(raw)

    def recv(self):
        raw = self.inbox.pop(0)
        if isinstance(raw, BaseException):
            raise raw
        return raw

    def close(self):
        self.close_calls += 1


@pytest.fixture
def client():
    with Client(api_key="sk-test", workspace="ws-test", timeout=10) as value:
        yield value


@pytest.fixture(params=["live", "realtime"])
def resource(request, client):
    return getattr(client.audio, request.param)


@pytest.mark.parametrize("raw", [b"\x00\xff", "not JSON", '{"type":"error"}'])
def test_transport_delivers_raw_frames_without_protocol_decisions(client, raw):
    ws = FrameSocket([raw])
    with WsSessionBase(
        client,
        model="route",
        connector=lambda u, h: ws,
    ) as transport:
        transport._send_raw(ws, raw)
        assert transport._recv_raw(ws) == raw
    assert ws.sent == [raw]
    assert ws.close_calls == 1


def test_transport_leaves_native_receive_errors_for_the_session(client):
    failure = WebSocketTimeoutException("read timed out")
    ws = FrameSocket([failure])
    with WsSessionBase(
        client,
        model="route",
        connector=lambda u, h: ws,
    ) as transport:
        with pytest.raises(WebSocketTimeoutException) as caught:
            transport._recv_raw(ws)
        assert caught.value is failure
        assert ws.close_calls == 0


def test_sessions_keep_distinct_timeout_exceptions(resource):
    failure = WebSocketTimeoutException("read timed out")
    ws = FrameSocket([failure])
    with resource.connect(model="route", connector=lambda u, h: ws) as session:
        expected = (
            APITimeoutError
            if resource is resource._client.audio.live
            else type(failure)
        )
        with pytest.raises(expected) as caught:
            session.recv_event()
        if expected is APITimeoutError:
            assert caught.value.__cause__ is failure
        else:
            assert caught.value is failure
    assert ws.close_calls == 1


def test_sessions_keep_distinct_idle_eof_contracts(resource):
    ws = FrameSocket([None])
    with resource.connect(model="route", connector=lambda u, h: ws) as session:
        if resource is resource._client.audio.live:
            with pytest.raises(APIConnectionError):
                session.recv_event()
        else:
            assert session.recv_event() is None
    assert ws.close_calls == 1


def test_recoverable_error_keeps_each_session_usable(resource):
    error = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "code": "InvalidParameter",
            "message": "correct the input",
        },
    }
    ws = FrameSocket(
        [
            json.dumps(error),
            '{"type":"session.updated","session":{"id":"s"}}',
        ],
    )
    with resource.connect(model="route", connector=lambda u, h: ws) as session:
        with pytest.raises(BadRequestError):
            session.recv_event()
        assert ws.close_calls == 0
        assert session.recv_event()["type"] == "session.updated"
        session.send_event({"type": "custom", "event_id": "next"})
        assert json.loads(ws.sent[0]) == {"type": "custom", "event_id": "next"}


def test_omni_response_acknowledged_during_send_does_not_remain_pending(
    client,
):
    sent, received = threading.Event(), threading.Event()
    failures = []

    class ReplySocket(FrameSocket):
        def send(self, raw):
            super().send(raw)
            sent.set()
            assert received.wait(
                2,
            ), "response was not read before send returned"

    ws = ReplySocket(
        [
            '{"type":"response.created","response":{"id":"r"}}',
            '{"type":"response.done","response":{"id":"r"}}',
            None,
        ],
    )
    with client.audio.realtime.connect(
        model="omni",
        connector=lambda u, h: ws,
    ) as session:

        def receive():
            try:
                assert sent.wait(2)
                assert session.recv_event()["type"] == "response.created"
                assert session.recv_event()["type"] == "response.done"
            except BaseException as exc:
                failures.append(exc)
            finally:
                received.set()

        receiver = threading.Thread(target=receive, daemon=True)
        receiver.start()
        try:
            session.create_response()
            receiver.join(2)
            assert not receiver.is_alive()
            assert failures == []
            assert session.recv_event() is None
        finally:
            session.close()
            receiver.join(3)


def test_close_sends_control_frame_before_unacknowledged_peer_timeout(
    resource,
):
    local, peer = socket.socketpair()
    ws, remote = WebSocket(), WebSocket()
    ws.sock, ws.connected = local, True
    remote.sock, remote.connected = peer, True
    peer.settimeout(2)
    session = resource.connect(
        model="route",
        connector=lambda u, h: ws,
    ).connect()
    closed = threading.Event()

    def close():
        session.close()
        closed.set()

    closer = threading.Thread(target=close, daemon=True)
    try:
        closer.start()
        frame = remote.recv_frame()
        assert frame.opcode == 8
        assert frame.data == b"\x03\xe8"  # Normal closure, 1000.
        # Do not acknowledge: the close deadline must release the descriptor.
        assert closed.wait(2), "close waited indefinitely for a peer Close"
        assert local.fileno() == -1
    finally:
        remote.shutdown()
        closer.join(3)
        session.close()
        local.close()


def test_close_still_releases_socket_when_deadline_cannot_start(
    resource,
    monkeypatch,
):
    local, peer = socket.socketpair()
    ws = WebSocket()
    ws.sock, ws.connected = local, True
    session = resource.connect(
        model="route",
        connector=lambda u, h: ws,
    ).connect()

    def unavailable(self):
        raise RuntimeError("cannot start a new thread")

    monkeypatch.setattr(threading.Timer, "start", unavailable)
    try:
        session.close()
        assert local.fileno() == -1
        assert session.recv_event() is None
    finally:
        peer.close()
        ws.shutdown()


def test_reconnect_is_rejected_until_old_close_completes(resource):
    entered, release = threading.Event(), threading.Event()
    failures = []

    class ClosingSocket(FrameSocket):
        def close(self):
            entered.set()
            assert release.wait(2)
            super().close()

    old, fresh = ClosingSocket(), FrameSocket()
    sockets = iter([old, fresh])
    session = resource.connect(
        model="route",
        connector=lambda u, h: next(sockets),
    ).connect()

    def close():
        try:
            session.close()
        except BaseException as exc:
            failures.append(exc)

    closer = threading.Thread(target=close, daemon=True)
    try:
        closer.start()
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="closing"):
            session.connect()
        release.set()
        closer.join(2)
        assert not closer.is_alive()
        assert failures == []
        session.connect()
        session.send_event({"type": "custom"})
        assert len(fresh.sent) == 1
        assert fresh.close_calls == 0
    finally:
        release.set()
        closer.join(3)
        session.close()


def test_close_waits_for_inflight_send_before_sending_control_frame(resource):
    entered, detached, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    failures = []

    class DelayedSocket(WebSocket):
        def send(self, payload, opcode=1):
            if opcode == 1:
                entered.set()
                assert release.wait(2)
            return super().send(payload, opcode)

    local, peer = socket.socketpair()
    ws, remote = DelayedSocket(), WebSocket()
    ws.sock, ws.connected = local, True
    remote.sock, remote.connected = peer, True
    peer.settimeout(2)
    session = resource.connect(
        model="route",
        connector=lambda u, h: ws,
    ).connect()
    lock = session._lock

    class ObservedLock:
        def __enter__(self):
            return lock.__enter__()

        def __exit__(self, *args):
            # Synchronize at detach without relying on thread scheduling delays.
            if session._ws is None:
                detached.set()
            return lock.__exit__(*args)

    session._lock = ObservedLock()

    def send():
        try:
            session.send_event({"type": "custom", "event_id": "last"})
        except BaseException as exc:
            failures.append(exc)

    sender = threading.Thread(target=send, daemon=True)
    closer = threading.Thread(target=session.close, daemon=True)
    try:
        sender.start()
        assert entered.wait(2)
        closer.start()
        assert detached.wait(2)
        release.set()
        first = remote.recv_frame()
        assert first.opcode == 1
        assert json.loads(first.data) == {"type": "custom", "event_id": "last"}
        final = remote.recv_frame()
        assert final.opcode == 8
        assert final.data == b"\x03\xe8"
        remote.send_close()
        sender.join(2)
        closer.join(2)
        assert not sender.is_alive()
        assert not closer.is_alive()
        assert failures == []
    finally:
        release.set()
        remote.shutdown()
        sender.join(3)
        if closer.ident is not None:
            closer.join(3)
        session.close()
        local.close()


def test_remote_close_reply_cannot_overtake_inflight_data(resource):
    entered, closing, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    failures = []

    class DelayedSocket(WebSocket):
        def send(self, payload, opcode=1):
            if opcode == 1:
                entered.set()
                assert release.wait(2)
            return super().send(payload, opcode)

        def send_close(self, *args, **kwargs):
            closing.set()
            return super().send_close(*args, **kwargs)

    local, peer = socket.socketpair()
    ws, remote = DelayedSocket(), WebSocket()
    ws.sock, ws.connected = local, True
    remote.sock, remote.connected = peer, True
    peer.settimeout(2)
    session = resource.connect(
        model="route",
        connector=lambda u, h: ws,
    ).connect()

    def send():
        try:
            session.send_event({"type": "custom", "event_id": "last"})
        except BaseException as exc:
            failures.append(exc)

    def receive():
        try:
            session.recv_event()
        except APIConnectionError:
            pass  # Live still reports an unexpected remote transport close.

    sender = threading.Thread(target=send, daemon=True)
    receiver = threading.Thread(target=receive, daemon=True)
    try:
        sender.start()
        assert entered.wait(2)
        receiver.start()
        remote.send_close()
        assert closing.wait(2)
        release.set()
        first = remote.recv_frame()
        assert first.opcode == 1
        assert json.loads(first.data) == {"type": "custom", "event_id": "last"}
        assert remote.recv_frame().opcode == 8
        sender.join(2)
        receiver.join(2)
        assert not sender.is_alive()
        assert not receiver.is_alive()
        assert failures == []
    finally:
        release.set()
        remote.shutdown()
        sender.join(3)
        if receiver.ident is not None:
            receiver.join(3)
        session.close()
        local.close()


def test_live_cancel_releases_socket_without_a_close_handshake(client):
    local, peer = socket.socketpair()
    ws = WebSocket()
    ws.sock, ws.connected = local, True
    peer.settimeout(0.5)
    session = client.audio.live.connect(
        model="route",
        connector=lambda u, h: ws,
    ).connect()
    try:
        session.cancel()
        assert peer.recv(1024) == b""
        assert local.fileno() == -1
        assert session.recv_event() is None
        session.cancel()
        session.close()
    finally:
        peer.close()
        session.cancel()
        local.close()


def test_local_close_wakes_a_native_blocked_receive(resource):
    entered, closed = threading.Event(), threading.Event()
    results, failures = [], []

    class ObservedSocket(WebSocket):
        def recv(self):
            entered.set()
            return super().recv()

    local, peer = socket.socketpair()
    ws = ObservedSocket()
    ws.sock, ws.connected = local, True
    session = resource.connect(
        model="route",
        connector=lambda u, h: ws,
    ).connect()

    def receive():
        try:
            results.append(session.recv_event())
        except BaseException as exc:
            failures.append(exc)

    def close():
        session.close()
        closed.set()

    receiver = threading.Thread(target=receive, daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    receiver.start()
    try:
        assert entered.wait(2)
        closer.start()
        assert closed.wait(2), "close waited for the blocked receive"
        receiver.join(2)
        assert not receiver.is_alive()
        assert results == [None]
        assert failures == []
        assert local.fileno() == -1
    finally:
        peer.close()
        receiver.join(3)
        if closer.ident is not None:
            closer.join(3)
        session.close()
        local.close()


def test_local_close_wakes_a_native_blocked_send(resource):
    entered, closed = threading.Event(), threading.Event()
    failures = []

    class ObservedSocket(WebSocket):
        def _send(self, raw):
            entered.set()
            return super()._send(raw)

    local, peer = socket.socketpair()
    local.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    ws = ObservedSocket()
    ws.sock, ws.connected = local, True
    session = resource.connect(
        model="route",
        connector=lambda u, h: ws,
    ).connect()

    def send():
        try:
            session.send_event({"type": "custom", "data": "x" * 1024 * 1024})
        except BaseException as exc:
            failures.append(exc)

    def close():
        session.close()
        closed.set()

    sender = threading.Thread(target=send, daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    sender.start()
    try:
        assert entered.wait(2)
        closer.start()
        assert closed.wait(2), "close waited for the blocked send"
        sender.join(2)
        if sender.is_alive():
            pytest.fail(
                "send did not stop:\n"
                + "".join(
                    traceback.format_stack(
                        sys._current_frames()[sender.ident],
                    ),
                ),
            )
        assert failures
        expected = (
            APIConnectionError
            if resource is resource._client.audio.live
            else (OSError, WebSocketException)
        )
        assert all(isinstance(error, expected) for error in failures)
        assert local.fileno() == -1
    finally:
        peer.close()
        sender.join(3)
        if closer.ident is not None:
            closer.join(3)
        session.close()
        local.close()
