# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 real-world scenario: a concurrent realtime meeting
transcription harness.

This is not "how to open one TingWu realtime session". It is an
industrial-grade base for a meeting platform that must transcribe many
rooms at once and keep each meeting's state and transcript intact even
when a socket drops. It stitches four production concerns into one
runnable blueprint:

1. **Concurrency that respects the protocol.** Realtime TingWu is NOT
   multiplexable: its wire protocol carries no task id, so one websocket
   is exactly one session and closes when the session ends (see the
   module note in dashscope/v2/transport/ws_connection.py). Concurrency
   therefore means one *dedicated* connection + worker per meeting, and
   the pool size is the admission control that caps how many live
   sockets -- and how much blast radius -- you hold open at once.
2. **Local state machine per meeting.** Every room moves through
   PENDING -> CONNECTING -> LISTENING -> TRANSCRIBING -> COMPLETED, with
   RECONNECTING and DEGRADED as managed off-ramps, so a supervisor can
   observe and report on hundreds of rooms uniformly.
3. **Cloud/local split for the transcript.** The cloud recognizes; the
   SDK side *assembles*. TingWu streams incremental ``recognize-result``
   events that re-send the same ``sentenceId`` with growing text until
   it is final. ``TranscriptAssembler`` merges those partials locally,
   which also makes the transcript survive a mid-stream reconnect.
4. **Rigid resilience + isolation.** A dropped socket
   (``APIConnectionError``) reconnects with exponential backoff; a
   business ``task-failed`` (bad app id, content policy) is *not*
   blindly retried -- it degrades that one room. Either way the failure
   is contained: one room falling over never takes down the others.

It runs **fully offline and deterministically by default** by injecting
a scripted duplex socket through TingWu's ``connector=`` seam (the same
seam tests/v2/test_tingwu.py uses), so you can *see* the concurrency,
reconnect and degradation without a microphone or a token. Set
``MEETING_HARNESS_LIVE=1`` to point the exact same engine at the real
gateway with ``DASHSCOPE_API_KEY`` + ``DASHSCOPE_TINGWU_APP_ID``.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dashscope
from dashscope.v2 import errors

# ---- configuration (override via env) ---- #
MODEL = os.environ.get("DASHSCOPE_TINGWU_MODEL", "tingwu-turbo")
APP_ID = os.environ.get("DASHSCOPE_TINGWU_APP_ID", "your-app-id")
TERMINOLOGY = os.environ.get("DASHSCOPE_TINGWU_TERMINOLOGY") or None
LIVE = os.environ.get("MEETING_HARNESS_LIVE") == "1"
AUDIO_FILE = os.environ.get(
    "DASHSCOPE_DEMO_AUDIO",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "data",
        "asr_example.wav",
    ),
)

# Bounded worker pool == bounded live sockets == admission control.
MAX_CONCURRENT = int(os.environ.get("MEETING_MAX_CONCURRENT", "4"))
RECONNECT_ATTEMPTS = 2  # per-meeting reconnects before degrading
BASE_DELAY = 0.0  # 0 keeps the offline demo instant; use ~0.5 in prod
SESSION_TIMEOUT = 30.0

SAMPLE_RATE = 16000
FRAME_BYTES = 3200  # 100 ms of 16 kHz 16-bit mono PCM
FRAME_INTERVAL = 0.1  # realtime cadence for the live path only

# Meeting states (a plain string enum keeps the report dependency-free).
PENDING = "pending"
CONNECTING = "connecting"
LISTENING = "listening"
TRANSCRIBING = "transcribing"
COMPLETED = "completed"
RECONNECTING = "reconnecting"
DEGRADED = "degraded"


# ===================== local transcript assembly ===================== #
class TranscriptAssembler:
    """Merge streaming partials into final segments, keyed by sentenceId.

    TingWu realtime re-sends the same ``sentenceId`` with growing text
    until the sentence is final. Keeping the latest text per id and
    preserving first-seen order is the local half of the cloud/local
    split -- and because the assembler outlives a single socket, the
    transcript keeps its place across a mid-stream reconnect.
    """

    def __init__(self):
        self._order = []
        self._text = {}

    def feed(self, sentence):
        """Fold one recognize-result sentence in; return its id/text."""
        sid = sentence.get("sentenceId")
        if sid is None:  # no id: fall back to positional so nothing is lost
            sid = len(self._order)
        if sid not in self._text:
            self._order.append(sid)
        self._text[sid] = sentence.get("text", "")
        return sid, self._text[sid], bool(sentence.get("end"))

    def segments(self):
        return [self._text[sid] for sid in self._order if self._text[sid]]

    def text(self):
        return "".join(self.segments())


# ===================== per-meeting state machine ===================== #
class MeetingSession:
    """One meeting: identity, state, transcript, and retry bookkeeping."""

    def __init__(self, meeting_id, room, frames, pace=False):
        self.meeting_id = meeting_id
        self.room = room
        self.frames = frames
        self.pace = pace  # True only for the live path (realtime cadence)
        self.state = PENDING
        self.assembler = TranscriptAssembler()
        self.attempts = 0
        self.reconnects = 0
        self.error = None

    def transition(self, state, note=""):
        """Move to ``state`` and log it, tagged so interleaving is readable.

        Meetings run on worker threads, so their log lines interleave;
        the ``[meeting_id]`` prefix is what makes a concurrent run still
        traceable per room.
        """
        self.state = state
        suffix = f" :: {note}" if note else ""
        print(f"  [{self.meeting_id}] -> {state}{suffix}")


def _is_reconnectable(exc):
    """Only a dropped/timed-out socket is worth reopening.

    A ``task-failed`` event (bad app id, content policy, parameter error)
    surfaces as a non-connection DashScopeError; reconnecting would just
    fail again, so those degrade instead of retry. A raw socket timeout
    from the websocket lib is not in the DashScope hierarchy, so it is
    classified by name.
    """
    if isinstance(exc, (errors.APIConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__
    return "Timeout" in name or "ConnectionClosed" in name


def _handle_event(session, event):
    output = (event.get("payload") or {}).get("output") or {}
    if output.get("action") != "recognize-result":
        return  # speech-end just terminates the iterator; nothing to store
    sentence = output.get("sentence") or {}
    _sid, text, is_end = session.assembler.feed(sentence)
    marker = "final" if is_end else "part "
    print(f"  [{session.meeting_id}] {marker} | {text}")


def _pump_audio(session, realtime):
    """Send every frame, then signal end of audio (stop).

    Production variant: to transcribe *live* rather than after stop,
    drain ``realtime.events()`` on a reader thread while frames are still
    being sent -- the socket is duplex. The sequential shape here matches
    01_atomic_capabilities/audio_speech/tingwu.py and keeps the offline
    mock deterministic.
    """
    for frame in session.frames:
        realtime.send_audio_frame(frame)
        if session.pace:  # only the live path paces to realtime cadence
            time.sleep(FRAME_INTERVAL)
    realtime.stop()


def _run_once(session, gateway):
    """Open one dedicated socket and drive the meeting over it."""
    session.transition(CONNECTING)
    with _open_realtime(gateway, session.meeting_id) as realtime:
        # __enter__ already waited for task-started, so audio may flow.
        session.transition(LISTENING)
        _pump_audio(session, realtime)
        session.transition(TRANSCRIBING)
        for event in realtime.events():
            _handle_event(session, event)


def run_meeting(session, gateway):
    """Drive one meeting to a terminal state; never raises.

    Containing every failure inside the per-meeting worker is what makes
    the supervisor's isolation guarantee real: a socket drop reconnects
    with backoff, anything else degrades this room and returns, leaving
    the other meetings untouched.
    """
    while True:
        session.attempts += 1
        try:
            _run_once(session, gateway)
            session.transition(
                COMPLETED,
                f"{len(session.assembler.segments())} segment(s)",
            )
            return session
        except Exception as exc:  # isolated per meeting by design
            session.error = exc
            reconnectable = _is_reconnectable(exc)
            if reconnectable and session.reconnects < RECONNECT_ATTEMPTS:
                session.reconnects += 1
                delay = BASE_DELAY * (2 ** (session.reconnects - 1))
                session.transition(
                    RECONNECTING,
                    f"{type(exc).__name__} backoff {delay:.2f}s "
                    f"(attempt {session.reconnects}/{RECONNECT_ATTEMPTS})",
                )
                if delay:
                    time.sleep(delay)
                continue
            reason = type(exc).__name__
            if not reconnectable:
                reason += " (business failure, not reconnectable)"
            session.transition(DEGRADED, f"{reason}: {exc}")
            return session


def _open_realtime(gateway, meeting_id):
    client = gateway.client()
    return client.tingwu.realtime(
        model=MODEL,
        app_id=gateway.app_id,
        format="pcm",
        sample_rate=SAMPLE_RATE,
        terminology=TERMINOLOGY,
        connector=gateway.connector(meeting_id),  # None => real websocket
        timeout=SESSION_TIMEOUT,
    )


# ===================== concurrency supervisor base ===================== #
class MeetingSupervisor:
    """Run many meetings concurrently over bounded dedicated sockets.

    The pool size is deliberately the concurrency *and* the admission
    cap: because each realtime session owns a socket + worker, holding
    the pool bounded is what keeps a burst of join requests from opening
    an unbounded number of live connections.
    """

    def __init__(self, gateway, max_concurrent):
        self._gateway = gateway
        self._max = max_concurrent

    def run(self, meetings):
        order = {m.meeting_id: i for i, m in enumerate(meetings)}
        results = []
        with ThreadPoolExecutor(max_workers=self._max) as pool:
            futures = [
                pool.submit(run_meeting, meeting, self._gateway)
                for meeting in meetings
            ]
            for future in as_completed(futures):
                results.append(future.result())  # run_meeting never raises
        results.sort(key=lambda s: order[s.meeting_id])
        return results


# ===================== scripted duplex mock gateway ===================== #
def _event(name, output=None):
    payload = {"output": output} if output is not None else {}
    return json.dumps(
        {
            "header": {"event": name, "task_id": "t", "request_id": "r"},
            "payload": payload,
        },
    )


def _recognize(text, sentence_id, end):
    return _event(
        "result-generated",
        {
            "action": "recognize-result",
            "sentence": {
                "text": text,
                "sentenceId": sentence_id,
                "end": end,
            },
        },
    )


_STARTED = _event("task-started")
_SPEECH_END = _event("result-generated", {"action": "speech-end"})


class FakeDuplexSocket:
    """A scripted duplex websocket that replays a canned event inbox.

    Mirrors tests/v2/test_tingwu.py's FakeWebSocket. ``recv`` returns the
    next JSON frame, or ``None`` once the inbox is exhausted -- which the
    SDK maps to ``APIConnectionError`` ("closed before task-finished"),
    so an inbox that stops short of ``speech-end`` simulates a mid-stream
    socket drop and drives the reconnect path.
    """

    def __init__(self, inbox):
        self._inbox = list(inbox)
        self.sent = []
        self.closed = False

    # pylint: disable=unused-argument
    def send(self, payload, opcode=1):
        # opcode must stay named ``opcode``: ws.py calls send(data,
        # opcode=2) for binary audio frames.
        self.sent.append(payload)

    def recv(self):
        if not self._inbox:
            return None  # simulate a dropped socket / premature EOF
        return self._inbox.pop(0)

    def settimeout(self, _seconds):
        pass

    # pylint: enable=unused-argument

    def close(self):
        self.closed = True


class MockMeetingGateway:
    """Serve a per-meeting, per-attempt event script through ``connector=``.

    Each call to the connector builds a fresh ``FakeDuplexSocket`` from
    the next scripted inbox for that meeting, so a meeting whose first
    attempt drops can be given a completing second attempt -- exactly the
    reconnect-and-resume flow the harness is meant to survive.
    """

    def __init__(self, scripts, app_id="app-mock"):
        self.app_id = app_id
        self._scripts = scripts  # meeting_id -> list of inboxes (per attempt)
        self._attempts = {}
        self._client = dashscope.Client(
            api_key="sk-mock",
            workspace="ws-demo",
            region="cn-beijing",
        )

    def client(self):
        return self._client

    def connector(self, meeting_id):
        def _connector(_url, _headers):
            return FakeDuplexSocket(self._next_inbox(meeting_id))

        return _connector

    def _next_inbox(self, meeting_id):
        attempt = self._attempts.get(meeting_id, 0)
        self._attempts[meeting_id] = attempt + 1
        script = self._scripts[meeting_id]
        return script[min(attempt, len(script) - 1)]


class LiveGateway:
    """Real gateway: no connector (the SDK opens a real websocket)."""

    def __init__(self, app_id):
        self.app_id = app_id
        self._client = dashscope.Client()

    def client(self):
        return self._client

    def connector(self, _meeting_id):
        return None


# ===================== audio frame source ===================== #
def _mock_frames(count=3):
    return [bytes(FRAME_BYTES) for _ in range(count)]


def _read_pcm(path):
    frames = []
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(FRAME_BYTES)
            if not chunk:
                break
            frames.append(chunk)
    return frames


# ===================== report ===================== #
def _report(results):
    print("\n" + "=" * 70)
    print("会议托管报告 / Meeting Supervisor Report")
    print("=" * 70)
    for session in results:
        full = session.assembler.text()
        preview = full[:36] + ("…" if len(full) > 36 else "")
        print(
            f"  {session.meeting_id:<12} state={session.state:<13} "
            f"attempts={session.attempts} reconnects={session.reconnects} "
            f"segments={len(session.assembler.segments())}",
        )
        print(f"      room      : {session.room}")
        print(f"      transcript: {preview or '(none)'}")
        if session.error is not None:
            print(f"      last error: {type(session.error).__name__}")
    completed = sum(1 for s in results if s.state == COMPLETED)
    degraded = sum(1 for s in results if s.state == DEGRADED)
    print("-" * 70)
    print(
        f"  {completed} completed / {degraded} degraded / "
        f"{len(results)} total -- one room failing never took down "
        f"the others.",
    )
    print("=" * 70)


# ===================== demo scenarios ===================== #
def _offline_scripts():
    """Four rooms that together exercise every resilience path.

    alpha/beta are clean; gamma drops mid-stream on attempt 0 and resumes
    on attempt 1 (the partial sentence ``s2`` is completed after the
    reconnect, proving the assembler survives it); delta hits a business
    ``task-failed`` that must degrade, not reconnect.
    """
    return {
        "room-alpha": [
            [
                _STARTED,
                _recognize("大家好，欢迎参加本次评审。", "s1", True),
                _recognize("我们先同步一下进度。", "s2", True),
                _SPEECH_END,
            ],
        ],
        "room-beta": [
            [
                _STARTED,
                _recognize("线上的同学能听到吗？", "s1", True),
                _recognize("可以，我们开始吧。", "s2", True),
                _recognize("先请产品同学介绍背景。", "s3", True),
                _SPEECH_END,
            ],
        ],
        "room-gamma": [
            # attempt 0: socket drops mid-sentence (no speech-end).
            [
                _STARTED,
                _recognize("这个季度的目标很明确。", "s1", True),
                _recognize("今天我们讨论", "s2", False),
            ],
            # attempt 1: resumes and completes sentence s2, then ends.
            [
                _STARTED,
                _recognize("今天我们讨论第三季度的路线图。", "s2", True),
                _recognize("结论下周同步给全员。", "s3", True),
                _SPEECH_END,
            ],
        ],
        "room-delta": [
            [
                _STARTED,
                _event(
                    "result-generated",
                    {
                        "action": "task-failed",
                        "errorCode": "DataInspectionFailed",
                        "errorMessage": "content policy blocked this room",
                    },
                ),
            ],
        ],
    }


def _run_offline_demo():
    print(
        f"离线并发实时会议转写演示（{MAX_CONCURRENT} 路并发上限，" f"每场会议独占一条 websocket）\n",
    )
    scripts = _offline_scripts()
    gateway = MockMeetingGateway(scripts)
    meetings = [
        MeetingSession(
            mid,
            f"Room {mid.split('-')[1].title()}",
            _mock_frames(),
        )
        for mid in scripts
    ]
    supervisor = MeetingSupervisor(gateway, MAX_CONCURRENT)
    started = time.monotonic()
    results = supervisor.run(meetings)
    elapsed = time.monotonic() - started
    print(f"\n并发跑完 {len(results)} 场会议，用时 {elapsed:.3f}s")
    _report(results)


def _run_live():
    if APP_ID in ("", "your-app-id"):
        print(
            "MEETING_HARNESS_LIVE=1 需要设置 DASHSCOPE_TINGWU_APP_ID "
            "（听悟应用 ID）与 DASHSCOPE_API_KEY。",
        )
        return
    if not os.path.exists(AUDIO_FILE):
        print(f"找不到音频文件：{AUDIO_FILE}（可用 DASHSCOPE_DEMO_AUDIO 指定）")
        return
    frames = _read_pcm(AUDIO_FILE)
    print(f"实时会议转写（live）：{len(frames)} 帧 x {FRAME_BYTES}B，paced")
    gateway = LiveGateway(APP_ID)
    meeting = MeetingSession("live-room", "Live Room", frames, pace=True)
    result = run_meeting(meeting, gateway)
    _report([result])


def main():
    if LIVE:
        _run_live()
    else:
        _run_offline_demo()


if __name__ == "__main__":
    main()
