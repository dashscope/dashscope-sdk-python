# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Loopback tests using aiohttp and the real websocket-client transport."""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
from concurrent.futures import Future
from contextlib import contextmanager

import pytest
import websocket
from aiohttp import WSMsgType, web
from websocket import WebSocketConnectionClosedException

from dashscope.v2 import Client
from dashscope.v2.errors import (
    APIConnectionError,
    BadRequestError,
    RateLimitError,
)


TIMEOUT = 2
SERVER_TIMEOUT = 4
LOCAL_KEY = "local-integration-not-a-real-key"


class _LocalWebSocketServer:
    """Bounded server thread preserving original protocol assertion errors."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.url = None
        self.loop = None
        self.stop_signal = None
        self.ready = threading.Event()
        self.handler_done = threading.Event()
        self.protocol_completed = threading.Event()
        self.errors = queue.Queue()
        self.thread = threading.Thread(
            target=self._run,
            name="audio-ws-integration",
            daemon=True,
        )

    def _record_error(self, error):
        self.errors.put((error, error.__traceback__))

    def _raise_server_error(self):
        try:
            error, traceback = self.errors.get_nowait()
        except queue.Empty:
            return
        raise error.with_traceback(traceback)

    async def _handle(self, request):
        ws = web.WebSocketResponse(timeout=TIMEOUT)
        try:
            await ws.prepare(request)
            assert request.headers["Authorization"] == f"Bearer {LOCAL_KEY}"
            await self.scenario(ws, request)
            transport = request.transport
            if (
                not ws.closed
                and transport is not None
                and not transport.is_closing()
            ):
                # Readers must stop on terminal events without waiting for EOF.
                message = await ws.receive(timeout=SERVER_TIMEOUT)
                assert message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED), (
                    "expected client close after scenario",
                    message,
                )
            self.protocol_completed.set()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            # aiohttp would otherwise mask handler assertions as disconnects.
            self._record_error(error)
        finally:
            try:
                await asyncio.wait_for(ws.close(), timeout=TIMEOUT)
            except Exception as error:
                self._record_error(error)
            self.handler_done.set()
        return ws

    async def _serve(self):
        self.stop_signal = asyncio.Event()
        app = web.Application()
        app.router.add_get("/audio", self._handle)
        runner = web.AppRunner(app, shutdown_timeout=TIMEOUT)
        try:
            await runner.setup()
            site = web.TCPSite(runner, host="127.0.0.1", port=0)
            await site.start()
            port = runner.addresses[0][1]
            self.url = f"ws://127.0.0.1:{port}/audio"
            self.ready.set()
            await self.stop_signal.wait()
        finally:
            await asyncio.wait_for(runner.cleanup(), timeout=SERVER_TIMEOUT)

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except BaseException as error:
            self._record_error(error)
        finally:
            self.ready.set()
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                _, remaining = self.loop.run_until_complete(
                    asyncio.wait(pending, timeout=TIMEOUT),
                )
                if remaining:
                    self._record_error(
                        AssertionError("server tasks did not stop"),
                    )
            self.loop.close()
            asyncio.set_event_loop(None)

    def start(self):
        self.thread.start()
        assert self.ready.wait(
            SERVER_TIMEOUT,
        ), "local WebSocket startup timed out"
        self._raise_server_error()
        assert self.url is not None, "local WebSocket server did not bind"

    def close(self, *, expect_completed):
        if expect_completed:
            self.handler_done.wait(SERVER_TIMEOUT)
        if self.loop is not None and not self.loop.is_closed():
            if self.stop_signal is not None:
                self.loop.call_soon_threadsafe(self.stop_signal.set)
        self.thread.join(SERVER_TIMEOUT + TIMEOUT)
        # Preserve the server assertion that caused the client-side EOF.
        self._raise_server_error()
        assert (
            not self.thread.is_alive()
        ), "local WebSocket thread did not stop"
        if expect_completed:
            assert (
                self.protocol_completed.is_set()
            ), "server scenario did not finish"


@pytest.fixture
def local_ws_server(monkeypatch):
    # A developer's HTTP proxy must never route these loopback connections out.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    @contextmanager
    def serve(scenario):
        server = _LocalWebSocketServer(scenario)
        failed = False
        try:
            server.start()
            yield server
        except BaseException:
            failed = True
            raise
        finally:
            server.close(expect_completed=not failed)

    return serve


@pytest.fixture(autouse=True)
def closed_client_sockets(monkeypatch):
    connect = websocket.create_connection
    sockets = []

    def tracked_connect(*args, **kwargs):
        ws = connect(*args, **kwargs)
        sockets.append(ws)
        return ws

    monkeypatch.setattr(websocket, "create_connection", tracked_connect)
    yield
    try:
        assert all(ws.sock is None for ws in sockets)
    finally:
        for ws in sockets:
            ws.shutdown()


@pytest.fixture
def client():
    with Client(
        api_key=LOCAL_KEY,
        workspace="ws-local-integration",
        region="cn-beijing",
        protocol="native",
        base_url="http://127.0.0.1:1",
        timeout=TIMEOUT,
        max_retries=0,
    ) as value:
        yield value


async def _receive_json(ws):
    message = await ws.receive(timeout=SERVER_TIMEOUT)
    assert message.type == WSMsgType.TEXT, (
        "expected JSON text opcode",
        message,
    )
    frame = json.loads(message.data)
    assert isinstance(frame, dict)
    return frame


async def _receive_binary(ws, expected):
    message = await ws.receive(timeout=SERVER_TIMEOUT)
    assert message.type == WSMsgType.BINARY, (
        "expected binary audio opcode",
        message,
    )
    assert message.data == expected


def _task_event(kind, task_id, payload=None):
    return {
        "header": {"event": kind, "task_id": task_id},
        "payload": {} if payload is None else payload,
    }


async def _receive_run(ws, *, task, model):
    frame = await _receive_json(ws)
    header = frame["header"]
    assert header["action"] == "run-task"
    assert header["streaming"] == "duplex"
    assert isinstance(header["task_id"], str) and header["task_id"]
    payload = frame["payload"]
    assert payload["task_group"] == "audio"
    assert payload["task"] == task
    assert payload["function"] == (
        "recognition" if task == "asr" else "SpeechSynthesizer"
    )
    assert payload["model"] == model
    assert payload["input"] == {}  # Required even for initially empty input.
    return frame


async def _receive_action(ws, task_id, action, input_data):
    frame = await _receive_json(ws)
    assert frame["header"] == {
        "action": action,
        "task_id": task_id,
        "streaming": "duplex",
    }
    assert frame["payload"] == {"input": input_data}


async def _wait_for_return(gate):
    # This gate opens only after the nonblocking client method returns.
    await asyncio.wait_for(asyncio.wrap_future(gate), timeout=SERVER_TIMEOUT)


async def _receive_realtime_event(ws, expected, event_ids):
    event = await _receive_json(ws)
    event_id = event.pop("event_id")
    assert isinstance(event_id, str) and event_id
    assert event_id not in event_ids
    event_ids.add(event_id)
    assert event == expected


def test_asr_binary_duplex_context_and_results_after_stop(
    client,
    local_ws_server,
):
    model = "fun-asr-realtime"
    first_audio = b"first-pcm\x00\x01"
    second_audio = b"second-pcm\x80\xff"
    context = {"context": {"text": "上一轮的识别上下文"}}
    partial_payload = {
        "output": {
            "sentence": {
                "text": "第一段",
                "sentence_end": False,
            },
        },
    }
    final_payload = {
        "output": {
            "sentence": {
                "text": "第一段，第二段。",
                "sentence_end": True,
            },
        },
    }
    usage = {"duration": 2.5}
    first_result_read = Future()
    stop_returned = Future()

    async def scenario(ws, _request):
        run = await _receive_run(ws, task="asr", model=model)
        task_id = run["header"]["task_id"]
        assert run["payload"]["parameters"]["format"] == "pcm"
        assert run["payload"]["parameters"]["sample_rate"] == 16000
        await ws.send_json(_task_event("task-started", task_id))
        await _receive_binary(ws, first_audio)
        await ws.send_json(
            _task_event("result-generated", task_id, partial_payload),
        )
        await _receive_action(ws, task_id, "continue-task", context)
        assert (
            first_result_read.done()
        ), "context sent before first result was read"
        await _receive_binary(ws, second_audio)
        await _receive_action(ws, task_id, "finish-task", {})
        await _wait_for_return(stop_returned)
        await ws.send_json(
            _task_event("result-generated", task_id, final_payload),
        )
        await ws.send_json(
            _task_event("task-finished", task_id, {"usage": usage}),
        )

    with local_ws_server(scenario) as server:
        with client.audio.asr.recognize(
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        ) as recognition:
            task_id = recognition.session.task_id
            assert recognition.last_response == _task_event(
                "task-started",
                task_id,
            )
            events = recognition.events()
            recognition.send_audio_frame(first_audio)
            assert next(events) == _task_event(
                "result-generated",
                task_id,
                partial_payload,
            )
            first_result_read.set_result(None)
            recognition.update_context(context)
            recognition.send_audio_frame(second_audio)
            recognition.stop()
            stop_returned.set_result(None)
            finished = _task_event("task-finished", task_id, {"usage": usage})
            assert list(events) == [
                _task_event("result-generated", task_id, final_payload),
                finished,
            ]
            assert recognition.last_response == finished
            assert recognition.usage == usage


@pytest.mark.parametrize("reader", ["events", "audio_chunks"])
def test_tts_reads_audio_before_second_text_and_drains_stop(
    client,
    local_ws_server,
    reader,
):
    model = "cosyvoice-v1"
    first_text, second_text = "第一段。", "第二段。"
    first_audio, second_audio = b"\x00\x81first-audio", b"\xff\x02second-audio"
    first_subtitle = {
        "output": {"sentence": {"text": first_text, "begin_time": 0}},
    }
    second_subtitle = {
        "output": {"sentence": {"text": second_text, "begin_time": 500}},
    }
    usage = {"characters": len(first_text + second_text)}
    first_audio_read = Future()
    stop_returned = Future()

    async def scenario(ws, _request):
        run = await _receive_run(ws, task="tts", model=model)
        task_id = run["header"]["task_id"]
        assert run["payload"]["parameters"]["voice"] == "longxiaochun"
        await ws.send_json(_task_event("task-started", task_id))
        await _receive_action(
            ws,
            task_id,
            "continue-task",
            {"text": first_text},
        )
        await ws.send_bytes(first_audio)
        await ws.send_json(
            _task_event("result-generated", task_id, first_subtitle),
        )
        await _receive_action(ws, task_id, "continue-task", {"flush": True})
        assert (
            first_audio_read.done()
        ), "flush sent before first audio was consumed"
        await _receive_action(
            ws,
            task_id,
            "continue-task",
            {"text": second_text},
        )
        assert (
            first_audio_read.done()
        ), "second text sent before first audio was consumed"
        await _receive_action(ws, task_id, "finish-task", {})
        await _wait_for_return(stop_returned)
        await ws.send_bytes(second_audio)
        await ws.send_json(
            _task_event("result-generated", task_id, second_subtitle),
        )
        await ws.send_json(
            _task_event("task-finished", task_id, {"usage": usage}),
        )

    with local_ws_server(scenario) as server:
        with client.audio.tts.connect(
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        ) as synthesis:
            task_id = synthesis.session.task_id
            assert synthesis.last_response == _task_event(
                "task-started",
                task_id,
            )
            frames = getattr(synthesis, reader)()
            synthesis.send_text(first_text)
            assert next(frames) == first_audio
            first_audio_read.set_result(None)
            synthesis.flush()
            synthesis.send_text(second_text)
            synthesis.stop()
            stop_returned.set_result(None)
            finished = _task_event("task-finished", task_id, {"usage": usage})
            remaining = list(frames)
            if reader == "events":
                assert remaining == [
                    _task_event("result-generated", task_id, first_subtitle),
                    second_audio,
                    _task_event("result-generated", task_id, second_subtitle),
                    finished,
                ]
            else:
                assert remaining == [second_audio]
                assert all(isinstance(chunk, bytes) for chunk in remaining)
            assert synthesis.last_response == finished
            assert synthesis.usage == usage


def test_tts_cancel_after_first_audio_uses_finish_directive(
    client,
    local_ws_server,
):
    model = "cosyvoice-v1"
    cancel_returned = Future()

    async def scenario(ws, _request):
        run = await _receive_run(ws, task="tts", model=model)
        task_id = run["header"]["task_id"]
        await ws.send_json(_task_event("task-started", task_id))
        await _receive_action(
            ws,
            task_id,
            "continue-task",
            {"text": "stop this speech"},
        )
        await ws.send_bytes(b"first-audio")
        await _receive_action(
            ws,
            task_id,
            "finish-task",
            {"directive": "cancel"},
        )
        await _wait_for_return(cancel_returned)
        await ws.send_json(_task_event("task-finished", task_id))

    with local_ws_server(scenario) as server:
        with client.audio.tts.connect(
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        ) as synthesis:
            task_id = synthesis.session.task_id
            events = synthesis.events()
            synthesis.send_text("stop this speech")
            assert next(events) == b"first-audio"
            synthesis.cancel()
            cancel_returned.set_result(None)
            assert list(events) == [_task_event("task-finished", task_id)]


def test_omni_manual_commit_two_responses_and_session_finish(
    client,
    local_ws_server,
):
    model = "qwen-omni-turbo-realtime"
    session_id = "local-omni-session"
    audio_inputs = [b"\x00\xffround-one", b"\x80\x01round-two"]
    settings = {
        "modalities": ["text", "audio"],
        "input_audio_format": "pcm16",
        "turn_detection": None,
    }
    created = {"type": "session.created", "session": {"id": session_id}}
    updated = {
        "type": "session.updated",
        "session": {"id": session_id, **settings},
    }
    finished = {"type": "session.finished", "session": {"id": session_id}}
    rounds = []
    for index in range(2):
        response_id = f"response-{index}"
        rounds.append(
            [
                {
                    "type": "input_audio_buffer.committed",
                    "item_id": f"item-{index}",
                },
                {"type": "response.created", "response": {"id": response_id}},
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(
                        b"\x00\xffreply" + bytes([index]),
                    ).decode("ascii"),
                },
                {
                    "type": "response.audio_transcript.delta",
                    "response_id": response_id,
                    "delta": f"回答{index}",
                },
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "usage": {"total_tokens": index + 3},
                    },
                },
            ],
        )
    response_read = [Future(), Future()]
    end_returned = Future()
    event_ids = set()

    async def receive_event(ws, kind):
        event = await _receive_json(ws)
        assert event["type"] == kind
        event_id = event.pop("event_id")
        assert isinstance(event_id, str) and event_id
        assert event_id not in event_ids
        event_ids.add(event_id)
        assert "header" not in event and "payload" not in event
        return event

    async def scenario(ws, request):
        assert request.query["model"] == model
        await ws.send_json(created)
        assert await receive_event(ws, "session.update") == {
            "type": "session.update",
            "session": settings,
        }
        await ws.send_json(updated)
        for index, audio in enumerate(audio_inputs):
            event = await receive_event(ws, "input_audio_buffer.append")
            assert set(event) == {"type", "audio"}
            assert base64.b64decode(event["audio"], validate=True) == audio
            if index:
                assert response_read[
                    index - 1
                ].done(), "next turn sent before response.done"
            assert await receive_event(ws, "input_audio_buffer.commit") == {
                "type": "input_audio_buffer.commit",
            }
            response = await receive_event(ws, "response.create")
            assert response["response"]["modalities"] == ["text", "audio"]
            for event in rounds[index]:
                await ws.send_json(event)
        assert await receive_event(ws, "session.finish") == {
            "type": "session.finish",
        }
        assert response_read[-1].done()
        await _wait_for_return(end_returned)
        await ws.send_json(finished)

    with local_ws_server(scenario) as server:
        with client.audio.realtime.connect(
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        ) as realtime:
            assert realtime.recv_event() == created
            realtime.send_event(
                {"type": "session.update", "session": settings},
            )
            events = realtime.events()
            assert next(events) == updated
            for index, audio in enumerate(audio_inputs):
                realtime.append_audio(audio)
                realtime.commit()
                realtime.create_response(modalities=["text", "audio"])
                received = [next(events) for _ in rounds[index]]
                assert received == rounds[index]
                assert base64.b64decode(
                    received[2]["delta"],
                    validate=True,
                ) == (b"\x00\xffreply" + bytes([index]))
                response_read[index].set_result(None)
            realtime.end_session()
            end_returned.set_result(None)
            assert list(events) == [finished]
            assert realtime.last_message == finished


def test_qwen_audio_manual_turns_recover_error_and_close(
    client,
    local_ws_server,
):
    model = "qwen-audio-3.1-realtime-plus"
    session_id = "local-qwen-audio-session"
    settings = {"modalities": ["text", "audio"], "turn_detection": None}
    instructions = "只描述本轮听到的声音。"
    created = {"type": "session.created", "session": {"id": session_id}}
    updated = {
        "type": "session.updated",
        "session": {"id": session_id, **settings},
    }
    instruction_updated = {
        "type": "session.updated",
        "session": {"id": session_id, "instructions": instructions},
    }
    invalid_update = {
        "type": "session.update",
        "session": {"instructions": 123},
    }
    rejected = {
        "type": "error",
        "event_id": "server-invalid-instructions",
        "error": {
            "type": "invalid_request_error",
            "code": "InvalidParameter",
            "message": "instructions must be a string",
            "param": "session.instructions",
        },
    }
    environment_transcript = json.dumps(
        {"text": "有人敲门", "environment": {"sound": "knocking"}},
        ensure_ascii=False,
    )
    tool_arguments = json.dumps({"sound": "knocking", "count": 2})
    audio_inputs = [b"\x00\xffknocking", b"\x80\x01footsteps"]
    rounds = []
    for index in range(2):
        response_id = f"audio-response-{index}"
        rounds.append(
            [
                {
                    "type": "input_audio_buffer.committed",
                    "item_id": f"audio-item-{index}",
                },
                {"type": "response.created", "response": {"id": response_id}},
                {
                    "type": "response.audio_transcript.delta",
                    "response_id": response_id,
                    "delta": "敲门声" if index == 0 else "脚步声",
                },
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(
                        b"\x00\xffreply" + bytes([index]),
                    ).decode("ascii"),
                },
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "status": "completed",
                        "usage": {
                            "total_tokens": 12 + index,
                            "input_tokens": 8,
                            "output_tokens": 4 + index,
                            "input_token_details": {"audio_tokens": 8},
                        },
                    },
                },
            ],
        )
    rounds[0][2:2] = [
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "audio-item-0",
            "content_index": 0,
            "transcript": environment_transcript,
        },
        {
            "type": "response.output_item.done",
            "response_id": "audio-response-0",
            "output_index": 0,
            "item": {
                "id": "tool-item-0",
                "type": "function_call",
                "status": "completed",
                "call_id": "call-0",
                "name": "describe_environment",
                "arguments": tool_arguments,
            },
        },
    ]
    response_read = [Future(), Future()]
    event_ids = set()

    async def scenario(ws, request):
        assert request.query["model"] == model
        await ws.send_json(created)
        await _receive_realtime_event(
            ws,
            {"type": "session.update", "session": settings},
            event_ids,
        )
        await ws.send_json(updated)
        for index, audio in enumerate(audio_inputs):
            if index:
                await _wait_for_return(response_read[0])
                await _receive_realtime_event(ws, invalid_update, event_ids)
                await ws.send_json(rejected)
                # A partial update must not resend modalities or old defaults.
                await _receive_realtime_event(
                    ws,
                    {
                        "type": "session.update",
                        "session": {"instructions": instructions},
                    },
                    event_ids,
                )
                await ws.send_json(instruction_updated)
            await _receive_realtime_event(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio).decode("ascii"),
                },
                event_ids,
            )
            await _receive_realtime_event(
                ws,
                {"type": "input_audio_buffer.commit"},
                event_ids,
            )
            # Qwen-Audio needs an explicit create after each manual commit.
            await _receive_realtime_event(
                ws,
                {"type": "response.create", "response": {}},
                event_ids,
            )
            for event in rounds[index]:
                await ws.send_json(event)
        message = await ws.receive(timeout=SERVER_TIMEOUT)
        # This protocol has no session.finish: context exit sends a close.
        assert message.type == WSMsgType.CLOSE
        assert message.data == 1000
        assert response_read[-1].done()

    with local_ws_server(scenario) as server:
        with client.audio.realtime.connect(
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        ) as realtime:
            assert realtime.recv_event() == created
            realtime.update_session(
                ["text", "audio"],
                enable_turn_detection=False,
            )
            events = realtime.events()
            assert next(events) == updated
            for index, audio in enumerate(audio_inputs):
                if index:
                    realtime.send_event(invalid_update)
                    with pytest.raises(
                        BadRequestError,
                        match="instructions must be a string",
                    ) as caught:
                        realtime.recv_event()
                    assert caught.value.body == rejected
                    assert caught.value.code == "InvalidParameter"
                    assert realtime.last_message == rejected
                    realtime.update_session(instructions=instructions)
                    assert next(events) == instruction_updated
                realtime.append_audio(audio)
                realtime.commit()
                realtime.create_response()
                assert [next(events) for _ in rounds[index]] == rounds[index]
                assert realtime.last_response_id == f"audio-response-{index}"
                response_read[index].set_result(None)
            assert realtime.session_id == session_id
            assert realtime.last_message == rounds[-1][-1]
        assert realtime.recv_event() is None


@pytest.mark.parametrize(
    "model, modality_key, text_event_type",
    [
        (
            "qwen3-livetranslate-flash-realtime",
            "modalities",
            "response.audio_transcript.text",
        ),
        (
            "qwen3.5-livetranslate-flash-realtime",
            "modalities",
            "response.text.text",
        ),
        (
            "qwen3.8-livetranslate-flash-realtime",
            "output_modalities",
            "response.audio_transcript.delta",
        ),
    ],
    ids=["qwen3", "qwen3.5", "qwen3.8"],
)
def test_livetranslate_auto_response_media_and_nonblocking_finish(
    client,
    local_ws_server,
    model,
    modality_key,
    text_event_type,
):
    session_id = "local-livetranslate-session"
    response_id = "translation-response"
    settings = {
        modality_key: ["text", "audio"],
        "translation": {"language": "en"},
    }
    audio_settings = {}
    if modality_key == "output_modalities":
        audio_settings = {
            "input_audio_config": {"type": "pcm"},
            "sample_rate": 16000,
        }
        settings["audio"] = {
            "input": {"format": {"type": "pcm", "sample_rate": 16000}},
        }
    created = {"type": "session.created", "session": {"id": session_id}}
    updated = {
        "type": "session.updated",
        "session": {"id": session_id, **settings},
    }
    audio = b"\x00\xffsource-speech"
    audio_b64 = base64.b64encode(audio).decode("ascii")
    image = b"\xff\xd8\xffloopback-image\xff\xd9"
    image_b64 = base64.b64encode(image).decode("ascii")
    translated_audio = b"\x80\x01translated-speech"
    is_legacy = modality_key == "modalities"
    translated_text = (
        {"text": "Good morning", "stash": ", everyone"}
        if is_legacy
        else {"delta": "Good morning, everyone"}
    )
    source_text = (
        {"text": "早上好", "stash": "，各位"} if is_legacy else {"delta": "早上好，各位"}
    )
    streaming_events = [
        {"type": "input_audio_buffer.committed", "item_id": "source-item"},
        {"type": "response.created", "response": {"id": response_id}},
        {
            "type": "conversation.item.input_audio_transcription."
            + ("text" if is_legacy else "delta"),
            "item_id": "source-item",
            "content_index": 0,
            "language": "zh",
            "begin_time": 0,
            "end_time": 640,
            "speaker_id": "speaker-0",
            **source_text,
        },
        {
            "type": text_event_type,
            "response_id": response_id,
            "item_id": "translation-item",
            "content_index": 0,
            "begin_time": 0,
            "end_time": 720,
            "speaker_id": "speaker-0",
            **translated_text,
        },
        {
            "type": "response.audio.delta",
            "response_id": response_id,
            "delta": base64.b64encode(translated_audio).decode("ascii"),
        },
    ]
    usage = {
        "total_tokens": 19,
        "input_tokens": 12,
        "output_tokens": 7,
        "input_token_details": {"audio_tokens": 10, "text_tokens": 2},
        "output_token_details": {"audio_tokens": 5, "text_tokens": 2},
    }
    finished = {
        "type": "session.finished",
        "event_id": "server-final-event",
        "session": {"id": session_id},
    }
    final_events = [
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "source-item",
            "transcript": "早上好，各位。",
            "begin_time": 0,
            "end_time": 800,
            "speaker_id": "speaker-0",
        },
        {
            "type": "response.audio_transcript.done",
            "response_id": response_id,
            "transcript": "Good morning, everyone.",
        },
        {
            "type": "response.done",
            "response": {"id": response_id, "usage": usage},
        },
        finished,
    ]
    end_returned = Future()
    event_ids = set()

    async def scenario(ws, request):
        assert request.query["model"] == model
        await ws.send_json(created)
        await _receive_realtime_event(
            ws,
            {"type": "session.update", "session": settings},
            event_ids,
        )
        await ws.send_json(updated)
        for _ in range(2):
            await _receive_realtime_event(
                ws,
                {"type": "input_audio_buffer.append", "audio": audio_b64},
                event_ids,
            )
        if is_legacy:
            for _ in range(2):
                await _receive_realtime_event(
                    ws,
                    {"type": "input_image_buffer.append", "image": image_b64},
                    event_ids,
                )
        await _receive_realtime_event(
            ws,
            {"type": "input_audio_buffer.commit"},
            event_ids,
        )
        # Live creates the response itself; the next client frame must finish.
        for event in streaming_events:
            await ws.send_json(event)
        await _receive_realtime_event(
            ws,
            {"type": "session.finish"},
            event_ids,
        )
        # Withhold trailing events until end_session has actually returned.
        await _wait_for_return(end_returned)
        for event in final_events:
            await ws.send_json(event)

    with local_ws_server(scenario) as server:
        with client.audio.realtime.connect(
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        ) as realtime:
            assert realtime.recv_event() == created
            realtime.update_session(
                output_modalities=["text", "audio"],
                translation_params={"language": "en"},
                **audio_settings,
            )
            events = realtime.events()
            assert next(events) == updated
            realtime.append_audio(audio)
            realtime.append_audio(audio_b64)
            if is_legacy:
                realtime.append_video(image)
                realtime.append_video(image_b64)
            realtime.commit()
            received = [next(events) for _ in streaming_events]
            assert received == streaming_events
            assert (
                base64.b64decode(received[-1]["delta"], validate=True)
                == translated_audio
            )
            assert realtime.last_first_text_delay is not None
            assert realtime.last_first_text_delay >= 0
            assert realtime.last_first_audio_delay is not None
            first_text_delay = realtime.last_first_text_delay
            realtime.end_session()
            end_returned.set_result(None)
            assert list(events) == final_events
            assert realtime.last_message == finished
            assert realtime.last_first_text_delay == first_text_delay
            assert realtime.recv_event() is None
            # session.finished releases the socket before context exit.
            assert server.handler_done.wait(SERVER_TIMEOUT)
            assert server.protocol_completed.is_set()


@pytest.mark.parametrize(
    "task, model",
    [
        ("asr", "fun-asr-realtime"),
        ("tts", "cosyvoice-v1"),
    ],
)
@pytest.mark.parametrize(
    "during_start",
    [True, False],
    ids=["handshake", "streaming"],
)
@pytest.mark.parametrize(
    "code, error_type",
    [
        ("InvalidParameter", BadRequestError),
        ("Throttling", RateLimitError),
    ],
)
def test_header_task_failure_maps_v2_error_and_preserves_task_id(
    client,
    local_ws_server,
    task,
    model,
    during_start,
    code,
    error_type,
):
    observed = {}
    message = "local server rejected this input"

    async def scenario(ws, _request):
        run = await _receive_run(ws, task=task, model=model)
        task_id = run["header"]["task_id"]
        failed = _task_event("task-failed", task_id)
        failed["header"].update(error_code=code, error_message=message)
        observed.update(task_id=task_id, failed=failed)
        if not during_start:
            await ws.send_json(_task_event("task-started", task_id))
            if task == "asr":
                await _receive_binary(ws, b"audio")
            else:
                await _receive_action(
                    ws,
                    task_id,
                    "continue-task",
                    {"text": "hello"},
                )
        # Deliberately no payload.code/message or header.request_id fallback.
        await ws.send_json(failed)

    connect = (
        client.audio.asr.recognize
        if task == "asr"
        else client.audio.tts.connect
    )
    with local_ws_server(scenario) as server:
        with pytest.raises(error_type, match=message) as caught:
            with connect(
                model=model,
                ws_url=server.url,
                timeout=TIMEOUT,
            ) as session:
                assert (
                    not during_start
                ), "task-failed handshake entered the context"
                if task == "asr":
                    session.send_audio_frame(b"audio")
                else:
                    session.send_text("hello")
                list(session.events())
        error = caught.value
        assert error.code == code
        assert error.message == message
        assert error.request_id == observed["task_id"]
        assert error.body == observed["failed"]
        assert error.body["header"]["task_id"] == observed["task_id"]
        assert observed["task_id"] in str(error)


@pytest.mark.parametrize("disconnect", ["close-frame", "tcp-abort"])
def test_tts_stream_disconnect_does_not_silently_return_truncated_audio(
    client,
    local_ws_server,
    disconnect,
):
    model = "cosyvoice-v1"
    first_audio_read = Future()

    async def scenario(ws, request):
        run = await _receive_run(ws, task="tts", model=model)
        task_id = run["header"]["task_id"]
        await ws.send_json(_task_event("task-started", task_id))
        await _receive_action(ws, task_id, "continue-task", {"text": "hello"})
        await _receive_action(ws, task_id, "finish-task", {})
        await ws.send_bytes(b"partial-audio")
        await _wait_for_return(first_audio_read)
        # Even a normal close without task-finished means truncated audio.
        if disconnect == "close-frame":
            await ws.close(code=1000)
        else:
            assert request.transport is not None
            request.transport.abort()

    with local_ws_server(scenario) as server:
        chunks = client.audio.tts.stream(
            "hello",
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        )
        try:
            assert next(chunks) == b"partial-audio"
            first_audio_read.set_result(None)
            # A timeout or StopIteration must not mask a premature disconnect.
            with pytest.raises(
                (APIConnectionError, WebSocketConnectionClosedException),
            ):
                list(chunks)
        finally:
            chunks.close()


def test_local_server_propagates_original_handler_assertion(
    client,
    local_ws_server,
):
    original = AssertionError("server-side protocol assertion sentinel")

    async def scenario(ws, _request):
        await _receive_run(ws, task="asr", model="fun-asr-realtime")
        raise original

    with pytest.raises(
        AssertionError,
        match="protocol assertion sentinel",
    ) as caught:
        with local_ws_server(scenario) as server:
            with client.audio.asr.recognize(
                model="fun-asr-realtime",
                ws_url=server.url,
                timeout=TIMEOUT,
            ):
                pytest.fail("server assertion should prevent the handshake")
    assert caught.value is original
    assert not server.thread.is_alive()


@pytest.mark.parametrize("active_response", [False, True])
def test_omni_peer_close_releases_socket(
    client,
    local_ws_server,
    active_response,
):
    async def scenario(ws, _request):
        await ws.send_json({"type": "session.created"})
        if active_response:
            await ws.send_json(
                {
                    "type": "response.created",
                    "response": {"id": "response-1"},
                },
            )
        await ws.close(code=1000)

    with local_ws_server(scenario) as server:
        with client.audio.realtime.connect(
            model="qwen-omni-turbo-realtime",
            ws_url=server.url,
        ) as realtime:
            if active_response:
                with pytest.raises(APIConnectionError, match="response.done"):
                    list(realtime.events())
            else:
                assert list(realtime.events()) == [{"type": "session.created"}]


@pytest.mark.parametrize(
    "model",
    [
        "qwen3-livetranslate-flash-realtime",
        "qwen3.5-livetranslate-flash-realtime",
        "qwen3.8-livetranslate-flash-realtime",
    ],
)
@pytest.mark.parametrize("disconnect", ["close-frame", "tcp-abort"])
def test_livetranslate_response_done_does_not_replace_finish_ack(
    client,
    local_ws_server,
    model,
    disconnect,
):
    async def scenario(ws, request):
        await ws.send_json({"type": "session.created"})
        await _receive_realtime_event(ws, {"type": "session.finish"}, set())
        await ws.send_json(
            {"type": "response.created", "response": {"id": "last-response"}},
        )
        await ws.send_json(
            {"type": "response.done", "response": {"id": "last-response"}},
        )
        if disconnect == "close-frame":
            await ws.close(code=1000)
        else:
            assert request.transport is not None
            request.transport.abort()

    with local_ws_server(scenario) as server:
        with client.audio.realtime.connect(
            model=model,
            ws_url=server.url,
            timeout=TIMEOUT,
        ) as realtime:
            assert realtime.recv_event() == {"type": "session.created"}
            realtime.end_session()
            with pytest.raises(
                APIConnectionError,
                match="before session.finished",
            ):
                list(realtime.events())
            assert realtime.last_message["type"] == "response.done"
