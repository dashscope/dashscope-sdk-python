# -*- coding: utf-8 -*-
"""Regression tests for the agentic-RL FC server hardening (review §6, §7).

- P1 (§6): admission control must be an atomic ``asyncio.Semaphore`` bounded
  to ``workers + queue`` (no probing of ``ThreadPoolExecutor._work_queue``),
  and a client disconnect must cancel the request while surfacing the
  "HTTP cancelled but the sync worker is still running" case.
- P1 (§7): the validation error handler must never log or echo the full
  request body by default; sensitive fields are deep-masked and truncated,
  and only emitted at DEBUG level.

The server module reads ``FUNC_TYPE`` / ``PROCESSOR_CLASS`` and builds its
FastAPI app at import time, so it is imported lazily through a fixture and
skipped when ``fastapi`` is not installed.
"""
import asyncio
import contextlib
import importlib
import inspect
import json
import os
import sys
import threading
from unittest.mock import MagicMock

import pytest

_SERVER_MODULE = (
    "dashscope.applications.finetune.reinforcement.component.server.server"
)
_DEMO_PROCESSOR = (
    "dashscope.applications.finetune.reinforcement.component.demo"
    ".reward_processor_demo.DemoRewardProcessor"
)


@pytest.fixture(scope="module")
def server_mod():
    """Import the FC server module without leaking process-wide state."""
    imported_before = set(sys.modules)
    pytest.importorskip("fastapi")
    env_before = {
        key: os.environ.get(key) for key in ("FUNC_TYPE", "PROCESSOR_CLASS")
    }
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        loop = None
    os.environ.setdefault("FUNC_TYPE", "reward")
    os.environ.setdefault("PROCESSOR_CLASS", _DEMO_PROCESSOR)
    module = importlib.import_module(_SERVER_MODULE)
    yield module
    for name in set(sys.modules) - imported_before:
        sys.modules.pop(name, None)
    for key, value in env_before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    if loop is not None:
        loop.close()
        asyncio.set_event_loop(None)


class _FakeRequest:
    """Minimal stand-in for a Starlette Request."""

    def __init__(self, body: bytes, headers=None):
        self._body = body
        self.headers = headers if headers is not None else {}

    async def body(self):
        return self._body


# ========================================================================== #
#                 P1 (§6): admission control (semaphore)                      #
# ========================================================================== #


class TestAdmissionControl:
    def test_does_not_probe_private_work_queue(self, server_mod):
        """The fix must not reach into ThreadPoolExecutor internals."""
        src = inspect.getsource(server_mod._check_queue_capacity)
        assert "_work_queue" not in src
        assert "qsize" not in src

    @pytest.mark.asyncio
    async def test_rejects_when_at_capacity(self, server_mod, monkeypatch):
        sem = asyncio.Semaphore(1)
        monkeypatch.setattr(server_mod, "_admission_semaphore", sem)

        # First request is admitted (counter 1 -> 0).
        await server_mod._check_queue_capacity()

        # At capacity -> 503, without acquiring.
        with pytest.raises(server_mod.HTTPException) as exc_info:
            await server_mod._check_queue_capacity()
        assert exc_info.value.status_code == 503

        # Releasing frees a slot again.
        server_mod._release_admission()
        await server_mod._check_queue_capacity()

    @pytest.mark.asyncio
    async def test_admission_is_atomic_under_concurrency(
        self,
        server_mod,
        monkeypatch,
    ):
        """Exactly N of N+M concurrent requests are admitted (no race that
        lets multiple requests pass a check-then-submit window)."""
        sem = asyncio.Semaphore(3)
        monkeypatch.setattr(server_mod, "_admission_semaphore", sem)

        async def admit():
            try:
                await server_mod._check_queue_capacity()
                return "in"
            except server_mod.HTTPException:
                return "out"

        results = await asyncio.gather(*[admit() for _ in range(6)])
        assert results.count("in") == 3
        assert results.count("out") == 3

    def test_release_admission_swallows_over_release(
        self,
        server_mod,
        monkeypatch,
    ):
        """_release_admission must not raise on an unmatched release."""
        sem = asyncio.BoundedSemaphore(1)
        monkeypatch.setattr(server_mod, "_admission_semaphore", sem)
        # BoundedSemaphore.release() raises ValueError when over-released;
        # the helper swallows it.
        server_mod._release_admission()


class TestDisconnectCancellation:
    @pytest.mark.asyncio
    async def test_normal_completion_returns_false(self, server_mod):
        disconnected = asyncio.Event()

        async def quick():
            return "done"

        async def never_disconnect():
            await asyncio.sleep(30)

        process_task = asyncio.create_task(quick())
        disconnect_listener = asyncio.create_task(never_disconnect())

        result = await server_mod._cancel_task_on_disconnect(
            process_task,
            disconnect_listener,
            disconnected,
        )

        assert result is False
        disconnect_listener.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_listener

    @pytest.mark.asyncio
    async def test_disconnect_cancels_cancellable_task(self, server_mod):
        disconnected = asyncio.Event()

        async def work():
            await asyncio.sleep(30)

        async def listener():
            disconnected.set()

        process_task = asyncio.create_task(work())
        disconnect_listener = asyncio.create_task(listener())

        result = await server_mod._cancel_task_on_disconnect(
            process_task,
            disconnect_listener,
            disconnected,
        )

        assert result is True
        assert process_task.done()

    @pytest.mark.asyncio
    async def test_warns_when_sync_worker_cannot_be_cancelled(
        self,
        server_mod,
        monkeypatch,
    ):
        """A worker that survives cancellation (like a sync processor already
        running in a thread) must be surfaced via a warning."""
        logger_mock = MagicMock()
        monkeypatch.setattr(server_mod, "logger", logger_mock)

        started = asyncio.Event()

        async def stubborn():
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                # Emulate an uninterruptible sync worker: it does not stop
                # on cancel and keeps running briefly.
                await asyncio.sleep(0.05)

        process_task = asyncio.create_task(stubborn())
        await started.wait()

        disconnected = asyncio.Event()

        async def listener():
            disconnected.set()

        disconnect_listener = asyncio.create_task(listener())

        async def fake_wait_for(awaitable, timeout):
            # Simulate the 5s grace elapsing while the worker is still busy.
            awaitable.cancel()
            raise asyncio.TimeoutError()

        monkeypatch.setattr(server_mod.asyncio, "wait_for", fake_wait_for)

        result = await server_mod._cancel_task_on_disconnect(
            process_task,
            disconnect_listener,
            disconnected,
        )

        assert result is True
        logged = " ".join(str(c) for c in logger_mock.warning.call_args_list)
        assert "still running" in logged

        with contextlib.suppress(asyncio.CancelledError):
            await process_task
        disconnect_listener.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_listener

    @pytest.mark.asyncio
    async def test_sync_cancellation_waits_for_worker(self, server_mod):
        started = threading.Event()
        release = threading.Event()

        def blocking_worker():
            started.set()
            release.wait(5)

        process_task = asyncio.create_task(
            server_mod.func_manager._run_sync(blocking_worker),
        )
        while not started.is_set():
            await asyncio.sleep(0.001)

        try:
            process_task.cancel()
            await asyncio.sleep(0)
            assert not process_task.done()
        finally:
            release.set()

        with contextlib.suppress(asyncio.CancelledError):
            await process_task

    @pytest.mark.asyncio
    async def test_disconnect_holds_admission_until_process_finishes(
        self,
        server_mod,
        monkeypatch,
    ):
        semaphore = asyncio.BoundedSemaphore(1)
        monkeypatch.setattr(server_mod, "_admission_semaphore", semaphore)
        created = []
        cancellation_seen = asyncio.Event()

        class Manager:
            async def processes(self, _input):
                created.append(asyncio.current_task())
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    cancellation_seen.set()
                    await asyncio.sleep(30)

        class Request(_FakeRequest):
            async def receive(self):
                return {"type": "http.disconnect"}

        async def parse_body(_request):
            return {}

        async def parse_input(_body):
            return object()

        async def trace_context(_request):
            return None, None

        async def noop(*_args, **_kwargs):
            return None

        async def fake_wait_for(awaitable, timeout):
            assert timeout == 5.0
            awaitable.cancel()
            raise asyncio.TimeoutError()

        monkeypatch.setattr(server_mod, "func_manager", Manager())
        monkeypatch.setattr(server_mod, "_parse_request_body", parse_body)
        monkeypatch.setattr(server_mod, "_parse_processor_input", parse_input)
        monkeypatch.setattr(
            server_mod,
            "_extract_trace_context",
            trace_context,
        )
        monkeypatch.setattr(server_mod, "_cleanup_trace_context", noop)
        monkeypatch.setattr(server_mod, "_log_request_metrics", noop)
        monkeypatch.setattr(server_mod.asyncio, "wait_for", fake_wait_for)

        try:
            response = await server_mod.handle_endpoint(Request(b"{}"))
            assert response.status_code == 499
            await cancellation_seen.wait()
            assert semaphore.locked()
        finally:
            for task in created:
                task.cancel()
            for task in created:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await asyncio.sleep(0)
        assert not semaphore.locked()


# ========================================================================== #
#             P1 (§7): validation handler body masking                        #
# ========================================================================== #


class TestValidationBodyMasking:
    @pytest.mark.asyncio
    async def test_full_body_not_logged_by_default(
        self,
        server_mod,
        monkeypatch,
    ):
        logger_mock = MagicMock()
        monkeypatch.setattr(server_mod, "logger", logger_mock)
        monkeypatch.setattr(server_mod, "LOG_LEVEL", "INFO")

        secret = "SUPER_SECRET_GROUND_TRUTH"
        body = json.dumps(
            {"ground_truth": secret, "prompt": "p"},
        ).encode()
        request = _FakeRequest(body, {"x-fc-request-id": "req-1"})
        exc = server_mod.RequestValidationError(
            [
                {
                    "loc": ("body", "agent_output"),
                    "msg": "field required",
                    "type": "value_error.missing",
                },
            ],
        )

        resp = await server_mod.validation_exception_handler(request, exc)

        assert resp.status_code == 422
        logs = " ".join(
            str(c)
            for c in (
                logger_mock.error.call_args_list
                + logger_mock.debug.call_args_list
            )
        )
        # The raw body / secret is never logged, and no DEBUG dump happens.
        assert secret not in logs
        assert not logger_mock.debug.called
        # Request id and body size ARE recorded for triage.
        assert "req-1" in logs
        # The 422 response must not echo the raw body.
        assert secret not in resp.body.decode()

    @pytest.mark.asyncio
    async def test_sensitive_fields_masked_at_debug(
        self,
        server_mod,
        monkeypatch,
    ):
        logger_mock = MagicMock()
        monkeypatch.setattr(server_mod, "logger", logger_mock)
        monkeypatch.setattr(server_mod, "LOG_LEVEL", "DEBUG")

        raw_key = "sk-abcdefgh123456"
        body = json.dumps({"api_key": raw_key, "prompt": "hello"}).encode()
        request = _FakeRequest(body, {"x-fc-request-id": "req-2"})
        exc = server_mod.RequestValidationError(
            [{"loc": ("body", "x"), "msg": "err", "type": "t"}],
        )

        resp = await server_mod.validation_exception_handler(request, exc)

        assert resp.status_code == 422
        assert logger_mock.debug.called
        dbg = " ".join(str(c) for c in logger_mock.debug.call_args_list)
        # Full key masked; only the secret_part_str form is present.
        assert raw_key not in dbg
        assert "sk-a****3456" in dbg

    @pytest.mark.asyncio
    async def test_debug_non_json_body_is_omitted(
        self,
        server_mod,
        monkeypatch,
    ):
        logger_mock = MagicMock()
        monkeypatch.setattr(server_mod, "logger", logger_mock)
        monkeypatch.setattr(server_mod, "LOG_LEVEL", "DEBUG")

        body = b'{"ground_truth": "SECRET-GROUND-TRUTH", broken'
        request = _FakeRequest(body, {"x-fc-request-id": "req-3"})
        exc = server_mod.RequestValidationError(
            [{"loc": ("body",), "msg": "err", "type": "t"}],
        )

        resp = await server_mod.validation_exception_handler(request, exc)

        assert resp.status_code == 422
        dbg = " ".join(str(c) for c in logger_mock.debug.call_args_list)
        assert "SECRET-GROUND-TRUTH" not in dbg
        assert "non-JSON body omitted" in dbg

    @pytest.mark.asyncio
    async def test_survives_unreadable_body(self, server_mod, monkeypatch):
        logger_mock = MagicMock()
        monkeypatch.setattr(server_mod, "logger", logger_mock)
        monkeypatch.setattr(server_mod, "LOG_LEVEL", "DEBUG")

        class _BadRequest:
            headers = {"x-fc-request-id": "req-4"}

            async def body(self):
                raise RuntimeError("stream closed")

        exc = server_mod.RequestValidationError(
            [{"loc": ("body",), "msg": "err", "type": "t"}],
        )

        resp = await server_mod.validation_exception_handler(
            _BadRequest(),
            exc,
        )

        # body read failed -> body_bytes None -> no DEBUG body dump.
        assert resp.status_code == 422
        assert not logger_mock.debug.called
