# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.audio.asr resource (design doc M3 3.4).

Two shapes on one resource: batch file transcription (async Job over
HTTP, ``transcribe`` + ``jobs``) and realtime recognition (duplex
websocket session used as a context manager, ``recognize``).
"""

# pylint: disable=redefined-builtin

from __future__ import annotations

import asyncio
import os
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field

from dashscope.v2.errors import DashScopeError, error_from_business_code
from dashscope.v2.resources._jobs import (
    CANCELED,
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    FAILED,
    AsyncJobs,
    Job,
    Jobs,
)
from dashscope.v2.transport.ws import Connector, EVENT_GENERATED, WsSession
from dashscope.v2.transport.ws_connection import WsConnection, WsTask

__all__ = [
    "Asr",
    "AsyncAsr",
    "AsyncTranscriptionJobs",
    "Recognition",
    "RecognitionResult",
    "TranscriptionJobs",
]

SERVICE = "audio/asr/transcription"

MODULE = "audio"

# Local-file chunk size, mirroring 1.x Recognition.call (recognition.py:519
# reads 12800 bytes per frame).
FILE_CHUNK_SIZE = 12800


def _input_for(file_urls: List[str]) -> Dict[str, Any]:
    if not file_urls:
        raise ValueError("file_urls must not be empty")
    return {"file_urls": list(file_urls)}


def _stream_file_frames(
    recognition: "Recognition",
    path: str,
) -> Iterator[Dict[str, Any]]:
    """Send ``path`` in 1.x-sized chunks, then drain the result events."""
    with recognition:
        with open(path, "rb") as audio_file:
            while True:
                chunk = audio_file.read(FILE_CHUNK_SIZE)
                if not chunk:
                    break
                recognition.send_audio_frame(chunk)
        recognition.stop()
        yield from recognition.events()


def _phrase_resources(
    phrase_id: Optional[str],
) -> Optional[List[Dict[str, str]]]:
    if not phrase_id:
        return None
    return [{"resource_id": phrase_id, "resource_type": "asr_phrase"}]


def _check_terminal(job: Job, *, raise_on_error: bool) -> Job:
    """Apply ASR's error contract even to caller-supplied snapshots."""
    if job.status == FAILED and raise_on_error:
        raise error_from_business_code(
            message=job.message or f"task {job.task_id} failed",
            code=job.code,
            body=job.model_dump(),
            headers=job.headers,
        )
    if job.status == CANCELED and raise_on_error:
        raise DashScopeError(
            f"task {job.task_id} was canceled while waiting",
            request_id=job.request_id,
            headers=job.headers,
        )
    return job


class TranscriptionJobs:
    """Async task facade over the batch transcription service."""

    def __init__(self, client) -> None:
        self._client = client
        self._jobs = Jobs(client, SERVICE, MODULE)

    def create(
        self,
        *,
        model: str,
        file_urls: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return self._jobs.create(
            model=model,
            input=_input_for(file_urls),
            parameters={} if parameters is None else parameters,
            resources=_phrase_resources(phrase_id),
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )

    def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return self._jobs.fetch(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )

    def wait(
        self,
        job: Union[Job, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        _sleep: Optional[Callable[[float], Any]] = None,
    ) -> Job:
        """Poll with backoff, without starting another poll after the deadline."""
        task_id = job.task_id if isinstance(job, Job) else job
        sleep_fn = _sleep or time.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, Job)
            else self.fetch(task_id, api_key=api_key, workspace=workspace)
        )
        interval = poll_interval
        while not current.finished:
            # Compat-mode error envelopes have no terminal task status.
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for task {task_id}",
                )
            sleep_fn(min(interval, remaining))
            if time.monotonic() >= deadline:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for task {task_id}",
                )
            interval = min(interval * 2, max_interval)
            current = self.fetch(task_id, api_key=api_key, workspace=workspace)
        return _check_terminal(
            current,
            raise_on_error=self._client.raise_on_error,
        )


class AsyncTranscriptionJobs:
    def __init__(self, client) -> None:
        self._client = client
        self._jobs = AsyncJobs(client, SERVICE, MODULE)

    async def create(
        self,
        *,
        model: str,
        file_urls: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return await self._jobs.create(
            model=model,
            input=_input_for(file_urls),
            parameters={} if parameters is None else parameters,
            resources=_phrase_resources(phrase_id),
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )

    async def fetch(
        self,
        task_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        return await self._jobs.fetch(
            task_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )

    async def wait(
        self,
        job: Union[Job, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        _sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> Job:
        """Async twin of :meth:`TranscriptionJobs.wait`."""
        task_id = job.task_id if isinstance(job, Job) else job
        sleep_fn = _sleep or asyncio.sleep
        deadline = time.monotonic() + timeout
        current = (
            job
            if isinstance(job, Job)
            else await self.fetch(
                task_id,
                api_key=api_key,
                workspace=workspace,
            )
        )
        interval = poll_interval
        while not current.finished:
            # Compat-mode error envelopes have no terminal task status.
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for task {task_id}",
                )
            await sleep_fn(min(interval, remaining))
            if time.monotonic() >= deadline:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for task {task_id}",
                )
            interval = min(interval * 2, max_interval)
            current = await self.fetch(
                task_id,
                api_key=api_key,
                workspace=workspace,
            )
        return _check_terminal(
            current,
            raise_on_error=self._client.raise_on_error,
        )


class RecognitionResult(BaseModel):
    """Typed view over one ``result-generated`` event payload.

    ``Recognition.events()`` keeps yielding raw dicts; this model adds the
    1.x ``RecognitionResult`` accessors (recognition.py:99-125) for
    callers that want them. Unknown payload fields stay readable as
    attributes (``extra="allow"``).
    """

    model_config = ConfigDict(extra="allow")

    output: Dict[str, Any] = Field(default_factory=dict)
    usage: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

    def get_sentence(self) -> Optional[Dict[str, Any]]:
        """The sentence dict of this result, or None."""
        if self.output and "sentence" in self.output:
            return self.output["sentence"]
        return None

    def is_sentence_end(self) -> bool:
        """True when the sentence carries its end marker (``end_time``)."""
        sentence = self.get_sentence()
        return sentence is not None and sentence.get("end_time") is not None

    def get_usage(self) -> Optional[Dict[str, Any]]:
        # 1.x joins a usages list by the sentence's end_time; v2 events
        # carry usage on the payload itself, so it comes back directly.
        return self.usage


class Recognition:
    """Duplex recognition; consume events on one thread while sending audio."""

    def __init__(
        self,
        client,
        *,
        model: str,
        format: str = "pcm",
        sample_rate: int = 16000,
        vocabulary_id: Optional[str] = None,
        disfluency_removal_enabled: Optional[bool] = None,
        diarization_enabled: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        input: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> None:
        config = client.config
        self._model = model
        merged = {"format": format, "sample_rate": sample_rate}
        if parameters:
            merged.update(parameters)
        # Named kwargs win over the ``parameters=`` escape hatch; the keys
        # match the 1.x Recognition wire names.
        if vocabulary_id is not None:
            merged["vocabulary_id"] = vocabulary_id
        if disfluency_removal_enabled is not None:
            merged["disfluency_removal_enabled"] = disfluency_removal_enabled
        if diarization_enabled is not None:
            merged["diarization_enabled"] = diarization_enabled
        self._parameters = merged
        self._input = dict(input or {})
        # Phrases ride as top-level ``resources`` on the run-task payload
        # (verified against the service), not inside ``parameters``.
        self._resources = _phrase_resources(phrase_id)
        # A shared connection runs this recognition as one task on it;
        # ws_url/connector/api_key/workspace/headers then belong to the
        # connection and the per-session values are ignored.
        self._connection = connection
        self._session_kwargs = {
            "url": ws_url,
            "api_key": api_key or config.api_key,
            "workspace": workspace or config.workspace,
            "region": config.region,
            "connector": connector,
            "module": MODULE,
            "timeout": config.timeout if timeout is None else timeout,
            "headers": headers,
        }
        self._session: Union[WsSession, WsTask, None] = None
        self.last_response: Optional[Dict[str, Any]] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.first_package_delay: Optional[float] = None
        self.last_package_delay: Optional[float] = None
        self._first_sent_at: Optional[float] = None
        self._stopped_at: Optional[float] = None
        self._request_id: Optional[str] = None

    @property
    def session(self) -> Union[WsSession, WsTask]:
        if self._session is None:
            raise RuntimeError("recognition session is not open")
        return self._session

    @property
    def request_id(self) -> Optional[str]:
        return self._request_id

    def start(self) -> "Recognition":
        if self._session is not None:
            raise RuntimeError("recognition session is already open")
        if self._connection is not None:
            self._session = self._connection.open_task(
                timeout=self._session_kwargs["timeout"],
            )
        else:
            self._session = WsSession(**self._session_kwargs)
        self.last_response = None
        self.usage = None
        self.first_package_delay = None
        self.last_package_delay = None
        self._first_sent_at = None
        self._stopped_at = None
        self._request_id = self._session.task_id
        try:
            self.last_response = self.session.run_task(
                task_group="audio",
                task="asr",
                function="recognition",
                model=self._model,
                parameters=self._parameters,
                input=self._input,
                resources=self._resources,
            )
            self._request_id = self.session.request_id
        except BaseException:
            self.close()
            raise
        return self

    def __enter__(self) -> "Recognition":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def send_audio_frame(self, data: bytes) -> None:
        if self._first_sent_at is None:
            self._first_sent_at = time.monotonic()
        self.session.send_binary(data)

    def update_context(self, context: Dict[str, Any]) -> None:
        """Apply context to subsequent audio without restarting recognition."""
        if not isinstance(context, dict):
            raise TypeError("context must be a dict")
        self.session.continue_task(context)

    def stop(self) -> None:
        """End input; keep reading events through task-finished."""
        if self._stopped_at is None:
            self._stopped_at = time.monotonic()
        self.session.finish_task()

    def events(self) -> Iterator[Dict[str, Any]]:
        session = self.session
        while not session.finished:
            event = session.recv_event()
            self.last_response = event
            self._request_id = session.request_id
            payload = event.get("payload") or {}
            if payload.get("usage") is not None:
                self.usage = payload["usage"]
            output = payload.get("output") or {}
            sentence = output.get("sentence") or {}
            has_text = bool(
                sentence.get("text") and not sentence.get("heartbeat"),
            )
            if (
                has_text
                and self.first_package_delay is None
                and self._first_sent_at is not None
            ):
                self.first_package_delay = (
                    time.monotonic() - self._first_sent_at
                ) * 1000
            if session.finished and self._stopped_at is not None:
                self.last_package_delay = (
                    time.monotonic() - self._stopped_at
                ) * 1000
            yield event

    def close(self) -> None:
        # With a shared connection this only detaches the task; the
        # socket belongs to the WsConnection and stays open.
        if self._session is not None:
            self._session.close()
            self._session = None

    def results(self) -> Iterator[RecognitionResult]:
        """Yield typed results until the task finishes.

        Non-result events (task-finished, ...) are skipped: 1.x reports
        them through ``on_complete`` instead of ``on_event``. Heartbeat
        sentences are dropped too, as 1.x does before invoking callbacks.
        """
        for event in self.events():
            if event["header"].get("event") != EVENT_GENERATED:
                continue
            data = dict(event.get("payload") or {})
            data.setdefault("request_id", event["header"].get("request_id"))
            result = RecognitionResult.model_validate(data)
            sentence = result.get_sentence()
            if sentence and sentence.get("heartbeat"):
                continue
            yield result


class Asr:
    """``client.audio.asr``: batch transcription + realtime recognition."""

    def __init__(self, client) -> None:
        self._client = client
        self.jobs = TranscriptionJobs(client)

    def transcribe(
        self,
        file_urls: List[str],
        *,
        model: str = "qwen-audio-3.1-asr-flash-filetrans",
        parameters: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        submit_timeout: Optional[float] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Submit and wait; output.results carries per-file transcripts.

        ``timeout`` is the wait budget — the same ``timeout`` as
        ``jobs.wait``. ``submit_timeout`` caps only the submit request.
        """
        job = self.jobs.create(
            model=model,
            file_urls=file_urls,
            parameters=parameters,
            phrase_id=phrase_id,
            timeout=submit_timeout,
            api_key=api_key,
            workspace=workspace,
        )
        return self.jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )

    def recognize(
        self,
        *,
        model: str,
        format: str = "pcm",
        sample_rate: int = 16000,
        vocabulary_id: Optional[str] = None,
        disfluency_removal_enabled: Optional[bool] = None,
        diarization_enabled: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        input: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Recognition:
        """Open recognition; send audio and consume events concurrently.

        With ``connection=`` the session runs as one task on the shared
        socket (binary uplink is single-owner per connection, so one
        active recognition per connection).
        """
        return Recognition(
            self._client,
            model=model,
            format=format,
            sample_rate=sample_rate,
            vocabulary_id=vocabulary_id,
            disfluency_removal_enabled=disfluency_removal_enabled,
            diarization_enabled=diarization_enabled,
            parameters=parameters,
            input=input,
            phrase_id=phrase_id,
            ws_url=ws_url,
            connector=connector,
            connection=connection,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )

    def recognize_file(
        self,
        path: str,
        *,
        model: str,
        format: str = "pcm",  # pylint: disable=redefined-builtin
        sample_rate: int = 16000,
        vocabulary_id: Optional[str] = None,
        disfluency_removal_enabled: Optional[bool] = None,
        diarization_enabled: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Recognize a local audio file, yielding the same events as
        ``Recognition.events()``.

        The file is auto-chunked and driven to completion in one session,
        the 1.x ``Recognition.call(file)`` equivalent. Validation happens
        eagerly; the session itself opens on first iteration.
        """
        if not os.path.isfile(path):
            raise ValueError(f"audio file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"audio file is empty: {path}")
        recognition = self.recognize(
            model=model,
            format=format,
            sample_rate=sample_rate,
            vocabulary_id=vocabulary_id,
            disfluency_removal_enabled=disfluency_removal_enabled,
            diarization_enabled=diarization_enabled,
            parameters=parameters,
            ws_url=ws_url,
            connector=connector,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        return _stream_file_frames(recognition, path)


class AsyncAsr:
    """Async ``client.audio.asr``.

    Realtime ``recognize`` sessions stay sync-only (design doc M3 3.5).
    """

    def __init__(self, client) -> None:
        self.jobs = AsyncTranscriptionJobs(client)

    async def transcribe(
        self,
        file_urls: List[str],
        *,
        model: str = "qwen-audio-3.1-asr-flash-filetrans",
        parameters: Optional[Dict[str, Any]] = None,
        phrase_id: Optional[str] = None,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        submit_timeout: Optional[float] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Job:
        """Submit and wait; output.results carries per-file transcripts.

        ``timeout`` is the wait budget — the same ``timeout`` as
        ``jobs.wait``. ``submit_timeout`` caps only the submit request.
        """
        job = await self.jobs.create(
            model=model,
            file_urls=file_urls,
            parameters=parameters,
            phrase_id=phrase_id,
            timeout=submit_timeout,
            api_key=api_key,
            workspace=workspace,
        )
        return await self.jobs.wait(
            job,
            poll_interval=poll_interval,
            max_interval=max_interval,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
