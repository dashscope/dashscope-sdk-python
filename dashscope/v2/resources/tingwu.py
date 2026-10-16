# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.tingwu resource (design doc M3 3.4 follow-up).

Offline summarization (``call``, one POST returning the parsed result)
and realtime speech understanding (``realtime``, a duplex websocket
session used as a context manager). Mirrors the 1.x
``dashscope.multimodal.tingwu`` pair.
"""

# "input" is the wire-protocol field name for task payloads.
# pylint: disable=redefined-builtin

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Union

from pydantic import ConfigDict, Field

from dashscope.v2.errors import check_response, error_from_business_code
from dashscope.v2.transport.ws import Connector, WsSession
from dashscope.v2.transport.ws_connection import WsConnection, WsTask
from dashscope.v2.types.common import (
    ResponseBase,
    Usage,
    response_envelope,
    with_headers,
)

__all__ = [
    "AsyncTingWu",
    "TingWu",
    "TingWuRealtime",
    "TingWuResult",
]

MODULE = "multimodal"

SERVICE = "aigc/multimodal-generation/generation"


class TingWuResult(ResponseBase):
    """Parsed offline TingWu response.

    ``output`` keeps the result dict (``TaskId`` / ``Result`` / ...) and
    ``usage`` the token counters, both as sent by the service.
    """

    model_config = ConfigDict(extra="allow")

    output: Dict[str, Any] = Field(default_factory=dict)
    usage: Optional[Usage] = None


def _realtime_start_input(
    app_id: str,
    workspace_id: Optional[str],
    data_id: Optional[str],
) -> Dict[str, Any]:
    input: Dict[str, Any] = {"appId": app_id, "directive": "start"}
    if workspace_id is not None:
        input["workspace_id"] = workspace_id
    if data_id is not None:
        input["dataId"] = data_id
    return input


class TingWuRealtime:
    """Realtime TingWu session over one duplex websocket connection.

    Usage::

        with client.tingwu.realtime(
            model="tingwu-turbo",
            app_id="your-app-id",
        ) as session:
            session.send_audio_frame(chunk)
            ...
            session.stop()
            for event in session.events():
                ...
    """

    def __init__(
        self,
        client,
        *,
        model: str,
        app_id: str,
        format: str = "pcm",  # pylint: disable=redefined-builtin
        sample_rate: int = 16000,
        terminology: Optional[str] = None,
        max_end_silence: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        data_id: Optional[str] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> None:
        config = client.config
        self._model = model
        self._app_id = app_id
        merged: Dict[str, Any] = {
            "format": format,
            "sampleRate": sample_rate,
        }
        if parameters:
            merged.update(parameters)
        # Named kwargs win over the ``parameters=`` escape hatch; the keys
        # match the 1.x TingWuRealtime wire names (asr convention).
        if terminology is not None:
            merged["terminology"] = terminology
        if max_end_silence is not None:
            merged["maxEndSilence"] = max_end_silence
        self._parameters = merged
        self._start_input = _realtime_start_input(
            app_id,
            workspace_id,
            data_id,
        )
        # A shared connection runs this session as one task on it;
        # ws_url/connector/api_key/workspace then belong to the
        # connection and the per-session values are ignored.
        self._connection = connection
        self._session_kwargs = {
            "url": ws_url,
            "api_key": api_key or config.api_key,
            "workspace": workspace or config.workspace,
            "region": config.region,
            "connector": connector,
            "module": MODULE,
            "timeout": timeout,
        }
        self._session: Optional[Union[WsSession, WsTask]] = None

    @property
    def session(self) -> Union[WsSession, WsTask]:
        if self._session is None:
            raise RuntimeError("tingwu realtime session is not open")
        return self._session

    def __enter__(self) -> "TingWuRealtime":
        if self._connection is not None:
            self._session = self._connection.open_task(
                timeout=self._session_kwargs["timeout"],
            )
        else:
            self._session = WsSession(**self._session_kwargs)
        # WsSession opens the socket in its constructor, so a run_task
        # failure (bad api_key/app_id/model, unexpected first frame) would
        # leak the connection: Python skips __exit__ when __enter__ raises.
        # Close it here, matching the guard Tts.stream / asr.Recognition use.
        try:
            self._session.run_task(
                task_group="aigc",
                task="multimodal-generation",
                function="generation",
                model=self._model,
                input=self._start_input,
                parameters=self._parameters,
            )
        except BaseException:
            self._session.close()
            self._session = None
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def send_audio_frame(self, data: bytes) -> None:
        """Send one raw audio frame (format/sampleRate as configured)."""
        self.session.send_binary(data)

    def stop(self) -> None:
        """Signal end of audio; results keep arriving as events."""
        # 1.x sends appId=None here (tingwu_realtime.py:139 reads the
        # never-assigned ``_Request.app_id``); v2 sends the real app_id.
        self.session.finish_task(
            input={"appId": self._app_id, "directive": "stop"},
        )

    def events(self) -> Iterator[Dict[str, Any]]:
        """Yield result events until ``speech-end`` (or session end).

        A payload ``task-failed`` action raises through the error catalog,
        like the header-level task-failed the session itself raises on.
        """
        session = self.session
        while not session.finished:
            event = session.recv_event()
            output = (event.get("payload") or {}).get("output") or {}
            action = output.get("action")
            if action == "task-failed":
                # The frame's request_id lives in the header; forward it
                # like WsSession does for header-level task-failed.
                header = event.get("header") or {}
                request_id = header.get("request_id")
                raise error_from_business_code(
                    message=output.get("errorMessage") or "task failed",
                    code=output.get("errorCode"),
                    body=event,
                    headers=(
                        {"x-request-id": request_id} if request_id else None
                    ),
                )
            yield event
            if action == "speech-end":
                break


class TingWu:
    """``client.tingwu``: offline summarization + realtime understanding."""

    def __init__(self, client) -> None:
        self._client = client

    def call(
        self,
        *,
        model: str,
        input: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> TingWuResult:
        """Run one offline TingWu request (1.x ``TingWu.call``)."""
        body: Dict[str, Any] = {"model": model, "input": dict(input)}
        if parameters:
            body["parameters"] = dict(parameters)
        response = self._client.transport.request(
            "POST",
            f"/services/{SERVICE}",
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        envelope = response.json()
        data = response_envelope(envelope)
        output = envelope.get("output")
        data["output"] = output if isinstance(output, dict) else {}
        if envelope.get("usage") is not None:
            data["usage"] = envelope["usage"]
        return with_headers(TingWuResult.model_validate(data), response)

    def realtime(
        self,
        *,
        model: str,
        app_id: str,
        format: str = "pcm",  # pylint: disable=redefined-builtin
        sample_rate: int = 16000,
        terminology: Optional[str] = None,
        max_end_silence: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        data_id: Optional[str] = None,
        ws_url: Optional[str] = None,
        connector: Optional[Connector] = None,
        connection: Optional[WsConnection] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> TingWuRealtime:
        """Open a realtime TingWu session (use as a context manager).

        With ``connection=`` the session runs as one task on the shared
        socket (binary uplink is single-owner per connection).
        """
        return TingWuRealtime(
            self._client,
            model=model,
            app_id=app_id,
            format=format,
            sample_rate=sample_rate,
            terminology=terminology,
            max_end_silence=max_end_silence,
            parameters=parameters,
            workspace_id=workspace_id,
            data_id=data_id,
            ws_url=ws_url,
            connector=connector,
            connection=connection,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )


class AsyncTingWu:
    """Async ``client.tingwu``.

    Realtime sessions stay sync-only, matching the rest of v2 (design
    doc M3 3.5).
    """

    def __init__(self, client) -> None:
        self._client = client

    async def call(
        self,
        *,
        model: str,
        input: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> TingWuResult:
        """Run one offline TingWu request (1.x ``TingWu.async_call``)."""
        body: Dict[str, Any] = {"model": model, "input": dict(input)}
        if parameters:
            body["parameters"] = dict(parameters)
        response = await self._client.transport.request(
            "POST",
            f"/services/{SERVICE}",
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        envelope = response.json()
        data = response_envelope(envelope)
        output = envelope.get("output")
        data["output"] = output if isinstance(output, dict) else {}
        if envelope.get("usage") is not None:
            data["usage"] = envelope["usage"]
        return with_headers(TingWuResult.model_validate(data), response)
