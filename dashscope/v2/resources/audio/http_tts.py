# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""HTTP one-shot and streaming speech synthesis (real HTTP POST, no ws).

V2 form of the two 1.x HTTP TTS families, sharing one entry:

- sambert/cosyvoice (1.x ``HttpSpeechSynthesizer``) ->
  ``services/audio/tts/SpeechSynthesizer``
- qwen-tts family (1.x qwen ``SpeechSynthesizer``: qwen-tts,
  qwen3-tts-flash) -> ``services/aigc/multimodal-generation/generation``

Both POST the same ``{model, input}`` envelope and answer with
``output.audio``; streaming rides SSE via the ``X-DashScope-SSE: enable``
header, framed exactly like the 1.x ``StreamEventMixin`` wire (see
``resources/_sse.py``).
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Iterator, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from dashscope.v2.errors import DashScopeError, check_response
from dashscope.v2.resources._sse import _iter_sse, _stream_error
from dashscope.v2.types.common import with_headers

__all__ = [
    "HttpSynthesis",
    "HttpSynthesisChunk",
    "HttpSynthesisStream",
    "HttpTts",
]

MODULE = "audio"

_SSE_HEADER = {"X-DashScope-SSE": "enable"}
_DEFAULT_PATH = "/services/audio/tts/SpeechSynthesizer"
_QWEN_PATH = "/services/aigc/multimodal-generation/generation"


def _path_for(model: str) -> str:
    # qwen-tts / qwen3-tts-flash ride the multimodal-generation service;
    # sambert/cosyvoice ride the audio tts service.
    if model.startswith("qwen"):
        return _QWEN_PATH
    return _DEFAULT_PATH


def _build_body(
    text: str,
    *,
    model: str,
    voice: str,
    format: Optional[str],  # pylint: disable=redefined-builtin
    sample_rate: Optional[int],
    language_type: Optional[str],
    parameters: Optional[Dict[str, Any]],
    extra_input: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    input_: Dict[str, Any] = dict(extra_input or {})
    # Named arguments win over the ``extra_input=`` escape hatch; the
    # keys match the 1.x wire names.
    input_["text"] = text
    input_["voice"] = voice
    if format is not None:
        input_["format"] = format
    if sample_rate is not None:
        input_["sample_rate"] = sample_rate
    if language_type is not None:
        input_["language_type"] = language_type
    body: Dict[str, Any] = {"model": model, "input": input_}
    if parameters is not None:
        body["parameters"] = parameters
    return body


def _audio_dict(output: Dict[str, Any]) -> Dict[str, Any]:
    audio = output.get("audio")
    return audio if isinstance(audio, dict) else {}


def _usage_dict(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = body.get("usage")
    return usage if isinstance(usage, dict) else None


class HttpSynthesis(BaseModel):
    """Non-streaming result: a URL reference to the rendered audio."""

    model_config = ConfigDict(extra="allow")

    output: Dict[str, Any] = {}
    usage: Optional[Dict[str, Any]] = None
    request_id: str = ""
    # Raw HTTP response headers, 1.x parity (``GenerationResponse.headers``);
    # filled via ``with_headers`` like every other non-streaming success.
    headers: Dict[str, str] = Field(default_factory=dict)

    @property
    def audio_url(self) -> Optional[str]:
        """Temporary URL of the rendered audio."""
        return _audio_dict(self.output).get("url")

    @property
    def audio_id(self) -> Optional[str]:
        return _audio_dict(self.output).get("id")

    @property
    def expires_at(self) -> Optional[int]:
        """Expiry timestamp of :attr:`audio_url`."""
        return _audio_dict(self.output).get("expires_at")


class HttpSynthesisChunk(BaseModel):
    """One SSE part of a streaming synthesis."""

    model_config = ConfigDict(extra="allow")

    output: Dict[str, Any] = {}
    audio: Optional[bytes] = None
    usage: Optional[Dict[str, Any]] = None
    request_id: str = ""

    @property
    def sentence(self) -> Optional[Dict[str, Any]]:
        """Sentence metadata of cosyvoice ``sentence-*`` parts."""
        sentence = self.output.get("sentence")
        return sentence if isinstance(sentence, dict) else None

    @property
    def finish_reason(self) -> Optional[str]:
        return self.output.get("finish_reason")

    @property
    def is_final(self) -> bool:
        """True on the closing part (``finish_reason == "stop"``)."""
        return self.finish_reason == "stop"

    @property
    def audio_url(self) -> Optional[str]:
        """Temporary URL of the whole audio; final part only."""
        return _audio_dict(self.output).get("url")

    @property
    def audio_id(self) -> Optional[str]:
        return _audio_dict(self.output).get("id")

    @property
    def expires_at(self) -> Optional[int]:
        return _audio_dict(self.output).get("expires_at")


def _synthesis_from_body(
    body: Dict[str, Any],
    header_request_id: str,
) -> HttpSynthesis:
    output = body.get("output")
    return HttpSynthesis(
        output=output if isinstance(output, dict) else {},
        usage=_usage_dict(body),
        request_id=body.get("request_id") or header_request_id,
    )


def _chunk_from_data(
    data: str,
    header_request_id: str,
) -> HttpSynthesisChunk:
    try:
        body = json.loads(data)
    except ValueError as error:
        raise DashScopeError(
            f"invalid synthesis stream data: {data!r}",
        ) from error
    if not isinstance(body, dict):
        raise DashScopeError(f"invalid synthesis stream data: {data!r}")
    output = body.get("output")
    output = output if isinstance(output, dict) else {}
    payload = _audio_dict(output).get("data")
    return HttpSynthesisChunk(
        output=output,
        audio=base64.b64decode(payload) if payload else None,
        usage=_usage_dict(body),
        request_id=body.get("request_id") or header_request_id,
    )


class HttpSynthesisStream:
    """Typed iterator over streaming synthesis parts, mirroring AppStream."""

    def __init__(
        self,
        inner: Iterator[HttpSynthesisChunk],
        response: Any,
    ) -> None:
        self._inner = inner
        self._response = response

    def __iter__(self) -> Iterator[HttpSynthesisChunk]:
        return iter(self._inner)

    def __next__(self) -> HttpSynthesisChunk:
        return next(self._inner)

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return dict(self._response.headers)

    def close(self) -> None:
        self._inner.close()


class HttpTts:
    """``client.audio.tts.http`` -- HTTP synthesis for both families."""

    def __init__(self, client) -> None:
        self._client = client

    def synthesize(
        self,
        text: str,
        *,
        model: str,
        voice: str,
        format: Optional[str] = None,  # pylint: disable=redefined-builtin
        sample_rate: Optional[int] = None,
        language_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> HttpSynthesis:
        """Synthesize in one POST; the audio comes back as a URL."""
        response = self._client.transport.request(
            "POST",
            _path_for(model),
            json=_build_body(
                text,
                model=model,
                voice=voice,
                format=format,
                sample_rate=sample_rate,
                language_type=language_type,
                parameters=parameters,
                extra_input=extra_input,
            ),
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            _synthesis_from_body(
                response.json(),
                response.headers.get("x-request-id", ""),
            ),
            response,
        )

    def stream(
        self,
        text: str,
        *,
        model: str,
        voice: str,
        format: Optional[str] = None,  # pylint: disable=redefined-builtin
        sample_rate: Optional[int] = None,
        language_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> HttpSynthesisStream:
        """Stream the synthesis over SSE, one chunk per part.

        Audio frames arrive base64-decoded on :attr:`HttpSynthesisChunk.audio`;
        the final chunk (``is_final``) carries the whole-audio URL reference.
        The envelope's raw HTTP headers are on
        :attr:`HttpSynthesisStream.headers`.
        """
        sse_headers = dict(_SSE_HEADER)
        if headers:
            sse_headers.update(headers)
        response = self._client.transport.open_stream(
            "POST",
            _path_for(model),
            json=_build_body(
                text,
                model=model,
                voice=voice,
                format=format,
                sample_rate=sample_rate,
                language_type=language_type,
                parameters=parameters,
                extra_input=extra_input,
            ),
            headers=sse_headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
        )
        return HttpSynthesisStream(self._events(response), response)

    def _events(self, response) -> Iterator[HttpSynthesisChunk]:
        header_request_id = response.headers.get("x-request-id", "")
        try:
            for is_error, status, data in _iter_sse(response):
                if is_error:
                    raise _stream_error(status, data)
                yield _chunk_from_data(data, header_request_id)
        finally:
            response.close()
