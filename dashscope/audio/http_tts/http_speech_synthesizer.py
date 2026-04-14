# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus
from typing import Dict, Iterator, List, Optional, Union

from dashscope.api_entities.dashscope_response import (
    DashScopeAPIResponse,
    SpeechSynthesisResponse,
)
from dashscope.client.base_api import BaseApi
from dashscope.common.constants import HTTPMethod


class HttpSpeechSynthesisResult:
    """The result of HTTP speech synthesis."""

    def __init__(
        self,
        audio_data: Optional[bytes] = None,
        audio_url: Optional[str] = None,
        audio_id: Optional[str] = None,
        expires_at: Optional[int] = None,
        sentences: Optional[List[Dict]] = None,
        response: Optional[SpeechSynthesisResponse] = None,
    ):
        self._audio_data = audio_data
        self._audio_url = audio_url
        self._audio_id = audio_id
        self._expires_at = expires_at
        self._sentences = sentences or []
        self._response = response

    @property
    def audio_data(self) -> Optional[bytes]:
        """Get the audio data (for streaming mode)."""
        return self._audio_data

    @property
    def audio_url(self) -> Optional[str]:
        """Get the audio URL (for non-streaming mode)."""
        return self._audio_url

    @property
    def audio_id(self) -> Optional[str]:
        """Get the audio ID."""
        return self._audio_id

    @property
    def expires_at(self) -> Optional[int]:
        """Get the URL expiration timestamp."""
        return self._expires_at

    @property
    def sentences(self) -> List[Dict]:
        """Get the sentence-level synthesis results (for streaming mode)."""
        return self._sentences

    @property
    def response(self) -> Optional[SpeechSynthesisResponse]:
        """Get the full API response."""
        return self._response


class HttpSpeechSynthesizer(BaseApi):
    """HTTP-based text-to-speech interface for CosyVoice."""

    class AudioFormat:
        WAV = "wav"
        PCM = "pcm"
        MP3 = "mp3"

    @classmethod
    def call(  # pylint: disable=arguments-renamed
        cls,
        model: str,
        text: str,
        voice: str,
        audio_format: str = "wav",
        sample_rate: int = 24000,
        stream: bool = False,
        workspace: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs,
    ) -> Union[HttpSpeechSynthesisResult, Iterator[HttpSpeechSynthesisResult]]:
        """Convert text to speech via HTTP API.

        Args:
            model (str): The speech synthesis model, e.g.,
                'cosyvoice-v3-flash'.
            text (str): The text to synthesize.
            voice (str): The voice to use for synthesis.
            audio_format (str): Audio encoding format ('wav', 'pcm', 'mp3').
                Defaults to 'wav'.
            sample_rate (int): Audio sample rate in Hz. Defaults to 24000.
            stream (bool): Whether to use streaming (SSE) mode.
                Defaults to False.
            workspace (str): The DashScope workspace ID.
            api_key (str): The DashScope API key.
            url (str): custom http url if needed.
            **kwargs: Additional parameters like volume, rate, pitch, etc.

        Returns:
            HttpSpeechSynthesisResult: For non-streaming mode.
            Iterator[HttpSpeechSynthesisResult]: For streaming mode.
        """
        # Build request body
        body = {
            "model": model,
            "input": {
                "text": text,
                "voice": voice,
                "format": audio_format,
                "sample_rate": sample_rate,
                **{k: v for k, v in kwargs.items() if v is not None},
            },
        }

        # Prepare headers
        headers = {}
        if stream:
            headers["X-DashScope-SSE"] = "enable"

        # Make the HTTP request
        response = cls._http_call(
            method=HTTPMethod.POST,
            body=body,
            headers=headers if headers else None,
            stream=stream,
            workspace=workspace,
            api_key=api_key,
            url=url,
        )

        if stream:
            return cls._handle_streaming_response(response)
        else:
            return cls._handle_non_streaming_response(response)

    @classmethod
    def _http_call(
        cls,
        method: str,
        body: Dict,
        headers: Optional[Dict] = None,
        stream: bool = False,
        workspace: Optional[str] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """Make HTTP API call using BaseApi infrastructure."""
        from dashscope.api_entities.http_request import HttpRequest

        # Get base URL
        import dashscope
        from dashscope.common.utils import join_url

        if url:
            base_url = url
        else:
            base_url = dashscope.base_http_api_url
        url_for_call = join_url(
            base_url,
            "services/audio/tts/SpeechSynthesizer",
        )

        # Get API key
        from dashscope.common.api_key import get_default_api_key

        if api_key is None:
            api_key = get_default_api_key()

        # Prepare workspace header
        workspace_headers = {}
        if workspace:
            workspace_headers["X-DashScope-Workspace"] = workspace

        # Create request
        request = HttpRequest(
            url=url_for_call,
            api_key=api_key,
            http_method=method,
            stream=stream,
        )

        # Add custom headers
        if headers:
            request.add_headers(headers)
        if workspace_headers:
            request.add_headers(workspace_headers)

        # Set request body
        request.data = _RequestData(body)

        return request.call()

    @classmethod
    def _handle_non_streaming_response(
        cls,
        response,
    ) -> HttpSpeechSynthesisResult:
        """Handle non-streaming response."""
        # Get the response
        api_response = response

        if isinstance(api_response, DashScopeAPIResponse):
            if api_response.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"Request failed with status {api_response.status_code}: "
                    f"{api_response.message}",
                )

            output = api_response.output or {}
            audio_info = output.get("audio", {})

            return HttpSpeechSynthesisResult(
                audio_url=audio_info.get("url"),
                audio_id=audio_info.get("id"),
                expires_at=audio_info.get("expires_at"),
            )
        else:
            # Handle raw dict response
            output = api_response.get("output", {})
            audio_info = output.get("audio", {})

            return HttpSpeechSynthesisResult(
                audio_url=audio_info.get("url"),
                audio_id=audio_info.get("id"),
                expires_at=audio_info.get("expires_at"),
            )

    @classmethod
    def _handle_streaming_response(
        cls,
        response,
    ) -> Iterator[HttpSpeechSynthesisResult]:
        """Handle streaming (SSE) response."""
        audio_data_parts = []
        sentences = []

        for part in response:
            if isinstance(part, DashScopeAPIResponse):
                if part.status_code != HTTPStatus.OK:
                    raise RuntimeError(
                        f"Stream error with status {part.status_code}: "
                        f"{part.message}",
                    )

                output = part.output or {}
                output_type = output.get("type", "")

                # Handle sentence events
                if output_type.startswith("sentence-"):
                    sentence_info = output.get("sentence", {})
                    if sentence_info:
                        sentences.append(sentence_info)

                    # Yield intermediate result with audio frame
                    audio_data = output.get("audio", {}).get("data")
                    if audio_data:
                        import base64

                        audio_bytes = base64.b64decode(audio_data)
                        audio_data_parts.append(audio_bytes)
                        yield HttpSpeechSynthesisResult(
                            audio_data=audio_bytes,
                            sentences=sentences.copy(),
                            response=(
                                SpeechSynthesisResponse.from_api_response(part)
                            ),
                        )

                # Handle final message
                elif output.get("finish_reason") == "stop":
                    audio_info = output.get("audio", {})

                    yield HttpSpeechSynthesisResult(
                        audio_data=(
                            b"".join(audio_data_parts)
                            if audio_data_parts
                            else None
                        ),
                        audio_url=audio_info.get("url"),
                        audio_id=audio_info.get("id"),
                        expires_at=audio_info.get("expires_at"),
                        sentences=sentences.copy(),
                    )
            else:
                # Handle raw dict
                output = part.get("output", {})
                output_type = output.get("type", "")

                if output_type.startswith("sentence-"):
                    sentence_info = output.get("sentence", {})
                    if sentence_info:
                        sentences.append(sentence_info)

                    audio_data = output.get("audio", {}).get("data")
                    if audio_data:
                        import base64

                        audio_bytes = base64.b64decode(audio_data)
                        audio_data_parts.append(audio_bytes)
                        yield HttpSpeechSynthesisResult(
                            audio_data=audio_bytes,
                            sentences=sentences.copy(),
                        )

                elif output.get("finish_reason") == "stop":
                    audio_info = output.get("audio", {})
                    yield HttpSpeechSynthesisResult(
                        audio_data=(
                            b"".join(audio_data_parts)
                            if audio_data_parts
                            else None
                        ),
                        audio_url=audio_info.get("url"),
                        audio_id=audio_info.get("id"),
                        expires_at=audio_info.get("expires_at"),
                        sentences=sentences.copy(),
                    )


class _RequestData:
    """Wrapper for request data to provide required interface."""

    def __init__(self, data: Dict):
        self._data = data

    def get_http_payload(self):
        """Return HTTP payload."""
        return False, None, self._data

    def get_aiohttp_payload(self):
        """Return aiohttp payload."""
        return False, self._data

    @property
    def parameters(self):
        """Return query parameters."""
        return {}
