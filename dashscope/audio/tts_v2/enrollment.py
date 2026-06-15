# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import time
from typing import List

import aiohttp

from dashscope.client.base_api import BaseApi
from dashscope.common.constants import ApiProtocol, HTTPMethod
from dashscope.common.logging import logger


class VoiceEnrollmentException(Exception):
    def __init__(
        self,
        request_id: str,
        status_code: int,
        code: str,
        error_message: str,
    ) -> None:
        self._request_id = request_id
        self._status_code = status_code
        self._code = code
        self._error_message = error_message

    def __str__(self):
        return f"Request: {self._request_id}, Status Code: {self._status_code}, Code: {self._code}, Error Message: {self._error_message}"  # noqa: E501  # pylint: disable=line-too-long


class VoiceEnrollmentService(BaseApi):
    """
    API for voice clone service
    """

    MAX_QUERY_TRY_COUNT = 3

    def __init__(
        self,
        api_key=None,
        workspace=None,
        model=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._workspace = workspace
        self._kwargs = kwargs
        self._last_request_id = None
        self.model = model
        if self.model is None:
            self.model = "voice-enrollment"

    def __call_with_input(  # pylint: disable=redefined-builtin
        self,
        input,
    ):
        try_count = 0
        while True:
            try:
                response = super().call(
                    model=self.model,
                    task_group="audio",
                    task="tts",
                    function="customization",
                    input=input,
                    api_protocol=ApiProtocol.HTTP,
                    http_method=HTTPMethod.POST,
                    api_key=self._api_key,
                    workspace=self._workspace,
                    **self._kwargs,
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
                logger.error(e)
                try_count += 1
                if try_count <= VoiceEnrollmentService.MAX_QUERY_TRY_COUNT:
                    time.sleep(2)
                    continue

            break
        logger.debug(">>>>recv %s", response)
        return response

    def create_voice(
        self,
        target_model: str,
        prefix: str,
        url: str,
        language_hints: List[str] = None,
        max_prompt_audio_length: float = None,
        **kwargs,
    ) -> str:
        """
        Create a new cloned voice.
        param: target_model TTS model version for the cloned voice
        param: prefix Custom voice prefix, only digits and lowercase
            letters allowed, less than 10 characters.
        param: url Audio file URL for voice cloning
        param: language_hints Target language for the cloned voice
        param: max_prompt_audio_length Max length of prompt audio output
            from audio preprocessing, in seconds. Default is 10s.
        param: kwargs Additional parameters
        return: voice_id
        """

        input_params = {
            "action": "create_voice",
            "target_model": target_model,
            "prefix": prefix,
            "url": url,
        }
        if language_hints is not None:
            input_params["language_hints"] = language_hints
        if max_prompt_audio_length is not None:
            input_params["max_prompt_audio_length"] = max_prompt_audio_length
        if kwargs:
            input_params.update(kwargs)
        response = self.__call_with_input(input_params)
        self._last_request_id = response.request_id
        if response.status_code == 200:
            return response.output["voice_id"]
        else:
            raise VoiceEnrollmentException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def list_voices(
        self,
        prefix=None,
        page_index: int = 0,
        page_size: int = 10,
    ) -> List[dict]:
        """
        List all created voices.
        param: page_index Page index for query
        param: page_size Page size
        return: List[dict] Voice list, including id, creation time,
            modification time, and status for each voice.
        """
        if prefix:
            # pylint: disable=no-value-for-parameter
            response = self.__call_with_input(
                input={
                    "action": "list_voice",
                    "prefix": prefix,
                    "page_index": page_index,
                    "page_size": page_size,
                },
            )
        else:
            # pylint: disable=no-value-for-parameter
            response = self.__call_with_input(
                input={
                    "action": "list_voice",
                    "page_index": page_index,
                    "page_size": page_size,
                },
            )
        self._last_request_id = response.request_id
        if response.status_code == 200:
            return response.output["voice_list"]
        else:
            raise VoiceEnrollmentException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def query_voice(self, voice_id: str) -> List[str]:
        """
        Query voice details.
        param: voice_id Voice ID to query
        return: bytes Audio used for voice registration
        """
        # pylint: disable=no-value-for-parameter
        response = self.__call_with_input(
            input={
                "action": "query_voice",
                "voice_id": voice_id,
            },
        )
        self._last_request_id = response.request_id
        if response.status_code == 200:
            return response.output
        else:
            raise VoiceEnrollmentException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def update_voice(self, voice_id: str, url: str) -> None:
        """
        Update voice.
        param: voice_id Voice ID
        param: url Audio file URL for cloning
        """
        # pylint: disable=no-value-for-parameter
        response = self.__call_with_input(
            input={
                "action": "update_voice",
                "voice_id": voice_id,
                "url": url,
            },
        )
        self._last_request_id = response.request_id
        if response.status_code == 200:
            return
        else:
            raise VoiceEnrollmentException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def delete_voice(self, voice_id: str) -> None:
        """
        Delete voice.
        param: voice_id Voice ID to delete
        """
        # pylint: disable=no-value-for-parameter
        response = self.__call_with_input(
            input={
                "action": "delete_voice",
                "voice_id": voice_id,
            },
        )
        self._last_request_id = response.request_id
        if response.status_code == 200:
            return
        else:
            raise VoiceEnrollmentException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def get_last_request_id(self):
        return self._last_request_id
