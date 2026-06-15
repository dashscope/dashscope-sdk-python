# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import time
from typing import List

import aiohttp

from dashscope.client.base_api import BaseApi
from dashscope.common.constants import ApiProtocol, HTTPMethod
from dashscope.common.logging import logger


class VocabularyServiceException(Exception):
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


class VocabularyService(BaseApi):
    """
    API for asr vocabulary service
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
            self.model = "speech-biasing"

    def __call_with_input(self, input):  # pylint: disable=redefined-builtin
        try_count = 0
        while True:
            try:
                response = super().call(
                    model=self.model,
                    task_group="audio",
                    task="asr",
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
                if try_count <= VocabularyService.MAX_QUERY_TRY_COUNT:
                    time.sleep(2)
                    continue

            break
        logger.debug(">>>>recv %s", response)
        return response

    def create_vocabulary(
        self,
        target_model: str,
        prefix: str,
        vocabulary: List[dict],
    ) -> str:
        """
        Create a hot word table.
        param: target_model ASR model version for the hot word table
        param: prefix Custom hot word table prefix, only digits and
            lowercase letters allowed, less than 10 characters.
        param: vocabulary Hot word table dictionary
        return: Hot word table identifier vocabulary_id
        """
        # pylint: disable=no-value-for-parameter
        response = self.__call_with_input(
            input={
                "action": "create_vocabulary",
                "target_model": target_model,
                "prefix": prefix,
                "vocabulary": vocabulary,
            },
        )
        if response.status_code == 200:
            self._last_request_id = response.request_id
            return response.output["vocabulary_id"]
        else:
            raise VocabularyServiceException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def list_vocabularies(
        self,
        prefix=None,
        page_index: int = 0,
        page_size: int = 10,
    ) -> List[dict]:
        """
        List all created hot word tables.
        param: prefix Custom prefix, if set only returns hot word table
            identifiers with the specified prefix.
        param: page_index Page index for query
        param: page_size Page size
        return: List of hot word table identifiers
        """
        if prefix:
            # pylint: disable=no-value-for-parameter
            response = self.__call_with_input(
                input={
                    "action": "list_vocabulary",
                    "prefix": prefix,
                    "page_index": page_index,
                    "page_size": page_size,
                },
            )
        else:
            # pylint: disable=no-value-for-parameter
            response = self.__call_with_input(
                input={
                    "action": "list_vocabulary",
                    "page_index": page_index,
                    "page_size": page_size,
                },
            )
        if response.status_code == 200:
            self._last_request_id = response.request_id
            return response.output["vocabulary_list"]
        else:
            raise VocabularyServiceException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def query_vocabulary(self, vocabulary_id: str) -> List[dict]:
        """
        Get hot word table contents.
        param: vocabulary_id Hot word table identifier
        return: Hot word table
        """
        # pylint: disable=no-value-for-parameter
        response = self.__call_with_input(
            input={
                "action": "query_vocabulary",
                "vocabulary_id": vocabulary_id,
            },
        )
        if response.status_code == 200:
            self._last_request_id = response.request_id
            return response.output
        else:
            raise VocabularyServiceException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def update_vocabulary(
        self,
        vocabulary_id: str,
        vocabulary: List[dict],
    ) -> None:
        """
        Replace existing hot word table with a new one.
        param: vocabulary_id Hot word table identifier to replace
        param: vocabulary Hot word table
        """
        # pylint: disable=no-value-for-parameter
        response = self.__call_with_input(
            input={
                "action": "update_vocabulary",
                "vocabulary_id": vocabulary_id,
                "vocabulary": vocabulary,
            },
        )
        if response.status_code == 200:
            self._last_request_id = response.request_id
            return
        else:
            raise VocabularyServiceException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def delete_vocabulary(self, vocabulary_id: str) -> None:
        """
        Delete hot word table.
        param: vocabulary_id Hot word table identifier to delete
        """
        # pylint: disable=no-value-for-parameter
        response = self.__call_with_input(
            input={
                "action": "delete_vocabulary",
                "vocabulary_id": vocabulary_id,
            },
        )
        if response.status_code == 200:
            self._last_request_id = response.request_id
            return
        else:
            raise VocabularyServiceException(
                response.request_id,
                response.status_code,
                response.code,
                response.message,
            )

    def get_last_request_id(self):
        return self._last_request_id
