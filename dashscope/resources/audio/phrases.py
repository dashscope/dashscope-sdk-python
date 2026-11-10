# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Paraformer hot-phrase management (1.x ``AsrPhraseManager``).

A compile job in disguise: ``create``/``update`` POST ``/fine-tunes``
with ``training_type=compile_asr_phrase`` and an empty file set; ``get``/
``list``/``delete`` hit ``/fine-tunes/outputs``. ``finetuned_output`` is
the compiled resource ID that ``asr.recognize`` consumes as ``phrase_id``;
``job_id`` identifies the compile job and is only a legacy fallback. 1.x
``list_phrases`` accepted ``page=`` but dropped it in ``**kwargs``
before it reached the wire; v2 sends the documented ``page_no``.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from pydantic import AliasChoices, Field

from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    JobStatus,
    ResponseBase,
    flatten_envelope,
    response_body,
    with_headers,
)

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

__all__ = [
    "Phrase",
    "PhraseList",
    "Phrases",
]

MODULE = "audio"

_CREATE_PATH = "/fine-tunes"
_OUTPUTS_PATH = "/fine-tunes/outputs"


def _require_id(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_phrases(phrases: Dict[str, Any]) -> None:
    if not phrases:
        raise ValueError("phrases must be a non-empty dict")


def _compile_body(
    *,
    model: str,
    phrases: Dict[str, Any],
    training_type: str,
    finetuned_output: Optional[str],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "training_file_ids": [],
        "validation_file_ids": [],
        "training_type": training_type,
        "hyper_parameters": {"phrase_list": phrases},
    }
    if finetuned_output is not None:
        body["finetuned_output"] = finetuned_output
    return body


class Phrase(ResponseBase):
    """One compiled phrase set; prefer its resource ID over the compile job ID."""

    phrase_id: Optional[str] = Field(
        None,
        validation_alias=AliasChoices(
            "phrase_id",
            "finetuned_output",
            "job_id",
        ),
    )
    status: Optional[str] = None


class PhraseList(ResponseBase):
    page_no: int = 1
    page_size: int = 10
    total_count: int = Field(
        0,
        validation_alias=AliasChoices("total_count", "total"),
    )
    phrases: List[Phrase] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "phrases",
            "jobs",
            "outputs",
            "finetuned_outputs",
        ),
    )


class Phrases:
    """``client.audio.phrases`` -- paraformer hot phrases."""

    def __init__(self, client) -> None:
        self._client = client

    def _parse(self, model_cls, body: Dict[str, Any], response):
        """Build one response model, degrading rather than raising."""
        return parse_model(
            model_cls,
            flatten_envelope(body),
            strict=self._client.config.strict_validation,
            response=response,
        )

    @overload
    def create(
        self,
        *,
        model: str,
        phrases: Dict[str, Any],
        training_type: str = "compile_asr_phrase",
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Phrase:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        phrases: Dict[str, Any],
        training_type: str = "compile_asr_phrase",
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def create(
        self,
        *,
        model: str,
        phrases: Dict[str, Any],
        training_type: str = "compile_asr_phrase",
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Phrase, "APIResponse"]:
        """Compile a new phrase set; the result carries ``phrase_id``.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. The non-raw path
        keeps ``response_body``'s envelope: it preserves non-JSON failure
        bodies and injects ``status_code``/``request_id``/``is_error``.
        """
        _require_phrases(phrases)
        resp = self._client.transport.request_api(
            "POST",
            _CREATE_PATH,
            json=_compile_body(
                model=model,
                phrases=phrases,
                training_type=training_type,
                finetuned_output=None,
            ),
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        body = response_body(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            self._parse(Phrase, body, resp.http_response),
            resp.http_response,
        )

    @overload
    def update(
        self,
        phrase_id: str,
        *,
        model: str,
        phrases: Dict[str, Any],
        training_type: str = "compile_asr_phrase",
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Phrase:
        ...

    @overload
    def update(
        self,
        phrase_id: str,
        *,
        model: str,
        phrases: Dict[str, Any],
        training_type: str = "compile_asr_phrase",
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def update(
        self,
        phrase_id: str,
        *,
        model: str,
        phrases: Dict[str, Any],
        training_type: str = "compile_asr_phrase",
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Phrase, "APIResponse"]:
        """Recompile the phrase set behind ``phrase_id``.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. The non-raw path
        keeps ``response_body``'s envelope: it preserves non-JSON failure
        bodies and injects ``status_code``/``request_id``/``is_error``.
        """
        _require_id(phrase_id, "phrase_id")
        _require_phrases(phrases)
        resp = self._client.transport.request_api(
            "POST",
            _CREATE_PATH,
            json=_compile_body(
                model=model,
                phrases=phrases,
                training_type=training_type,
                finetuned_output=phrase_id,
            ),
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        body = response_body(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            self._parse(Phrase, body, resp.http_response),
            resp.http_response,
        )

    @overload
    def get(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Phrase:
        ...

    @overload
    def get(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def get(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Phrase, "APIResponse"]:
        """Fetch one compiled phrase set (1.x ``query_phrases``).

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. The non-raw path
        keeps ``response_body``'s envelope: it preserves non-JSON failure
        bodies and injects ``status_code``/``request_id``/``is_error``.
        """
        _require_id(phrase_id, "phrase_id")
        resp = self._client.transport.request_api(
            "GET",
            f"{_OUTPUTS_PATH}/{phrase_id}",
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        body = response_body(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            self._parse(Phrase, body, resp.http_response),
            resp.http_response,
        )

    @overload
    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> PhraseList:
        ...

    @overload
    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[PhraseList, "APIResponse"]:
        """List compiled phrase sets.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. The non-raw path
        keeps ``response_body``'s envelope: it preserves non-JSON failure
        bodies and injects ``status_code``/``request_id``/``is_error``.
        """
        resp = self._client.transport.request_api(
            "GET",
            _OUTPUTS_PATH,
            params={"page_no": page_no, "page_size": page_size},
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        body = response_body(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            self._parse(PhraseList, body, resp.http_response),
            resp.http_response,
        )

    @overload
    def delete(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobStatus:
        ...

    @overload
    def delete(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def delete(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobStatus, "APIResponse"]:
        """Delete one compiled phrase set.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. The non-raw path
        keeps ``response_body``'s envelope: it preserves non-JSON failure
        bodies and injects ``status_code``/``request_id``/``is_error``.
        """
        _require_id(phrase_id, "phrase_id")
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_OUTPUTS_PATH}/{phrase_id}",
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        body = response_body(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            self._parse(JobStatus, body, resp.http_response),
            resp.http_response,
        )
