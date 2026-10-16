# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Paraformer hot-phrase management (1.x ``AsrPhraseManager``).

A compile job in disguise: ``create``/``update`` POST ``/fine-tunes``
with ``mode=compile_asr_phrase`` and an empty file set; ``get``/
``list``/``delete`` hit ``/fine-tunes/outputs``. The resulting job id
doubles as the ``phrase_id`` that ``asr.recognize`` consumes. 1.x
``list_phrases`` accepted ``page=`` but dropped it in ``**kwargs``
before it reached the wire; v2 sends the documented ``page_no``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import AliasChoices, Field

from dashscope.v2.errors import check_response
from dashscope.v2.types.common import (
    JobStatus,
    ResponseBase,
    flatten_envelope,
    with_headers,
)

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
    """One compiled hot-phrase set; the job id doubles as ``phrase_id``."""

    phrase_id: Optional[str] = Field(
        None,
        validation_alias=AliasChoices(
            "phrase_id",
            "job_id",
            "finetuned_output",
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
        validation_alias=AliasChoices("phrases", "jobs", "outputs"),
    )


class Phrases:
    """``client.audio.phrases`` -- paraformer hot phrases."""

    def __init__(self, client) -> None:
        self._client = client

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
    ) -> Phrase:
        """Compile a new phrase set; the result carries ``phrase_id``."""
        _require_phrases(phrases)
        response = self._client.transport.request(
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
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            Phrase(**flatten_envelope(response.json())),
            response,
        )

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
    ) -> Phrase:
        """Recompile the phrase set behind ``phrase_id``."""
        _require_id(phrase_id, "phrase_id")
        _require_phrases(phrases)
        response = self._client.transport.request(
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
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            Phrase(**flatten_envelope(response.json())),
            response,
        )

    def get(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Phrase:
        """Fetch one compiled phrase set (1.x ``query_phrases``)."""
        _require_id(phrase_id, "phrase_id")
        response = self._client.transport.request(
            "GET",
            f"{_OUTPUTS_PATH}/{phrase_id}",
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            Phrase(**flatten_envelope(response.json())),
            response,
        )

    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> PhraseList:
        """List compiled phrase sets."""
        response = self._client.transport.request(
            "GET",
            _OUTPUTS_PATH,
            params={"page_no": page_no, "page_size": page_size},
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            PhraseList(**flatten_envelope(response.json())),
            response,
        )

    def delete(
        self,
        phrase_id: str,
        *,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        """Delete one compiled phrase set."""
        _require_id(phrase_id, "phrase_id")
        response = self._client.transport.request(
            "DELETE",
            f"{_OUTPUTS_PATH}/{phrase_id}",
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(
            JobStatus(**flatten_envelope(response.json())),
            response,
        )
