# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Audio customization management plane: vocabularies, voices, voice design.

V2 form of 1.x ``VocabularyService`` (asr hot-word tables,
``audio/asr/vocabulary.py``) and ``VoiceEnrollmentService`` (voice
cloning, ``audio/tts_v2/enrollment.py``), plus voice design (no 1.x SDK
class). All three ride the same envelope: ``POST /services/audio/{asr|tts}/
customization`` with ``{model, input: {action, ...}, parameters?}``.

Cloning and design are two doors onto one registry: both post
``model="voice-enrollment"`` to the tts path with the same action words and
the same ``voice_id``, and they share a single per-account voice quota.
They differ only in how a voice is created -- from a sample ``url``, or from
a ``voice_prompt`` describing the wanted sound.

1.x retried transport failures inside these services; v2 leaves that to the
transport's retry loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import AliasChoices, Field

from dashscope.types.common import (
    JobStatus,
    ResponseBase,
    flatten_envelope,
    response_body,
    with_headers,
)

__all__ = [
    "DesignedVoice",
    "DesignedVoiceList",
    "Vocabularies",
    "Vocabulary",
    "VocabularyList",
    "Voice",
    "VoiceDesign",
    "VoiceList",
    "Voices",
]

MODULE = "audio"


def _require_id(value: str, name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


class Vocabulary(ResponseBase):
    """One hot-word table; ``create``/``get`` return it."""

    vocabulary_id: Optional[str] = None
    vocabulary: Optional[List[Dict[str, Any]]] = None


class VocabularyList(ResponseBase):
    vocabularies: List[Dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("vocabularies", "vocabulary_list"),
    )


class Voice(ResponseBase):
    """One cloned voice; ``create``/``get`` return it."""

    voice_id: Optional[str] = None
    status: Optional[str] = None
    preview_audio: Optional[Dict[str, Any]] = None


class VoiceList(ResponseBase):
    voices: List[Dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("voices", "voice_list"),
    )


class DesignedVoice(ResponseBase):
    """One designed voice.

    The same registry entry as a cloned :class:`Voice` -- one endpoint, one
    ``voice_id``, one shared quota -- plus the two texts it was designed
    from.
    """

    voice_id: Optional[str] = None
    status: Optional[str] = None
    preview_audio: Optional[Dict[str, Any]] = None
    target_model: Optional[str] = None
    voice_prompt: Optional[str] = None
    preview_text: Optional[str] = None
    gmt_create: Optional[str] = None
    gmt_modified: Optional[str] = None


class DesignedVoiceList(ResponseBase):
    voices: List[Dict[str, Any]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("voices", "voice_list"),
    )


class _CustomizationBase:
    """Shared POST helper for the customization envelope."""

    PATH = ""
    DEFAULT_MODEL = ""

    def __init__(self, client) -> None:
        self._client = client

    def _call(
        self,
        action: str,
        fields: Dict[str, Any],
        *,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Any:
        body: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "input": {**fields, "action": action},
        }
        if parameters is not None:
            body["parameters"] = parameters
        response = self._client.transport.request(
            "POST",
            self.PATH,
            json=body,
            headers=headers,
            timeout=timeout,
            sdk_module=MODULE,
            api_key=api_key,
            workspace=workspace,
        )
        return response

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        return flatten_envelope(
            response_body(
                response,
                raise_on_error=self._client.raise_on_error,
            ),
        )


class Vocabularies(_CustomizationBase):
    """``client.audio.vocabularies`` -- asr hot-word tables."""

    PATH = "/services/audio/asr/customization"
    DEFAULT_MODEL = "speech-biasing"

    def create(
        self,
        *,
        target_model: str,
        prefix: str,
        vocabulary: List[Dict[str, Any]],
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Vocabulary:
        """Create a table; the result carries ``vocabulary_id``."""
        response = self._call(
            "create_vocabulary",
            {
                "target_model": target_model,
                "prefix": prefix,
                "vocabulary": vocabulary,
            },
            model=model,
            parameters=parameters,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            Vocabulary(**self._parse_response(response)),
            response,
        )

    def list(
        self,
        *,
        prefix: Optional[str] = None,
        page_index: int = 0,
        page_size: int = 10,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> VocabularyList:
        """List tables, optionally narrowed to one ``prefix``."""
        fields: Dict[str, Any] = {
            "page_index": page_index,
            "page_size": page_size,
        }
        if prefix:
            fields["prefix"] = prefix
        response = self._call(
            "list_vocabulary",
            fields,
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            VocabularyList(**self._parse_response(response)),
            response,
        )

    def get(
        self,
        vocabulary_id: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Vocabulary:
        """Fetch one table's contents (1.x ``query_vocabulary``)."""
        _require_id(vocabulary_id, "vocabulary_id")
        response = self._call(
            "query_vocabulary",
            {"vocabulary_id": vocabulary_id},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            Vocabulary(**self._parse_response(response)),
            response,
        )

    def update(
        self,
        vocabulary_id: str,
        *,
        vocabulary: List[Dict[str, Any]],
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        """Replace one table's contents."""
        _require_id(vocabulary_id, "vocabulary_id")
        response = self._call(
            "update_vocabulary",
            {"vocabulary_id": vocabulary_id, "vocabulary": vocabulary},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            JobStatus(**self._parse_response(response)),
            response,
        )

    def delete(
        self,
        vocabulary_id: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        """Delete one table."""
        _require_id(vocabulary_id, "vocabulary_id")
        response = self._call(
            "delete_vocabulary",
            {"vocabulary_id": vocabulary_id},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            JobStatus(**self._parse_response(response)),
            response,
        )


class Voices(_CustomizationBase):
    """``client.audio.voices`` -- voice cloning."""

    PATH = "/services/audio/tts/customization"
    DEFAULT_MODEL = "voice-enrollment"

    def create(
        self,
        *,
        target_model: str,
        prefix: str,
        url: str,
        language_hints: Optional[List[str]] = None,
        max_prompt_audio_length: Optional[float] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Voice:
        """Clone a voice from the audio at ``url``; carries ``voice_id``."""
        fields: Dict[str, Any] = dict(extra_input or {})
        # Named arguments win over the ``extra_input=`` escape hatch.
        fields.update(
            {"target_model": target_model, "prefix": prefix, "url": url},
        )
        if language_hints is not None:
            fields["language_hints"] = language_hints
        if max_prompt_audio_length is not None:
            fields["max_prompt_audio_length"] = max_prompt_audio_length
        response = self._call(
            "create_voice",
            fields,
            model=model,
            parameters=parameters,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            Voice(**self._parse_response(response)),
            response,
        )

    def list(
        self,
        *,
        prefix: Optional[str] = None,
        page_index: int = 0,
        page_size: int = 10,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> VoiceList:
        """List cloned voices, optionally narrowed to one ``prefix``."""
        fields: Dict[str, Any] = {
            "page_index": page_index,
            "page_size": page_size,
        }
        if prefix:
            fields["prefix"] = prefix
        response = self._call(
            "list_voice",
            fields,
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            VoiceList(**self._parse_response(response)),
            response,
        )

    def get(
        self,
        voice_id: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Voice:
        """Fetch one voice's details (1.x ``query_voice``)."""
        _require_id(voice_id, "voice_id")
        response = self._call(
            "query_voice",
            {"voice_id": voice_id},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            Voice(**self._parse_response(response)),
            response,
        )

    def update(
        self,
        voice_id: str,
        *,
        url: str,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        """Re-clone one voice from the audio at ``url``."""
        _require_id(voice_id, "voice_id")
        response = self._call(
            "update_voice",
            {"voice_id": voice_id, "url": url},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            JobStatus(**self._parse_response(response)),
            response,
        )

    def delete(
        self,
        voice_id: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        """Delete one cloned voice."""
        _require_id(voice_id, "voice_id")
        response = self._call(
            "delete_voice",
            {"voice_id": voice_id},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            JobStatus(**self._parse_response(response)),
            response,
        )


class VoiceDesign(_CustomizationBase):
    """``client.audio.voice_design`` -- voices described in words, not audio.

    Shares the voice-enrollment registry with :class:`Voices`: same path,
    same ``model``, same action words, same ``voice_id``, one quota. Only
    ``create`` differs -- a text prompt where cloning has a sample ``url``
    -- so ``list`` returns cloned voices as well as designed ones.
    """

    PATH = "/services/audio/tts/customization"
    DEFAULT_MODEL = "voice-enrollment"

    def create(
        self,
        *,
        target_model: str,
        prefix: str,
        voice_prompt: str,
        preview_text: str,
        language_hints: Optional[List[str]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DesignedVoice:
        """Design a voice from a text prompt; carries ``voice_id`` + preview.

        ``prefix`` becomes part of the generated id,
        ``{target_model}-vd-{prefix}-{unique}``. Only the first element of
        ``language_hints`` is honoured, and it must match the language of
        ``preview_text``. The voice is unusable until its ``status`` leaves
        ``DEPLOYING``.
        """
        fields: Dict[str, Any] = dict(extra_input or {})
        # Named arguments win over the ``extra_input=`` escape hatch.
        fields.update(
            {
                "target_model": target_model,
                "prefix": prefix,
                "voice_prompt": voice_prompt,
                "preview_text": preview_text,
            },
        )
        if language_hints is not None:
            fields["language_hints"] = language_hints
        response = self._call(
            "create_voice",
            fields,
            model=model,
            parameters=parameters,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            DesignedVoice(**self._parse_response(response)),
            response,
        )

    def list(
        self,
        *,
        prefix: Optional[str] = None,
        page_index: int = 0,
        page_size: int = 10,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DesignedVoiceList:
        """List voices, optionally narrowed to one ``prefix``."""
        fields: Dict[str, Any] = {
            "page_index": page_index,
            "page_size": page_size,
        }
        if prefix:
            fields["prefix"] = prefix
        response = self._call(
            "list_voice",
            fields,
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            DesignedVoiceList(**self._parse_response(response)),
            response,
        )

    def get(
        self,
        voice_id: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DesignedVoice:
        """Fetch one voice's details, including its review ``status``."""
        _require_id(voice_id, "voice_id")
        response = self._call(
            "query_voice",
            {"voice_id": voice_id},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            DesignedVoice(**self._parse_response(response)),
            response,
        )

    def delete(
        self,
        voice_id: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> JobStatus:
        """Delete one designed voice, freeing its quota."""
        _require_id(voice_id, "voice_id")
        response = self._call(
            "delete_voice",
            {"voice_id": voice_id},
            model=model,
            timeout=timeout,
            headers=headers,
            api_key=api_key,
            workspace=workspace,
        )
        return with_headers(
            JobStatus(**self._parse_response(response)),
            response,
        )
