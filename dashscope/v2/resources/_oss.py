# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Local-file → OSS upload preprocessing for media inputs.

DashScope serves media (image / audio / video) from URLs; a local path or
``file://`` URI must be uploaded to OSS first and the request opted into
OSS resolution. This reuses the 1.x uploader (``dashscope.utils.oss_utils``)
so 2.0 keeps the same upload behavior instead of re-implementing it.

Two things are anchored to the v2 client rather than to 1.x globals:

* The upload certificate is fetched from the client's own ``base_url``
  (``base_address=`` below), so a region- or gateway-scoped client uploads
  to the matching regional OSS and the returned ``oss://`` URL resolves
  under the same endpoint the inference request is sent to. The 1.x global
  endpoint would otherwise pin every upload to the default region.
* The effective per-call ``api_key`` / ``workspace`` (the override, else the
  client default) drive both the certificate fetch and the upload, so the
  asset lands in the same account the request runs under.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from dashscope.utils.oss_utils import check_and_upload_local
from dashscope.v2.types.messages import (
    AudioPart,
    ImagePart,
    Message,
    VideoPart,
)

__all__ = [
    "OSS_RESOLVE_HEADERS",
    "upload_embedding_media",
    "upload_media_value",
    "upload_message_media",
]

# The service only resolves oss:// inputs when the request opts in.
OSS_RESOLVE_HEADERS = {"X-DashScope-OssResourceResolve": "enable"}

_EMBEDDING_MEDIA_KEYS = ("image", "audio", "video")


def _effective(
    client,
    api_key: Optional[str],
    workspace: Optional[str],
) -> Tuple[Optional[str], Optional[str], str]:
    """Resolve the per-call override against the client defaults.

    Returns ``(api_key, workspace, base_url)``. The certificate fetch and
    the upload both run under these, so the asset lands in the same
    account/region the request is sent to.
    """
    return (
        api_key or client.config.api_key,
        workspace or client.config.workspace,
        client.config.base_url,
    )


def upload_message_media(
    client,
    messages: List[Message],
    model: str,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
) -> bool:
    """Upload local/oss refs in chat media parts in place.

    Returns True when any part holds an ``oss://`` URL afterwards (uploaded
    or already remote), so the caller opts the request into OSS resolution.
    """
    key, space, base_url = _effective(client, api_key, workspace)
    has_upload = False
    cert: Optional[dict] = None
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, (ImagePart, AudioPart)):
                if part.url is None:
                    # Inline base64 audio has no URL to upload.
                    continue
                is_up, new, cert = check_and_upload_local(
                    model,
                    part.url,
                    key,
                    cert,
                    base_address=base_url,
                    workspace=space,
                )
                if is_up:
                    part.url = new
                    has_upload = True
            elif isinstance(part, VideoPart):
                if part.url is not None:
                    is_up, new, cert = check_and_upload_local(
                        model,
                        part.url,
                        key,
                        cert,
                        base_address=base_url,
                        workspace=space,
                    )
                    if is_up:
                        part.url = new
                        has_upload = True
                if part.frames:
                    frames = []
                    for frame in part.frames:
                        is_up, new, cert = check_and_upload_local(
                            model,
                            frame,
                            key,
                            cert,
                            base_address=base_url,
                            workspace=space,
                        )
                        has_upload = has_upload or is_up
                        frames.append(new)
                    part.frames = frames
    return has_upload


def _upload_value(
    value: Any,
    model: str,
    api_key: Optional[str],
    workspace: Optional[str],
    base_url: str,
    cert: Optional[dict],
) -> Tuple[Any, bool, Optional[dict]]:
    if isinstance(value, str):
        is_up, new, cert = check_and_upload_local(
            model,
            value,
            api_key,
            cert,
            base_address=base_url,
            workspace=workspace,
        )
        return new, is_up, cert
    if isinstance(value, list):
        has_upload = False
        out = []
        for item in value:
            new_item, is_up, cert = _upload_value(
                item,
                model,
                api_key,
                workspace,
                base_url,
                cert,
            )
            has_upload = has_upload or is_up
            out.append(new_item)
        return out, has_upload, cert
    if isinstance(value, dict):
        has_upload = False
        out = {}
        for key, item in value.items():
            new_item, is_up, cert = _upload_value(
                item,
                model,
                api_key,
                workspace,
                base_url,
                cert,
            )
            has_upload = has_upload or is_up
            out[key] = new_item
        return out, has_upload, cert
    return value, False, cert


def upload_media_value(
    client,
    value: Any,
    model: str,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Recursively upload local/oss refs in a media container.

    Walks strings, lists and dicts (``image_urls`` / ``media`` /
    ``extra_input``); each string goes through the 1.x uploader, which only
    rewrites ``file://`` URIs, existing local paths and ``oss://`` URLs and
    passes http(s) and other strings through unchanged.
    """
    key, space, base_url = _effective(client, api_key, workspace)
    new_value, has_upload, _ = _upload_value(
        value,
        model,
        key,
        space,
        base_url,
        None,
    )
    return new_value, has_upload


def upload_embedding_media(
    client,
    contents: List[Dict[str, Any]],
    model: str,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
) -> None:
    """Upload the image/audio/video field of each embedding item in place."""
    key, space, base_url = _effective(client, api_key, workspace)
    cert: Optional[dict] = None
    for item in contents:
        for field_key in _EMBEDDING_MEDIA_KEYS:
            field = item.get(field_key)
            if isinstance(field, str) and field:
                is_up, new, cert = check_and_upload_local(
                    model,
                    field,
                    key,
                    cert,
                    base_address=base_url,
                    workspace=space,
                )
                if is_up:
                    item[field_key] = new
