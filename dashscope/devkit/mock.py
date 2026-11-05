# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dev-time mock for dashscope 2.0 API calls.

Three modes:
  static  -- synthesize a schema-valid wire response locally, zero network.
             This is an HTTP *parser* smoke mock: it models the chat /
             multimodal shapes only, so a dev loop can exercise protocol
             adapters, SSE parsing and typed models offline. It does not
             simulate tool calls, structured output, async job state
             machines or non-chat resources -- use record/replay for those.
  replay  -- answer from a recorded JSONL fixture; a miss raises
             DevMockMiss (never silently falls through to the network).
  record  -- pass through to the real service and append the exchange to
             the fixture for later replay.

Where the 1.x mock short-circuited ``BaseApi.call`` and returned an
already-typed ``DashScopeAPIResponse``, the 2.0 mock hooks the transport
(the single choke point every HTTP resource flows through) and returns a
raw ``httpx.Response`` carrying wire bytes. The real protocol adapters
and typed-model parsers then run on it unchanged, so a dev loop exercises
the same decode path production does -- for every protocol (native /
openai / anthropic), streaming and non-streaming.

Scope: HTTP only. The websocket transport (``WsConnection``, realtime
audio) is not intercepted and gets no mock coverage.

Activation is process-wide with near-zero overhead when off: the
transport checks ``DASHSCOPE_DEV_MOCK`` / ``dashscope.devkit.is_active()``
before each send, and only builds the mock seam / imports this module
once a mock is actually requested.

Record mode writes full response bodies to the fixture, which can carry
generated content, tool-call arguments or account data; see the warning
emitted by :func:`enable` and the scrubbing hook :func:`set_redactor`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import uuid
import warnings
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

import httpx

from dashscope.errors import DashScopeError

MOCK_MARKER = "[MOCK]"
DEV_MODE_ENV = "DASHSCOPE_DEV_MOCK"
FIXTURE_PATH_ENV = "DASHSCOPE_DEV_MOCK_FIXTURE"
ALLOW_PARTIAL_ENV = "DASHSCOPE_DEV_MOCK_ALLOW_PARTIAL"
DEFAULT_FIXTURE = "dashscope_mock.jsonl"

_MODES = ("static", "replay", "record")
_EVENT_STREAM = "text/event-stream; charset=utf-8"

# Semantics-bearing request headers folded into the record/replay
# fingerprint. Identity headers (Authorization, User-Agent, the per-call
# client request id) are deliberately absent: they change every call and
# would make every replay miss.
_FINGERPRINT_HEADERS = ("x-dashscope-async", "x-dashscope-sse", "content-type")

# Serializes fixture appends within the process; the O_APPEND write
# itself is what keeps concurrent processes from interleaving a line.
_FIXTURE_LOCK = threading.Lock()

_mode: Optional[str] = None
_fixture_path: Optional[str] = None
_redactor: Optional[Callable[[Any], Any]] = None
_warned_record = False


class DevMockMiss(DashScopeError):
    """Replay has no recorded response, or static cannot synthesize one.

    Subclasses the v2 error base, so ``except DashScopeError`` covers it.
    """


def set_redactor(fn: Optional[Callable[[Any], Any]]) -> None:
    """Register a hook that scrubs a recorded JSON body before it is saved.

    Record mode persists the full response body, which can contain
    generated content, tool-call arguments or account fields. The
    redactor receives the parsed JSON (``dict`` / ``list``) and returns
    the value to write. It runs only when the body parses as a single
    JSON document -- SSE streams and binary bodies are stored verbatim,
    so scrub those at the source or do not record them. Pass ``None`` to
    clear.
    """
    global _redactor
    _redactor = fn


def enable(mode: str = "static", fixture_path: Optional[str] = None) -> None:
    global _mode, _fixture_path
    if mode not in _MODES:
        raise ValueError(
            f"Unknown dev mock mode {mode!r}; expected one of {_MODES}",
        )
    _mode = mode
    _fixture_path = fixture_path


def disable() -> None:
    global _mode, _fixture_path
    _mode = None
    _fixture_path = None


def active() -> bool:
    return _resolve_mode() is not None


def _warn_record_once() -> None:
    global _warned_record
    if _warned_record:
        return
    _warned_record = True
    warnings.warn(
        "dev mock 'record' mode writes full response bodies to "
        f"{_resolve_fixture()!r}. Responses can contain generated "
        "content, tool-call arguments or account data, and the file is "
        "easy to commit by accident. Register "
        "dashscope.devkit.set_redactor(...) to scrub bodies, keep the "
        "fixture out of version control, and never record with "
        "production credentials.",
        stacklevel=4,
    )


def _resolve_mode() -> Optional[str]:
    mode = _mode
    if mode is None:
        env = os.environ.get(DEV_MODE_ENV)
        if not env:
            return None
        if env not in _MODES:
            raise ValueError(
                f"Invalid {DEV_MODE_ENV}={env!r}; expected one of {_MODES}",
            )
        mode = env
    if mode == "record":
        _warn_record_once()
    return mode


def _resolve_fixture() -> str:
    return _fixture_path or os.environ.get(FIXTURE_PATH_ENV) or DEFAULT_FIXTURE


def _allow_partial_replay() -> bool:
    return bool(os.environ.get(ALLOW_PARTIAL_ENV))


# ---------------------------------------------------------------------------
# request fingerprint. Identity headers (Authorization / User-Agent /
# per-call client id) are excluded: they are volatile and would defeat
# replay hits. Everything that can change the *response* is included --
# endpoint, JSON body, query params, the raw/form/multipart channels and
# a small allowlist of semantics-bearing headers.
# ---------------------------------------------------------------------------


def _bytes_digest(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return {"__sha256__": hashlib.sha256(bytes(value)).hexdigest()}
    return value


def _one_file(field: Any, value: Any) -> Dict[str, Any]:
    # httpx accepts bytes, a file object, or a (filename, payload[, ctype])
    # tuple. Digest bytes directly; for file objects record identity only --
    # reading them here would drain the stream the real upload needs.
    filename: Any = None
    payload = value
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        filename, payload = value[0], value[1]
    name = filename or getattr(payload, "name", None)
    entry: Dict[str, Any] = {
        "field": str(field),
        "filename": str(name) if name else None,
    }
    if isinstance(payload, (bytes, bytearray)):
        entry["sha256"] = hashlib.sha256(bytes(payload)).hexdigest()
    return entry


def _files_digest(files: Any) -> Any:
    if not files:
        return None
    items = list(files.items()) if isinstance(files, Mapping) else list(files)
    out = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out.append(_one_file(item[0], item[1]))
        else:
            out.append(_one_file(item, None))
    return out


def _semantic_headers(headers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if not headers:
        return {}
    lowered = {str(k).lower(): str(v) for k, v in headers.items()}
    return {h: lowered[h] for h in _FINGERPRINT_HEADERS if h in lowered}


def _fingerprint(
    method: str,
    path: str,
    body: Any,
    params: Optional[Any],
    *,
    base_url: Optional[str] = None,
    content: Any = None,
    data: Any = None,
    files: Any = None,
    headers: Optional[Mapping[str, str]] = None,
) -> str:
    """Stable request identity for record/replay.

    ``base_url`` is part of the key so a capture made against one region
    (e.g. Hangzhou) never replays for a request routed to another
    (e.g. Singapore) that happens to share the same path and body.
    """
    payload = {
        "method": method,
        "path": path,
        "base_url": base_url,
        "body": body,
        "params": dict(params) if params else {},
        "content": _bytes_digest(content),
        "data": dict(data)
        if isinstance(data, Mapping)
        else _bytes_digest(data),
        "files": _files_digest(files),
        "headers": _semantic_headers(headers),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# static synthesis: build a wire body the active protocol's parser accepts.
# the protocol is inferred from the request path (the same signal the
# adapters themselves key off), so the mock needs no client reference.
# ---------------------------------------------------------------------------


def _protocol_of(path: str) -> Optional[str]:
    """Wire protocol for a chat endpoint, or ``None`` if it is not one.

    Keyed off the same path signal the adapters use. static synthesis
    only models the chat/multimodal shapes, so a non-chat path (embeddings,
    images, files, ...) yields ``None`` and fails loudly instead of
    returning a body the endpoint's typed parser would silently misread.
    """
    if path.endswith("/messages"):
        return "anthropic"
    if path.endswith("/chat/completions"):
        return "openai"
    if (
        "text-generation/generation" in path
        or "multimodal-generation/generation" in path
    ):
        return "native"
    return None


def _model_of(body: Any) -> str:
    if isinstance(body, dict):
        return str(body.get("model") or "unknown")
    return "unknown"


def _mock_text(model: str) -> str:
    return f"{MOCK_MARKER} static response for model={model}"


def _usage(body: Any, text: str) -> Dict[str, int]:
    in_tokens = max(
        1,
        len(json.dumps(body, ensure_ascii=False, default=str)) // 4,
    )
    out_tokens = max(1, len(text) // 4)
    return {
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
    }


def _pieces(text: str) -> List[str]:
    # Five incremental pieces mirror the server's SSE cadence, so
    # consumers exercising stream assembly see realistic chunking.
    step = max(1, len(text) // 5)
    return [text[i : i + step] for i in range(0, len(text), step)]


def _native_frames(rid: str, model: str, text: str, usage: Dict[str, int]):
    frames: List[Dict[str, Any]] = []
    pieces = _pieces(text)
    for piece in pieces[:-1]:
        frames.append(
            {
                "output": {
                    "choices": [
                        {
                            "finish_reason": None,
                            "message": {
                                "role": "assistant",
                                "content": piece,
                            },
                        },
                    ],
                },
                "model": model,
                "request_id": rid,
            },
        )
    frames.append(
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": pieces[-1],
                        },
                    },
                ],
            },
            "usage": usage,
            "model": model,
            "request_id": rid,
        },
    )
    return frames


def _openai_frames(rid: str, model: str, text: str, usage: Dict[str, int]):
    frames: List[Dict[str, Any]] = []
    pieces = _pieces(text)
    for i, piece in enumerate(pieces):
        delta: Dict[str, Any] = {"content": piece}
        if i == 0:
            delta["role"] = "assistant"
        frames.append(
            {
                "id": rid,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": None},
                ],
            },
        )
    frames.append(
        {
            "id": rid,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        },
    )
    return frames


def _anthropic_frames(rid: str, model: str, text: str, usage: Dict[str, int]):
    frames: List[Dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": rid,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": 0,
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ]
    for piece in _pieces(text):
        frames.append(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            },
        )
    frames.append({"type": "content_block_stop", "index": 0})
    frames.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": usage["output_tokens"]},
        },
    )
    frames.append({"type": "message_stop"})
    return frames


_STREAM_FRAMES = {
    "native": _native_frames,
    "openai": _openai_frames,
    "anthropic": _anthropic_frames,
}


def _static_body(
    proto: str,
    rid: str,
    model: str,
    text: str,
    usage: Dict[str, int],
) -> Dict[str, Any]:
    if proto == "openai":
        return {
            "id": rid,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        }
    if proto == "anthropic":
        return {
            "id": rid,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
            },
        }
    return {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                },
            ],
        },
        "usage": usage,
        "request_id": rid,
    }


def _sse_text(frames: List[Dict[str, Any]], done_marker: bool) -> str:
    out = []
    for i, frame in enumerate(frames, 1):
        out.append(
            f"id: {i}\ndata: {json.dumps(frame, ensure_ascii=False)}\n\n",
        )
    if done_marker:
        out.append("data: [DONE]\n\n")
    return "".join(out)


def _response(
    method: str,
    url: str,
    *,
    status_code: int = 200,
    content: Optional[bytes] = None,
    json_body: Optional[Any] = None,
    content_type: Optional[str] = None,
    request_id: Optional[str] = None,
) -> httpx.Response:
    headers: Dict[str, str] = {}
    if content_type:
        headers["content-type"] = content_type
    if request_id:
        headers["x-request-id"] = request_id
    kwargs: Dict[str, Any] = {"headers": headers}
    if json_body is not None:
        kwargs["json"] = json_body
    elif content is not None:
        kwargs["content"] = content
    return httpx.Response(
        status_code,
        request=httpx.Request(method, url),
        **kwargs,
    )


def _synthesize(method: str, url: str, path: str, body: Any, stream: bool):
    proto = _protocol_of(path)
    model = _model_of(body)
    if proto is None or not isinstance(body, dict) or "model" not in body:
        raise DevMockMiss(
            f"static dev mock cannot synthesize a response for {path}; "
            "it covers the chat/multimodal endpoints only. Use "
            "mode='record' once against the real service, then "
            "mode='replay' for this endpoint.",
        )
    rid = f"mock-{uuid.uuid4().hex[:12]}"
    text = _mock_text(model)
    usage = _usage(body, text)
    if not stream:
        return _response(
            method,
            url,
            json_body=_static_body(proto, rid, model, text, usage),
            request_id=rid,
        )
    frames = _STREAM_FRAMES[proto](rid, model, text, usage)
    return _response(
        method,
        url,
        content=_sse_text(frames, proto == "openai").encode("utf-8"),
        content_type=_EVENT_STREAM,
        request_id=rid,
    )


# ---------------------------------------------------------------------------
# fixture read/write. Request headers (Authorization / User-Agent) are
# never persisted; the response body is, so it may carry sensitive data
# -- see the record-mode warning and :func:`set_redactor`.
# ---------------------------------------------------------------------------


def _encode_body(raw: bytes) -> Dict[str, Any]:
    """Store wire bytes losslessly: text as UTF-8, anything else base64.

    A strict decode is the test -- only bytes that round-trip exactly are
    stored as text (keeping JSON/SSE fixtures human-readable). Binary
    payloads (audio, images, file downloads) go in as base64 so replay
    reproduces them byte for byte instead of corrupting them through
    ``decode(errors="replace")``.
    """
    try:
        return {"body": raw.decode("utf-8"), "body_encoding": "utf-8"}
    except UnicodeDecodeError:
        return {
            "body": base64.b64encode(raw).decode("ascii"),
            "body_encoding": "base64",
        }


def _decode_body(entry: Dict[str, Any]) -> bytes:
    body = entry.get("body", "")
    if entry.get("body_encoding") == "base64":
        return base64.b64decode(body.encode("ascii"))
    # Fixtures recorded before body_encoding existed are UTF-8 text.
    return body.encode("utf-8")


def _record_body(raw: bytes) -> Dict[str, Any]:
    if _redactor is not None:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            parsed = None
        if parsed is not None:
            redacted = _redactor(parsed)
            return {
                "body": json.dumps(redacted, ensure_ascii=False),
                "body_encoding": "utf-8",
            }
    return _encode_body(raw)


def _append_fixture(entry: Dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    path = _resolve_fixture()
    with _FIXTURE_LOCK:
        # O_CREAT mode 0600 so a fixture holding recorded response bodies
        # is never group/world readable (umask can only tighten it). The
        # single O_APPEND write is atomic on POSIX, so concurrent
        # processes each land a whole line rather than interleaving.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line)


def _lookup_fixture(key: str) -> Optional[Dict[str, Any]]:
    """Return the newest recorded entry for ``key``, or ``None``.

    The fixture is append-only, so re-recording a request leaves several
    lines under the same key. The last one is the most recent capture and
    wins -- scanning the whole file instead of returning on the first hit
    is what makes a re-record actually take effect.
    """
    found: Optional[Dict[str, Any]] = None
    try:
        with open(_resolve_fixture(), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("key") == key:
                    found = entry
    except FileNotFoundError:
        return None
    return found


def _capture(
    key: str,
    response: httpx.Response,
    raw: bytes,
    stream: bool,
    *,
    complete: bool = True,
) -> None:
    entry = {
        "key": key,
        "stream": stream,
        "complete": complete,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
    }
    entry.update(_record_body(raw))
    _append_fixture(entry)


def _replay(key: str, method: str, url: str, stream: bool) -> httpx.Response:
    entry = _lookup_fixture(key)
    if entry is None:
        raise DevMockMiss(
            f"No recorded response for request fingerprint {key} in "
            f"{_resolve_fixture()}. Run once with mode='record' to "
            "capture it.",
        )
    if bool(entry.get("stream")) != stream:
        raise DevMockMiss(
            "Fixture holds a "
            f"{'streaming' if entry.get('stream') else 'non-streaming'} "
            f"response but this call is {'streaming' if stream else 'non-streaming'}; "
            "re-record with the matching call.",
        )
    # A stream captured from a consumer that stopped early, cancelled or
    # errored before EOF is truncated; replaying it as if whole produces
    # a half SSE stream that is miserable to debug. Refuse by default.
    if stream and not entry.get("complete", True):
        if not _allow_partial_replay():
            raise DevMockMiss(
                "The recorded stream for this request is incomplete (it "
                "was captured before EOF -- the consumer stopped early, "
                "cancelled, or errored). Re-record it, or set "
                f"{ALLOW_PARTIAL_ENV}=1 to replay the truncated capture "
                "anyway.",
            )
    return _response(
        method,
        url,
        status_code=entry.get("status_code", 200),
        content=_decode_body(entry),
        content_type=entry.get("content_type") or None,
    )


# ---------------------------------------------------------------------------
# record: pass through to the real transport, tee the wire bytes, persist.
# ---------------------------------------------------------------------------


class _TeeSyncStream(httpx.SyncByteStream):
    def __init__(self, inner, on_done: Callable[[bytes, bool], None]) -> None:
        self._inner = inner
        self._on_done = on_done
        self._buf = bytearray()
        self._done = False
        self._complete = False

    def __iter__(self):
        try:
            for chunk in self._inner:
                self._buf += chunk
                yield chunk
        except BaseException:
            # Errored mid-stream: what was buffered is partial.
            self._finish()
            raise
        # Inner stream drained to natural EOF: a complete recording.
        self._complete = True
        self._finish()

    def close(self) -> None:
        # Closed before EOF (early stop / cancel): partial by definition.
        self._finish()
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._on_done(bytes(self._buf), self._complete)


class _TeeAsyncStream(httpx.AsyncByteStream):
    def __init__(self, inner, on_done: Callable[[bytes, bool], None]) -> None:
        self._inner = inner
        self._on_done = on_done
        self._buf = bytearray()
        self._done = False
        self._complete = False

    async def __aiter__(self):
        try:
            async for chunk in self._inner:
                self._buf += chunk
                yield chunk
        except BaseException:
            self._finish()
            raise
        self._complete = True
        self._finish()

    async def aclose(self) -> None:
        self._finish()
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._on_done(bytes(self._buf), self._complete)


def _record_sync(
    key: str,
    stream: bool,
    real: Callable[[], httpx.Response],
) -> httpx.Response:
    response = real()
    if not stream:
        response.read()
        _capture(key, response, response.content, False)
        return response
    tee = _TeeSyncStream(
        response.stream,
        lambda raw, complete: _capture(
            key,
            response,
            raw,
            True,
            complete=complete,
        ),
    )
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        stream=tee,
        request=response.request,
    )


async def _record_async(
    key: str,
    stream: bool,
    real: Callable[[], Awaitable[httpx.Response]],
) -> httpx.Response:
    response = await real()
    if not stream:
        await response.aread()
        _capture(key, response, response.content, False)
        return response
    tee = _TeeAsyncStream(
        response.stream,
        lambda raw, complete: _capture(
            key,
            response,
            raw,
            True,
            complete=complete,
        ),
    )
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        stream=tee,
        request=response.request,
    )


# ---------------------------------------------------------------------------
# transport entry points
# ---------------------------------------------------------------------------


def _abs_url(base_url: Optional[str], path: str) -> str:
    if base_url is None:
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def intercept(
    *,
    method: str,
    path: str,
    json_body: Any = None,
    content: Any = None,
    data: Any = None,
    files: Any = None,
    params: Optional[Any] = None,
    headers: Optional[Mapping[str, str]] = None,
    base_url: Optional[str] = None,
    stream: bool = False,
    real: Callable[[], httpx.Response],
) -> httpx.Response:
    """Handle one sync transport send in dev mock mode.

    ``real`` performs the actual (retried) httpx call; only record mode
    invokes it. static/replay never touch the network. ``content`` /
    ``data`` / ``files`` / ``headers`` feed the request fingerprint so a
    multipart, binary or region-specific request cannot collide with a
    different one that shares its path and JSON body.
    """
    mode = _resolve_mode()
    if mode is None:
        return real()
    url = _abs_url(base_url, path)
    if mode == "static":
        return _synthesize(method, url, path, json_body, stream)
    key = _fingerprint(
        method,
        path,
        json_body,
        params,
        base_url=base_url,
        content=content,
        data=data,
        files=files,
        headers=headers,
    )
    if mode == "replay":
        return _replay(key, method, url, stream)
    return _record_sync(key, stream, real)


async def aintercept(
    *,
    method: str,
    path: str,
    json_body: Any = None,
    content: Any = None,
    data: Any = None,
    files: Any = None,
    params: Optional[Any] = None,
    headers: Optional[Mapping[str, str]] = None,
    base_url: Optional[str] = None,
    stream: bool = False,
    real: Callable[[], Awaitable[httpx.Response]],
) -> httpx.Response:
    """Async counterpart of :func:`intercept`."""
    mode = _resolve_mode()
    if mode is None:
        return await real()
    url = _abs_url(base_url, path)
    if mode == "static":
        return _synthesize(method, url, path, json_body, stream)
    key = _fingerprint(
        method,
        path,
        json_body,
        params,
        base_url=base_url,
        content=content,
        data=data,
        files=files,
        headers=headers,
    )
    if mode == "replay":
        return _replay(key, method, url, stream)
    return await _record_async(key, stream, real)
