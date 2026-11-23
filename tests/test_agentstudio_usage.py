# -*- coding: utf-8 -*-
import json
import asyncio
import io
import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from dashscope import Client, AsyncClient


# Every resource goes through the real HTTP adapter, rather than a fake
# transport.request() that would bypass the migration under test.
RESOURCE_CASES = []


def _resource_cases(
    resource,
    path,
    scope=(),
    create=None,
    update=None,
    operations=("create", "retrieve", "update", "list", "archive", "delete"),
):
    for operation in operations:
        args = scope if operation in ("create", "list") else scope + ("item",)
        kwargs = (
            create or {}
            if operation == "create"
            else update or {}
            if operation == "update"
            else {}
        )
        method = (
            "GET"
            if operation in ("retrieve", "list")
            else "DELETE"
            if operation == "delete"
            else "POST"
        )
        target = path if operation in ("create", "list") else path + "/item"
        if operation == "archive":
            target += "/archive"
        RESOURCE_CASES.append(
            (resource, operation, args, kwargs, method, target),
        )


_resource_cases(
    "agents",
    "/agents",
    create={"name": "test", "model": "model"},
    update={"version": 1, "name": "changed"},
    operations=("create", "retrieve", "update", "list", "archive"),
)
_resource_cases(
    "environments",
    "/environments",
    create={"name": "test", "config": {"type": "cloud"}},
    update={"name": "changed"},
)
_resource_cases(
    "sessions",
    "/sessions",
    create={"agent": "agent-test"},
    update={"title": "changed"},
)
_resource_cases(
    "vaults",
    "/vaults",
    create={"display_name": "test"},
    update={"display_name": "changed"},
)
_resource_cases(
    "vaults.credentials",
    "/vaults/scope/credentials",
    ("scope",),
    create={"auth": {"type": "api_key", "api_key": "secret"}},
    update={"display_name": "changed"},
)
_resource_cases(
    "memory_stores",
    "/memory_stores",
    create={"name": "test"},
    update={"name": "changed"},
)
_resource_cases(
    "memory_stores.memories",
    "/memory_stores/scope/memories",
    ("scope",),
    create={"path": "/test.md", "content": "test"},
    update={"content": "changed", "content_sha256": "sha"},
    operations=("create", "retrieve", "update", "list", "delete"),
)
_resource_cases(
    "skills",
    "/skills",
    create={"file_id": "file-test"},
    operations=("create", "retrieve", "list", "delete"),
)
_resource_cases(
    "skills.versions",
    "/skills/scope/versions",
    ("scope",),
    create={"file_id": "file-test"},
    operations=("create", "retrieve", "list"),
)
_resource_cases("files", "/files", operations=("retrieve", "list", "delete"))
_resource_cases(
    "deployments",
    "/deployments",
    create={"name": "test", "agent": "agent-test", "initial_events": []},
    update={"name": "changed"},
    operations=("create", "retrieve", "update", "list", "archive"),
)
_resource_cases(
    "deployment_runs",
    "/deployment_runs",
    operations=("retrieve", "list"),
)
_resource_cases(
    "sessions.threads",
    "/sessions/scope/threads",
    ("scope",),
    operations=("retrieve", "list", "archive"),
)
_resource_cases(
    "memory_stores.memory_versions",
    "/memory_stores/scope/memory_versions",
    ("scope",),
    operations=("retrieve", "list"),
)
_resource_cases(
    "webhook_endpoints",
    "/webhook_endpoints",
    create={"url": "https://example.test/hook", "events": ["session.idle"]},
    update={"description": "changed"},
    operations=("create", "retrieve", "update", "list", "delete"),
)
RESOURCE_CASES = [
    (
        *c[:4],
        "PUT" if c[0] == "webhook_endpoints" and c[1] == "update" else c[4],
        c[5],
    )
    for c in RESOURCE_CASES
]
RESOURCE_CASES += [
    (
        "agents",
        "list_versions",
        ("scope",),
        {},
        "GET",
        "/agents/scope/versions",
    ),
    (
        "sessions.events",
        "send",
        (
            "scope",
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "test"}],
                },
            ],
        ),
        {},
        "POST",
        "/sessions/scope/events",
    ),
    (
        "sessions.events",
        "list",
        ("scope",),
        {},
        "GET",
        "/sessions/scope/events",
    ),
    (
        "sessions.events",
        "stream",
        ("scope",),
        {},
        "GET",
        "/sessions/scope/events/stream",
    ),
    (
        "sessions.threads.events",
        "list",
        ("scope", "item"),
        {},
        "GET",
        "/sessions/scope/threads/item/events",
    ),
    (
        "sessions.resources",
        "add",
        ("scope",),
        {"file_id": "file-test"},
        "POST",
        "/sessions/scope/resources",
    ),
    (
        "sessions.resources",
        "retrieve",
        ("scope", "item"),
        {},
        "GET",
        "/sessions/scope/resources/item",
    ),
    (
        "sessions.resources",
        "list",
        ("scope",),
        {},
        "GET",
        "/sessions/scope/resources",
    ),
    (
        "sessions.resources",
        "delete",
        ("scope", "item"),
        {},
        "DELETE",
        "/sessions/scope/resources/item",
    ),
    (
        "skills.versions",
        "download",
        ("scope", "item"),
        {},
        "GET",
        "/skills/scope/versions/item/content",
    ),
    ("files", "upload", (), {}, "POST", "/files"),
    ("files", "download", ("item",), {}, "GET", "/files/item/content"),
    (
        "memory_stores.memory_versions",
        "redact",
        ("scope", "item"),
        {},
        "POST",
        "/memory_stores/scope/memory_versions/item/redact",
    ),
    ("deployments", "pause", ("item",), {}, "POST", "/deployments/item/pause"),
    (
        "deployments",
        "unpause",
        ("item",),
        {},
        "POST",
        "/deployments/item/unpause",
    ),
    ("deployments", "run", ("item",), {}, "POST", "/deployments/item/run"),
    (
        "deployments",
        "list_runs",
        ("item",),
        {},
        "GET",
        "/deployments/item/runs",
    ),
    ("security", "overview", (), {}, "GET", "/security/overview"),
    (
        "security",
        "list_agent_logs",
        (),
        {"page_size": 1},
        "GET",
        "/security/agent_logs",
    ),
    (
        "usage",
        "summary",
        (),
        {"dimension": "session", "session_id": "scope"},
        "POST",
        "/usage/summary",
    ),
    (
        "usage",
        "trend",
        (),
        {
            "dimension": "workspace",
            "time_range": {
                "start_time": "2026-11-02T00:00:00Z",
                "end_time": "2026-11-03T00:00:00Z",
            },
        },
        "POST",
        "/usage/trend",
    ),
]
for operation in ("enable", "disable", "test", "reset_secret", "list_events"):
    suffix = "events" if operation == "list_events" else operation
    RESOURCE_CASES.append(
        (
            "webhook_endpoints",
            operation,
            ("item",),
            {},
            "GET" if operation == "list_events" else "POST",
            "/webhook_endpoints/item/" + suffix,
        ),
    )


def _resource(client, path):
    for part in path.split("."):
        client = getattr(client, part)
    return client


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("entry", ["mounted", "standalone", "legacy"])
@pytest.mark.parametrize(
    "case",
    RESOURCE_CASES,
    ids=lambda c: c[0] + "." + c[1],
)
@pytest.mark.asyncio
async def test_all_resource_methods_through_http(
    entry,
    asynchronous,
    case,
    tmp_path,
):
    from dashscope.applications.agentstudio import (
        Client as Studio,
        AsyncClient as AsyncStudio,
    )
    from dashscope.agentstudio import (
        Client as Legacy,
        AsyncClient as AsyncLegacy,
    )

    resource, operation, args, kwargs, method, path = case
    requests = []

    def handler(request):
        requests.append(request)
        if path.endswith("/events/stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"type":"message","role":"assistant",'
                    b'"content":[{"type":"text","text":"done"}]}\n\n'
                ),
            )
        if resource == "files" and operation == "download":
            return httpx.Response(
                200,
                content=b"download-bytes",
                headers={"content-type": "application/octet-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "item",
                "name": "test",
                "model": {"id": "model"},
                "version": 1,
                "data": [],
                "next_page": None,
                "request_id": "req-matrix",
            },
        )

    pool = (httpx.AsyncClient if asynchronous else httpx.Client)(
        transport=httpx.MockTransport(handler),
    )
    classes = {
        "mounted": (Client, AsyncClient),
        "standalone": (Studio, AsyncStudio),
        "legacy": (Legacy, AsyncLegacy),
    }
    client = classes[entry][asynchronous](
        api_key="sk-matrix",
        base_url="https://example.test/api/v1/agentstudio",
        http_client=pool,
    )
    studio = client.agentstudio if entry == "mounted" else client
    options = dict(kwargs)
    if resource == "files" and operation == "upload":
        options["file"] = ("test.bin", io.BytesIO(b"upload-bytes"))
    try:
        result = getattr(_resource(studio, resource), operation)(
            *args,
            **options,
        )
        if inspect.isawaitable(result):
            result = await result
        if operation == "stream":
            events = (
                [event async for event in result]
                if asynchronous
                else list(result)
            )
            assert len(events) == 1 and events[0].type == "message"
        elif resource == "files" and operation == "download":
            assert result == b"download-bytes"
            assert (
                result.write_to_file(tmp_path / "download.bin").read_bytes()
                == result
            )
        else:
            decoded = (
                result
                if isinstance(result, dict)
                else result.to_dict()
                if hasattr(result, "to_dict")
                else {"request_id": result.request_id}
            )
            assert decoded["request_id"] == "req-matrix"
        assert len(requests) == 1
        request = requests[0]
        assert (
            request.method == method
            and request.url.path == "/api/v1/agentstudio" + path
        )
        assert request.headers["Authorization"] == "Bearer sk-matrix"
        assert request.headers["x-dashscope-sdk-client"].endswith(
            "/agentstudio",
        )
        if resource == "files" and operation == "upload":
            assert request.headers["content-type"].startswith(
                "multipart/form-data;",
            )
            assert b"upload-bytes" in request.content
        elif request.content:
            assert isinstance(json.loads(request.content), dict)
    finally:
        if asynchronous:
            await client.aclose()
            assert not pool.is_closed
            await pool.aclose()
        else:
            client.close()
            assert not pool.is_closed
            pool.close()


def test_resource_matrix_has_no_missing_public_methods():
    from dashscope.applications.agentstudio import Client as Studio

    with Studio(api_key="test", base_url="https://example.test") as client:
        for path in {case[0] for case in RESOURCE_CASES}:
            resource = _resource(client, path)
            expected = {
                name
                for name, value in type(resource).__dict__.items()
                if not name.startswith("_")
                and callable(value)
                and name not in ("get", "call")
            }
            assert expected == {
                case[1] for case in RESOURCE_CASES if case[0] == path
            }, path
            if hasattr(resource, "get"):
                assert resource.get.__func__ is resource.retrieve.__func__
            if hasattr(resource, "call"):
                assert resource.call.__func__ is resource.create.__func__


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_real_http_connection_pool_and_concurrent_responses(
    asynchronous,
):
    state = {"active": 0, "max_active": 0, "ports": set(), "ids": set()}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["ports"].add(self.client_address[1])
                state["ids"].add(self.headers["x-dashscope-client-request-id"])
            try:
                time.sleep(0.01)
                payload = json.dumps(
                    {"id": self.path.rsplit("/", 1)[-1]},
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
            finally:
                with lock:
                    state["active"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    pool = (httpx.AsyncClient if asynchronous else httpx.Client)(
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
    )
    client = (AsyncClient if asynchronous else Client)(
        api_key="local-test",
        base_url=f"http://127.0.0.1:{server.server_port}/api/v1",
        http_client=pool,
    )
    try:
        if asynchronous:
            results = await asyncio.gather(
                *(
                    client.agentstudio.agents.retrieve(f"agent-{i}")
                    for i in range(24)
                ),
            )
        else:
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        client.agentstudio.agents.retrieve,
                        [f"agent-{i}" for i in range(24)],
                    ),
                )
        assert [item.id for item in results] == [
            f"agent-{i}" for i in range(24)
        ]
        assert len(state["ids"]) == 24
        assert len(state["ports"]) <= 2
        assert 1 < state["max_active"] <= 2
    finally:
        if asynchronous:
            await client.aclose()
            assert not pool.is_closed
            await pool.aclose()
        else:
            client.close()
            assert not pool.is_closed
            pool.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("source", ["path", "stream", "tuple"])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.asyncio
async def test_multipart_progress_and_file_ownership(
    asynchronous,
    source,
    failure,
    tmp_path,
    monkeypatch,
):
    from dashscope.applications.agentstudio import (
        Client as Studio,
        AsyncClient as AsyncStudio,
    )
    from dashscope.applications.agentstudio.resources import files as module
    from dashscope.errors import APIStatusError

    content = b"upload-content" * 12000
    path = tmp_path / "upload.bin"
    path.write_bytes(content)
    borrowed = io.BytesIO(content)
    file = (
        path
        if source == "path"
        else borrowed
        if source == "stream"
        else ("named.bin", borrowed)
    )
    opened = []
    original = module._open_file

    def tracked(value):
        result = original(value)
        opened.append(result)
        return result

    monkeypatch.setattr(module, "_open_file", tracked)
    progress = []

    def handler(request):
        assert request.headers["content-type"].startswith(
            "multipart/form-data;",
        )
        assert b"application/x-test" in request.content
        assert content in request.content
        if failure:
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "code": "invalid_request_error",
                        "message": "bad file",
                    },
                },
            )
        return httpx.Response(200, json={"id": "file-upload"})

    pool = (httpx.AsyncClient if asynchronous else httpx.Client)(
        transport=httpx.MockTransport(handler),
    )
    client = (AsyncStudio if asynchronous else Studio)(
        api_key="test",
        base_url="https://example.test",
        http_client=pool,
    )
    try:

        async def upload():
            result = client.files.upload(
                file,
                mime_type="application/x-test",
                progress=lambda sent, total: progress.append((sent, total)),
            )
            return await result if asynchronous else result

        if failure:
            with pytest.raises(APIStatusError):
                await upload()
        else:
            assert (await upload()).id == "file-upload"
        assert progress[-1] == (len(content), len(content))
        assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))
        assert opened[0][1].closed == (source == "path")
    finally:
        borrowed.close()
        if asynchronous:
            await client.aclose()
            await pool.aclose()
        else:
            client.close()
            pool.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("first_failure", ["429", "network"])
@pytest.mark.asyncio
async def test_shared_retry_replays_request_once(asynchronous, first_failure):
    from dashscope.applications.agentstudio import (
        Client as Studio,
        AsyncClient as AsyncStudio,
    )

    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if first_failure == "network":
                raise httpx.ReadError("connection reset", request=request)
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"code": "rate_limit", "message": "retry"}},
            )
        return httpx.Response(
            200,
            json={"request_id": "req-retry", "data": {"usage": {}}},
        )

    pool = (httpx.AsyncClient if asynchronous else httpx.Client)(
        transport=httpx.MockTransport(handler),
    )
    client = (AsyncStudio if asynchronous else Studio)(
        api_key="retry-key",
        base_url="https://example.test",
        http_client=pool,
        max_retries=1,
    )
    try:
        result = client.usage.summary(
            dimension="session",
            session_id="session-retry",
        )
        if asynchronous:
            result = await result
        assert result["request_id"] == "req-retry"
        assert len(requests) == 2
        assert requests[0].content == requests[1].content
        assert all(
            r.headers["Authorization"] == "Bearer retry-key" for r in requests
        )
        assert (
            requests[0].headers["x-dashscope-client-request-id"]
            == requests[1].headers["x-dashscope-client-request-id"]
        )
    finally:
        if asynchronous:
            await client.aclose()
            await pool.aclose()
        else:
            client.close()
            pool.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_usage_uses_parent_transport_and_preserves_cursor(asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "request_id": "req-usage",
                "data": {
                    "points": [],
                    "next_page_token": "opaque-cursor",
                    "has_more": True,
                },
            },
        )

    http_class = httpx.AsyncClient if asynchronous else httpx.Client
    sdk_class = AsyncClient if asynchronous else Client
    pool = http_class(transport=httpx.MockTransport(handler))
    client = sdk_class(
        api_key="sk-test",
        base_url="https://example.test/api/v1",
        http_client=pool,
    )
    usage = client.agentstudio.usage
    query = {
        "dimension": "workspace",
        "time_range": {
            "start_time": "2026-11-02T10:00:00+08:00",
            "end_time": "2026-11-02T12:00:00+08:00",
        },
        "page_size": 1,
    }
    try:
        summary = usage.summary(dimension="session", session_id="sesn-test")
        if asynchronous:
            summary = await summary
        assert summary["request_id"] == "req-usage"
        page = usage.trend(**query)
        if asynchronous:
            page = await page
        result = usage.trend(
            **query,
            page_token=page["data"]["next_page_token"],
        )
        if asynchronous:
            await result
        assert requests[0].url.path == "/api/v1/agentstudio/usage/summary"
        assert json.loads(requests[0].content) == {
            "dimension": "session",
            "session_id": "sesn-test",
        }
        assert requests[2].url.path == "/api/v1/agentstudio/usage/trend"
        assert json.loads(requests[2].content) == dict(
            query,
            page_token="opaque-cursor",
        )
        assert client.agentstudio.transport._client is pool
    finally:
        if asynchronous:
            await client.aclose()
            assert not pool.is_closed
            await pool.aclose()
        else:
            client.close()
            assert not pool.is_closed
            pool.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "failure",
    [None, httpx.ConnectError, httpx.ReadTimeout],
)
@pytest.mark.asyncio
async def test_transport_preserves_absolute_url_and_error_request_id(
    asynchronous,
    stream,
    failure,
):
    from dashscope.applications.agentstudio import Client as StudioClient
    from dashscope.applications.agentstudio import (
        AsyncClient as AsyncStudioClient,
    )
    from dashscope.errors import APIConnectionError, APITimeoutError

    requests = []

    def handler(request):
        requests.append(request)
        if failure:
            raise failure("network probe", request=request)
        return httpx.Response(200, json={"id": "ok"})

    http_class = httpx.AsyncClient if asynchronous else httpx.Client
    client_class = AsyncStudioClient if asynchronous else StudioClient
    pool = http_class(transport=httpx.MockTransport(handler))
    client = client_class(
        api_key="sk-test",
        base_url="https://example.test/api/v1/agentstudio",
        http_client=pool,
        max_retries=0,
    )
    target = "https://example.test/absolute/path?cursor=a%2Fb"
    try:
        if failure:
            error_class = (
                APITimeoutError
                if failure is httpx.ReadTimeout
                else APIConnectionError
            )
            with pytest.raises(error_class) as caught:
                result = client.transport.request("GET", target, stream=stream)
                if asynchronous:
                    await result
            assert caught.value.request_id
            assert caught.value.request_id == caught.value.__cause__.request_id
        else:
            result = client.transport.request("GET", target, stream=stream)
            if asynchronous:
                result = await result
            if stream:
                if asynchronous:
                    await result.aclose()
                else:
                    result.close()
            else:
                assert result.data == {"id": "ok"}
        assert str(requests[0].url) == target
    finally:
        if asynchronous:
            await client.aclose()
            await pool.aclose()
        else:
            client.close()
            pool.close()
