# DashScope Python SDK — HTTP API 错误码/消息透传审查

## 目标

审查 SDK 是否对 HTTP API 输出的错误码（code）和错误消息（message）做了修改，确保原样透传。

## 架构概览

SDK 有 **两套独立的错误处理管线**：

1. **核心 SDK 管线** — `dashscope/api_entities/`、`dashscope/common/`、`dashscope/client/`（用于 `dashscope.Generation`、`dashscope.Application` 等）
2. **AgentStudio 管线** — `dashscope/agentstudio/`（AgentStudio 子 SDK，使用 `httpx`）

---

## 正常透传的路径（GOOD）

以下代码路径都**正确保留**了 API 返回的 `code` 和 `message`：

### 1. `_handle_error_message()` — 中央 JSON 错误解析器

**文件:** `dashscope/common/utils.py`, 行 270–291

```python
def _handle_error_message(error, status_code, flattened_output, headers):
    code = None
    msg = ""
    request_id = ""
    if flattened_output:
        error["status_code"] = status_code
        return error
    if "message" in error:
        msg = error["message"]
    if "msg" in error:
        msg = error["msg"]
    if "code" in error:
        code = error["code"]
    if "request_id" in error:
        request_id = error["request_id"]
    return DashScopeAPIResponse(
        request_id=request_id,
        status_code=status_code,
        code=code,
        message=msg,
        headers=headers,
    )
```

**结论:** 直接从 API JSON body 提取 `code` 和 `message`，原样透传。也处理了 `"msg"` 作为替代键。

### 2. `_handle_http_failed_response()` — sync 非 2xx 处理器

**文件:** `dashscope/common/utils.py`, 行 294–336

对 JSON 响应完全委托给 `_handle_error_message()`。对非 JSON/非 SSE 响应，`code="Unknown"` 但保留原始 body 作为 `message`。

### 3. `_handle_aiohttp_failed_response()` — async 非 2xx 处理器

**文件:** `dashscope/common/utils.py`, 行 361–408

与 sync 版本相同逻辑，JSON 和 SSE 内容类型委托给 `_handle_error_message()`。

### 4. HttpRequest SSE 流错误（sync）

**文件:** `dashscope/api_entities/http_request.py`, 行 409–418

```python
if is_error:
    yield DashScopeAPIResponse(
        request_id=request_id,
        status_code=status_code,
        output=None,
        code=msg["code"] if "code" in msg else None,
        message=msg["message"] if "message" in msg else None,
        headers=headers,
    )
```

直接从 SSE event JSON 提取。

### 5. HttpRequest SSE 流错误（async）

**文件:** `dashscope/api_entities/http_request.py`, 行 299–306

```python
if is_error:
    yield DashScopeAPIResponse(
        request_id=request_id,
        status_code=status_code,
        code=msg["code"],
        message=msg["message"],
        headers=headers,
    )
```

### 6. AioHttpRequest SSE 流错误（async）

**文件:** `dashscope/api_entities/aiohttp_request.py`, 行 172–178

```python
if is_error:
    yield DashScopeAPIResponse(
        request_id=request_id,
        status_code=status_code,
        code=msg["code"],
        message=msg["message"],
    )
```

### 7. AioHttpRequest 非 2xx JSON 错误

**文件:** `dashscope/api_entities/aiohttp_request.py`, 行 222–246

```python
if "application/json" in response.content_type:
    error = await response.json()
    yield DashScopeAPIResponse(
        request_id=request_id,
        status_code=response.status,
        code=error["code"],
        message=message,
    )
```

### 8. `_handle_http_stream_response()` — sync stream 错误路径

**文件:** `dashscope/common/utils.py`, 行 483–493

```python
msg = json.loads(event.data)
yield event.eventType, DashScopeAPIResponse(
    request_id=request_id,
    status_code=status_code,
    output=None,
    code=msg["code"] if "code" in msg else None,
    message=msg["message"] if "message" in msg else None,
    headers=headers,
)
```

### 9. `DashScopeAPIResponse` — 响应容器

**文件:** `dashscope/api_entities/dashscope_response.py`, 行 80–122

纯数据容器，不做任何转换。所有 `from_api_response()` 类方法（约 15 个响应类型）都是直接 `code=api_response.code, message=api_response.message` 拷贝。

### 10. AgentStudio `from_response()` — 异常构造

**文件:** `dashscope/agentstudio/exceptions.py`, 行 177–244

当 API 响应 body 包含 `error.code` 和 `error.message` 时，原样保留。fallback（`f"HTTP {status_code}"`、状态码映射）仅在响应 body 缺少结构化错误信息时触发。

---

## 发现的问题

### 问题 1 — StreamEventMixin：错误码和消息被丢弃（HIGH）

**文件:** `dashscope/client/base_api.py`, 行 1557–1565

```python
for is_error, status_code, data in cls._handle_stream(response):
    if is_error:
        yield DashScopeAPIResponse(
            request_id=request_id,
            status_code=status_code,
            output=None,
            code="",          # ← 原始 code 丢弃
            message="",       # ← 原始 message 丢弃
        )
```

**影响:** 使用 `StreamEventMixin.stream_events()` 的 API（如任务日志流）在流错误时，用户**完全看不到**错误信息。`data` 变量中有完整的 JSON 错误但没有解析。

**对比正确代码:** `http_request.py:409-418`（相同模式，但正确解析了 `msg["code"]` 和 `msg["message"]`）。

### 问题 2 — WebSocket 握手：错误消息被替换（MODERATE）

**文件:** `dashscope/api_entities/websocket_request.py`, 行 204–218

```python
except aiohttp.WSServerHandshakeError as e:
    code = e.status
    msg = e.message
    if e.status in [HTTPStatus.FORBIDDEN, HTTPStatus.UNAUTHORIZED]:
        msg = "Unauthorized, your api-key is invalid!"     # ← 替换了服务端消息
    elif e.status == HTTPStatus.SERVICE_UNAVAILABLE:
        msg = SERVICE_503_MESSAGE                           # ← 替换了服务端消息
    yield DashScopeAPIResponse(
        request_id=task_id,
        status_code=code,
        code=code,                                          # ← HTTP 状态码作为 code，非 API code
        message=msg,
    )
```

**影响:** WebSocket 鉴权/服务失败时，用户丢失了服务端提供的诊断信息。`code` 被设为 HTTP 状态码整数，而非 API 自身的错误码。

### 问题 3 — `iter_over_async` 桥接：包装为自定义格式（LOW）

**文件:** `dashscope/common/utils.py`, 行 124–131

```python
def iter_over_async(ait):
    if error is not None:
        yield DashScopeAPIResponse(
            -1,
            "",
            "Unknown",
            message=f"Error type: {type(error)}, message: {error}",
        )
```

**影响:** async→sync 转换异常时，`code="Unknown"`，message 被拼接为自定义格式字符串。这是传输层故障而非 API 错误修改，但会丢失内部异常携带的结构化错误信息。

### 问题 4 — 非 JSON 响应：`code="Unknown"`（LOW）

**文件:** 多处 — `utils.py:215, 322, 333, 388, 405`、`aiohttp_request.py:168, 252`、`http_request.py:294, 404`

当 API 返回非 JSON/非 SSE 响应（如 HTML 错误页、网关错误）时，`code` 设为 `"Unknown"`，原始 body 保留在 `message` 中。作为 fallback 可接受，但可考虑改为 `code=None` 以明确表示 API 未提供错误码。

---

## 汇总表

| 位置 | Code | Message | 结论 |
|---|---|---|---|
| `_handle_error_message()` (utils.py:270) | 保留 | 保留 | GOOD |
| `_handle_http_failed_response()` JSON (utils.py:300) | 保留 | 保留 | GOOD |
| `_handle_aiohttp_failed_response()` JSON (utils.py:367) | 保留 | 保留 | GOOD |
| HttpRequest SSE sync (http_request.py:409) | 保留 | 保留 | GOOD |
| HttpRequest SSE async (http_request.py:299) | 保留 | 保留 | GOOD |
| AioHttp SSE async (aiohttp_request.py:172) | 保留 | 保留 | GOOD |
| AioHttp 非 2xx JSON (aiohttp_request.py:241) | 保留 | 保留 | GOOD |
| `_handle_http_stream_response()` (utils.py:483) | 保留 | 保留 | GOOD |
| 所有 `from_api_response()` (dashscope_response.py) | 保留 | 保留 | GOOD |
| AgentStudio `from_response()` (exceptions.py:177) | 保留 | 保留 | GOOD |
| **StreamEventMixin** SSE (base_api.py:1559) | **丢弃 → `""`** | **丢弃 → `""`** | **PROBLEM (HIGH)** |
| **WebSocket 握手** (websocket_request.py:207) | HTTP 状态码 | **替换为硬编码字符串** | **PROBLEM (MODERATE)** |
| `iter_over_async` (utils.py:125) | `"Unknown"` | 自定义格式包装 | Minor |
| 非 JSON fallback (多文件) | `"Unknown"` | 保留原始 body | 可接受的 fallback |

---

## 修复建议

1. **修复 StreamEventMixin（HIGH）** — `client/base_api.py:1559`：解析 `data` 为 JSON（参考 `http_request.py:409-418` 的做法），而非设置 `code=""` 和 `message=""`。

2. **修复 WebSocket 握手（MODERATE）** — `websocket_request.py:208`：保留 `e.message`（服务端消息）而非替换。如需友好提示，应追加而非替换。

3. **考虑移除 `code="Unknown"` fallback（LOW）** — 对真正无结构的响应可接受，但可改为 `code=None` 明确表示 API 未提供错误码。

---

## CLI 层审查（`dashscope/cli/`，分支 dev/sdk-cli-1）

### 核心错误处理机制

CLI 层有 3 个统一的错误处理函数，所有命令模块共用：

#### 1. `print_failed_message(rsp)` — 通用错误打印

**文件:** `dashscope/cli/common.py`, 行 35–41

```python
def print_failed_message(rsp):
    err_console.print(
        f"[red]Failed[/red] request_id: {rsp.request_id}, "
        f"status_code: {rsp.status_code}, "
        f"code: {rsp.code}, message: {rsp.message}",
    )
```

**结论:** 直接输出 `rsp.code` 和 `rsp.message`，原样透传。

#### 2. `ensure_ok(rsp)` — 统一响应检查与退出

**文件:** `dashscope/cli/common.py`, 行 44–91

```python
def ensure_ok(rsp, check_business_error=True):
    if rsp.status_code != HTTPStatus.OK:
        print_failed_message(rsp)      # ← 透传
        raise typer.Exit(1)

    if check_business_error:
        error_code = output.get("code")
        message = output.get("message", "Unknown error")
        if error_code and error_code != "":
            err_console.print(
                f"[red]Business Error[/red] ... code: {error_code}, message: {message}",
            )
            raise typer.Exit(1)
    return output
```

**结论:** HTTP 错误和业务错误都直接透传原始 code/message。`"Unknown error"` 仅作为 message 的 fallback（当 API 未返回 message 时）。

#### 3. `handle_sdk_error(action)` — SDK 异常装饰器

**文件:** `dashscope/cli/common.py`, 行 110–125

```python
def handle_sdk_error(action: str):
    def decorator(command_function):
        @wraps(command_function)
        def wrapper(*args, **kwargs):
            try:
                return command_function(*args, **kwargs)
            except typer.Exit:
                raise
            except Exception as exception:
                error(f"{action}: {exception}")
        return wrapper
    return decorator
```

**结论:** `action` 是操作描述标签（如 "Generation request failed"），`{exception}` 是原始异常消息，不修改 API 错误码/消息。

### 各命令模块审查

| 模块 | 错误处理方式 | 结论 |
|---|---|---|
| `generation.py` | `print_failed_message(rsp)` | GOOD |
| `fine_tunes.py` | `ensure_ok` + `print_failed_message` | GOOD |
| `files.py` | `ensure_ok` | GOOD |
| `deployments.py` | `ensure_ok` + `print_failed_message` | GOOD |
| `models.py` | `ensure_ok` | GOOD |
| `embeddings.py` | `ensure_ok` | GOOD |
| `application.py` | `ensure_ok` | GOOD |
| `image_synthesis.py` | `ensure_ok` | GOOD |
| `image_generation.py` | `ensure_ok` | GOOD |
| `multimodal_conversation.py` | `ensure_ok` | GOOD |
| `multimodal_embedding.py` | `ensure_ok` | GOOD |
| `rerank.py` | `ensure_ok` | GOOD |
| `tokenization.py` | `ensure_ok` | GOOD |
| `understanding.py` | `ensure_ok` | GOOD |
| `transcription.py` | `ensure_ok` | GOOD |
| `speech_synthesis.py` | `print_failed_message` | GOOD |
| `video_synthesis.py` | `ensure_ok` | GOOD |
| `oss.py` | `error()` 自定义消息（非 API 错误，本地校验） | GOOD |
| `code_generation.py` | `ensure_ok` | GOOD |

### 轻微注意点

**agentic_rl.py** — 用自定义格式拼接错误：

```python
raise OutputError(f"API returned status {result.status_code}: {result.message}")
```

原始 `message` 保留在拼接字符串中未修改，但 `result.code`（API 业务错误码）未单独展示。影响较小，因为 `status_code` 和 `message` 都已包含。

### CLI 层结论

**CLI 层没有修改 API 的 error code 和 error message，全部原样透传。** 所有命令通过 `print_failed_message` 和 `ensure_ok` 两个统一入口处理错误，直接输出 SDK 返回的 `rsp.code` 和 `rsp.message`。
