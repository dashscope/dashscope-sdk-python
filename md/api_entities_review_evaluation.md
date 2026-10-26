# dashscope/api_entities 代码 Review 评估报告

> 对原始 review 报告中 11 条风险建议的逐条核对结果（已对照源码验证）。
> 日期：2026-07-27

## 逐条评估

| # | 原严重度 | 文件/位置 | 问题 | 评估结论 | 说明 |
|---|----------|-----------|------|----------|------|
| 1 | 🔴 严重 | `encryption.py:94` | HTTP 请求无 timeout | ✅ 属实，P0 同意 | `Encryption.initialize()` 在构建请求时同步调用（`api_request_factory.py:126`），无 timeout 会阻塞调用线程。建议复用 `DEFAULT_REQUEST_TIMEOUT_SECONDS` 常量而非硬编码 30 |
| 2 | 🟡 中等 | `http_request.py:126-127` | query 模式 `replace("api","api-task")` | ⚠️ 属实但应降级 | 全 SDK 内**没有任何调用方传 `query=True`**（仅 factory 从 kwargs 透传），近似死代码，只有用户显式传入才触发。默认 URL 只含一个 "api"，误替换需自定义 base_address 含 "api"（如 `api.example.com`）。建议降为低优先级；原 review 给出的修复片段过于复杂 |
| 3 | 🟡 中等 | `aiohttp_request.py:80-82` | 同上 | ⚠️ 同 #2 | 同上 |
| 4 | 🟡 中等 | `websocket_request.py:75` | Authorization 用小写 "bearer" | ✅ 属实 | RFC 7235 中 auth scheme 大小写不敏感，服务端已兼容多年，属一致性问题。改为 "Bearer" 安全，可修 |
| 5 | 🟡 中等 | `dashscope_response.py:43` | `DictMixin.pop` 的 default 无默认值 | ✅ 属实，但**原建议修复有误** | 若按 review 建议改为 `default=None`，`pop(missing_key)` 会静默返回 None，而 `dict.pop` 无 default 时应抛 KeyError，语义不符。正确做法用 sentinel：`_MISSING = object()`，default 缺省时走 `super().pop(key)` |
| 6 | 🟡 中等 | `api_request_factory.py:107-114` | 手动拼接 URL 不健壮 | ❌ 不成立，不建议改 | 拼接输入全部是 SDK 内部受控常量（task_group/task/function），现状简单可靠。`urljoin` 对无尾斜杠路径会**吞掉末段路径**，改用它反而风险更高 |
| 7 | 🟡 中等 | `aio_session.py:59-86` | GC 回调清理逻辑复杂 | ✅ 属实 | `weakref.finalize` 触发时 session 已不可达，回调内遍历 `_aio_sessions.values()` 按 `id()` 匹配基本不可能命中，该回调实际近乎无效。同意"待优化"（如 atexit 或显式 shutdown API），不阻塞发布 |
| 8 | 🟡 中等 | `http_request.py:438-441` | task_id 被 output 覆盖 | ✅ 属实但是预期行为 | 响应同时含两字段时 `output` 优先是正确语义；改为 if/elif 仅提升可读性。建议降为低优先级 |
| 9 | 🟡 中等 | `api_request_data.py:167-168` | `_only_parameters` pop 修改实例状态 | ✅ 属实，但**原建议修复有 bug** | review 建议只 `get` 不删，会导致 `raw_input` 同时残留在 `parameters` 中被序列化进请求体，改变请求行为。正确修复：`params = dict(self.parameters)` 复制后再 pop。目前该方法仅在 `get_http_payload`（第 129 行）调用一次，仅对象复用/重试场景受影响 |
| 10 | 🟢 低 | `aiohttp_request.py` | `_handle_response` 未传 headers | ✅ 属实 | 已确认该文件所有 `DashScopeAPIResponse(...)` 构造均未传 `headers`，与 `http_request.py` 不一致，建议补齐 |
| 11 | 🟢 低 | `http_request.py:92-96` | headers 合并顺序可能被用户覆盖 | ❌ 不成立 | `**self.headers` 展开的是 `base_request.py:19` 中 SDK 自己设置的 `user-agent` / `X-DashScope-DataInspection`，**不是用户自定义 headers**；用户 headers 通过 `add_headers()` 后置合并覆盖，行为正确，无需修改 |

## 总体结论

原报告 11 条中 **8 条属实、2 条不成立（#6、#11）、2 条属实但重要性应下调（#2/#3、#8）**。另有三点必须注意：

1. **#6、#11 不成立**，建议不做改动。
2. **#5、#9 的"建议修复代码"本身会引入新 bug**（改变 dict 语义 / 改变请求体），采纳前必须按上表说明改写。
3. **#2/#3 的 query 分支在 SDK 内无调用方**，修复前建议先确认是否存在外部使用者，否则可考虑降级或标记为死代码。

## 修正后的建议修复顺序

| 优先级 | 事项 |
|--------|------|
| P0 | #1 `encryption.py` 添加 timeout（复用 `DEFAULT_REQUEST_TIMEOUT_SECONDS`） |
| P1 | #4 WebSocket "Bearer" 大写；#10 aiohttp 响应补 headers |
| P2 | #5 `pop` 用 sentinel 修复；#9 `_only_parameters` 复制字典后再 pop |
| P3 | #7 aio_session 清理机制优化；#8 if/elif 可读性调整 |
| 观察 | #2/#3 确认 query 分支是否有外部使用者后再决定 |
| 不修 | #6 URL 拼接维持现状；#11 headers 合并顺序行为正确 |
