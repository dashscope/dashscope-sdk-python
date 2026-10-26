# dashscope/finetune/agentic_rl.py 错误码与错误注册表对齐评估

## 评估目标

审查 `dashscope/finetune/agentic_rl.py` 对外 SDK 接口中抛出的错误（error/message）是否与 `/Users/zhansheng.lzs/ali/pro/ptm/dashscope-error-registry/python/dashscope_error_registry/error_registry.py` 中定义的错误体系对齐。

参考：`error.md` 中对 SDK 错误透传原则的审查结论。

---

## 错误注册表结构

`error_registry.py` 定义了两类错误：

- `PublicError`: 包含 `key`, `status_code`, `error_code`（字符串类名）, `error_msg`, `anthropic_error_code`
- `InternalErrorDef`: 包含 `name`, `external`（对应 PublicError）, `message`, `allow_retry`, `cause`, `solution`

核心 `PublicError` 包括：

| key | status_code | error_code | error_msg |
|---|---|---|---|
| invalid_request | 400 | BadRequestError | The request is invalid. Please check the request and try again. |
| missing_parameter | 400 | BadRequestError | Missing required parameter: {parameter}. |
| auth_failed | 401 | AuthenticationError | Authentication failed. Please provide valid authentication credentials. |
| invalid_api_key | 401 | AuthenticationError | Incorrect API key provided. |
| permission_denied | 403 | PermissionDeniedError | You do not have permission to access this resource. |
| resource_not_found | 404 | NotFoundError | The requested resource was not found: {resource}. |
| request_too_large | 413 | RequestTooLargeError | The request exceeds the maximum allowed size of {limit}. |
| rate_limit_exceeded | 429 | RateLimitError | Rate limit exceeded. Please slow down your requests. |
| internal_error | 500 | InternalServerError | The server encountered an internal error. Please try again later. |
| service_unavailable | 503 | ServiceUnavailableError | The service is temporarily unavailable. Please try again later. |
| request_timeout | 504 | GatewayTimeoutError | The request timed out. Please try again later. |

---

## agentic_rl.py 当前错误处理清单

| 位置 | 错误码 | 当前 message | 错误类型 | 触发场景 |
|---|---|---|---|---|
| `__init__:61-64` | 3001 | Invalid API key configuration | `ValueErrorWithCode` | API key 配置无效 |
| `register_functions:111-114` | 3002 | Function registration failed | `RegistrationError` | 函数注册失败 |
| `upload_datasets:144-147` | 3003 | Datasets upload failed | `DatasetsError` | 数据集上传失败 |
| `submit_job:203-207` | 3004 | Duplicate function names detected... | `ValueErrorWithCode` | 函数名重复（本地校验） |
| `submit_job:256-259` | 3005 | Job submission failed | `RuntimeErrorWithCode` | 任务提交失败 |
| `run:321-324` | 3006 | RL tuning workflow failed | `RuntimeErrorWithCode` | 工作流执行失败 |
| `test_functions:440-443` | 3007 | Unsupported function type: {functype} | `ValueErrorWithCode` | 函数类型不支持（本地校验） |
| `test_functions:459-462` | 3008 | Function test failed | `ValidationError` | 函数测试失败 |

---

## 对齐评估

### 1. 错误码格式未对齐（HIGH）

`error_registry.py` 使用**字符串类名**作为 `error_code`，例如 `BadRequestError`、`AuthenticationError`、`InternalServerError`。

`agentic_rl.py` 使用**自定义整数错误码** `3001-3008`，与注册表的字符串体系完全不兼容。

**结论：未对齐。**

### 2. 缺少注册表要求的标准字段（HIGH）

`agentic_rl.py` 的异常类（`ValueErrorWithCode`、`RuntimeErrorWithCode`、`RegistrationError`、`DatasetsError`、`ValidationError`）仅包含：

- `message`
- `error_code`（整数）
- `__cause__`

缺少 `error_registry.py` 中 `PublicError` 定义的字段：

- `status_code`
- `key`
- `anthropic_error_code`

**结论：未对齐。**

### 3. 输入校验类错误未映射到 INVALID_REQUEST（HIGH）

以下场景属于**客户端输入校验失败**，应映射到注册表的 `INVALID_REQUEST` / `BadRequestError`：

| 当前错误码 | 场景 | 建议映射 |
|---|---|---|
| 3004 | 函数名重复 | `INVALID_REQUEST` / `BadRequestError` |
| 3007 | 不支持的函数类型 | `INVALID_REQUEST` / `BadRequestError` |
| 3008 | 函数测试输入校验失败 | `INVALID_REQUEST` / `BadRequestError` |

当前使用 3004/3007/3008 等自定义整数码，用户无法通过统一错误码识别这是请求参数问题。

**结论：未对齐。**

### 4. API key 错误未映射到 INVALID_API_KEY/AuthenticationError（HIGH）

`__init__:61-64` 在 API key 配置无效时抛出 `error_code=3001`、message 为 `"Invalid API key configuration"`。

注册表中对应错误为 `INVALID_API_KEY`：

- `status_code=401`
- `error_code="AuthenticationError"`
- `error_msg="Incorrect API key provided."`

当前实现未使用标准 `AuthenticationError` 错误码，也未携带 `401` 状态码。

**结论：未对齐。**

### 5. 服务端/网络类错误未映射到标准服务端错误码（MODERATE）

以下场景属于**服务端调用或底层依赖失败**，应根据实际失败原因映射到 `INTERNAL_ERROR`、`SERVICE_UNAVAILABLE`、`REQUEST_TIMEOUT` 等：

| 当前错误码 | 场景 | 建议映射 |
|---|---|---|
| 3002 | 函数注册失败 | 透传底层 API 错误，或 fallback 到 `INTERNAL_ERROR` |
| 3003 | 数据集上传失败 | 透传底层 OSS/API 错误，或 fallback 到 `INTERNAL_ERROR` / `INVALID_FILE` |
| 3005 | 任务提交失败 | 透传底层 API 错误，或 fallback 到 `INTERNAL_ERROR` |
| 3006 | RL 调优工作流失败 | 透传底层 API 错误，或 fallback 到 `INTERNAL_ERROR` |

当前实现使用固定整数码和硬编码 message，无法区分服务端内部错误、服务不可用、超时等情形。

**部分缓解：** 这些异常在捕获前会检查 `hasattr(e, "error_code")`，如果底层异常已有 `error_code` 则直接抛出。但：

- 该机制依赖底层异常是否携带 `error_code`，未明确与注册表字段兼容；
- 新抛出的 `RuntimeErrorWithCode` / `RegistrationError` / `DatasetsError` 仍然使用整数码。

**结论：部分对齐，但包装层未对齐。**

### 6. 错误消息未使用注册表标准模板（MODERATE）

注册表为每类错误提供了标准化 `error_msg`（部分支持变量替换，如 `{parameter}`、`{resource}`、`{limit}`）。

`agentic_rl.py` 的消息均为硬编码英文，例如：

- `"Invalid API key configuration"`
- `"Duplicate function names detected. All function names must be unique."`
- `"Unsupported function type: {functype}"`

这些消息语义正确，但未与注册表统一，导致：

- 多语言/多 SDK 难以保持一致；
- 无法通过注册表的 `key` 定位错误定义和解决方案。

**结论：未对齐。**

### 7. 异常类继承体系与注册表结构不匹配（LOW-MODERATE）

`agentic_rl.py` 的异常定义在 `dashscope/finetune/reinforcement/common/errors.py` 中，形成了独立的层级：

- `AgenticRLError`（基类，整数 `error_code`）
  - `DeploymentError` → `RegistrationError` / `DatasetsError`
  - `ValidationError`
  - `InputError`
  - `OutputError`
  - 等

该层级未继承或引用注册表中的 `PublicError` / `InternalErrorDef`，无法直接用于与注册表一致的错误序列化。

**结论：未对齐。**

---

## 与 error.md 原则的对比

`error.md` 的核心结论是：

> SDK 应原样透传 HTTP API 返回的 `code` 和 `message`，不得修改。

`agentic_rl.py` 的问题在于：

- 对于**本地 SDK 输入校验错误**（如 3004、3007），它没有使用 API 侧统一约定的 `BadRequestError` 等错误码；
- 对于**服务端调用失败**（如 3002、3003、3005、3006），虽然保留了带 `error_code` 的底层异常，但新包装的错误码和消息是 SDK 自定义的，覆盖了 API 原本可能返回的结构化错误。

因此，`agentic_rl.py` 的错误处理方式与 `error.md` 主张的"透传 API 错误"原则存在冲突。

---

## 修复建议

### 短期（保持兼容性前提下对齐）

1. **将整数错误码映射为注册表字符串错误码**：
   - 3001 → `AuthenticationError`
   - 3004 / 3007 / 3008 → `BadRequestError`
   - 3002 / 3003 / 3005 / 3006 → 透传底层错误，或 fallback 到 `InternalServerError`

2. **在异常中补充 `status_code` 和 `key` 字段**：
   - 使异常对象能够被统一错误处理中间件识别。

3. **使用注册表的标准 message 模板**：
   - 例如 API key 错误使用 `"Incorrect API key provided."`；
   - 输入校验错误使用 `"The request is invalid. Please check the request and try again."` 或 `"Missing required parameter: {parameter}."`。

### 长期（推荐）

1. **直接复用 `dashscope_error_registry` 中的 `PublicError` / `InternalErrorDef`**：
   - 本地输入校验错误直接抛出基于 `INVALID_REQUEST` 的异常；
   - 认证失败直接抛出基于 `INVALID_API_KEY` 的异常；
   - 服务端错误透传 API 返回的 `code`/`message`，不重新包装。

2. **移除 `3001-3008` 自定义整数码**：
   - 这些码与注册表不兼容，会造成多 SDK / 多语言不一致。

3. **统一异常基类**：
   - 让 AgenticRL 相关异常继承自一个与注册表结构兼容的基类，支持 `status_code`, `error_code`, `message`, `anthropic_error_code` 等字段。

---

## 结论

`dashscope/finetune/agentic_rl.py` 对外 SDK 接口的错误处理**未与 `dashscope_error_registry/error_registry.py` 对齐**。主要问题包括：

- 使用自定义整数错误码（3001-3008），而非注册表要求的字符串错误码；
- 缺少 `status_code`、`key`、`anthropic_error_code` 等标准字段；
- 输入校验、认证、服务端错误未映射到注册表中的标准错误类型；
- 错误消息为硬编码，未使用注册表标准模板；
- 与 `error.md` 主张的"原样透传 API 错误"原则存在冲突。

建议按上述修复建议逐步改造，优先处理错误码格式和输入校验类错误的映射。
