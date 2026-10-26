# AgentStudio / AgenticRL 迁移 v2 与统一 2.0 错误体系

> 状态：设计稿（v2，已按评审意见修订）。
> 约束：`dashscope-error-registry` 仓库本次不动；catalog 以 v2 vendored 副本
> （`dashscope/v2/_error_registry.py`，上游 1.0.11@cd873b6）为准。
>
> 已确认的方向：
> 1. agentstudio / agentic_rl 实现归 v2，**1.x 路径保留反向引用 shim，不删除**。
> 2. reinforcement 的 1.x 路径由 `dashscope/finetune/reinforcement/` 调整为
>    **`dashscope/reinforcement/`**。
> 3. v2 目录下全部遵循 `dashscope.v2.errors`；agentstudio 与
>    agentic_rl（reinforcement）同样遵循新错误体系。

## 1. 背景与现状

两个子系统的实现都已经在 v2 下，1.x 路径是反向引用的 shim；但**错误体系仍是 1.x 风格**，靠 mount 边界的翻译层把子系统异常转成 v2 错误。本次的核心工作就是消除这两套独立错误体系和翻译层。

### 1.1 AgentStudio

| 项 | 现状 |
|---|---|
| 实现 | `dashscope/v2/agentstudio/`（client / transport / resources / streaming / pagination） |
| 1.x 路径 | `dashscope/agentstudio/__init__.py`：纯 shim，re-export v2（**保留**） |
| 错误体系 | `dashscope/v2/agentstudio/exceptions.py`：独立层级，基于 `dashscope._shared._errors.DashScopeException`，与 `dashscope.v2.errors` **无血缘** |
| 翻译层 | `dashscope/v2/resources/_agentstudio.py`：透明代理。HTTP 层（`APIStatusError` 子类）走 `error_from_status`；本地/流错误（APIConnectionError / APITimeoutError / StreamError / StreamClosedError）走 `error_from_sdk_exception(family="sdk.agentstudio")` |
| 网关错误码 | anthropic 风格字符串（`invalid_request_error` / `authentication_error` / `overloaded_error` …），`exceptions.from_response` 里手维护 `_CODE_TO_CLASS` / `_STATUS_TO_DEFAULT` 两张表 |

### 1.2 AgenticRL

| 项 | 现状 |
|---|---|
| 实现 | `dashscope/v2/finetune/agentic_rl.py`（`Client(...).rl` 所用）；`submit_job` 返回 `SubmitResult` |
| 1.x 路径 | `dashscope/finetune/agentic_rl.py`：v2 子类 + 5 个 1.x 动词（`submit_job`/`cancel`/`list`/`get`/`delete`/`logs`），委托 1.x 根 API `FineTunes`，返回 1.x `FineTune` dataclass（**保留**） |
| reinforcement 包 | `dashscope/finetune/reinforcement/`：`sys.meta_path` finder 按需别名到 v2，**同一 module 对象**。**本次移动到 `dashscope/reinforcement/`** |
| 错误体系 | `common/errors.py`：独立层级。`AgenticRLError` 带**数字** `error_code`（1000/1800…）+ timestamp + `_RootCauseMixin` root-cause 格式化；另有 `RuntimeErrorWithCode` / `ValueErrorWithCode` 混入内置类型 |
| 翻译层 | `resources/reinforcement_learning.py` 的 `_translate_rl_errors` → `error_from_sdk_exception(family="sdk.agentic_rl")` |
| catalog | 27 个 `sdk.agentic_rl.*` 条目：18 个类名精确命中、6 个 condition-only、1 个家族 fallback（`sdk.agentic_rl.Error`） |

v2 侧 `client.rl.jobs` 已原生覆盖任务全生命周期（create/fetch/wait/list/cancel/delete/logs）。

## 2. 目标态

1. **单一实现**：实现只在 v2；1.x 路径保留反向引用 shim（agentstudio / agentic_rl / reinforcement 三者都是），shim 不含第二份实现。
2. **单一错误体系**：两个子系统原生抛 `dashscope.v2.errors`；删除两个 mount 翻译层；错误码事实源只剩 catalog。
3. **reinforcement 新 1.x 路径**：`dashscope.reinforcement`（反向引用 `dashscope.v2.finetune.reinforcement`）。
4. 旧路径 import 到的错误类就是新的 v2 错误类（同一对象），保证 `except` / `isinstance` 行为一致。

## 3. AgentStudio 迁移方案

### 3.1 异常类映射

| 旧（agentstudio.exceptions） | 新（dashscope.v2.errors） | 说明 |
|---|---|---|
| `AgentStudioError` | `DashScopeError` | 基类 |
| `APIConnectionError` | `APIConnectionError` | catalog `sdk.ConnectionError` 支撑 |
| `APITimeoutError` | `APITimeoutError` | catalog `sdk.Timeout` 支撑 |
| `APIStatusError` | `APIStatusError` | HTTP 层基类 |
| `InvalidRequestError` (400) | `BadRequestError` | **改名** |
| `AuthenticationError` (401) | `AuthenticationError` | 同名 |
| `PermissionDeniedError` (403) | `PermissionDeniedError` | 同名 |
| `NotFoundError` (404) | `NotFoundError` | 同名 |
| `ConflictError` (409) | `APIStatusError`（基类） | catalog 无 409 条目（registry 缺口，本次不动 registry，先落基类） |
| `RateLimitError` (429) | `RateLimitError` | 同名 |
| `OverloadedError` (503) | `ServiceUnavailableError` | **改名** |
| `InternalServerError` (500) | `InternalServerError` | 同名 |
| `StreamError` | 新 `SdkError` 子类 | catalog `sdk.agentstudio.StreamError` 支撑 |
| `StreamClosedError` | 同上（子类） | catalog `sdk.agentstudio.StreamClosedError` 支撑 |

属性对应：`code/message/request_id/status_code` 两边同名；旧 `raw` → v2 `body`。

### 3.2 wire code 交叉表

catalog public 条目自带 `anthropic_error_code` 字段，交叉表**从 catalog 派生**，不再手维护。两个对不上的例外显式补：

- `permission_denied_error`（agentstudio 网关实际值）→ `permission_denied` 条目（catalog 字段值是 `permission_error`）
- `conflict_error` / 409 → 无条目，落 `APIStatusError` 基类

`from_response` 的 CMA/旧版 `error_code`/`error_message`/Spring 三种 body 形状的归一化逻辑保留（迁入 agentstudio transport），类选择改为「crosswalk → 状态映射 → 基类」。

### 3.3 改动清单

- 重写 `dashscope/v2/agentstudio/exceptions.py`：独立层级替换为 `dashscope.v2.errors` 体系；旧类名保留为 **deprecated alias**（指向新类，§6 决策点 3），使 1.x shim re-export 的旧名字不断裂
- transport / streaming 的 raise 点改抛 v2 错误
- 删除 `dashscope/v2/resources/_agentstudio.py` 翻译代理；`client.agentstudio` 直接挂载
- **保留** `dashscope/agentstudio/` shim（re-export v2，含错误类新名字与旧别名）
- 测试：`tests/v2/test_agentstudio_shim.py` / `test_agentstudio_errors.py` 重写（翻译层断言改为原生错误断言）；`tests/unit/test_agentstudio_*` 走 1.x shim，继续可用，按需调整异常断言

## 4. AgenticRL 迁移方案

### 4.1 API 面

- `dashscope/finetune/agentic_rl.py` **保留**：1.x 动词（`FineTunes` 委托、`FineTune` 返回）是 1.x 独有面；底层实现继续复用 v2，无第二份逻辑
- `dashscope/v2/finetune/agentic_rl.py` 里对 `dashscope.api_key` 根全局的现有规避保持不变

### 4.2 错误体系统一

- `common/errors.py` 重写：异常类改为 `dashscope.v2.errors.SdkError` 子类，`code`/`solution`/`allow_retry` 由 catalog 条目支撑（沿用 `APIConnectionError` 的 `_SDK_ERROR_NAME` 模式）；condition-only 条目不设专用类，由 `sdk_error()` 工厂构造
- **数字 `error_code` 处理**：catalog 用 `sdk.agentic_rl.*` 名称作 code。旧数字码建议保留 property 兼容映射一个版本周期（§6 决策点 2）；`{inner_code}` 占位符改由包装时的内部错误名填充
- `RuntimeErrorWithCode` / `ValueErrorWithCode` 的内置类型混入取消（v2 体系不入内置层级）
- `_RootCauseMixin` 的 root-cause 遍历/格式化评估并入 `SdkError.__str__` 或删除
- 删除 `_translate_rl_errors` 挂载翻译；工作流（register/upload/submit/run）原生抛 v2 错误
- 旧类名保留为 deprecated alias（18 个精确命中类名 + `AgenticRLError` 基类名），1.x 路径 `except` 旧类名继续命中同一对象

### 4.3 reinforcement 路径迁移：`dashscope/finetune/reinforcement/` → `dashscope/reinforcement/`

- 新包 `dashscope/reinforcement/`：整体平移现有 meta_path 别名 shim，目标仍为 `dashscope.v2.finetune.reinforcement`（实现位置不变）
- **旧路径 `dashscope/finetune/reinforcement/` 是否保留别名，见 §6 决策点 4**。外部进程按模块名字符串 import 的既有消费方：
  - FC sandbox `start.sh` 的 `FC_SERVER_CLASSPATH`
  - `examples/workspace/scripts/start_local_*.sh`
  - `server.py` 传给 uvicorn 的 dotted path（worker 进程内 import）
  这些脚本若同步改为 `dashscope.reinforcement.*`，需同时更新模板与示例；已部署 sandbox 用其环境内旧版本，不受影响
- `dashscope/finetune/agentic_rl.py` 内对 `...reinforcement` 的 re-export 改指新路径（保持两个旧名字可 import 的过渡方案由决策点 4 决定）

## 5. 破坏性变更清单

| 变更 | 影响面 |
|---|---|
| agentstudio 异常改名（InvalidRequestError→BadRequestError、OverloadedError→ServiceUnavailableError、ConflictError 取消） | `except` 旧类名的代码（deprecated alias 可缓解） |
| agentstudio 异常属性 `raw` → `body` | 读 `.raw` 的代码 |
| agentstudio 异常基类从 `DashScopeException` 换到 `DashScopeError` | 基类捕获链（1.x 根 `DashScopeException` 捕获将失效，alias 无法解决基类切换） |
| rl 数字 `error_code` 语义切换为 catalog 名 | 按数字判断错误种类的代码（property 映射可缓解） |
| rl `*WithCode` 不再继承内置 RuntimeError/ValueError | `except RuntimeError` 兜底 rl 错误的代码 |
| `AgenticRLError` 基类从 `Exception` 换到 `SdkError` | 基类捕获链 |
| reinforcement 1.x 路径调整为 `dashscope.reinforcement` | 按旧 dotted path 启动的脚本（决策点 4） |

## 6. 决策点（评审确认）

1. ~~1.x import 路径是否删除~~ → **已确认：全部保留反向引用 shim**
2. **rl 数字 `error_code`**：直接取消，还是保留 property 兼容映射到 catalog 名？
3. **旧异常类名** deprecated alias 的范围与保留周期（agentstudio 的 InvalidRequestError/OverloadedError、rl 的 18 个类名）
4. **旧 `dashscope.finetune.reinforcement` 路径**：保留到 `dashscope.reinforcement` 的别名（推荐，外部脚本无感），还是直接切换并要求同步改脚本？
5. registry 缺口（409/conflict_error、permission_denied_error 别名）：本次落基类 + 显式 crosswalk，后续是否推 registry 补条目？

## 7. 实施步骤（建议分 PR）

1. **PR-1 agentstudio 错误统一**：重写 exceptions + transport/streaming raise 点 + crosswalk；删 `_agentstudio.py` 翻译层；v2 测试改写；1.x shim 保持可用
2. **PR-2 agentic_rl 错误统一**：重写 `common/errors.py`；工作流 raise 点改造；删 `_translate_rl_errors`；registry 完整性测试同步（`test_agentic_rl_reachability_counts` 等）
3. **PR-3 reinforcement 路径迁移**：新增 `dashscope/reinforcement/` shim；按决策点 4 处理旧路径；更新 FC 模板 / examples 脚本 / acli skill 文档中的模块名

每步验收：`pytest tests/v2 tests/unit` 全绿 + CLI 冒烟（rl 提交/查询）+ agentstudio 样例冒烟。
