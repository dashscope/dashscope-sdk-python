# V2 errors 层按 `error_registry.py` 标准改写计划

> **前提（三条硬边界）**
> 1. **V2 分支是接口标准**：资源树、方法名、模型名、`ResponseBase` 信封一律不动。
> 2. **只修 errors 相关**：`errors.py` 及其消费者，其余不碰。
> 3. **不参考 `V2-errors` 分支的实现**：唯一输入是 `error_registry.py` 这个目录文件本身
>    （新 errors 标准）+ V2 现有代码。因此本计划**不做分支合并**，
>    `md/dashscope-2.0-errors-merge-plan.md` 的合并路线已被本计划取代。
> 4. **不需要向后兼容**：V2 是全新版本，尚未发布、无存量用户。
>    异常继承关系只为**语义正确**服务，不为"存量 `except` 子句继续生效"服务；
>    不做兼容别名、不做 deprecation shim；既有测试**允许随设计一起改**。
>    （本条推翻了本文早先以兼容性为依据的 D3 / D5 / §5 / §6，下文已按此重写。）

---

## 1. 输入：registry 到底约定了什么

`error_registry.py`（vendored 到 `dashscope/v2/_error_registry.py`，649 行）
是**生成制品**，顶部带 provenance：

```python
__version__ = "1.0.6"
__commit__  = "56b7208"
__snapshot__ = False
```

两张表、两个 frozen dataclass：

**`PublicError` ×16**（`PUBLIC_ERRORS`）

| 字段 | 含义 | 对 V2 的约束 |
| --- | --- | --- |
| `key` | 内部规范键（`invalid_request` / `auth_failed` / …） | **不是**网关下发的 wire code |
| `status_code` | 400/401/403/404/**413**/429/500/**503**/**504** | V2 必须每个都能落类 |
| `error_code` | **异常类名**（字符串） | 类名一字不差由目录决定 |
| `error_msg` | 默认文案，带 `{parameter}` / `{resource}` / `{limit}` 占位 | 配 `format_msg()` |
| `anthropic_error_code` | `invalid_request_error` / `authentication_error` / `permission_error` / `not_found_error` / `request_too_large` / `rate_limit_error` / `billing_error` / `api_error` / `overloaded_error` / `timeout_error` | anthropic 兼容协议的错误形状 |

`error_code` 一共要求 **9 个异常类**：
`BadRequestError` `AuthenticationError` `PermissionDeniedError` `NotFoundError`
`RequestTooLargeError` `RateLimitError` `InternalServerError`
`ServiceUnavailableError` `GatewayTimeoutError`

**`SdkErrorDef` ×32**（`SDK_ERRORS`，1.0.5 里叫 `INTERNAL_ERRORS`）：`name`
（`sdk.InvalidApiKey` ×1 /
`sdk.agentic_rl.*` ×27 / `sdk.agentstudio.*` ×4）、`message`、`solution`、
`allow_retry`、`vars`，配 `format_message()`。
**注意：这张表没有 status_code，也没有异常类名**——本地/SDK 侧失败，不是 HTTP 响应。

### 1.1 最关键的一条约束（决定整体设计）

**`PublicError` 没有 wire code 字段。** 所以：

- `_STATUS_MAP`（HTTP 状态 → 异常类）**可以完全由目录推导** ✅
- `_CODE_MAP`（网关 wire code → 异常类）**无法由目录推导** ❌
  ——`Throttling.RateQuota`、`DataInspectionFailed`、`Model.NotExist` 这些
  真实下发码到规范键的对应关系，是 V2 自己的知识，目录里没有。

结论：改写后 V2 要**新增一份显式 crosswalk**（wire code → catalog entry），
它是手维护的，但**类选择仍然由目录驱动**（crosswalk 只负责选到哪个
`PublicError`，异常类和状态码都从该条目取）。这份 crosswalk 必须配完整性测试，
否则它会像现在的 `_CODE_MAP` 一样悄悄漏项（见 §2 差距 4）。

### 1.2 第二条决定性事实：32 条 `sdk.*` 只有 23 条可由异常类型到达

已逐条核对 `SDK_ERRORS.name` 的后缀与 1.x 实际存在的异常类：

| 分组 | 条数 | 可由 `type(exc).__name__` 到达 |
| --- | --- | --- |
| `sdk.agentstudio.*` | 4 | **4/4**（`dashscope/agentstudio/exceptions.py:75,81,135,141`） |
| `sdk.agentic_rl.*` | 27 | **18/27** |
| `sdk.InvalidApiKey` | 1 | 由 v2 挂载层自己抛，不依赖 1.x 类 |

`dashscope/finetune/reinforcement/common/errors.py` 里**不存在**这 9 个类：

- 8 条是**操作结果条件**，不是异常类型：`FunctionRegistrationFailed`
  `DatasetsUploadFailed` `DuplicateFunctionNames` `JobSubmissionFailed`
  `WorkflowFailed` `UnsupportedFunctionType` `FunctionTestFailed`
  `FunctionTestTimeout`。1.x 是用**通用类 + 具体消息 + 数字 `error_code`**
  表达它们的，例如"Duplicate function names detected"由
  `common/model.py:1589` 抛 `ValidationError`，`InputError(f"...", error_code=4011)`
  （`common/utils.py:187`）。
- 1 条是 `sdk.agentic_rl.Error`：1.x 对应的基类叫 `AgenticRLError`
  （`common/errors.py:35`），**不叫 `Error`**。

后果（决定 D6）：

1. **不能**声称"类名与目录后缀逐条对得上"——早先版本的 D6 这么写，是错的。
2. 类型推导只覆盖 18 + 4 = 22 条；`AgenticRLError` 及其未收录子类要有
   **家族级兜底**（落到 `sdk.agentic_rl.Error`）。
3. 那 8 条条件级条目**不要**靠嗅探 message 文本或数字 `error_code` 去反推——
   脆弱、且属于 1.x 内部编码。它们在 V2 侧就是**目录有、代码不产出**的条目
   （目录是跨 SDK 共享制品，存在不可达条目是正常的）。
4. 因此 P0 的完整性测试方向必须反过来：**不是**"每条 `name` 都能找到 1.x 类"
   （会红），而是"每个 v2 挂载层可能捕获到的 1.x 类，都能在目录里找到落点或兜底"。

### 1.3 第三条输入约束：`SdkErrorDef` 不带 `external`，**不要照 `InternalErrorDef` 补**

曾考虑把 `SdkErrorDef` 改成 1.0.6 里 `InternalErrorDef` 的形状（多一个
`external: PublicError` 和 `cause: str`），让 sdk.* 也能投影到公开错误。
**结论：不改。** 这个不对称是上游 schema 层面的刻意设计，不是生成器漏了字段：

| 证据 | 位置 | 内容 |
| --- | --- | --- |
| proto 里两个 message 本就不同 | `proto/error_registry.proto:25-32` vs `:41-53` | `InternalError` 有 `string external = 2;  // Reference to PublicError.key`；`SdkError` **没有 `external`**，改成 `solution` 必填 + 四个分语言 `solution_python/java/go/cpp` |
| 三层模型把两者分在不同子图 | `README.md:9-20` 的 mermaid | `InternalError → external → PublicError` 全在 **服务端** 子图里；`SdkError → solution` 单独待在 **客户端SDK** 子图，**没有连向 PublicError 的边** |
| 措辞明确 | `README.md:24` | SdkError 是"客户端 SDK **独立**错误码，包含 `solution`（**必填**修复建议）" |
| 校验只针对 InternalError | `tools/generate.py` 的 `main()` | 任何 internal error 的 `external` 不在已知 public key 里就 `sys.exit(1)`；**sdk error 没有任何等价校验**，因为 schema 不给它这个字段 |
| 字段顺序也对得上 | 生成制品 | 生成的 `SdkErrorDef(name, message, solution, allow_retry, vars)` 与 `InternalErrorDef(name, external, message, allow_retry, vars, cause, solution)` 的差异逐字对应各自 proto 的字段序，说明是 schema 决定的，不是生成器随手排的 |

语义上也讲得通：`external` 描述的是"服务端组件级错误对外暴露成哪个公开错误码"，
它的前提是**存在一个 HTTP 响应**要按公开码下发；sdk.* 全是本地/客户端侧失败
（配置缺失、上传失败、函数注册失败……），**根本没有响应可下发**，
所以没有公开投影可言，只有 `solution` 这种"给人看的修复建议"。

对 D6 的影响（重要）：**`SdkError.status_code = None` 不是本地偷懒，
而是目录给出的唯一忠实读法。** 目录里没有为 sdk.* 定义任何公开投影，
V2 若自己编一个状态码，等于在生成制品之外私加语义。

若将来确实需要 sdk.* 的公开投影（比如要让 `rl` 失败也能进统一的可观测口径），
正确做法是**向上游 registry 仓库提 RFC 加一个 optional `external`**，
由 proto + textproto 生成，而不是改 `_error_registry.py` 这个标了
`DO NOT EDIT` 的制品——手改会让下次重新生成时静默丢失。

---

## 2. V2 errors 现状与差距（逐条已核实）

> **这是改写前的基线**：下表 `errors.py` 的行号一律指 commit `d2088de` 之前的
> 那份文件，改写后已不对应任何现有代码。差距 1–5、8、10 的闭环见 §5 与 §6，
> 差距 6、7、9 按 §7 决策 1/2/4 **仍不接线**。

| # | 差距 | 证据 | 后果 |
| --- | --- | --- | --- |
| 1 | 缺 3 个目录要求的异常类 | `errors.py:82-107` 只有 6 个 `APIStatusError` 子类 | 413/503/504 无处落 |
| 2 | `_STATUS_MAP` 缺 413/500/503/504 | `errors.py:110-116` 只有 400/401/403/404/429 | 413 落到裸 `APIStatusError`；503/504 靠 `>=500` 兜底成 `InternalServerError`，**无法区分** |
| 3 | `status_code` 是硬编码字面量 | `errors.py:83,87,91,95,99,103` | 目录改值，代码不跟随（正是"标准"要消除的） |
| 4 | `_CODE_MAP` 漏掉 2 个已声明的 wire code | `ErrorCode.INVALID_PARAMETER`(:24)、`MODEL_NOT_EXIST`(:25) 不在 `_CODE_MAP`(:118-130) | `InvalidParameter` 只能靠状态码兜底；`Model.NotExist` **不会**变成 `NotFoundError` |
| 5 | 503/504 被压平 | `errors.py:128-129` 都指向 `InternalServerError` | 用户无法按类区分服务不可用与网关超时 |
| 6 | `anthropic_error_code` 无消费者 | `protocols/anthropic.py` 只 import `BadRequestError, DashScopeError, ErrorCode`，全文无错误体输出 | anthropic 兼容协议不产出 anthropic 形状的错误 |
| 7 | `allow_retry` 无消费者 | `transport/_retry.py` 的 `RETRYABLE_STATUS_CODES` 只看 HTTP 状态 | sdk.* 错误的可重试性信息丢失 |
| 8 | **rl / agentstudio 的 1.x 异常裸奔** | `reinforcement_learning.py` 467 行 **0 个 `except`**；1.x 抛 `ValueErrorWithCode` / `RuntimeErrorWithCode` / `RegistrationError` / `DatasetsError` / `ValidationError`（定义在 `dashscope/finetune/reinforcement/common/errors.py`） | `except DashScopeError` **抓不到** rl 失败；32 条 `sdk.*` 目录无人使用 |
| 9 | 占位符文案未使用 | 目录 `format_msg()` 支持 `{parameter}` 等 | 默认消息不含具体参数名/资源名 |
| 10 | 无目录完整性测试 | `tests/v2/` 无对应文件 | 目录与异常类脱钩不会被发现 |

差距 8 是里面唯一的**功能性缺陷**（不是精度问题）：rl 是 1.x 直通层，
用户拿到的异常不在 2.0 异常族里。差距 1–5、10 是"按标准改写"的主体。
差距 6、7、9 是目录里已有、V2 尚未接线的字段（见 §7 决策项）。
**其中差距 9 已于 P3 接线**（见 §7 决策 4），差距 6、7 仍按决策 1/2 不接。

---

## 3. 目标设计

**D1｜registry 以生成制品身份落地，provenance = 上游 1.0.6。**

已落地：`dashscope/v2/_error_registry.py`，649 行，最长行 78 字符，
**零非 ASCII**，flake8 / black(79) / pylint 全绿（pylint 10.00/10）。
`git hash-object` = `908fbf161dae98c2e2bae5337671fdb533b352a8`。

文件头：首行与上游逐字一致，provenance 三行是上游独立 `_version.py` 的
内容内联进来（适配 4）：

```python
"""Auto-generated by tools/generate.py -- DO NOT EDIT."""
...
__version__ = "1.0.6"
__commit__ = "56b7208"
__snapshot__ = False
```

权威源是**独立仓库** `gitlab.alibaba-inc.com/dashscope/dashscope-error-registry`
（`tools/generate.py` + `proto/error_registry.proto` + `data/*.textproto`，
产出 python/go/java/cpp 四语言）。原文即
`python/dashscope_error_registry/error_registry.py`。生成器不在 SDK 仓库里。

本仓库各 ref 里的历史副本（仅供考古，均**未采用**）：

| 来源 | 路径 | 版本 |
| --- | --- | --- |
| `upstream/dev/errors` | `dashscope/common/error_registry.py` | 1.0.3（`8e9c1a0`） |
| `V2-errors` | `dashscope/v2/error_registry.py` | 1.0.5（`ededf0f`） |

**与 1.0.6 原文的一致性已逐字段核实**（不是靠版本号声明）：

| 表 | 本文件 | 原文 1.0.6 | 核实结果 |
| --- | --- | --- | --- |
| `PublicError` schema | 5 字段 | 5 字段 | 字段名逐个相同 |
| `PUBLIC_ERRORS` | 16 条 | 16 条 | 4 个字段**逐条全同，含顺序** |
| `SdkErrorDef` schema | 5 字段 | 5 字段 | 字段名逐个相同 |
| `SDK_ERRORS` | 32 条 | 32 条 | 5 个字段**逐条全同，含顺序** |
| `InternalErrorDef` / `INTERNAL_ERRORS` | **不含** | 277 条 | 见下 |

**为什么裁掉那张 277 条的表**（1.0.6 相对 1.0.5 的唯一实质增量）：

- 它描述的是**服务端组件**的错误：`gateway.*` / `mint.*` / `api.*` /
  `turbo.*` / `model.*`。客户端 SDK 一条也不会抛。
- 实测原文整份 vendor 的代价：black(79) 之后 **3553 行**、
  **388 行中文**（`mint.*` 的 `cause` / `solution`）、
  **97 处 E501**（black 拆不开长字符串字面量，最长行 267 字符）。
  而裁剪后的文件是 649 行 / 0 中文 / 0 E501。
- 也就是说，直接采用原文会让 `dashscope/v2/` **第一次出现 CJK**，
  并把一个 lint 干净的文件换成 97 处 E501。
- 裁剪已由 `tests/v2/test_error_registry.py::
  test_vendored_copy_is_pruned_to_the_tables_v2_consumes` 钉死，
  将来谁要 vendor 全量，必须有意识地改这条测试。

若确实需要那张表（例如要让 SDK 解释网关内部码），正解不是把原文塞进
`dashscope/v2/`，而是：让上游提供分表产出或纯 ASCII 变体，
或改为依赖 `dashscope-error-registry` wheel（`python/build.sh` 已在打包上传）。

**相对上游原文的 5 处本地适配**（整文件替换时会全部回退，需重新打上；
在此记录避免届时误判为"目录被污染"）：

1. `substitutions: Dict[str, str] = None` → 显式 `Optional[Dict[str, str]]`
   （隐式 Optional 被现代 mypy 拒）。
2. `format_msg(self, vars=...)` / `format_message(self, vars=...)` 形参
   **改名**为 `substitutions` / `variables`——上游用 `vars` 遮蔽内建名，
   仓库的 pylint disable 列表里**没有** W0622，会直接报错。
3. 长字符串与目录列表按仓库 pin 的 black 23.3.0 / `--line-length=79` 重排
   （原文的 `PUBLIC_ERRORS = [...]` 是单行数千字符）。
4. 上游把 provenance 放在独立的 `python/dashscope_error_registry/_version.py`；
   这里内联进文件头，因为 vendoring 只取一个文件。
5. **文件名** `error_registry.py` → `_error_registry.py`（§8 已完成）：
   v2 根下内部模块一律带 `_` 前缀，这份不导出、只被 `errors.py` 与测试引用。

顶部 provenance 视为不可手改；所有适配逻辑写在 `errors.py`。

**D2｜异常类名 = 目录的 `error_code`，一字不差。** 新增 3 个：
`RequestTooLargeError`、`ServiceUnavailableError`、`GatewayTimeoutError`。
`status_code` 类属性与 `_STATUS_MAP` 全部由 `PUBLIC_ERRORS` 推导，删掉字面量。

**D3｜扁平继承：9 个类全部直接挂 `APIStatusError`。**

```python
class BadRequestError(APIStatusError):          status_code = 400
class AuthenticationError(APIStatusError):      status_code = 401
class PermissionDeniedError(APIStatusError):    status_code = 403
class NotFoundError(APIStatusError):            status_code = 404
class RequestTooLargeError(APIStatusError):     status_code = 413   # 新增
class RateLimitError(APIStatusError):           status_code = 429
class InternalServerError(APIStatusError):      status_code = 500
class ServiceUnavailableError(APIStatusError):  status_code = 503   # 新增
class GatewayTimeoutError(APIStatusError):      status_code = 504   # 新增
```

理由（按前提 4，不再以兼容性为依据）：

- 目录给的是 `(status_code, error_code)` 二元组，**扁平结构正好一一对应**，
  不引入目录没说的东西。
- 目录里 `INTERNAL_ERROR` 就是 500。把 503/504 挂在它下面，等于把
  `InternalServerError` 的含义偷偷扩成"任意 5xx"——这是 **invent 出来的分类学**，
  恰恰是"标准"要消除的。
- 想要"任意服务端失败"的兜底，用 `except APIStatusError` + `err.status_code >= 500`
  显式表达，而不是靠继承意外获得。

配套删掉 `error_from_status` 里的 `>= 500` 特例
（改写前那行是 `cls = InternalServerError if status_code >= 500 else APIStatusError`；
改写后类选择只剩 `errors.py:286` 的 `_CODE_MAP` → `_STATUS_MAP` → `APIStatusError` 三级查表）：
未映射的状态码一律落 `APIStatusError`，**不再猜成 500**。502/507 这类
目录未收录的码因此不会被贴上 `InternalServerError` 的名字。

**D4｜两套命名空间都保留，用 crosswalk 连接。**
`ErrorCode`（wire code 枚举，13 项）是网关真实下发值；目录 `key`（16 项）是内部规范键。
**不合并**。新增 `_WIRE_TO_PUBLIC: Dict[str, PublicError]` 显式表达对应关系，
`_CODE_MAP` 由它派生。同时补上差距 4 的两个漏项：
`InvalidParameter → INVALID_REQUEST`、`Model.NotExist → RESOURCE_NOT_FOUND`。

**D5｜`ContentFilterError` 改成 `BadRequestError` 的子类——目录特例消失。**

现状：`ContentFilterError(DashScopeError)`，与目录冲突。目录里
`CONTENT_POLICY_VIOLATION` 的 `error_code` 是 `BadRequestError`（400）。
早先的做法是让 crosswalk "V2 覆盖目录"，那是为了不改既有测试；按前提 4 不再需要。

```python
class ContentFilterError(BadRequestError):
    """内容审核拦截（wire code DataInspectionFailed）。"""
```

这样 `isinstance(err, BadRequestError)` 成立，**目录契约天然满足**，
同时保留 DashScope 确实需要的细粒度区分（内容审核是高频码，且会在
HTTP 200 上以业务错误形式下发）。D4 的 crosswalk 里
`DataInspectionFailed → CONTENT_POLICY_VIOLATION` 直接取目录类，
V2 只是**在它下面细化**，不是覆盖。

由此得到一条**可测试的不变式**（进 P0 完整性测试）：

> crosswalk 为任一 wire code 选出的异常类，必须是该条目 `error_code`
> 所指类的**子类或同类**。

即：目录决定"至少是哪一类"，V2 只能往下细化，不能横向改判。

HTTP 200 业务错误路径不受影响：`error_from_business_code` 显式传
`status_code=200`，实例属性覆盖类属性，`ContentFilterError.status_code`
在该路径上读作 200（与现状一致）。

**D6｜sdk.* 错误 → 单一新类 `SdkError(DashScopeError)`，两级识别。**

```python
class SdkError(DashScopeError):
    solution: Optional[str]      # 目录 SdkErrorDef.solution
    allow_retry: bool            # 目录 SdkErrorDef.allow_retry
```

`status_code=None`——这些是本地/SDK 侧失败，**没有 HTTP 语义，不编造状态码**；
`code=def_.name`（如 `sdk.agentic_rl.InputError`）。

识别按 §1.2 的事实分两级，**不手维护 32 条映射表**：

```python
# 一级：家族命名空间 + 类名精确命中（覆盖 18 + 4 = 22 条）
name = f"{family}.{type(exc).__name__}"      # family = "sdk.agentic_rl" / "sdk.agentstudio"
def_ = _SDK_ERRORS.get(name)
# 二级：家族兜底条目
if def_ is None:
    def_ = _SDK_ERRORS[f"{family}.Error"]    # sdk.agentic_rl.Error
```

要点：

- `AgenticRLError`（基类）及其未被目录收录的子类，一律落 `sdk.agentic_rl.Error`。
  这样**任何** rl 失败都保证拿到 `SdkError`，不会漏成裸 1.x 异常。
- 那 8 条条件级条目（`DuplicateFunctionNames` 等）**故意不接**：
  要靠 message 文本或 1.x 的数字 `error_code`（4011/4012…）反推，
  既脆弱又侵入 1.x 内部编码。目录是跨 SDK 共享制品，有不可达条目是正常的。
- `sdk.InvalidApiKey` 不由 1.x 类型到达，而是 v2 挂载层在 api_key 缺失时**自己抛**。

**agentstudio 家族没有 `.Error` 兜底条目**，两级方案对它不成立。已核实
`dashscope/agentstudio/exceptions.py` 自己就分了两层（源码里有分节注释）：

| 层 | 类 | 目录归属 | 应译成 |
| --- | --- | --- | --- |
| 传输层（无 HTTP 响应） | `APIConnectionError`(:75) `APITimeoutError`(:81) `StreamError`(:135) `StreamClosedError`(:141) | `SDK_ERRORS` 的 4 条 `sdk.agentstudio.*`，**恰好一一对应** | `SdkError` |
| HTTP 层（有响应） | `APIStatusError`(:92) 及其 8 个子类 | 不在 `SDK_ERRORS` 里 | **按 `status_code` 映射到 v2 状态类** |

关键：agentstudio 的 HTTP 层异常带真实 `status_code`，
**不能**压进 `SdkError`（`status_code=None`），那会丢掉状态码。
它们的 `code` 取值（`invalid_request_error` / `authentication_error` / …）
还正好是目录的 `anthropic_error_code` 词表，说明目录设计者也是这么分层的。

**D7｜翻译只发生在 v2 挂载层，不动 1.x；但两个挂载点代价不同。**

- **rl（P2 做）** ✅ 已完成：`resources/reinforcement_learning.py` 是 v2 自己的资源模块，
  方法体在 v2 手里，直接 `try/except` 包住 1.x 调用即可
  （**改写前**全文 467 行、0 个 `except`；现 513 行，
  翻译集中在 `:111-121` 的 `_translate_rl_errors()`）。
- **agentstudio（本次不做，单列 P2b）**：`client.py:113-128` 的
  `agentstudio` property **把整个 1.x `AgentStudioClient` 对象返回给用户**，
  用户随后自己调它的方法——v2 根本不在调用路径上，
  `try/except` 无处可加。要翻译就必须包一层**透明代理**
  （转发全部属性/方法、同步与异步两套），那是资源层的结构改动，
  超出"只修 errors"的范围，也会碰前提 1 的资源树。
  该处代码本身已写明这是待办：
  > "Until the dedicated errors-alignment task (design doc M3 3.12) lands,
  > 1.x error types and HTTP behavior propagate as-is"

  所以 P2 只修 rl；agentstudio 保持现状（缺口继续挂在 design doc M3 3.12），
  4 条 `sdk.agentstudio.*` 目录条目在本次改写后仍不可达。

1.x 代码零改动，符合既有约束。

---

## 4. 分阶段执行

**P0｜落地目录（不改任何行为）** ✅ 已完成
- `error_registry.py` 原样进 V2（生成制品，provenance 三行不手改）
- 新增 `tests/v2/test_error_registry.py`，**只放当前就能绿的用例**：
  - 目录自身完整性：`key` 唯一、`status_code` 合法、`error_code` 非空、
    `anthropic_error_code` 非空；`SDK_ERRORS.name` 唯一
  - `SDK_ERRORS` 可达性（方向按 §1.2，**不是**"每条 `name` 都有 1.x 类"）：
    `sdk.agentic_rl.*` 中恰有 18 条命中 `common/errors.py` 的真实类，
    `sdk.agentstudio.*` 4 条全部命中 `agentstudio/exceptions.py`，
    且家族兜底 `sdk.agentic_rl.Error` 存在。
    把这三个数字**钉死**，上游重新生成目录时会立刻暴露漂移。
  - `format_msg()` / `format_message()` 的占位符替换行为
- 注意：`PUBLIC_ERRORS.error_code` → 真实类的解析用例**不能**放这一阶段，
  3 个类还不存在，会红。随 P1 一起落。
- 此阶段结束时应全绿且行为无变化

**P1｜按目录重写 `errors.py`（主体）** ✅ 已完成
- 新增 3 个异常类，9 个类扁平挂 `APIStatusError`（D2/D3），`status_code` 由目录取值
- `ContentFilterError` 改挂 `BadRequestError`（D5）
- `_STATUS_MAP` 改为由 `PUBLIC_ERRORS` 推导（同状态多条去重）
- 新增 `_WIRE_TO_PUBLIC` crosswalk，`_CODE_MAP` 由它派生（D4）
- 删掉 `error_from_status` 的 `>= 500` 兜底特例（D3）
- 补差距 4、5
- 目录 ↔ 代码的耦合用例**落在 `test_errors.py`，不放 `test_error_registry.py`**：
  后者的 docstring 已把自己限定为"只验目录自身，不断言 `errors.py` 如何映射目录"，
  耦合用例依赖 `errors.py`，放进去会让"目录完整性"这层被下游代码污染。
  已落的耦合用例：
  - 每条 `PUBLIC_ERRORS.error_code` 都能解析成一个真实异常类
  - 解析出的类的 `status_code` 属性与目录条目一致
  - `_WIRE_TO_PUBLIC` 覆盖 `ErrorCode` 全部 13 项（**锁死差距 4 不再复发**）
  - **D5 不变式**：crosswalk 为每个 wire code 选出的类，是对应目录条目
    `error_code` 类的子类或同类；未被 refine 的必须**恰好等于**该类
  - 目录 `key` 与 wire code 两个命名空间不相交（crosswalk 存在的理由）
  - `_STATUS_MAP` 由目录推导，而非手写
- **同步改 `tests/v2/test_errors.py`**（前提 4：测试随设计改，不冻结）。改后必须覆盖：
  - 413 → `RequestTooLargeError`、503 → `ServiceUnavailableError`、
    504 → `GatewayTimeoutError`，三者互不为子类（锁死 D3 的扁平性）
  - 未收录状态码（如 502）→ 裸 `APIStatusError`，**不是** `InternalServerError`
  - `ContentFilterError` 同时 `isinstance BadRequestError`（锁死 D5）
  - HTTP 200 + `InvalidParameter` → `BadRequestError`；
    HTTP 200 + `Model.NotExist` → `NotFoundError`

**P2｜rl 的 sdk.* 翻译（差距 8，唯一功能缺陷）** ✅ 已完成
- 新增 `SdkError` + `error_from_sdk_exception(exc, *, family)`（D6）
- `resources/reinforcement_learning.py`：**8 个**公开工作流方法全部包住
  （`Rl`/`AsyncRl` 的 `register_functions`/`test_functions`/`upload_data`
  ＋ `RlJobs`/`AsyncRlJobs` 的 `create`）。用的是**一个普通同步
  `@contextlib.contextmanager`**，同步体（`asyncio.run(...)`）和异步体
  （`await ...`）都能包，不需要两套。
- **catch 的是 3 个基类，不是 1 个**：`AgenticRLError` 只 root 了
  `common/errors.py` 19 个异常类里的 17 个，`RuntimeErrorWithCode` /
  `ValueErrorWithCode` 分别继承 `RuntimeError` / `ValueError`，**不在其下**。
  三元组正好覆盖 19/19（已用 `issubclass` 逐条验过）。
  不再放宽的依据：`common/utils.py:139` 已经把 transport / timeout /
  unexpected 全部 `except Exception` 收进 `RuntimeErrorWithCode`，
  所以捕得更宽只会把**本模块自己的 bug** 贴成 `SdkError`。
  另：pydantic 的 `ValidationError` 也是 `ValueError` 子类，但不在三元组内，
  已验 `issubclass(pydantic.ValidationError, _RL_EXCEPTIONS) is False`。
- **v2 模型适配留在 `with` 外面**：`_to_job(...)` / `UploadedData(...)`
  不被包，pydantic 校验失败不会被误译成 SDK 错误。
- 测试（`test_errors.py` 5 个机制用例 + `test_rl.py` 8 个接线用例）：
  `code` 为 `sdk.agentic_rl.*`、`status_code is None`、原始 message 保留、
  `solution` / `allow_retry` 取自目录、`__cause__` 链住原异常、
  未收录类落 `sdk.agentic_rl.Error` 兜底、`except DashScopeError` 可捕获、
  同步与异步两路都覆盖、**非 rl 家族的 `ValueError` 原样穿透**。
- **偏离 D6 一处：`sdk.InvalidApiKey` 本次不接线。**
  D6 说它"由 v2 挂载层在 api_key 缺失时自己抛"，但挂载层无法可靠判断
  "缺失"：`Config.api_key` 是 `resolve_config()` **当时**读的
  `os.environ[DASHSCOPE_API_KEY]`，而 1.x 的 `set_api_key` 是**调用时**读环境变量。
  用户先建 `Client()`、后 export key，今天能用；加了前置检查就会误报
  `sdk.InvalidApiKey`。要避免误报就得在 rl 模块里复制 `_config` 的解析逻辑
  （含私有 `_API_KEY_ENV`），代价大于收益。
  现状已可接受：这条路径 1.x 抛
  `ValueErrorWithCode("Invalid API key configuration")`
  （`finetune/agentic_rl.py:61`，非 `cli/agentic_rl.py`），
  经翻译后是 `SdkError(code="sdk.agentic_rl.ValueErrorWithCode")`，
  仍被 `except DashScopeError` 捕获，只是 `code` 不够专名。
  真要修，正确位置是 `_config.py` 的 key 解析（超出"只修 errors"范围）。

**P2b｜agentstudio 翻译（本次不做，已论证）**
需要透明代理才能 intercept，属资源层结构改动，超出"只修 errors"范围。
缺口继续挂在 design doc M3 3.12 与 `client.py:114-119` 的既有注释上。

**现状的用户可见危害（比"没接"本身更值得记一笔）**：1.x
`dashscope.agentstudio` 导出 14 个异常类，其中 **8 个与
`dashscope.v2.errors` 同名**——`APIConnectionError`、`APIStatusError`、
`APITimeoutError`、`AuthenticationError`、`InternalServerError`、
`NotFoundError`、`PermissionDeniedError`、`RateLimitError`。实测这 14 个对
`issubclass(x, dashscope.v2.errors.DashScopeError)` **全为 False**（它们挂在
`AgentStudioError(DashScopeException)` 这条 1.x 基类下），所以
`from dashscope.v2.errors import RateLimitError` 去接 `client.agentstudio`
的限流**永远接不到**，且从 import 语句上完全看不出问题。

**更隐蔽的是其中 6 个连父类名都一样**：`APITimeoutError` 两侧都继承
`APIConnectionError`；`AuthenticationError` / `InternalServerError` /
`NotFoundError` / `PermissionDeniedError` / `RateLimitError` 两侧都继承
`APIStatusError`。于是 traceback 前两层完全同名，只有第三层
（`AgentStudioError` vs `DashScopeError`）能区分，读栈的人几乎必然看漏。
这与下面"只包调用层比完全不包更糟"是同一类风险的**现成版本**——
不必等接线，今天就已经在误导用户。

已核实的实现成本（留给后来人，不必重新踩）：

- **翻译分两条路（分层依据见 §3 D7 的表，不在此重述）。**
  可操作的落点：4 条非 HTTP 类走
  `error_from_sdk_exception(exc, family="sdk.agentstudio")` → `SdkError`；
  HTTP 层走
  `error_from_status(exc.status_code, message=exc.message,
  body=exc.raw, headers={"x-request-id": exc.request_id})`。
  **陷阱**：`sdk.agentstudio` 没有 `.Error` 家族兜底（`sdk.agentic_rl` 有），
  所以对那 4 条之外的类调用 `error_from_sdk_exception` 会 `KeyError`，
  **反而把原始异常盖掉**。现状已被测试钉死：
  `test_errors.py:477-485`（断言 `KeyError`）、
  `test_error_registry.py:265-287`（断言无兜底、且 HTTP 层类与目录不相交）。
  接线时要改的是这两个用例，不是绕过它们。
- **rl 的做法搬不过来。** rl 是逐方法手写 `@contextmanager`
  （`reinforcement_learning.py:111-121`）；agentstudio 是 11 个资源类
  × 同步/异步双胞胎、约 75 个 `async def`，且**没有共同基类或 mixin**
  （都是裸类 + `self._client`）。逐方法重写等于复制整个子 SDK，
  所以只有**递归透明代理**可行。
- **代理会破坏现有断言。** `test_apps.py:146-158` 与
  `smoke/test_full_surface.py:560-599` 断言
  `isinstance(client.agentstudio, AgentStudioClient)`；
  `test_apps.py:233-242`、`:246-255` 通过 `mounted.transport.is_closed`
  验证关闭，代理必须原样透传 `transport`。
- **决定性的那条：流式错误在迭代中途抛，代理拦不到。**
  `SessionEvents.stream()` 是**急切**发请求（`session_events.py:127`）
  然后返回 `_TypedEventStream`；此后的失败发生在 `for event in stream`
  期间——`StreamClosedError` 抛在生成器体里（`streaming.py:88`、`:169`），
  生产者线程的 transport 异常也是迭代中 `raise item` 重抛
  （`streaming.py:125`、`:208`）。只包 `stream()` 这一层，
  仅能覆盖"建流即失败"，**覆盖不到流式的主要失败模式**。
  要覆盖就得再套一层鸭子类型代理，转发
  `__enter__` / `__exit__` / `__iter__` / `close` / `text_stream`
  （异步侧再加 `__aenter__` / `__aexit__` / `__aiter__` / `__anext__` /
  `aclose`），而 `text_stream` 是**属性、其 getter 本身是生成器**
  （`session_events.py:155-178`）。注意隐式 dunder 查找走**类型**而非
  `__getattr__`，所以每个 dunder 都必须显式声明，**漏一个就是一个静默缺口**。
- **因此"只包调用层"比"完全不包"更糟**：它会让用户相信
  `except DashScopeError` 已覆盖 agentstudio，然后在最主要的流式路径上被漏网。
  加上这里无法对真实后端做集成验证，本次维持不接线，
  保持上述两个"未解决"用例原样，作为对现状的显式声明。

**P3｜可选接线（取决于 §7 决策）**
- `anthropic_error_code` → anthropic 兼容协议输出 anthropic 形状错误体（差距 6；
  **仍不做**，决策 1）
- `allow_retry` → 是否影响重试策略（差距 7；**仍不做**，决策 2。
  注意 `_retry.py` 是 HTTP 层，与 sdk.* 不同层，**不要混用**）
- `format_msg` / `format_message` 占位符填充（差距 9）✅ 已完成，
  见 `errors.py` 的 `_catalog_message()` 与决策 4

**P4｜文档同步** ✅ 已完成

三份文档都已同步，且**每段示例都用 mock transport 对真实实现跑过**（9.1 的
import 名单、9.2 业务层 → `RateLimitError` 且 `status_code == 200`、9.1 未收录
502 → 裸 `APIStatusError`、9.5 `SdkError` 的 `code`/`solution`/`allow_retry`/
`__cause__`、9.6 `raise_on_error=False` → `is_error=True` 且连接错误仍然抛）。

- `md/dashscope-2.0-design.md`：**新增 §2.7 错误模型（目录驱动，长期约束）**——
  目录是唯一事实源（1.0.6 / `56b7208`、`PUBLIC_ERRORS` 16 条、`SDK_ERRORS`
  32 条、`InternalError` 为何不 vendor）、派生规则、三层来源一个异常族、
  crosswalk 与目录 key 的命名空间分离（附 13 项对照表 + `Arrearage` 为何走
  `permission_denied`）、`ContentFilterError` 是唯一细化且必须子类化、
  `SdkError` 无状态码的标准依据、1.x 挂载层翻译义务与当前进度、
  新增错误代码时的 6 步自检。另同步 5 处：§0 调研共识表的落点
  （§2.2 → §2.7）、§2.2 错误模型摘要（9 个扁平类 + 修掉不存在的
  `dashscope.errors.ErrorCode` 路径）、§2.6 自检清单加第 ⑥ 步、
  §7 目录树补 `_error_registry.py`、里程碑 1.3 与 3.12（记录 rl 半边已完成、
  剩 agentstudio）。
- `md/dashscope-2.0-naming-review.md`：**偏离计划一处**——计划写"§6.4 记录
  新增异常类名"，但 §6.4 是**输出字段**改名清单，异常类名塞进去会破坏该节
  的分类；改为**新增 §6.5 异常类名**，10 行按 P0/P1/P2 分级，并点明本节
  记录的不是"改成什么名"而是"哪些名从 SDK 手写变成目录派生"。
- `md/dashscope-2.0-interfaces-examples.md`：**偏离计划一处**——计划写"异常族
  清单与 `except` 示例"，但该文档原先**完全没有**错误处理章节（只有开头约定里
  一句"未捕获的 HTTP 错误默认抛出异常族"）；改为**新增 §9 错误处理**六节
  （9.1 异常族与 import 路径 + 类/状态码表、9.2 两层错误一个族、9.3 异常属性、
  9.4 重试、9.5 `SdkError`、9.6 `raise_on_error=False`），并在 §6 Agentic RL
  补"两半错误行为统一"、§7 AgentStudio 补"异常仍是 1.x 原生类型、
  `except DashScopeError` 抓不到"。

**写文档过程中查出并改正一处事实错误**：初稿在 design §2.7 与 naming-review
§6.5 都写了"去掉 `>= 500` 猜测可以避免连带触发重试"。**这是错的**——
`transport/_retry.py:12` 的 `RETRYABLE_STATUS_CODES`（408/409/429/500/502/
503/504）由 `http.py:173,323` 按**原始响应状态码**判定，与 `errors.py` 最终
选出哪个异常类**完全无关**；502 改前改后都会被重试。两处已改为陈述真实危害：
猜出来的类让 `except InternalServerError` 抓到并非 500 的失败，且实例
`status_code`（502）与类声明的（500）矛盾。

`md/dashscope-2.0-m1-m3-review.md` **故意不改**：该文第 76 行记录的
"`APIConnectionError`/`APITimeoutError` 全库零引用"缺陷如今已修（传输层
`_translate_transport_error` 已接线），但文首明写"本文结论按当时状态保留，
**不作回填**"，属带日期的评审快照，遵守其自身约定。

**P5｜resource 层旁路点接线** ✅ 已完成

先核实"要不要改"：`dashscope/v2/resources/` 下 **48 处** `check_response()`
调用全部经 `response_error()` → `error_from_status()` /
`error_from_business_code()`，P0/P1 的目录改写**自动生效**，无需逐点改。
真正要动的只有**绕过 `check_response` 自己造异常**的旁路点，共 3 处：

| 文件 | 改前 | 改后 |
| --- | --- | --- |
| `models.py`（`get`，同步 + 异步） | 手写 `NotFoundError(..., status_code=404)` | `error_from_business_code(code=ErrorCode.MODEL_NOT_EXIST.value, headers=response.headers)` |
| `audio/realtime.py:128` | 裸 `DashScopeError`（WS `type:error` 事件） | `error_from_business_code(...)`，与 `transport/sse.py:69-76` 的 `error` 分支同形 |
| `fine_tunes.py` | ① `_check_terminal` 丢弃 `job.message`（同步 + 异步）② SSE 无 `status:` 行时默认 500（`_iter_sse` + 异步内联副本） | ① `message=job.message or ...`，对齐 `_jobs.py:222` ② 新增模块级 `_stream_error()` 供同步/异步共用，删掉 `_raise_stream_error` 方法 |

**`fine_tunes` 的 500 是本次唯一实质缺陷**：`status:` 在 1.x
`StreamEventMixin` 线格式里是**可选行**，缺省时把失败说成服务端 500——与 D3
拒绝"未收录状态码猜 `InternalServerError`"是同一条理由：**没有任何一方声明过
这个状态码**。现在缺省即走业务错误层，类由 body 的 `code` 选，
`status_code == 200`。

**故意不改（9 处裸 `raise DashScopeError`，分 3 类）**

1. **wait 超时 / wait 期间被取消**（各 4 处：`_jobs.py:210,300` 与
   `:230,320`、`fine_tunes.py:370,554` 与 `:387,571`）。这两类**正是**
   `SdkError` 的目标场景（客户端自己判定、服务端没回失败），但目录
   `SDK_ERRORS` 32 条只有 `sdk.InvalidApiKey` + `sdk.agentic_rl.*`（27）+
   `sdk.agentstudio.*`（4），**没有 jobs / fine-tune 家族**。
   `_error_registry.py` 是"DO NOT EDIT"生成物，按既定约束**不能在 SDK 里造
   目录条目**——要补得去上游 `dashscope-error-registry` 仓库提。故此处
   **阻塞于上游**，不是遗漏。
2. **本地入参校验**（`fine_tunes.py:189` `_job_id`）。这不是网关失败，
   resource 层其余同类一律用内建 `ValueError` / `RuntimeError`；改成
   `ValueError` 会变更对外异常类型，超出本次"仅改 errors 相关、不动函数名与
   输入输出"的范围。
3. **`transport/ws.py` 的 `WsEventError`**。在 `transport/` 而非
   `resources/`，不属 P5 范围——**已在 P6 单独处理**。

另记一处**行为不对称、但不属 errors**：`_jobs.py` 把 FAILED 检查放在 `_parse`
（每次 `get()` 都抛）、CANCELED 放在 `_check_canceled`（只有 `wait()` 抛）；
`fine_tunes.py` 两者都在 `_check_terminal`（只有 `wait()` 抛）。这是"何时抛"
的行为决策，统一它会改变 `get()` 的可观测行为，不在本次范围。

**P6｜transport 层接线（`ws.py`）** ✅ 已完成

P5 收口后全 v2 复审发现的最大缺口，与 realtime 那一处同类：
`transport/ws.py` 的 `_check_failed()` 把**所有**服务端 `task-failed` 事件写死
成 `WsEventError`，payload 里的 wire `code` 只被存进实例属性、**从不参与选类**。
实测 `payload.code == "Throttling"`：

```
e.code == "Throttling"                    # 拿到了
e.status_code is None                     # 没有
isinstance(e, APIStatusError)  -> False
isinstance(e, RateLimitError)  -> False   # except 抓不到
```

影响 **`client.audio.tts` 与 `client.audio.asr` 两个公开接口**（`_check_failed`
被 `run_task` / `recv_event` / `recv` 三处调用，是唯一收口点，改一处即全覆盖）。

改法与 realtime 一致：走 `error_from_business_code`，类由 `payload.code` 经
crosswalk 选。**一处非显然的接线**：WS 帧把 `request_id` 放在 `header` 而非
`payload`，而 `error_from_business_code` 只从 body 或 `x-request-id` 读它，
websocket 又没有 HTTP 响应头可传——所以**仅在有值时**用
`headers={"x-request-id": ...}` 这条既定回退通道送进去，`body` 仍保留原始整帧。

**`WsEventError` 收窄语义而非删除**：它仍在 `ws.py` 的 `__all__` 里，且 `:141`
那处（`run_task` 收到非 `task-started` 事件）**没有 wire code**，属本地判定的
协议时序违规，crosswalk 无从下手。一个类同时表示"服务端说任务失败"和"服务端
破坏了帧协议"正是原先混淆的根源，现在 docstring 明确排除前者。

**随之而来的行为变更**：`task-failed` 的 code 若未命中 crosswalk，现在落裸
`DashScopeError` 而非 `WsEventError`——与 `sse.py` / realtime 的未命中行为一致。
无消费者受损：`WsEventError` 在 `ws.py` 之外**零 import、零测试断言**
（`tts.py` / `asr.py` 只 import `Connector, WsSession`，`transport/__init__.py`
的 `__all__` 也没有它），除基类外本来就没人能按名字 catch 它。

**故意留下的一处（非目录问题，待决）**：`ws.py:186` 与 `:205` 的
`json.loads(raw)` 让裸 `json.JSONDecodeError` 直接逃出去，而对应的 SSE 路径
（`sse.py:45-51`）把"畸形报文"和"非 dict 报文"都包成 `DashScopeError`。
所以畸形 WS 帧目前 `except DashScopeError` **抓不到**。这是健壮性一致性缺口、
不是目录接线问题，且会改变对外异常类型，故未并入本次。

---

## 5. 行为变更清单（不做兼容性评估）

前提 4：V2 未发布、无存量用户，**不评估"是否破坏存量"**。
下表只列出相对当前 V2 的**可观测行为差异**，供 review 时逐条确认是有意为之。

| 变更 | 现状 → 改后 | 依据 |
| --- | --- | --- |
| 503 | 压平成 `InternalServerError` → `ServiceUnavailableError` | D2/D3，差距 5 |
| 504 | 压平成 `InternalServerError` → `GatewayTimeoutError` | D2/D3，差距 5 |
| 413 | 落裸 `APIStatusError` → `RequestTooLargeError` | D2，差距 2 |
| 未收录状态码（502/507…） | 猜成 `InternalServerError` → 裸 `APIStatusError` | D3 |
| `InvalidParameter`（HTTP 200 业务错误） | 基类 `DashScopeError` → `BadRequestError` | D4，差距 4 |
| `Model.NotExist` | 不映射（靠状态码兜底）→ `NotFoundError` | D4，差距 4 |
| `ContentFilterError` 的父类 | `DashScopeError` → `BadRequestError` | D5 |
| rl 失败 | 裸 1.x 异常（`except DashScopeError` 抓不到）→ `SdkError` | D6/D7，差距 8 |
| agentstudio 失败 | **不变**，仍裸奔 1.x 异常 | P2b 延后（D7 已论证需透明代理） |
| 空 body 的失败（如网关直接回 503） | `message == ""` → 目录 `error_msg` 兜底 | 决策 4，差距 9 |
| `models.get` 查无此模型 | 仍是 `NotFoundError`，但 `status_code` 由伪造的 404 → 真实的 **200**，并补上 `code == "Model.NotExist"` | P5 |
| fine-tune SSE 错误帧缺 `status:` 行 | 伪造 500 → `InternalServerError`；改后由 body 的 `code` 选类，`status_code == 200` | P5 |
| fine-tune `wait()` 遇 FAILED | 丢弃服务端 `job.message`，只给本地套话 → 优先 `job.message` | P5 |
| realtime WS `type:error` 事件 | 恒为基类 `DashScopeError` → 按事件 `code` 选目录类（如 `Throttling` → `RateLimitError`） | P5 |
| audio `task-failed` 事件（tts / asr） | 恒为 `WsEventError`（`status_code is None`，不是 `APIStatusError`）→ 按 payload 的 `code` 选目录类，`status_code == 200` | P6 |
| `task-failed` 的 code 未命中 crosswalk | `WsEventError` → 裸 `DashScopeError`（与 `sse.py` / realtime 的未命中行为一致） | P6 |
| 公开异常类数量 | 11 → 15（`APIStatusError` 子类 6 → 9；另加 `SdkError`） | D2/D6 |

**`APIStatusError` 的语义随之放宽（P1 已改其 docstring）。**
D5 把 `ContentFilterError` 挂到 `BadRequestError` 之后，HTTP 200 业务错误
造出的实例**也是** `APIStatusError`，其 `status_code == 200`。
所以基类不能再读作"非 2xx 响应"，而是"**网关声明的失败**（两层都算），
实例的 `status_code` 说明是哪一层"。这样 `except ContentFilterError`
一处即可覆盖两种上报方式——正是统一异常族的目的。

**新增类不从 `dashscope/v2/__init__.py` 导出（P1 决定，推翻早先建议）。**
理由是一致性而非成本：该文件现在只导出 `errors` 模块 +
`DashScopeError` + `ErrorCode`（`except` 需要的两样），9 个状态类一律走
`dashscope.v2.errors`；包根 `dashscope/__init__.py:84` 也只再导出
`AsyncClient, Client, chat, settings`，且属 1.x 不可改范围。
只加 3 个会显得随意，加 9 个会让 `__all__` 失焦。全部 v2 消费者
本来就直接 `from dashscope.v2.errors import ...`，无需中转。

---

## 6. 验证清单

- [x] `pytest tests/v2`：305 → **324**（P0 加 19 个目录用例）→ **336**
      （P1 加 12 个耦合用例）→ **349**（P2 加 13 个：5 个翻译机制 + 8 个 rl
      接线）→ **356**（决策 4 加 7 个目录文案兜底）→ **359**（P5 新加 3 个、
      加强 2 个）→ **360 passed / 3 skipped**（P6 新加 1 个、加强 1 个），
      无回归。P5/P6 这 7 个用例都用 `git stash` 只回退对应源文件验证过
      **改前必失败**（P5 命中 `assert 500 == 200`，P6 命中 `raise WsEventError`）
- [x] `pytest tests/unit`（**476 passed / 6 skipped**）/
      `tests/legacy`（**65 passed / 31 skipped**）不回归
- [x] **基名冲突已修**：`tests/v2/test_messages.py` 与 1.x 的
      `tests/unit/test_messages.py`（Assistants Messages 资源）同基名，
      `tests/` 下无 `__init__.py`，pytest 的 prepend 导入模式两者都叫
      `test_messages` —— `pytest.ini` 的 `testpaths = tests` 意味着裸跑
      `pytest` 会被 `import file mismatch` **中断收集**（实测 922 collected
      + 1 error，一个用例都不执行）。已把 v2 那份改名
      `test_types_messages.py`（它测的是 `dashscope/v2/types/messages.py`
      这个规范消息模型，按 v2 测试"以被测模块命名"的惯例补上包名限定），
      1.x 那份不动。裸跑收集恢复：**934 collected / 0 error**
- [x] **D3 扁平性**：`test_catalog_classes_are_flat_siblings` 断言 9 个类
      `__bases__ == (APIStatusError,)` 且两两互不为子类
- [x] **D5 不变式**：`test_wire_code_never_leaves_the_catalogued_class` +
      `test_unrefined_wire_codes_map_to_exactly_the_catalogued_class`
- [x] `test_errors.py` 已按新继承关系改写（503/502 两处旧断言已更新），
      §5 表中 **errors 层**每条变更都有对应用例；
      `rl 失败` 一行由 P2 的 8 个 `test_rl.py` 用例覆盖，
      仅 `agentstudio` 一行按 P2b 延后
- [x] crosswalk 完整性：`test_crosswalk_covers_every_declared_wire_code`
      断言 `set(_WIRE_TO_PUBLIC) == {m.value for m in ErrorCode}`（13 项）
- [x] pylint（pinned disable + 本地 `R0917`）：`dashscope/v2` 整棵源码树
      = **10.00/10，零告警**
- [x] pylint：`tests/v2` 整棵树同样 **10.00/10**，但**需要把仓库根放进
      `sys.path`**：`python3 -m pylint --disable=<pinned>,R0917
      --init-hook='import sys; sys.path.insert(0, ".")' tests/v2`。
      不加 `--init-hook` 时本机会报 **66 处 `E0611`（`No name 'v2' in
      module 'dashscope'`）+ 1 处 `E1102`（`dashscope.Client is not
      callable`）**——`tests/` 下没有 `__init__.py`，pylint 把 `tests/v2`
      而不是仓库根插入 `sys.path`，astroid 于是解析到本机 user
      site-packages 里装的 **dashscope 1.26.3**（该版本没有 `v2` 子包），
      而不是仓库里的这份。属本机环境污染，不是仓库缺陷，也与本次改写无关
      （未改动的 `test_transport.py` 同样如此）。lint `dashscope/v2` 时
      pylint 会沿 `dashscope/__init__.py` 上溯到仓库根，所以源码树不受影响
- [x] flake8：`dashscope/v2` + `tests/v2` **零告警**。原先记录的两处既有
      E501 已修（`_config.py:54` 的 `WORKSPACE_WS_GATEWAY_TEMPLATE` 加括号
      换行、`test_protocols.py:23` 的尾注释提到上一行），两处都不改值。
      全仓（排除 `tests/legacy`、`samples`）只剩 2 处 `F824`，都在
      `tests/unit/test_multimodal_dialog.py`——1.x 测试代码，不在本次改动范围
- [x] black `--line-length=79 --check`：P2 的 4 个改动文件
      （`errors.py` / `reinforcement_learning.py` / `test_errors.py` /
      `test_rl.py`）**clean**；
      全仓仍有 7 个文件是 25.x drift（含 `_error_registry.py` 的
      docstring 后空行），仓库 pin 23.3.0，**不采纳**
- [x] `_error_registry.py` 落地后**未被手改**：`git hash-object` 仍为 D1 记录的
      `908fbf1…`，§8 的改名也没动内容（`git mv` 只改路径，相似度 100%）。
      注意这个 hash 钉的是"P0 落地后没人再动过目录内容"，**不是**"等于上游原文"——
      vendored 这份带 5 处本地适配（D1），本来就不等于原文。
      provenance 的比对基准是上游 registry 仓库 1.0.6 的
      `python/dashscope_error_registry/error_registry.py`，
      **不是** SDK 仓库的 `upstream/dev/errors`，那份是过时的 1.0.3
- [x] 1.x 代码零改动：`git status --porcelain` 只有 `dashscope/v2/*`、
      `tests/v2/*`、`md/*`
- [x] V2 接口面是**纯超集**：对 `errors.py` 做 AST 模块级名字 diff（vs `HEAD`），
      `removed: none`，`added:` 5 个公开名（`GatewayTimeoutError`
      `RequestTooLargeError` `ServiceUnavailableError` `SdkError` 四个类
      + `error_from_sdk_exception` 一个函数）+ 8 个下划线内部名；
      资源树 / 方法名 / 模型名 / `ResponseBase` 未碰
- [x] `SdkError` 相关用例：`test_errors.py` 5 个（`…has_no_http_semantics`
      钉 `status_code is None`、`…resolves_the_catalog_by_class_name`、
      `…keeps_the_original_message`、`…falls_back_to_the_family_entry`、
      `…needs_a_family_fallback` 钉 `sdk.agentstudio` 缺家族兜底会 `KeyError`）
      + `test_rl.py` 8 个（`…translates_1x_failure` 钉 `code`/`__cause__`、
      `…catchable_as_dashscope_error`、`…carries_its_own_solution` 钉
      `solution`/`allow_retry`、`…falls_back_to_the_family_entry`、
      `test_non_rl_exceptions_pass_through` 反向钉"非 rl 异常不翻译"、
      以及 `test_functions` 与 sync/async 两条接线）
- [x] **P4 文档同步已验证**：三份文档的交叉链接目标全部存在
      （design / registry-plan / interfaces-examples / interfaces-mapping /
      naming-review / m0-1x-cli-hotfix）；章节编号无冲突（design 新增 §2.7 后
      §3 顺延不变、naming-review §6.1–6.5、examples §0–9）；文档里写下的每条
      行为都**跑过真实实现**（mock transport），不是读代码推断出来的
- [x] **接口级覆盖审计（P5/P6 收口后做的全 v2 复审）**

      **方法**：AST 扫 `dashscope/v2/resources/**` 每个资源类的每个公开方法，
      再**手工解开委托链**。只数 raise 点是不够的，有两个盲区：
      ① 多数方法体内并不直接调用目录构造器，而是委托给 `_jobs` /
      `_checked_completion` / `WsSession`，按 raise 点统计会把它们误判成未接；
      ② **透传型接口在 v2 代码里根本不 `raise`**（`agentstudio` 就是），
      grep 全库也扫不到——所以必须从 `client.py` 的公开面反向逐接口核。

      已解开的委托链：`multimodal.chat` → 复用 `chat.completions` 的
      `_send`/`_asend`/`_build_request`/`ChatStream`；`chat.completions` →
      `_checked_completion` → `response_error`；`images`/`videos`/
      `embeddings`/`asr`(离线转写) 的 Jobs → `_jobs`；`rl` 的 Jobs →
      `fine_tunes`/`_jobs`，1.x 挂载调用 → `_translate_rl_errors`；
      `audio.tts` + `audio.asr`(实时识别) → `WsSession._check_failed`（P6）。

      | v2 接口 | 错误路径 | 状态 |
      | --- | --- | --- |
      | `chat.completions` | `_checked_completion` → `response_error` | ✅ |
      | `multimodal.chat` | 复用 completions 的 `_send`/`ChatStream` | ✅ |
      | `multimodal.embeddings` | `check_response` | ✅ |
      | `embeddings` / `images` / `videos` | `check_response`；作业委托 `_jobs` | ✅ |
      | `rerank` / `tokenization` / `files` / `deployments` | `check_response` | ✅ |
      | `models` | `check_response`；`get` 另走 crosswalk（P5） | ✅ |
      | `apps` | `check_response` ×2 + `error_from_business_code` | ✅ |
      | `fine_tunes` | `check_response` ×6 + `_stream_error`（P5） | ✅＊ |
      | `audio.tts` | `WsSession._check_failed` → crosswalk（P6） | ✅ |
      | `audio.asr` | 离线走 `_jobs`，实时走 `WsSession` | ✅ |
      | `audio.realtime` | `error_from_business_code`（P5） | ✅ |
      | `rl` / `reinforcement_learning` | `_translate_rl_errors` + `_jobs` | ✅ |
      | **`agentstudio`** | **返回 1.x 客户端对象本身，异常原样透传** | ❌ P2b |

      **结论**：15 个 resource 里 **14 个的全部公开接口都在目录上**；
      凡"服务端报了失败"（有 wire `code` 或 HTTP 状态）的路径，
      **100% 经 crosswalk 选类**，"报了 code 却写死类"这个缺陷类已清零。
      唯一未接的是 `agentstudio`（既定延后，见 P2b 的同名类危害）。

      ＊两处"接了一半"，按口径分开记：
      ① **8 处 wait 超时/取消**（`_jobs.py:210,230,300,320`、
      `fine_tunes.py:370,387,554,571`）抛的是 v2 `DashScopeError`，
      `except DashScopeError` **接得到**，但没有目录 `code`/`solution`——
      阻塞于上游 `SDK_ERRORS` 缺 jobs/fine-tune 家族（见 P5）。
      ② **`ws.py:186`/`:205` 的 `json.loads`** 让裸 `JSONDecodeError` 逃出去，
      `except DashScopeError` **接不到**，而 `sse.py:45-51` 的对等场景是包好的
      （见 P6 末，待决）。

---

## 7. 决策项（已按最小范围默认采纳，可推翻）

这 4 项**都不是兼容性问题**，前提 4 不解决它们。默认全部采纳"最小范围"选项，
因为它们都符合前提 2（只修 errors 相关）；若要推翻，只影响 P3，P0–P2 不受影响。

1. **P3 是否纳入本次？默认：不纳入。** `anthropic_error_code` 目前无消费者，
   接线意味着 anthropic 兼容协议要开始输出错误体（新增行为），
   范围会外溢到协议层。单独排期。

   已核实（比"没人用"更硬的一条）：**根本没有可实现的对端契约**。
   `protocols/anthropic.py` 只 import `BadRequestError, DashScopeError,
   ErrorCode`，两处抛出（`:187-191`、`:204`）都只用于**请求侧校验**，
   全文不产出任何错误响应体；design doc 也没规定 anthropic 形状错误体
   的字段与嵌套。所以接线不是"补一个消费者"，而是**先发明一份协议契约**
   ——那必须先和 anthropic 兼容协议的定义方对齐，不能由 errors 改写顺手决定。
2. **`allow_retry` 是否接重试？默认：不接。** `_retry.py` 是 HTTP 状态层，
   `allow_retry` 是 sdk.* 本地错误层，两层语义不同。
   仅作为 `SdkError` 的只读属性暴露，由用户自行判断。
3. **`SdkError` 是否按 `sdk.agentic_rl.*` / `sdk.agentstudio.*` 再分子类？默认：不分。**
   一个类 + `code` 字段足够；分子类会把目录的 31 条变成 31 个类，
   与"目录驱动、类名稳定"的初衷相反。
4. **`format_msg` 占位符谁来填？默认：body 的 `message` 优先。** ✅ 已实施
   目录文案带 `{parameter}` / `{resource}` / `{limit}`，但网关下发时通常已带
   `message`，V2 现有逻辑就是 body 优先（`errors.py:311`
   `message = data.get("message") or message`，业务层同理见 `:398`）。
   占位符只在 body 无 message 时兜底，且填不出变量就原样保留占位
   （**不要静默丢字段**）。

   实现为 `errors.py` 的 `_catalog_message(code, status_code)`，两处构造
   （`error_from_status` / `error_from_business_code`）都写成
   `message or _catalog_message(...)`。**条目识别顺序**：先按 wire code 查
   `_WIRE_TO_PUBLIC`；查不到才按状态码查 `_STATUS_ENTRY`，而 `_STATUS_ENTRY`
   **只收录恰有一条条目声明的状态**（403/404/413/500/503/504）。
   400/401/429 各有 2–5 条同状态条目、文案互不相同，只凭状态码挑一条
   等于替网关编一个它没说过的原因，因此这三种状态在没有 wire code 时
   **不兜底**（保持空 message）。同理，目录 key（如 `missing_parameter`）
   不是 wire code，`_CODE_MAP` 认不出它，兜底也认不出——
   **兜底的识别能力与选类严格一致，不会出现"类选对了、文案是别人的"**。

   实际触发面：`response_error()` 对 4xx/5xx 传 `message=response.text`
   （`errors.py:390`），所以**空 body 的响应**（负载均衡/网关直接回 503
   之类）会走到兜底；带 body 时 `response.text` 非空，兜底不触发。
   业务层经 `response_error()` 进来的 message 恒非空，兜底只对直接调用
   公开 API 的人生效（仓内 `_jobs.py` / `apps.py` / `fine_tunes.py`
   的调用点都自带非空 message）。
   **SSE 有意不接**：`transport/sse.py:64,71` 传的是
   `body.get("message") or "stream error"`，字面量恒真，流内错误继续用
   transport 自己的文案，不改。

---

## 8. 附：`error_registry.py` 的位置 ✅ 已完成（独立 commit）

只能放 `dashscope/v2/`：它不 import 任何 1.x 内容，1.x 有独立错误体系，
放 `dashscope/` 根会暗示两边共用。

但原位置有一处约定不一致：v2 根下内部模块一律带 `_` 前缀
（`_config.py` / `_logging.py` / `_quickstart.py`），不带前缀的是公开面
（`client.py` / `errors.py`）。`error_registry.py` 无前缀，却也没从
`dashscope/v2/__init__.py` 导出（只导出 `errors` / `DashScopeError` / `ErrorCode`），
实际只被 `errors.py` 和测试引用——**读起来像公开 API，其实是内部数据**。

已按建议改名 `_error_registry.py`，与 errors 改写分成两个 commit。改动面：
`git mv` + **4 处 import**（`errors.py` 两处、`test_errors.py` 一处、
`test_error_registry.py` 一处）+ 三处 docstring 路径引用 + 四份 md 文档的
路径引用（design §2.7 / 目录树 / 里程碑 1.3、naming-review §6.5、
interfaces-examples §9）。测试文件名 `test_error_registry.py` **不改**：
pytest 的 `test_*.py` 发现规则不要求与被测模块名前缀一致。

**代价**：改名本身成了第 5 处本地适配（见 D1），整份替换升级时要一起重打。
若将来要让用户查错误码，走 `errors.ErrorCode` 即可，不需要公开目录本身。
