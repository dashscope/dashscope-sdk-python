# DashScope v2 借鉴 Ark 重构任务列表

> 目标仓库：`/Users/zhansheng.lzs/ali/pro/ptm/moe-dashscope-sdk-python`（`dashscope/v2/`）
> 依据文档：`ark-vs-dashscope-v2-comparison.md`
> 拆分原则：**一个任务一个 PR**，均可独立合并；除标注的依赖外无隐式耦合。
> 全局约束：全部改动保持向后兼容（只新增 kwarg / 子类 / 模块，不改既有签名语义）；测试放 `tests/v2/`，风格沿用现有 `httpx.MockTransport` 注入。

## 任务总览

| # | 任务 | 优先级 | 依赖 | 工作量 | 影响面 |
|---|---|---|---|---|---|
| T1 | `RequestOptions` 统一请求定制通道 | ★★★ | 无 | M | transport + 全部 resources |
| T2 | 容错式模型构造 `parse_model()` | ★★★ | T3 | M | types + 全部 resources |
| T3 | `APIResponseValidationError` 异常类 | ★★ | 无 | S | errors.py |
| T4 | 分页抽象 `BasePage` | ★★ | 无 | M | pagination.py + 3 个 list 资源 |
| T5 | 响应封装 `APIResponse.parse(to=)` | ★★ | T1、T2 | L | transport + resources 入口 |
| T6 | 每请求 `x-dashscope-client-request-id` | ★ | 无 | S | _ident.py + http.py |
| T7 | async 孪生漂移治理（unasync） | ★ | 无 | L（分阶段） | 构建/CI |

推荐执行顺序：**T3 → T6 → T1 → T2 → T4 → T5 → T7**（先小后大，T5 依赖 T1/T2，T7 独立排最后）。

---

## T1：`RequestOptions` 统一请求定制通道

**目标**：借鉴 ark `_request_options.py:44`，给每个资源方法一致的逃生舱 `extra_headers / extra_query / extra_body`，并支持按请求覆盖 `max_retries`。

**现状痛点**：
- `SyncTransport.request()` 只接受 `headers/params/timeout/api_key/workspace`（`transport/http.py:156-172`），无 extra 通道；网关要加一个实验性 header 必须改 SDK 发版。
- `max_retries` 只有 client 级（`http.py:149`），无法按请求关闭（如长任务提交不想重试）。

**改动范围**：
1. 新增 `dashscope/v2/_request_options.py`：
   - `RequestOptions` dataclass：`extra_headers: Mapping[str,str]`、`extra_query: Mapping[str,Any]`、`extra_body: Mapping[str,Any]`、`timeout`、`max_retries: Optional[int]`。
   - 深合并函数 `_deep_merge(base: dict, extra: dict)`（对齐 ark `_base_client.py:236-243` 语义：dict 递归合并，其余覆盖）。
2. `transport/http.py`：
   - `SyncTransport.request()` / `AsyncTransport.request()` / `open_stream()` 增加 `options: Optional[RequestOptions] = None`。
   - header 合并顺序固定为：`sdk_headers(module)` < auth（api_key/workspace）< 调用方 `headers` < `extra_headers`（后者胜）。
   - `extra_query` 并入 `params`；`extra_body` 与 `json` dict 深合并（`json` 非 dict 时忽略并告警）。
   - 重试循环用 `options.max_retries if options and options.max_retries is not None else self._max_retries`（`http.py:174`、`http.py:223` 两处）。
3. resources：方法签名统一追加 `extra_headers=None, extra_query=None, extra_body=None`（保留既有 `timeout/api_key/workspace` kwarg 不动）；内部组装 `RequestOptions` 传给 transport。从 `chat/completions.py`、`embeddings.py`、`models.py`、`files.py`、`_jobs.py` 做起，其余资源后续跟进（可在 PR 内一次改完，模式相同）。

**验收标准**：
- `tests/v2/test_request_options.py`：
  - extra_headers 出现在最终请求头且覆盖默认值；
  - extra_query 并入 query string；extra_body 深合并（嵌套 dict 合并、标量覆盖）；
  - `max_retries=0` 时 503 不重试；`max_retries=1` 时重试一次；不设置时走 client 默认；
  - 优先级冲突用例：调用方 headers 与 extra_headers 同键时 extra_headers 胜。

**风险**：低。纯新增参数；合并顺序需在 docstring 写死，避免后续歧义。

---

## T2：容错式模型构造 `parse_model()`

**目标**：借鉴 ark `_models.py` 的 `construct_type` 哲学——服务端新增枚举值、判别字段新值、结构小漂移时不炸用户进程。

**现状痛点**：resources 直接 `Model(**data)` 构造（如 `models.py:67-73`），pydantic v2 校验失败会把原始 `ValidationError` 抛给用户；`extra="allow"` 只兜住了"多字段"，兜不住"字段值类型漂移"和"Literal/判别新值"（`types/messages.py:30-55` 的 `extra="forbid"` + Literal 判别最脆）。

**改动范围**：
1. 新增 `dashscope/v2/types/_parse.py`：
   ```python
   def parse_model(cls: type[_M], data: Mapping[str, Any], *, strict: bool = False) -> _M
   ```
   - 默认路径：`model_validate(data)`；失败时降级为 `cls.model_construct()` + 逐字段 best-effort 赋值（能 validate 的字段 validate，不能的原样塞入），并 `logger.debug` 记录漂移详情（模型名、字段、原始值类型）。
   - `strict=True`：校验失败抛 T3 的 `APIResponseValidationError`（保留 response body 供排障）。
   - 不移植 ark 的 700 行递归强转器，也不做 v1 双栈——pydantic v2 的 `model_construct` 已提供 90% 价值。
2. 替换 resources 中的裸构造调用（`ModelList(**...)`、`Model(**models[0], ...)` 等，典型位置 `resources/models.py:67-73,157-161`、`resources/files.py`、`resources/_jobs.py`、`resources/chat/completions.py` 的 chunk 解析路径）。
3. 配置项：`Config` / `Settings` 增加 `strict_validation: bool = False`（env `DASHSCOPE_STRICT_VALIDATION`），让用户可切回严格模式排障。

**验收标准**：
- `tests/v2/test_parse_model.py`：
  - 未知枚举值、未知 message part `type`、字段类型漂移（int 给了 str）三种场景下默认不抛错、字段可读取、其余字段正常校验；
  - `strict=True` 或 `strict_validation` 开启时抛 `APIResponseValidationError`，异常带 body；
  - 现有 `tests/v2/` 全部用例不回归。

**风险**：中。容错会掩盖真实的服务端回归——必须有 debug 日志 + strict 开关双保险；CHANGELOG 明确行为变化。

---

## T3：`APIResponseValidationError` 异常类

**目标**：把"SDK 解析失败"与"API 调用失败"分成两类，借鉴 ark `_exceptions.py` 的 `ArkAPIResponseValidationError`。

**改动范围**：
1. `errors.py` 新增：
   ```python
   class APIResponseValidationError(DashScopeError):
       """Response body failed SDK-side validation (strict mode)."""
   ```
   携带 `response`（httpx.Response 或 None）、`body`、`cause`（原始 `ValidationError`）。
2. 评估是否在 `_error_registry.py` 的 `SDK_ERRORS` 注册条目（`sdk.response_validation_error`，带 solution 文案）——与错误目录的"SDK 侧错误"约定对齐；若注册则改为继承 `SdkError`。
3. `errors.py` 顶部 docstring 的"third source"段落补充该类说明。

**验收标准**：`tests/v2/test_errors.py` 新增：异常可捕获、`str()` 含 request_id/body 摘要、`isinstance(err, DashScopeError)` 成立、（若注册目录）目录不变量测试通过。

**风险**：低。新增类，无既有行为变化。

---

## T4：分页抽象 `BasePage`

**目标**：借鉴 ark `pagination.py`，给 list 接口统一的可迭代分页对象。

**现状痛点**：`models.list` 返回裸 `ModelList`（`models.py:48-52`，`page_no/page_size/total_count`），翻页要用户手工循环；`files.list`、`jobs.list`（`_jobs.py:182-207`）同构但各写各的。

**改动范围**：
1. 新增 `dashscope/v2/pagination.py`：
   - `BasePage(Generic[T], ResponseBase)`：字段 `page_no/page_size/total_count`；方法 `has_next_page`（`page_no * page_size < total_count`）、`next_page()`（回调拉下一页，末页返回 None）、`__iter__`（委托 `self.items`）、`iter_pages()`（生成器跨页）。
   - `SyncPage` / `AsyncPage` 两个实现，构造时注入"取页回调"（`Callable[[int, int], BasePage]`），回调由资源方法闭包提供。
2. 资源接入（保持返回类型向后兼容——让现有 List 模型继承/替换为 Page 子类）：
   - `models.list` → `ModelList(BasePage[Model])`；
   - `files.list` → `FileList` 同理；
   - `_jobs.py` 的 `JobList` 同理（注意 `UNKNOWN` 终态语义不受影响）。
3. `provider` 客户端过滤（`models.py:122-125`）与分页的交互写进 docstring：过滤只作用于当前页，`total_count` 语义不变。

**验收标准**：
- `tests/v2/test_pagination.py`：`for m in page` 迭代当前页；`iter_pages()` 跨 3 页聚合且请求参数 page_no 递增；`total_count` 边界（整除、不足一页、空页）正确停止；`next_page()` 末页返回 None。

**风险**：低-中。`ModelList` 改继承关系是公开 API 变化但向后兼容（字段全保留）；注意 `model_dump()` 序列化结果不要多出内部字段。

---

## T5：响应封装 `APIResponse.parse(to=)` 与 raw 访问

**目标**：借鉴 ark `_response.py`，收敛目前散装的 `check_response` + 手工解析 + `with_headers` 三件套，并提供记忆化的按类型解析。

**现状痛点**：每个资源方法重复"request → check_response → response.json() → 构造 → with_headers"五步（`models.py:111-126` 是典型），共十几处；用户拿不到原始 response（无法读非标准响应头、无法自行二次解析）。

**改动范围**：
1. 新增 `dashscope/v2/_response.py`：
   - `APIResponse`：包装 `httpx.Response` + client 引用；属性 `headers / status_code / request_id / elapsed`；方法 `parse(cast_to=None)`——`cast_to` 为 None 时返回 `response.json()`，否则走 T2 的 `parse_model`，解析结果按 `cast_to` 缓存在 `_parsed_by_type: dict`（对齐 ark `_response.py:305-321` 的记忆化语义）。
   - 同步/异步共用一个类（httpx.Response 在 async 路径同样是已物化的对象，无需孪生）。
2. transport 新增 `request_api()` 返回 `APIResponse`（内部就是 `request()` + 包装，不重写重试逻辑）；`open_stream()` 的 `Stream` 对象补 `headers` 属性（`transport/sse.py` 已有 `Stream.headers`，对齐即可）。
3. resources 改造分两小步（可同 PR 内两个 commit）：
   - a) 内部路径切到 `request_api()` + `resp.parse(Model)`，删除重复的 `with_headers` 调用（`parse_model` 内统一挂 headers）；
   - b) 资源方法增加 `raw: bool = False` kwarg（或 `with_raw_response` 风格），为 True 时返回 `APIResponse` 而非模型。
4. `with_headers`（`types/common.py:100`）保留为公开工具不删（websocket 来源模型仍在用），docstring 标注 HTTP 路径已内建。

**验收标准**：
- `tests/v2/test_response.py`：`parse()` 两次调用返回同一缓存对象；`parse(OtherModel)` 按类型分别缓存；`request_id` 从 `x-request-id` 头提取；`raw=True` 时资源方法返回 `APIResponse` 且不抛 `check_response` 错误。
- `tests/v2/` 存量用例不回归（重点：headers 字段在异常与成功路径行为不变，`types/common.py:100-116` 文档承诺的语义保持）。

**风险**：中。触碰所有资源的公共路径；建议先只切 `models` + `embeddings` 两个资源验证模式，再全量推广（可拆 T5a/T5b 两个 PR）。

---

## T6：每请求 `x-dashscope-client-request-id`

**目标**：借鉴 ark `_base_client.py:195` 的 `X-Client-Request-Id`，给每次请求一个客户端生成的 UUID，与服务端 `x-request-id` 互补（网络层失败、请求未到达网关时也有 ID 可查）。

**改动范围**：
1. `transport/_ident.py`：新增 `CLIENT_REQUEST_ID_HEADER = "x-dashscope-client-request-id"` 与 `client_request_id() -> str`（`uuid.uuid4().hex`）；同样受 `DASHSCOPE_DISABLE_SDK_HEADERS` 开关控制。
2. `transport/http.py:184` 附近：在每次 `self._client.request()` 调用时生成并注入（**注意放重试循环内每次 attempt 生成一次，或循环外一次**——建议循环外一次：重试属于同一逻辑请求，与服务端 x-request-id 的关联更稳定；决策写进注释）。
3. debug 日志（`http.py:214`）并排输出 client_request_id 与 x-request-id。
4. 错误对象：连接层错误（`APIConnectionError`/`APITimeoutError`）无服务端 request_id 时，把 client_request_id 填入 `err.request_id` 便于排障（`http.py:199` 的 `_map_transport_error` 路径）。

**验收标准**：`tests/v2/test_ident.py`：header 存在且为 32 位 hex；同客户端两次请求 ID 不同；`DASHSCOPE_DISABLE_SDK_HEADERS` 设置后 header 消失；连接异常时 `err.request_id` 非空。

**风险**：低。网关在服务网格中可能剥离未知 header——不影响功能（纯客户端侧排障增强），但先与网关同学确认该 header 名未被占用。

---

## T7：async 孪生漂移治理（分阶段）

**目标**：治理 sync/async 人肉复制的漂移风险（`client.py:51 vs 272`、`http.py:128 vs 320`、`_jobs.py`、`resources/*` 全套孪生）。ark 的教训是：机器复制至少保证一致，人肉复制必然漂移。**不上 Stainless**，用轻量方案。

**阶段一（本任务内必做）：漂移检测**
1. 新增 `tests/v2/test_sync_async_parity.py`：对指定的成对模块（`client` / `transport.http` / `_jobs` / `resources.models` 等），用 `ast` 解析后做归一化比较（剔除 `async/await` 关键字、`Async` 类名前缀、`__aenter__/__aexit__`），方法集合与参数列表必须一致。
2. CI 挂为必过项——先止血，保证新增方法两端同步。

**阶段二（试点）：unasync 生成**
1. 评估并引入 [`unasync`](https://github.com/python-trio/unasync)（或等价脚本）：以 sync 代码为源，token 替换规则生成 async（`AsyncModels`→`Models` 逆规则、`await `→``、`async def`→`def`）。
2. 选 1 个简单资源试点（建议 `models.py`：无流式、无 WS，孪生结构最规整，`models.py:91-228`）；生成物不手写、不进 review diff（pre-commit/CI 生成并校验一致）。
3. 试点稳定后推广到 `files.py`、`_jobs.py` 等。

**不在范围内**：`transport/ws.py` / `ws_connection.py`（纯同步线程模型，无 async 孪生）；`open_stream()`（SSE 的 async 生成规则涉及 `async with`/`__aiter__`，试点后再定规则）。

**验收标准**：
- 阶段一：parity 测试能捕获人为制造的漂移（删一个 async 方法，测试红）。
- 阶段二：试点模块的 async 文件头标注 "GENERATED — DO NOT EDIT"，CI 校验生成物与 sync 源一致。

**风险**：中-高。触及构建流程；阶段一零风险先行，阶段二试点失败可回退，不影响已合并的阶段一。

---

## 附：明确不做的事

| 事项 | 原因 |
|---|---|
| 引入 Stainless 全量代码生成 | 会冲掉 protocols/Job/WebSocket/OSS 直传等手写差异化能力；ark 自己的生成代码也存在双客户端复制、双序列化约定问题 |
| pydantic v1 兼容层 | v2 只支持 pydantic v2，ark 的 v1 路径已是死代码 |
| `_legacy_response.py` 式新旧双实现长期并存 | ark 的前车之鉴（1000+ 行历史包袱）；T5 要求一步切干净 |
| POST 重试幂等区分 | ark 同样没做（`idempotency_key` 是死字段）；需网关先支持幂等键，单独立项 |
| mypy/pyright 类型门禁 | 属于两个 SDK 的共同短板，不在"借鉴 ark"范围；如需治理另立任务 |
