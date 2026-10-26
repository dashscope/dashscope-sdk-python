# dashscope 2.0 M1–M3 深度 Review

> 评审对象：`V2` 分支 `dashscope/v2/`（HEAD = 0c58a74），对照 [`dashscope-2.0-design.md`](dashscope-2.0-design.md) 的 M1/M2/M3 任务表与 P1–P6 原则。
> 评审方法：逐文件通读 v2 全部源码（约 6000 行）+ 对照设计文档逐任务核对 + 运行测试套件（**254 passed / 3 skipped**，skipped 为 env-gated 的 live 集成测试）+ git 考古验证零接触承诺。
> 评审日期：2026-08-28。
>
> **命名已过时**：本文引用的方法名（`apps.call`、`embeddings.create_multimodal`、`asr.recognition`、`asr.transcription.transcribe` 等）在随后的命名统一中已改名，现行名称与设计约束见 [`dashscope-2.0-naming-review.md`](dashscope-2.0-naming-review.md) 与 [`dashscope-2.0-design.md`](dashscope-2.0-design.md) §2.6。本文结论按当时状态保留，不作回填。

---

## 0. 总体结论

**M1–M3 的功能骨架已按设计落地，资源树 100% 映射（除计划下线的 Assistants），测试覆盖扎实；但设计的最高优先级原则 P2（契约正确性优先）仍有三处未闭合的"静默错误"缺口，另有若干健壮性问题集中在重试、流式聚合与兼容开关路径上。**

| 维度 | 评价 |
| --- | --- |
| 功能实现完整度 | ★★★★☆ 任务表 21/21 项有形交付；agentstudio 的 errors/HTTP 对齐例外条款未兑现，`with_options()` 缺失，AsyncAudio 面不全 |
| 代码健壮性 | ★★★☆☆ 主干路径稳；3 处 P2 级静默错误 + 重试/聚合/兼容开关路径若干中危问题 |
| 架构与理念一致性 | ★★★★☆ 分层/隔离/资源树严格兑现；P6 codegen 进度偏弱，1 行 1.x 触碰超出白名单 |

---

## 1. 功能实现完整度（按任务表逐项核对）

### M1 — SDK 2.0 骨架（1.1–1.10）

| # | 任务 | 状态 | 证据与备注 |
| --- | --- | --- | --- |
| 1.1 | Client/AsyncClient + 配置解析 | ✅ 有偏差 | `client.py`、`v2/_config.py` 齐备。**偏差**：`resolve_config` 中 `protocol/timeout/max_retries` 三个键的 env 槽位硬编码为 `None`（`_config.py:220-259`），只有 `api_key/base_url/workspace` 读环境变量——与 3.11 声明的优先级链「显式 > settings > env > config.toml」不一致（env 环断了三键）。 |
| 1.2 | transport（httpx 双栈 + 重试） | ✅ | `transport/http.py`：Retry-After 解析、指数退避 + 抖动、`x-should-retry` 头均在。健壮性问题见 §2-①。 |
| 1.3 | 异常体系 + ErrorCode + 双层检查 + 兼容开关 | ✅ | `errors.py`：`ErrorCode` 枚举、`_STATUS_MAP`/`_CODE_MAP`、HTTP-200-业务失败双层检查、`raise_on_error` 开关全部落地。 |
| 1.4 | 日志改造 | ✅ | `_logging.py`：request_id/状态码/重试决策入日志，`SensitiveHeadersFilter` 同时挂 handler 与子 logger（22fc5d8 修复），`DASHSCOPE_LOG` 一键开启。 |
| 1.5 | 内部规范消息模型 | ✅ | `types/messages.py`：归一化 + 边界加固（dcb8b33）。 |
| 1.6 | client.chat | ✅ | Pydantic 响应、`@overload` 区分流式/非流式、`output_text`、`text_stream`/`get_final_message`、`reasoning_content` 全部在。聚合缺陷见 §2-③。 |
| 1.7 | embeddings / rerank / tokenization | ✅ | 三资源齐备；tokenization 的 prompt→messages 合并与 rerank instruct 参数经 909110d 修正。 |
| 1.8 | files / models | ✅ | 齐备（69f1711 修正 wire 协议）。`provider` 过滤为客户端实现，见 §2-⑦。 |
| 1.9 | 包根导出 | ✅ | `dashscope/__init__.py:84` 导出 `Client/AsyncClient/chat/settings`；附 705 行手写 `__init__.pyi`。 |
| 1.10 | specs 元数据原型 | ⚠️ 偏弱 | `specs/chat.yaml` 存在，但 M1 之后无新增——与 P6「specs 从 M1 起同步建设」有差距，目前仍是单文件试点。 |

### M2 — 协议兼容层（2.1–2.6）

| # | 任务 | 状态 | 证据与备注 |
| --- | --- | --- | --- |
| 2.1 | ChatProtocol ABC + native | ✅ | `protocols/base.py` + `native.py`，接口收敛干净。 |
| 2.2 | openai adapter | ✅ | `openai.py`：strict 开关与透传（e786de8）、`reasoning_content` 回显（9aba4cd）、`stream_options.include_usage`。小问题见 §2-⑪。 |
| 2.3 | anthropic adapter | ✅ | `anthropic.py`（449 行）：per-stream 状态隔离（bf6cbab）、system 提取、tool_use/tool_result 合并、默认 max_tokens。8.27 修复后是三 adapter 中最扎实的。 |
| 2.4 | SSE 统一解析 | ✅ | `transport/sse.py`：流内错误抛异常、request_id 透传、畸形帧包装（1a2daef），另有 435 行专项验证套件。 |
| 2.5 | acli providers 收敛 | ✅ | 严格遵守「不改同步副本」：协议翻译只在 `v2/protocols/`，`dashscope/acli/providers/` 零改动。 |
| 2.6 | 兼容端点集成测试 | ⚠️ 形式达标 | `tests/v2/protocols/test_integration_endpoints.py` 存在（百炼 compatible-mode + 通用网关 3 例），但全部 env-gated——CI 默认 skip，「兼容端点可直连」无持续回归保障。建议至少在发版流水线中跑一次真实 key。 |

### M3 — SDK 全量覆盖（3.1–3.11）

| # | 任务 | 状态 | 证据与备注 |
| --- | --- | --- | --- |
| 3.1 | multimodal.chat | ✅ | content parts 全链路；与 chat 的复制问题见 §3-P6。 |
| 3.2 | Job 基建 | ✅ | `_jobs.py`：`Job` 统一对象 + 退避轮询（1s→8s 封顶）。`DEFAULT_WAIT_TIMEOUT=86400` 与 1.x 的语义差异见 §2-⑫。 |
| 3.3 | images / videos | ✅ | `generate` 同步封装 + `jobs.create/fetch/wait`，模式统一。 |
| 3.4 | audio.asr | ✅ | 批转写走 Job 基建；实时 Recognition 为 ws 上下文管理器，fake connector 可测。 |
| 3.5 | audio.tts + audio.realtime | ⚠️ 有缺口 | 同步面齐备（synthesize/stream + omni session），fe03bc6 已修 realtime 吞传输错误。**缺口**：`AsyncAudio` 只挂了 `asr.transcription`（`audio/__init__.py:55-57`）——异步用户没有 `tts`、`realtime`、`asr.recognition`。ws 实时接口以同步为主可辩护，但资源树 sync/async 不对称且文档未声明，属未解释的缺口。 |
| 3.6 | fine_tunes / deployments | ✅ | 完整 CRUD + `stream_events` + `wait`（sleep 可注入，测试友好）。 |
| 3.7 | rl 薄挂载 | ✅ | 复用 1.x `agentic_rl`，job 管理（get/list/cancel/logs）走 v2 transport + 错误模型，`client.reinforcement_learning` 别名在。符合「业务逻辑不动」的约束。asyncio.run 风险见 §2-⑥。 |
| 3.8 | apps + agentstudio 挂载 | ⚠️ 例外未兑现 | apps 完整（含流式 + compat 错误事件）。**agentstudio 未兑现设计例外条款**：3.8 明确要求「errors 与底层 HTTP 调用对齐 2.0」，但 `client.py:112-122` 的懒挂载只传了 `api_key/workspace`，1.x AgentStudio 的 HTTP 与错误体系原样保留——挂载层形同 pure pass-through。要么实现对齐，要么修订设计文档豁免此项。 |
| 3.9 | 映射核对表 + 全量冒烟 | ✅ | `dashscope-2.0-interfaces-mapping.md` 声明无遗留缺口（除 Assistants 下线）；`tests/v2/smoke/test_full_surface.py`（641 行）逐资源过一遍。 |
| 3.10 | embeddings 缺口补齐 | ✅ | `create_multimodal` + 批量 `jobs.*`（68c0e87）。 |
| 3.11 | settings 全局门面 | ✅ | 懒读 env、setter 即时生效、包根导出。线程安全与 clear() 语义见 §2-⑨。 |

**任务表外的小缺口**：设计 §2.2 提到「`Client(max_retries=N)` / `with_options()` 可调」——`with_options()`（派生修改配置的客户端副本，OpenAI/Anthropic SDK 均有）未实现。要么补上（很便宜：Config 是 frozen dataclass，`dataclasses.replace` 即可），要么从设计文档删去。

---

## 2. 代码健壮性（按严重度分级）

### 高危（P2 级——设计原则点名要消灭的"静默错误"）

**① 传输层错误逃逸 SDK 异常体系**
`transport/http.py:113-122`（同步）与 `:252-261`（异步）：重试耗尽后 `raise` 原样抛出裸 `httpx.TransportError`。而 `errors.py:68-72` 定义的 `APIConnectionError`/`APITimeoutError` **全库零引用**——定义了不用。后果：用户按文档 `except DashScopeError` 无法捕获连接错误/超时，异常驱动契约在最常见的失败路径上破防（OpenAI SDK 此处映射为 `APIConnectionError`）。
**修法**：重试耗尽与 `open_stream` 连接失败时映射进 `APIConnectionError`（超时映射 `APITimeoutError`），保留 `__cause__`。

**② chat 非流式在 `raise_on_error=False` 下静默吞错**
`chat/completions.py` 非流式路径：`check_response(response, raise_on_error=False)` 静默返回后，紧接着 `protocol.parse_completion(response.json())` 把错误体解析成一个**空的、看似成功的** ChatCompletion 返回。这正是 P2 点名的最高危类别在兼容开关下复发。对比 apps 资源 compat 模式至少产出 `is_error=True` 事件（f3805d8），chat 连标记都没有。
**修法**：compat 模式下检查响应体 `code` 字段，将其原样放进返回模型（1.x `GenerationResponse` 语义），或在 ChatCompletion 上补 `is_error/code/message`。

**③ `get_final_message()` 对 OpenAI 协议 tool_calls 聚合错误**
`chat/completions.py:144-145`：`_aggregate` 对 `tool_calls` 是「最后一个 chunk 覆盖」。native/anthropic 的流式 tool_calls 是累积式，碰巧正确；OpenAI 协议是非累积片段（按 index 分发、arguments 逐段追加），覆盖式聚合会丢工具、截断参数。
**修法**：按 `index` 归并、字符串字段追加（OpenAI 语义），或按协议分发聚合策略。

### 中危

**④ `files.upload` 重试会发空 body**
`files.py:136-143`：`with open(...) as f` 把活句柄塞进 multipart；transport 重试时句柄已消耗，第二次尝试发空文件（httpx 不会重读流式 body）。
**修法**：上传路径重试前 `f.seek(0)` 不可控——应读入 bytes 或对该请求禁用重试。

**⑤ 流内错误处理 chat 与 apps 行为分裂**
设计 §2.2：「流内错误统一抛异常，不 yield 错误对象」。chat/SSE 遵守（且 `raise_on_error=False` 也不影响，流内错误照样 raise）；apps 在 compat 模式 yield `is_error=True` 事件。两种语义并存且均无文档说明——用户换资源即踩坑。需要统一，或在设计文档中把 apps 列为显式例外。

**⑥ rl 同步方法用 `asyncio.run` 包 1.x 异步实现**
`reinforcement_learning.py:136/153/168/191`：在已有运行中 event loop 的环境（Jupyter、FastAPI、pytest-asyncio）直接 `RuntimeError`。1.x 用户群里有 notebook 场景，这是真实风险。
**修法**：短期文档标注「同步 rl 方法不可在运行中 loop 内调用」；中期给 1.x agentic_rl 补同步入口（属 M 外工作）。

**⑦ `models.list(provider=...)` 客户端过滤误导分页**
`models.py:90-93`：只过滤当前页结果，`total_count` 不调整——过滤后 `total_count=100` 但本页只有 3 条，分页消费者会被误导。服务端不支持 provider 过滤时，应至少在文档/docstring 中声明「total_count 为过滤前值」。

**⑧ `resolve_config` 三键不读环境变量**
同 1.1 偏差（`_config.py:220-259`）。`DASHSCOPE_PROTOCOL`/`DASHSCOPE_TIMEOUT` 之类的 env 不生效，运维侧无法用 env 统一切协议。

### 低危（记录即可）

- **⑨** `Settings` 用普通 dict 无锁（`_config.py:52` 起），并发赋值非线程安全；`log_level` setter 有副作用（调 `set_log_level`）但 `clear()` 不恢复 logger 级别——状态泄漏。
- **⑩** `transport/ws.py:180-183` `recv()` 仍 broad-except（finished 时吞异常返回 None）；8.27 修了 `recv_event` 与 realtime，`recv()` 的宽松行为是 TTS 路径在用，可接受但应加注释说明取舍。
- **⑪** openai adapter `parse_completion` 用 `payload["id"]`（completion id）充当 `request_id`（`openai.py:136/153`）——排障工单对不上，语义错位。
- **⑫** `DEFAULT_WAIT_TIMEOUT=86400` 与 1.x SDK `wait(timeout=-1)`（无限等待）不对齐；1.x CLI 是 3600。86400 是有意的有限等待折中（防挂死），方向正确，但应在迁移指南中明示。
- **⑬** `fine_tunes.py` 的 `_check_terminal` 在 sync/async 两个类中逐字重复（`:340-353` 与 `:520-533`）——sync/async 手写复制已开始产生实际重复，M6 codegen 之前的漂移风险现实存在。
- **⑭** `__init__.pyi` 705 行手写 stub 与实现双维护，codegen 缺位期间必然漂移。

---

## 3. 架构设计与理念一致性

### P1 对齐行业基线 —— ✅ 高

Client 实例化 + 资源树导航、httpx 双栈、Pydantic 响应、异常层级、自动重试、`output_text` 便捷属性、`dashscope.chat()` 2 行上手——OpenAI/Anthropic 验证过的基线全部落地，未自创风格。扣分项：`with_options()` 缺失；传输错误未纳入 SDK 异常体系（§2-①）——这两项恰是 Stainless 系 SDK 的标配。

### P2 契约正确性优先 —— ⚠️ 部分违反

设计的自我定位是「修复静默错误的优先级高于任何新能力」，且开发过程确实在践行：8.25–8.27 的修复提交链（1a2daef 流内错误带 request_id、e786de8 错误信封解析、bf6cbab 流状态隔离、f3805d8 apps compat 错误事件、fe03bc6 realtime 传输错误传播）全部是契约类修复。但 §2-①②③ 三处残留恰好都在 P2 点名的类别里（吞错、错误逃逸、聚合撒谎），**建议作为 M3 收尾项清零后再进 M4**。

### P3 渐进迁移 / 零接触隔离 —— ✅ 基本达标（1 处轻微越界）

git 考古（`git log master..HEAD -- 'dashscope/**' ':!dashscope/v2/**'`）确认 2.0 工作对 1.x 的触碰仅 2 处：

1. `dashscope/__init__.py` 追加导出（f68a311、e6142da）——设计 §2.4 规则 3 明确要求，合规。
2. `dashscope/finetune/reinforcement/common/constants.py` 的 `LOGGER_NAME` 从 `"dashscope"` 改为 `"dashscope.finetune.reinforcement"`（0c58a74）——动机正当（避免 1.x rl 的 logger 名与包根 logger 混淆），但规则 1 只允许安全补丁、规则 10 只允许 DeprecationWarning，这一行是 v2 动机驱动的 1.x 行为变更，**超出白名单**。影响极小（日志名变化），建议在设计文档白名单中补记豁免，不宜静默破例。

另注：V2 分支同时承载 1.x 常规迭代（v1.26.6–v1.27.2 发布、CLI 打包、Recognition 参数显式化等提交），2.0 与 1.x 发布共用分支——不违反隔离（文件层面无交叉），但发布节奏耦合值得注意。

### P4 保留自有优势 —— ✅ 高

长任务统一收敛为 `Job` + `jobs.create/fetch/wait`（images/videos/embeddings 批量/asr 转写四处复用同一基建），多模态 content parts 进规范消息模型，RL/微调/部署全流程完整迁入。有限化 wait 超时（86400）是增强而非削弱。无能力下降项。

### P5 差异化 CLI 定位 —— 本期范围外（M4 起）

### P6 codegen 是主线 —— ⚠️ 偏弱

`specs/` 目前只有 M1 的 `chat.yaml` 一个试点，M2/M3 期间无新增。与此同时手写复制面在扩大：sync/async × chat/multimodal 的 create/stream/aggregate 脚手架近乎重复（normalize 后 diff 仍有 232 行）、`__init__.pyi` 手写 705 行、`_check_terminal` 已出现逐字重复（§2-⑬）。「specs 从 M1 起同步建设」的执行力度落后于计划，复制债务在 codegen 接管前会持续累积。

### §3.3 第三方厂商结论落实 —— ✅

reasoning_content 回显（9aba4cd）、`models.list(provider=)` 过滤（客户端）、ErrorCode 覆盖审核/配额类错误——三个 SDK 侧落点都在；CLI 示例属 M4 范围。

### 目录布局 —— ✅

`v2/` 独立实现、`tests/v2/` 独立测试、包根只导新面，与 §7 目标布局一致。形式偏差：设计写 `resources/images/`、`resources/videos/` 目录，实际是单文件——无关紧要。

---

## 4. 建议行动项（按优先级）

| 优先级 | 事项 | 出处 |
| --- | --- | --- |
| P0 | 传输错误映射进 `APIConnectionError`/`APITimeoutError`，删除零引用死类或启用之 | §2-① |
| P0 | chat 非流式 compat 模式不得静默吞错（返回体带 code/is_error） | §2-② |
| P0 | 修 `get_final_message()` 的 OpenAI tool_calls 片段聚合 | §2-③ |
| P1 | `files.upload` 重试空 body（读 bytes 或禁重试） | §2-④ |
| P1 | 统一 chat/apps 流内错误语义，或在设计文档显式豁免 apps | §2-⑤ |
| P1 | 兑现 3.8 agentstudio「errors/HTTP 对齐 2.0」或修订设计文档 | §1-3.8 |
| P2 | 补 `with_options()` 或从设计文档删除 | §1 |
| P2 | AsyncAudio 补齐 tts/realtime 或文档声明 ws 资源仅同步 | §1-3.5 |
| P2 | `resolve_config` 三键 env 补齐或文档化 | §1-1.1 |
| P2 | rl 同步方法 event-loop 冲突写入文档限制 | §2-⑥ |
| P3 | `models.list` 文档声明 total_count 为过滤前值 | §2-⑦ |
| P3 | Settings 线程安全与 `clear()` 恢复 logger | §2-⑨ |
| P3 | 设计文档白名单补记 0c58a74 的 1.x LOGGER_NAME 豁免 | §3-P3 |
| P3 | specs 元数据恢复节奏（P6 纪律）；openai request_id 语义修正 | §3-P6、§2-⑪ |
| P3 | 迁移指南写明 `DEFAULT_WAIT_TIMEOUT=86400` 与 1.x 的差异 | §2-⑫ |

---

## 5. 测试证据

- `pytest tests/v2`：**254 passed / 3 skipped**（skipped 为 M2 2.6 的 live 集成测试，env-gated）。
- 测试资产：23 个测试文件，含协议层 947 行（`test_protocols.py`）+ SSE 专项 435 行 + 全资源冒烟 641 行；openai/anthropic 各有 conformance 套件。
- 冒烟测试覆盖核对表全部资源（chat/multimodal/embeddings/rerank/tokenization/files/models/images/videos/audio asr·tts·realtime/fine_tunes/deployments/rl/apps/agentstudio/async 面）。
