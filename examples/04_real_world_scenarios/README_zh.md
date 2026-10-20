# 04 真实场景

> [English](README.md) | **中文**

这一层不是"怎么调某个能力"——而是**把一个真实业务端到端建起来**：把 `01` 的原子能力、`02` 的工程封装和 `03` 的 agent 配方组装成完整、可直接对标的案例。每个案例都带着生产环境真实存在的权衡（护栏、上下文预算、容错、可观测性），并附一段"为什么这样写"的讲解。

| 案例 | 场景 | 联网 |
| --- | --- | --- |
| [`customer_service_assistant.py`](./customer_service_assistant.py) | 电商在线客服：护栏化人设 + 业务工具 + 多轮记忆 + 受管工具循环 | 是 |
| [`production_ready_rag_harness.py`](./production_ready_rag_harness.py) | 工业级 RAG 问答引擎：流式语义检索 + 本地 token 水位 + 流式退避重试 + 多地域熔断与故障切换 + 优雅降级 | 否（默认离线 mock；`RAG_HARNESS_LIVE=1` 打真实网关） |
| [`realtime_audio_meeting_harness.py`](./realtime_audio_meeting_harness.py) | 高并发实时会议转写底座：每场会议一条独立 websocket + 会议状态机 + 重连退避 + 跨重连本地转写稿合并 + 按房间故障隔离与降级 | 否（默认离线 mock；`MEETING_HARNESS_LIVE=1` 打真实网关） |
| [`ticket_triage_harness.py`](./ticket_triage_harness.py) | 决策模型工单分诊：每张工单一次决策 round（单次前向推理回答 choice / noul / score）+ 本地阈值策略 + 失败封闭降级到人工队列 + 单工单隔离与精确的 input-token 成本账本 | 否（默认脚本化决策后端；`TRIAGE_HARNESS_LIVE=1` 打真实网关） |

---

## 案例：电商在线客服助手

`customer_service_assistant.py` 演示了一个"示例商城"的售后服务机器人。用户可以用自然语言查订单、跟踪物流、询问能否退款、要求转人工；机器人在回答前会调用业务工具取**真实数据**。

### 它解决的真实问题

| 生产关切 | 示例的做法 |
| --- | --- |
| **护栏** | system prompt 把回答限定在售后话题内，订单事实必须走工具调用，禁止编造，并定义了何时转人工 |
| **业务工具集成** | `query_order` / `track_shipment` / `check_refund_eligible` / `create_ticket`，全部跑在一个内存"订单库"上——换成你自己的服务即可 |
| **上下文预算** | `ConversationMemory` 使用滑动窗口：system 消息固定保留，只保留最近 `MAX_HISTORY_TURNS` 轮，长对话永远不会撑爆上下文 |
| **报文完整性** | 从头部截断可能留下"孤儿" `role="tool"` 消息（其 assistant 工具调用轮已被丢弃），API 会拒绝这种请求；`_trim` 丢弃头部孤儿 tool 消息以保证请求合法 |
| **失控兜底** | `MAX_TURNS` 防止模型无限调用工具；工具执行使用防御式 JSON 解析，对未知工具 / 错误参数返回结构化错误而不是抛异常 |

### 架构一览

```text
用户输入
  │
  ▼
ConversationMemory ──(system + 最近若干轮)──► client.chat.completions.create(tools=TOOLS)
  ▲                                                    │
  │                                 ┌──────────────────┴──────────────────┐
  │                                 │ 有 tool_calls 吗？                  │
  │                          yes ◄──┤                                     ├──► no
  │                                 └─────────────────────────────────────┘
  │ 本地执行工具 → 把 role="tool" 结果喂回
  └──────────────────────────────────────────────────────────────────────┐
                                                             最终答案 ◄──┘
```

### 为什么工具循环是非流式的

工具调用轮必须在本地执行前**读到完整的 `message.tool_calls`**，因此流式增量在循环内没有意义；循环使用非流式 `create`。如果你想要纯聊天的打字机效果，见 [`01_atomic_capabilities/text_generation/chat.py`](../01_atomic_capabilities/text_generation/chat.py) 里的 `stream=True` + `text_stream`；当你同时需要流式和合并后的 `tool_calls` 时，用 `stream.get_final_message()`。

### 运行

```bash
export DASHSCOPE_API_KEY="sk-xxx"
python3 examples/04_real_world_scenarios/customer_service_assistant.py
```

演示会跑一段三轮对话，其中第二轮——"这个订单现在能退款吗？"——不再重复订单号：它能被正确解答，完全是因为 `ConversationMemory` 把第一轮的 `SO2024001` 带进了上下文。这正是记忆层存在的意义。

### 改造成你自己的客服系统

1. **换数据源**：把 `ORDERS` 和四个 `*_tool` 函数换成真实的订单 / 物流 / 工单服务调用（保持返回 JSON 可序列化的 dict）。
2. **改人设与政策**：编辑 `SYSTEM_PROMPT` 和 `REFUND_WINDOW_DAYS` 等业务常量。
3. **加工具**：写好函数 → 把 schema 加进 `TOOLS` → 在 `TOOL_IMPL` 里注册；循环会自动接管。
4. **接持久化记忆**：`ConversationMemory` 目前是进程内列表；生产环境换成按 session id 存取的 Redis / 数据库，保持接口（`add` / `extend` / `messages`）不变。

---

## 案例：工业级生产就绪 RAG harness

`production_ready_rag_harness.py` 不是"怎么拼一个 RAG 调用"——它把企业知识库问答要上生产必须做对的四件事缝成一个可运行的蓝图：**流式语义检索 -> 本地 token 水位 -> 刚性弹性治理 -> 优雅降级**。它默认离线运行（`httpx.MockTransport` 提供罐头响应并按需注入故障），所以不需要 API key 你就能**亲眼看到**重试、故障切换和水位裁剪真实发生。

### 四个生产维度

| 维度 | 做法 | 为什么重要 |
| --- | --- | --- |
| **流式语义检索** | `LocalVectorStore` 一次向量 + 余弦 top-k，然后 `client.rerank` 交叉编码器精排 | 快速召回 + 精确排序——RAG 质量的地基 |
| **本地 token 水位** | `TokenWatermark` 用 `client.tokenization.count` 逐段实测，只发送装得下的段落 | 再胖的上下文也不会撞窗口上限——或账单 |
| **刚性弹性** | 非流式调用依赖 SDK 内置 `max_retries`（感知 Retry-After 的退避）；**SDK 不重试流式调用**，所以 harness 自带 `retry_with_backoff`；其上再由 `ResilientGateway` 按地域熔断并跨地域切换 | 429/5xx / 整地域故障都拖不垮应用 |
| **优雅降级** | `answer()` 把检索 / 精排 / 生成包进一个护栏；`RateLimitError` / `ContentFilterError` / 全地域故障都变成干净的降级答案 | 上层拿到的是结果，而不是堆栈 |

### 不读源码一定会踩的坑

> **流式请求不走 SDK 的自动重试。** `dashscope/v2/transport/http.py` 里的 `open_stream` 只检查一次初始状态就返回（源码注释："only the initial status is checked here, no retries"），而普通 `request` 会跑 `max_retries` 退避循环。所以一个不自己处理 429/503 的流式 RAG，会把异常直接甩到用户脸上。harness 的 `retry_with_backoff` 正是为这个缺口而存在。

同样，流是**一次性迭代器**：先消费 `text_stream` 再调 `get_final_message()` 只能拿到空尾巴，而 usage 计数在**最后一个 chunk** 而不是第一个。因此 `_consume_stream` 恰好迭代一次，边打印增量边把 usage 折叠进来。

### 运行

```bash
# 离线（默认，无需密钥，输出确定，依次演示场景 A/B/C/D）
python3 examples/04_real_world_scenarios/production_ready_rag_harness.py

# 打真实网关（需要 DASHSCOPE_API_KEY；只跑场景 A，无故障注入）
RAG_HARNESS_LIVE=1 python3 examples/04_real_world_scenarios/production_ready_rag_harness.py
```

离线演示依次进行：

- **场景 A** 标准问答：召回 -> 水位裁剪（观察 `dropped`）-> 精排 -> 带引用的流式生成。
- **场景 B** 注入一次 429：harness 指数退避后成功（因为流式不被 SDK 重试）。
- **场景 C** 主地域 503：回答途中熔断器 CLOSED -> OPEN；后续步骤直接跳过死亡地域并切换到副地域。
- **场景 D** 全部地域不可用：不抛异常——返回降级答案。

### 改造成你自己的知识库

1. **换向量库**：`LocalVectorStore` 是内存余弦；生产换成 Milvus / AnalyticDB / pgvector，保持 `index` / `search` 接口。
2. **打真实网关**：去掉 `MockGateway`；`build_engine` 已内置 `RAG_HARNESS_LIVE=1` 分支（单地域）；真正的多地域容灾需要每个地域各自的密钥 + 工作空间（见 `legacy_1x/set_region_example.py`）。
3. **调水位与退避**：`CONTEXT_TOKEN_BUDGET` 由模型最大上下文减去指令和预留回答长度推导；`GEN_RETRIES` / `base_delay` 生产建议 `base_delay≈0.5`，并尊重服务端的 Retry-After。
4. **批量计数**：演示为了可观测性逐段计数；高频路径应批量计数或用本地 tokenizer 估算。

---

## 案例：高并发实时会议转写底座

`realtime_audio_meeting_harness.py` 把 legacy 听悟 realtime 脚本（`legacy_1x/tingwu_realtime.py`，回调 + `time.sleep` 轮询的测试风格）重建成**会议平台级并发监管器**：同时实时转写 N 个会议室，每个房间有自己的状态机和转写稿，任何一个房间的断线或审核拦截都不能拖垮其他房间。它同样默认离线运行——通过听悟 realtime 的 `connector=` 注入一条脚本化双工 socket（与 `tests/v2/test_tingwu.py` 用的是同一个接缝），没有麦克风、没有密钥也能看到并发、重连和降级真实发生。

### 四个生产维度

| 维度 | 做法 | 为什么重要 |
| --- | --- | --- |
| **尊重协议的并发** | 一场会议 = 一条独立 websocket + 一个 worker；`ThreadPoolExecutor(max_workers=MAX_CONCURRENT)` 同时承担并发与准入 | 见下面的坑：realtime 无法复用，所以池大小就是活跃 socket 的上限 |
| **本地会议状态机** | 每个房间走 `PENDING→CONNECTING→LISTENING→TRANSCRIBING→COMPLETED`，`RECONNECTING`/`DEGRADED` 是受管侧分支 | 上百个房间通过同一套状态词汇变得可观测、可汇报 |
| **云端识别 / 本地拼装** | `TranscriptAssembler` 按 `sentenceId` 合并增量句子（partial 被 final 覆盖） | 听悟会增量地重发同一个 `sentenceId`；只有本地合并才能得到完整转写稿——并让转写跨重连续上 |
| **刚性弹性 + 隔离** | 断线（`APIConnectionError`）触发指数退避重连；业务性 `task-failed` 让该房间降级而不是盲目重连；异常封死在每个房间的 worker 内 | 传输故障该重连，业务故障该降级——搞混了就是白忙；隔离保证一个房间的失败不外溢 |

### 不读源码一定会踩的坑

> **realtime 家族无法复用连接。** `dashscope/v2/transport/ws_connection.py` 的头注释写明了：realtime 协议没有 task id——一条 socket 就是一个会话，会话结束即关闭——所以它总是为每个会话开一条独立连接。（可复用的 `WsConnection` 只服务 JSON 帧的 asr/tts/translation 任务，且二进制上行是单属主的。）因此"高并发实时音频"的正确形态是**每场会议一条独立连接 + 一个监管线程池**，而不是把多路音频塞进一条 socket——协议做不到。harness 的 `MeetingSupervisor` 用有界线程池把"并发"绑定到"活跃 socket / 爆炸半径"上，作为准入控制。

同样，当 socket 在中途断开时，`WsSession.recv` 抛 `APIConnectionError`（"closed before task-finished"），而 `task-failed` 事件会经错误目录映射成 `status_code=200` 的业务异常（如 `ContentFilterError`）。`_is_reconnectable` 正是据此区分：前者重连，后者降级——离线演示里的 `room-gamma`（断线重连后，半句 `今天我们讨论` 被补全为 `今天我们讨论第三季度的路线图。`）和 `room-delta`（内容审核 `task-failed` 立即降级）分别演示了这两条路径。

### 运行

```bash
# 离线（默认，无需密钥，输出确定，4 个房间并发 + 重连 + 单房间降级）
python3 examples/04_real_world_scenarios/realtime_audio_meeting_harness.py

# 打真实网关（需要 DASHSCOPE_API_KEY + DASHSCOPE_TINGWU_APP_ID，跑一个真实音频房间）
MEETING_HARNESS_LIVE=1 python3 examples/04_real_world_scenarios/realtime_audio_meeting_harness.py
```

离线演示并发跑 4 个房间，分别汇报 `completed` / `degraded`，展示每个房间的尝试次数 / 重连次数 / 句段数和转写预览——`room-delta` 被审核拦截降级，其余 3 个完成，直观地证明"一个房间的失败不外溢"。

### 改造成你自己的会议平台

1. **换音频源**：`_read_pcm` 目前读本地 16 kHz PCM/WAV；生产环境从麦克风 / RTC 下行 / 消息队列字节流向 `_pump_audio` 喂帧，保持 `pace=True` 的实时节奏。
2. **真正的边说边转写**：演示是"发完全部音频 → stop → 读事件"（与 `01_atomic_capabilities/audio/tingwu.py` 一致，为了离线确定性）；实时字幕风格需要在发送的同时用读线程排空 `realtime.events()`——socket 本身是全双工的。
3. **调并发与准入**：`MAX_CONCURRENT` 是活跃 socket 上限——按你机器的 fd / 线程预算设定；`RECONNECT_ATTEMPTS` 和 `BASE_DELAY`（生产建议 ≈0.5s）控制重连激进程度。
4. **接持久化与可观测性**：`MeetingSession.transition` 是状态的唯一出口；生产环境接到日志 / 指标 / 会议记录存储上，按 `meeting_id` 持久化转写稿与状态。

---

## 案例：决策模型工单分诊 harness

`ticket_triage_harness.py` 把决策模型（`decision-model-preview`）变成工单系统的前台：每张进线工单只做一次决策 round——归哪个团队（`choice`）、要不要呼叫值班工程师（`noul`）、有多严重（`score`）——再由本地策略引擎把返回的概率分布变成路由动作。它默认离线运行：mock 缝在"一方法决策后端"上，`MockDecisionBackend` 提供脚本化答案（一张低于置信度下限的模糊工单，外加一次注入的后端故障），而被测的策略、降级和账本逻辑就是原封不动的生产代码。

### 四个生产维度

| 维度 | 做法 | 为什么重要 |
| --- | --- | --- |
| **一次 round，类型化问题** | `client.decision.create` 对同一个 `state` 在单次前向推理中回答 choice + noul + score；不生成文本 | 每张工单的延迟和成本是平的——不像"让大模型生成 JSON 再解析" |
| **本地阈值策略** | `TriagePolicy` 拥有置信度下限、P(yes) 升级阈值和严重度分档 | 模型返回的是分布，永远不是决策；阈值是可审计的业务决策，调整不需要重新部署任何东西 |
| **失败封闭降级** | 路由置信度不足送人工队列；异常**或** `is_error` 信封都降级到人工队列并附带错误 | 没有工单会被凭空猜测地自动路由——见下面的坑 |
| **单工单隔离 + token 账本** | 后端故障只降级那一张工单；`usage.input_tokens` 逐票计入滚动账本 | 一张坏工单永远拖不停整个批次，而且账本就是精确账单——input token 是决策模型唯一计费的量 |

### 不读源码一定会踩的坑

> **DashScope 可能用 HTTP 200 报告失败。** `dashscope/v2/types/common.py` 的 `ResponseBase` 在每个响应上都带 `code` / `message` / `is_error`，而 `raise_on_error=False` 时 SDK 会填充这个信封而不是抛异常。一个只包 `try/except` 的分诊循环会把失败的调用当成答案为空的成功调用——然后基于"什么都没有"自动路由工单。因此 harness 即使在默认 raising 模式下也检查 `response.is_error`：这是双保险，因为当某天有人把 `raise_on_error=False` 打开时，这道防线必须已经就位。

同样，**`score` 答案是概率加权等级索引，不是整数类别**——`2.3` 表示"介于等级 2 和 3 之间，偏向 2"，`legend` 把索引映射回你给的等级文本。像 `P1_SCORE = 2.5` 这样的分数线因此是连续轴上的阈值，这正是严重度分档想要的东西。

还有一条路由说明：System One 端点在**每个网关的 compatible-mode 路径下**（由 `dashscope/v2/_config.py` 的 `compatible_mode_base_url` 推导），所以改变客户端的聊天 `protocol` 不会移动它。

### 运行

```bash
# 离线（默认，无需密钥，输出确定：5 张工单，一张低置信度，一次注入故障）
python3 examples/04_real_world_scenarios/ticket_triage_harness.py

# 打真实网关（需要 DASHSCOPE_API_KEY；同一个 harness，无故障注入）
TRIAGE_HARNESS_LIVE=1 python3 examples/04_real_world_scenarios/ticket_triage_harness.py
```

离线演示分诊 5 张工单：T-2001 / T-2002 路由到各自团队，T-2003（"你们这个系统到底怎么回事"）低于置信度下限进入人工复核，T-2004 作为 P1 呼叫值班工程师，T-2005 遭遇注入的后端故障并降级到人工队列——最后汇总打印状态分布和精确的 input-token 账本。

### 改造成你自己的工单系统

1. **换进线队列**：`TICKETS` 是占位列表；接入真实工单流，保持模型需要评判的 `state` 形状（这里是 `ticket_id` / `content`）。
2. **调策略而不是调模型**：`CONFIDENCE_FLOOR` / `ESCALATE_THRESHOLD` / `P1_SCORE` / `P2_SCORE` 是业务决策——根据观察到的路由质量标定它们。
3. **接出口**：目前判定只是打印；生产环境把 `triaged` 推到团队队列、`escalated` 推到值班呼叫、`human_review` / `degraded` 推到人工分诊收件箱——verdict 对象已经带齐了任何下游需要的一切。
4. **横向扩展**：决策 round 按工单互相独立，`run` 可以并行化；保持有界线程池，并让 SDK 的 `max_retries` 在每个非流式调用内部吸收 429/5xx。
