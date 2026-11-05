# DashScope SDK 示例 —— 百炼（Model Studio）生产最佳实践

> [English](README.md) | **中文**

这不是一份"API 调用参考"。在 AI 辅助开发的时代，任何人都能几秒钟生成一行 `call(model="xxx")`——本目录的价值在于**提前、正确地把一个百炼应用推向生产所需的那几百行"脏活代码"写好**：每个进阶文件都是一个微型的生产级 harness 蓝图，带有严格的工程防御，面向一个具体的开发或业务场景。全部示例基于 2.0 资源风格客户端 `dashscope.Client()`（与包内 `dashscope` 一致）；旧版 1.x 示例完整归档在 `legacy_1x/` 下。

一个合格的最佳实践案例必须同时展现三个维度：

1. **解决真实业务/工程痛点**（企业知识库问答、高并发实时会议转写、在线智能客服等）。
2. **云端大模型 × 本地 SDK 工程协同**（本地 token 水位裁剪、流式增量优雅合并、跨重连转写稿拼装）。
3. **生产级刚性防御与弹性治理**（429/500 优雅处理、超时退避重试、多地域熔断容灾、状态机管理、按房间故障隔离）。

## 旗舰案例（先读这三个）

| 案例 | 一句话 | 离线运行 |
| --- | --- | --- |
| [`04_real_world_scenarios/production_ready_rag_harness.py`](./04_real_world_scenarios/production_ready_rag_harness.py) | 工业级 RAG：流式语义检索 + 本地 token 水位 + 流式退避重试（SDK 不重试流式）+ 多地域熔断与故障切换 + 优雅降级 | ✅ |
| [`04_real_world_scenarios/realtime_audio_meeting_harness.py`](./04_real_world_scenarios/realtime_audio_meeting_harness.py) | 高并发实时会议转写：每个会议室一条独立 websocket（realtime 无法复用）+ 会议状态机 + 断线重连 + 跨重连转写稿合并 + 按房间隔离与降级 | ✅ |
| [`04_real_world_scenarios/customer_service_assistant.py`](./04_real_world_scenarios/customer_service_assistant.py) | 在线智能客服：护栏化人设 + 业务工具 + 滑动窗口多轮记忆（含孤儿 tool 消息修剪）+ 受管工具循环 | 需要密钥 |

> 三个旗舰案例都嵌入了"代码为什么这样写"的源码级洞察（例如 **SDK 不会自动重试流式请求**、**realtime 协议无法复用连接**）；详见 [`04_real_world_scenarios/README.md`](./04_real_world_scenarios/README.md)。

## 目录结构

```
examples/
├── 01_atomic_capabilities/   原子能力速查（最小可运行；也是喂给 AI 的权威 wire-shape 种子）
├── 02_advanced_engineering/  工程模式：本地 mock 调试、语义缓存、熔断 + 多地域容灾
├── 03_agent_recipes/         Agent 配方：工具调用循环、RAG、MCP server
├── 04_real_world_scenarios/  旗舰最佳实践：把上面三层组装成端到端生产 harness
├── 05_cli/                   `dashscope` 命令行：输出格式、退出码、配置与 CI 配方
└── legacy_1x/                1.x 模块级 API 示例（dashscope.Generation 等），仅供参考
```

各层定位（一条从"速查"到"生产蓝图"的进阶路径）：

- **01 原子能力（速查种子）**：每个能力一个文件，最小但可运行。刻意保持简单——它是查询 wire-shape 的权威种子，也是喂给通义灵码 / Cursor 让它们生成正确代码的种子。每个文件末尾有"production springboard（生产化跳板）"页脚，指向加固该能力的 02/03/04 案例。
- **02 进阶工程**：把 SDK 包进真实生产关心的模式里（省钱、抗压、离线调试）。
- **03 Agent 配方**：多步骤、可组合的端到端 agent 流程。
- **04 真实场景**：不是"怎么调某个能力"，而是把一个完整业务端到端建成生产级 harness（护栏、上下文预算、容错、容灾、状态机等生产权衡）；每个案例配有自己的 `README.md`，讲清"为什么这样写"。
- **05 命令行**：同一批能力的 shell/CI 契约面，而不是 Python 契约面——机器可读输出、分层退出码、固定的一行 stderr 错误，以及写明的配置优先级；两个文件都离线可跑。
- **legacy_1x**：旧的 1.x 模块级风格，子目录布局原样保留，仅供参考；新项目请优先使用上面五层。

## 准备工作

1. 安装 SDK：

   ```bash
   pip install -U dashscope
   ```

2. 配置百炼 API key（[获取方式](https://help.aliyun.com/zh/model-studio/get-api-key)）：

   ```bash
   export DASHSCOPE_API_KEY="sk-xxx"
   ```

3. 直接运行任意示例文件（从仓库根目录）：

   ```bash
   python examples/01_atomic_capabilities/text_generation/chat.py
   ```

> **离线运行（无需 API key）**：`02_advanced_engineering/local_mock_debugging.py`、`02_advanced_engineering/circuit_breaker_retry.py`、`03_agent_recipes/mcp_server_demo/client_demo.py`、`04_real_world_scenarios/production_ready_rag_harness.py`、`04_real_world_scenarios/realtime_audio_meeting_harness.py`、`04_real_world_scenarios/ticket_triage_harness.py`、`05_cli/cli_output_and_errors.py` 和 `05_cli/cli_ci_harness.py` 使用 mock transport / 本地 mock 网关 / 脚本化接缝 / 子进程——完全离线且输出确定，适合先动手感受调用形态。

## 常用环境变量

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 百炼 API key（联网示例必需） | 无 |
| `DASHSCOPE_MODEL` | 覆盖聊天/生成模型（chat、缓存、RAG、工具循环、决策等） | 各文件默认值（多为 `qwen-plus`） |
| `DASHSCOPE_EMBED_MODEL` | 覆盖向量模型（`client_side_cache.py`、`rag_expert_system.py`） | `text-embedding-v4` |
| `DASHSCOPE_RERANK_MODEL` | 覆盖重排模型（`rag_expert_system.py`） | `gte-rerank-v2` |
| `DASHSCOPE_WORKSPACE` | AgentStudio 工作空间 | `trial` |
| `DASHSCOPE_APP_ID` | 百炼应用 ID（application.py） | 无 |
| `DASHSCOPE_FILE_ID` | 文件 ID（assistant_api_file.py） | 无 |
| `DASHSCOPE_DEMO_AUDIO` | 本地音频路径（tingwu_realtime.py、tingwu.py） | `tests/data/asr_example.wav` |
| `DASHSCOPE_TINGWU_APP_ID` | 听悟应用 ID（tingwu_realtime.py、tingwu.py） | 无 |
| `DASHSCOPE_TTS_VOICE` | TTS 音色（audio_tts.py） | `zhichu` |
| `DASHSCOPE_TRAINING_FILE_ID` | 微调训练文件 ID（fine_tunes.py） | 无 |
| `DASHSCOPE_DEMO_IMAGE` | 本地图片路径（cv_models.py） | `~/Downloads/cat.png` |
| `DASHSCOPE_DEMO_VIDEO_URL` | 视频 URL（multimodal_conversation/video.py） | 示例视频 |
| `RAG_HARNESS_LIVE` | 置 `1` 时 `production_ready_rag_harness.py` 打真实网关（否则离线 mock） | 未设置（离线） |
| `MEETING_HARNESS_LIVE` | 置 `1` 时 `realtime_audio_meeting_harness.py` 打真实网关（否则离线 mock） | 未设置（离线） |
| `TRIAGE_HARNESS_LIVE` | 置 `1` 时 `ticket_triage_harness.py` 打真实网关（否则脚本化决策后端） | 未设置（离线） |
| `CLI_HARNESS_LIVE` | 置 `1` 时 `cli_ci_harness.py` 打真实网关，仅只读场景（否则本地 mock 网关） | 未设置（离线） |
| `DASHSCOPE_TINGWU_MODEL` | 听悟 realtime 模型（`realtime_audio_meeting_harness.py`） | `tingwu-turbo` |
| `MEETING_MAX_CONCURRENT` | 实时会议并发上限（即活跃 socket 上限） | `4` |

## 01_atomic_capabilities —— 原子能力

统一风格：`client = dashscope.Client()`（默认从 `DASHSCOPE_API_KEY` 环境变量读取密钥）。

| 路径 | 内容 |
| --- | --- |
| `text_generation/chat.py` | 文本生成：一行 `dashscope.chat()`、非流式、流式（text_stream） |
| `text_generation/tokenization.py` | token 计数 |
| `multimodal_vision/multimodal_chat.py` | 多模态对话（qwen-vl 图像理解） |
| `audio/` | 语音示例：ASR/TTS/realtime/音色 CLI（见 `audio/README.md`；`direct_call/` 为不使用 `with` 的副本） |
| `audio/tingwu.py` | 听悟离线任务与实时会话 |
| `image_video_synthesis/images.py` | 图像生成：一键生成并等待 + 手动 jobs 流程 |
| `image_video_synthesis/videos.py` | 视频生成：一键生成并等待 + 手动 jobs 流程 |
| `embeddings/embeddings.py` | 文本向量（含批量任务入口说明） |
| `rerank/rerank.py` | 文档重排 |
| `decision/decision.py` | 决策模型（System One）：一次调用对业务状态回答类型化的 choice / noul / score 问题 |
| `files/files.py` | 文件管理：上传 / 列表 / 查询 / 删除 |
| `applications/apps.py` | 百炼应用调用（非流式 + 流式） |
| `fine_tuning/fine_tunes.py` | 微调任务：list / get、create / wait / delete（取消注释即可运行） |
| `fine_tuning/reinforcement_learning.py` | 智能体 RL：任务 list / get；注册函数 -> 上传数据 -> create / wait / delete（取消注释即可运行） |
| `models/models.py` | 模型列表与详情 |

## 02_advanced_engineering —— 进阶工程

| 路径 | 内容 | 联网 |
| --- | --- | --- |
| `local_mock_debugging.py` | 注入 `httpx.MockTransport`，零成本离线跑通成功路径与错误路径（400 / 限流 / 内容安全）；把 `_mock_client()` 换成 `dashscope.Client()` 即可打真实网关 | 否 |
| `client_side_cache.py` | 客户端语义缓存：向量 + 余弦相似度捕获重复/近似问题，命中即直接返回不调模型，直接省 token | 是（仅缓存未命中时） |
| `circuit_breaker_retry.py` | 熔断器 + 多地域容灾：在 SDK 内置的 Retry-After 退避之上，按地域 CLOSED/OPEN/HALF_OPEN 熔断并自动切换；演示用 mock transport（主地域 503、副地域 200），离线可运行 | 否 |

## 03_agent_recipes —— Agent 配方

| 路径 | 内容 | 联网 |
| --- | --- | --- |
| `tool_call_loop.py` | 受管多轮工具调用状态机：声明 `tools` schema，模型请求调用时本地执行，把 `role="tool"` 结果喂回，循环直到最终答案（带 `MAX_TURNS` 兜底） | 是 |
| `rag_expert_system.py` | 标准 RAG：语料向量 -> 余弦 top-k 召回 -> `client.rerank` 精排 -> "仅依据资料作答"的聊天 prompt -> 带引用的答案 | 是 |
| `mcp_server_demo/` | 零依赖 MCP（Model Context Protocol）stdio server + client 演示：`server.py` 讲 JSON-RPC 2.0；`client_demo.py` 拉起子进程并跑 initialize -> tools/list -> tools/call。见其 `README.md` | 否 |
| `agentstudio.py` | AgentStudio 端到端 quickstart（经 `client.agentstudio`）：环境 -> 智能体 -> 会话 -> 流式回复（另附可选的 memory store 演示）；AgentStudio 为 1.x/2.0 两代共用模块，挂载在 2.0 客户端上 | 是 |
| `agentstudio_deployments.py` | 托管智能体部署：创建 cron 定时部署、触发运行、列出运行记录 | 是 |

## 04_real_world_scenarios —— 真实场景

把 `01`–`03` 组装成完整、可直接对标的业务案例，带护栏、上下文预算、容错等生产权衡；每个案例配有自己的 `README.md`，讲清设计权衡与改造方法。

| 路径 | 内容 | 联网 |
| --- | --- | --- |
| `customer_service_assistant.py` | 电商在线客服：护栏化人设 + 业务工具（订单 / 物流 / 退款 / 工单）+ 滑动窗口多轮记忆（含孤儿 tool 消息修剪）+ 受管工具循环；演示跑一段依赖上下文的三轮对话 | 是 |
| `production_ready_rag_harness.py` | 工业级 RAG 引擎：流式语义检索 + 本地 token 水位（用 tokenization 逐段实测）+ 流式退避重试（SDK 不重试流式）+ 多地域熔断与故障切换 + 优雅降级；默认离线 mock 运行，演示场景 A/B/C/D | 否 |
| `realtime_audio_meeting_harness.py` | 高并发实时会议转写底座：每场会议一条独立 websocket（realtime 无法复用）+ 会议状态机 + 重连退避 + 跨重连本地转写稿合并 + 按房间故障隔离与降级；默认离线 mock 运行，4 个房间并发演示重连与降级 | 否 |
| `ticket_triage_harness.py` | 决策模型工单分诊：每张工单一次决策 round（单次前向推理回答 choice / noul / score）+ 本地阈值策略（置信度下限、升级阈值、严重度分档）+ 失败封闭降级到人工队列 + 单工单隔离与精确的 input-token 成本账本；默认用脚本化决策后端离线运行 | 否 |

详见 [`04_real_world_scenarios/README.md`](./04_real_world_scenarios/README.md)。

## 05_cli —— 命令行

同一批能力，只是契约面从 Python 换成了 **shell**：三档输出格式并有明确的 stdout/stderr 分工、分层退出码（0 成功 / 1 服务端 / 2 用法 / 3 鉴权 / 4 不存在）、stderr 上一行固定的 `Error [category]: message`，以及四层配置和写明的优先级。两个文件都离线、确定性运行——不需要密钥，不联网。

| 路径 | 内容 | 联网 |
| --- | --- | --- |
| `cli_output_and_errors.py` | 从内部看契约：一个假 client 通过 `typer.testing.CliRunner` 在进程内驱动，把 table / json / yaml 三档、NDJSON 流式、`@file` / stdin / `--messages` 输入形态，以及包括破坏性命令拒绝在内的每一档失败的 stdout / stderr / 退出码并排打印 | 否 |
| `cli_ci_harness.py` | 从外部看契约：在临时端口用标准库起一个 mock 网关，在真实子进程里驱动真实 CLI——解析 `-o json`、按退出码分支与重试、逐行消费 NDJSON 流、用 `DASHSCOPE_CONFIG` 隔离配置，并在 CI 没有终端时传 `--yes` | 否 |

建议按表格顺序读两个示例。第一个演示每个通道**吐出什么**，第二个演示**流水线该怎么消费它**。

### 安装与第一次调用

包里注册了 console script，所以 `pip install -U dashscope` 之后 `PATH` 上就有一个 `dashscope` 命令：

```bash
dashscope --help
dashscope version
```

在没装 console script 的源码检出里，模块入口完全等价：

```bash
python -m dashscope.cli --help
python -m dashscope.cli version
```

> 现有 1.x 命令形式同样使用 `dashscope` 入口。1.x 专有名称仍由冻结的 1.x 实现执行，共享名称则通过 2.0 命令树兼容旧参数拼写。1.x 模块级 API 示例见 [`legacy_1x/`](./legacy_1x/)。

密钥配一次，两种方式都行：

```bash
export DASHSCOPE_API_KEY="sk-xxx"
dashscope chat create -m qwen-plus --prompt "用一句话介绍杭州"
```

**全局选项必须放在子命令前面。**它们属于 app，不属于 `chat create`：

```bash
dashscope -k sk-xxx --output json chat create -m qwen-plus --prompt "hi"   # 正确
dashscope chat create -m qwen-plus --prompt "hi" --output json             # exit 2，usage
```

最后这条不是小怪癖——它是这个 CLI 最容易让人一头雾水的失败方式，`cli_output_and_errors.py` 专门演示了它。

### 命令地图

| 命令 | 用途 |
| --- | --- |
| `dashscope version` | 打印已安装的 SDK 版本。什么都不碰，所以是你手上最便宜的冒烟测试 |
| `dashscope chat create` | 一次 chat 补全：`--prompt` / `--messages`、`--stream`、`--tools`、采样参数，其余走 `--parameters` |
| `dashscope ft create` | 起一个微调作业；`--wait` 带 spinner 阻塞到终态，`--timeout` 限定等待时长 |
| `dashscope ft get` | 查看单个作业 |
| `dashscope ft wait` | 阻塞等待一个不是本进程创建的作业——`create --wait` 的后半段，适用于「一步提交、另一步收结果」的流水线。作业失败或被取消时退出码为 1 |
| `dashscope ft list` | 分页列出作业：`--page`、`--size` |
| `dashscope ft stream` | 跟随服务端事件流，直到服务端关闭它 |
| `dashscope ft logs` | 读作业的训练日志：`--offset`、`--lines`。table 模式原样打印日志文本——不画表——所以 `> train.log` 拿到的就是训练器打出来的东西 |
| `dashscope ft cancel` | 取消运行中的作业。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope ft delete` | 删除作业及其产出模型。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope files upload` | 上传微调用的文件；返回 `ft create -t` 需要的 `file_id` |
| `dashscope files list` | 分页列出已上传文件：`--page`、`--size` |
| `dashscope files get` | 按 id 查看单个文件 |
| `dashscope files delete` | 删除文件。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope models list` | 分页浏览模型目录：`--page`、`--size`、`--provider` |
| `dashscope models get` | 按名称查找模型；不存在时 exit 4 |
| `dashscope embeddings create` | 同步生成向量：`--input`（文本或 JSON 数组）、`--text-type`、`--dimension` |
| `dashscope embeddings jobs create` | 提交异步批量向量任务（文件 URL）；`--wait` 阻塞到完成 |
| `dashscope embeddings jobs get` | 查看批量任务状态 |
| `dashscope embeddings jobs wait` | 阻塞等待上一步拿到的批量任务 id，而不是循环轮询 `jobs get` |
| `dashscope embeddings jobs list` | 分页列出批量任务：`--page`、`--size`、`--status` |
| `dashscope embeddings jobs cancel` | 取消 pending 的批量任务。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope multimodal chat create` | 打到多模态端点的 `chat create`，用于那些即使不带媒体分片也必须走这个端点的会话 |
| `dashscope multimodal embeddings create` | 把文本、图片、音频、视频嵌到同一个向量空间：可重复的 `--text`/`--image`/`--audio`/`--video`，或 `--input` JSON |
| `dashscope rerank create` | 文档重排：`--query`、`--documents`（JSON 数组）或可重复的 `--document` |
| `dashscope tokenization count` | 计算 prompt 或 messages 的 token 数 |
| `dashscope apps completions create` | 按 `--app-id` 调用百炼**应用**：`--prompt`/`--messages`、可重复的 `--image`/`--file`、`--session-id`/`--memory-id`、`--rag-options`、`--biz-params`。带 `--stream` 时会同时要求增量帧，因为服务端默认每次重发整段答案 |
| `dashscope images generate` | 生成图片并等到出图；一行一个 URL，所以 `$(...)` 和 `xargs wget` 直接能用 |
| `dashscope images jobs create` | 只提交图片任务、立刻拿到 task_id；`--wait` 阻塞，或之后用 `jobs get`/`jobs list` 取结果 |
| `dashscope images jobs get` / `list` / `wait` | 收取上一步提交的任务：查看单个、分页列出，或阻塞到它结束并像 `generate` 一样打印图片 URL |
| `dashscope images jobs cancel` | 取消 pending 的图片任务。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope videos generate` | 生成视频并等到出片；打印视频 URL，所以 `$(...)` 和 `xargs wget` 直接能用。`--timeout` 默认一小时 |
| `dashscope videos jobs create` | 只提交视频任务、立刻拿到 task_id；`--wait` 阻塞，或之后用 `jobs get`/`jobs list` 取结果 |
| `dashscope videos jobs get` / `list` / `wait` | 收取上一步提交的任务：查看单个、分页列出，或阻塞到它结束并像 `generate` 一样打印视频 URL |
| `dashscope videos jobs cancel` | 取消 pending 的视频任务。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope deployments create` | 为模型开通独享容量：`--capacity`，MU 部署还要 `--deploy-spec`/`--billing-method` |
| `dashscope deployments get` / `list` | 查看单个部署，或分页列出 |
| `dashscope deployments stream` | 跟随部署的服务端事件流，直到服务端关闭它 |
| `dashscope deployments scale` | 调整部署容量。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope deployments delete` | 下线部署。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope rl jobs create` | 跑完整的 agentic-RL 流程（注册函数、上传数据、提交）；`--set key=value` 覆盖单个配置项，`--wait` 阻塞 |
| `dashscope rl jobs get` / `list` / `logs` | 查看单个 RL 作业、分页列出，或读它的训练日志 |
| `dashscope rl jobs wait` | 阻塞等待上一步拿到的 RL 作业 id；作业失败或被取消时退出码为 1 |
| `dashscope rl jobs cancel` / `delete` | 取消或删除 RL 作业。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope rl functions register` | 上传 rollout/reward 代码；返回 entity id 和 instance id |
| `dashscope rl functions test` | 用一份样例 `--input` 跑一个已注册的函数实例 |
| `dashscope rl functions upload-data` | 上传数据集；返回 `rl jobs create` 要用的 file id |
| `dashscope rl functions logs` | 读函数实例的日志；`--all` 聚合所有分页 |
| `dashscope rl functions delete` | 删除函数实例。**破坏性**——会确认，CI 里需要 `--yes` |
| `dashscope audio tts synthesize` | 把 `TEXT` 合成语音并写出音频：`--out speech.mp3`；不给 `--out` 就写 stdout，所以 `dashscope audio tts synthesize hi > hi.mp3` 能用。`--voice`、`--format`、`--sample-rate`、`--volume`、`--rate`、`--pitch` |
| `dashscope audio tts http` | 走非流式端点的同一个请求，打印 `audio_url` 而不是音频字节。这里 `-m` 和 `--voice` 是**必填**的——流式端点那套 cosyvoice 默认值不适用 |
| `dashscope audio asr transcribe` | 转写一个或多个音视频 URL 并**等待**结果；`--no-wait` 只提交、返回 task_id |
| `dashscope audio asr get` / `wait` | 立刻读一个转写任务，或带 spinner 阻塞到它落定 |
| `dashscope audio asr recognize` | 走实时识别器转写**本地文件**；table 模式只打句子，`-o json` 一行一个原始事件 |
| `dashscope audio realtime run` | 在双工 Omni socket 上跑**一轮**：喂进一整个 `--audio` 文件（路径、`@路径`，或管道 stdin），打印答案文本和输入转写，把语音回复写进 `--out`。`-o json` 是这一轮里每个事件的 NDJSON 轨迹 |
| `dashscope audio vocabularies create` / `list` / `get` / `update` / `delete` | 热词表，由 `asr recognize --vocabulary-id` 消费。`--vocabulary` 是词条的 JSON **数组**。`delete` 是**破坏性**的 |
| `dashscope audio phrases create` / `list` / `get` / `update` / `delete` | 编译好的短语集，由 `asr transcribe --phrase-id` 消费。`--phrases` 是短语列表的 JSON **对象**。`delete` 是**破坏性**的 |
| `dashscope audio voices create` / `list` / `get` / `update` / `delete` | 从 `--url` 上的一段样本录音克隆出来的音色。`delete` 是**破坏性**的 |
| `dashscope audio voice-design create` / `list` / `get` / `delete` | 用文字描述（`--voice-prompt` + `--preview-text`）而不是录音造出来的音色——和 `voices` 共用同一个注册表、同一套 id 命名空间和同一份配额。没有 `update`，因为服务端就没有。`delete` 是**破坏性**的 |
| `dashscope tingwu call` | 跑一个离线 TingWu 任务：`-m`，加上一份由业务定义的载荷，要么写成 `--input @task.json`，要么走 `--file-url` / `--app-id` 这一对——它们的优先级高于 JSON。实时 websocket 只有 SDK |
| `dashscope decision create` | 针对一份 `--state`（自由文本、一段对话，或 JSON）回答**带类型**的问题：`--questions` 为每个 id 声明 `choice` / `noul` / `score`，每个答案回一行并带上它的置信度。模型不生成文本 |
| `dashscope agentstudio agents create` / `get` / `list` / `versions` / `update` / `archive` | 管理一个**托管 agent**：`--name`、`-m`、`--system-prompt`，以及作为 JSON 数组的 `--tools` / `--mcp-servers` / `--skills`。`versions` 列出 `get --version` 和 `update --version` 要填的版本号。`update` 必须给 `--version`——版本对不上就是 409。`archive` 是**破坏性**的 |
| `dashscope agentstudio sessions create` / `get` / `list` / `update` / `archive` / `delete` | 在某个 agent 上开一个会话（`--agent`、`--environment-id`、`--vault-ids`、`--env`、`--mcp-configs`）并查看它。`update` 打补丁，没给的字段保持原值；它的 `--agent-config` 是会话级的 agent 覆盖项（`system` / `tools` / `mcp_servers` / `skills`），**不是** agent id。`archive` 和 `delete` 都是**破坏性**的 |
| `dashscope agentstudio sessions events send` | 投递一轮对话：`--message` 会被替你包成一条 user 消息，`--events` 原样发送一个 JSON 数组且优先级更高，`--thread-id` 指定线程 |
| `dashscope agentstudio sessions events list` | 把会话已持久化的事件读回来：`--type`（可重复）、`--order`、`--limit`、`--page` |
| `dashscope agentstudio sessions events stream` | 跟随这次运行直到落定：table 模式打印答案文本，`--deltas message` 选择接收增量帧，`-o json` 一行一个事件 |
| `dashscope config set` | 往 `~/.dashscope/config.toml` 写一个键 |
| `dashscope config get` | 读回一个键（api_key 默认打码，除非 `--reveal`） |
| `dashscope config list` | 每个键、它的生效值，以及这个值来自哪一层 |

一个 `dashscope` 入口同时服务两代命令。1.x 专有命令组（`generation`、`application`、`image-synthesis`、`multimodal-conversation`、`speech-synthesis`、`transcription`、`video-synthesis`、`image-generation`、`multimodal-embedding`）和 Typer 之前的点式拼写保留原始 argv，由冻结的 1.x 实现执行。`ft`/`fine-tunes`、`files`、`deployments`、`models`、`rl`/`agentic-rl`、`auth`、`oss` 等共享命令组由 2.0 执行；兼容层接受旧参数拼写，并在默认值不同时保留旧默认。新的 2.0 命令按 SDK 资源树命名，因此应用调用是 `apps completions create`，不是简化的 `apps call`。不带子命令的裸 `dashscope` 会交互式打开内置的 `expert` agent。

这里*没有*的东西是一个决定，而不是一堆待办，而且这些空缺长得都一样：一条长连接的双向 socket，或者一长串 shell 加不出任何价值的 CRUD。`audio realtime run` 驱动双工 Omni socket 的**一轮**然后挂断；真正的实时麦克风——它跨多轮保持打开，并且要求音频以实时速率到达——是 Python 里的 `client.audio.realtime.connect()`，`tingwu realtime` 也是同一个故事。`agentstudio` 覆盖了你真会拿去写脚本的那个回路（agents、sessions、session events）；它另外九个顶层资源（deployments 及其 runs、environments、files、skills、vaults、memory stores、webhook endpoints、security），加上 session threads 和 session resources，仍然只有 SDK。SDK 暴露的其余东西都有命令了。

### 输出格式

`-o table`（默认）给人看，`-o json` 和 `-o yaml` 给机器看。这个切分是刻意的，并且在三个地方被强制执行：

| 行为 | table | json | yaml |
| --- | --- | --- | --- |
| 渲染方式 | Rich，会折行会上色 | stdout 上 `json.dumps(indent=2, ensure_ascii=False)` | stdout 上 `yaml.safe_dump(allow_unicode=True, sort_keys=False)` |
| 回答文本 | stdout | 在文档里 | 在文档里 |
| `reasoning_content`、`usage:`、分页信息 | **只走 stderr** | 是文档的一部分 | 是文档的一部分 |
| 流式 | 增量文本写 stdout | **NDJSON** | **NDJSON**（不是 YAML 文档） |
| 传输层 `headers` | 永不出现 | 剥掉，除非 `-v` | 剥掉，除非 `-v` |

有两个后果值得内化：

1. table 模式下 `dashscope chat create ... > answer.txt` 只会捕获回答本身。推理过程和 token 计数走 stderr，所以重定向永远不会捞到装饰性内容。
2. `--stream -o yaml` 是个阅读陷阱：它每行吐一个 **JSON** 对象，和 `--stream -o json` 一模一样。流式输出天然是按行的，而多文档 YAML 流对 `while read` 极不友好。

管道配方：

```bash
dashscope -o json chat create -m qwen-plus --prompt "hi" | jq -r '.choices[0].message.content'
dashscope -o json chat create -m qwen-plus --prompt "hi" | jq -r '.usage.total_tokens'
dashscope -o json ft list --size 50 | jq -r '.jobs[] | "\(.job_id)\t\(.status)"'
dashscope -o json chat create -m qwen-plus --prompt "hi" --stream \
  | jq -j '.choices[0].delta.content // empty'
```

需要原始 HTTP 响应头（提工单用）和失败堆栈时加 `-v` / `--verbose`。流水线里别加：`headers` 是传输元数据，不属于结果。

### 退出码与错误格式

退出码才是契约，不要去 grep 错误文本。

| 退出码 | 含义 | 可重试？ |
| --- | --- | --- |
| `0` | 成功 | — |
| `1` | `server`、`ratelimit`、`network`——服务或链路失败 | 可以，带退避。`ratelimit` 要尊重 `Retry-After` |
| `2` | `usage`——你的命令行、你的输入文件，或缺了 `--yes` | 不行，改调用方式 |
| `3` | `auth`——凭据被拒 | 不行，改密钥 |
| `4` | `notfound`——这个 id 不存在 | 不行，改 id |

每次失败都会在 stderr 上打印一行固定形状的内容，后面跟可选的 dim 上下文：

```text
Error [ratelimit]: Rate limit exceeded.
  request_id: 7574ee8f-38a3-4b1e-9280-11c33ab46e51
  solution: Reduce your request frequency.
```

`Error [category]: message` 是稳定的；两行 dim 只在有内容可说时才出现。`request_id` 是提工单需要的东西，`solution` 是注册表自带的指引——单独一行，这样你永远能分清哪些字来自网关。

HTTP 状态码如何变成 category（`cli/errors.py` 的 `_STATUS_CATEGORY`）：

| 状态码 | category | 退出码 |
| --- | --- | --- |
| 400、413 | `usage` | 2 |
| 401、403 | `auth` | 3 |
| 404 | `notfound` | 4 |
| 429 | `ratelimit` | 1 |
| 500、503、504 | `server` | 1 |

SDK 侧的失败没有 HTTP 状态码，所以按注册表 code 归类：`sdk.ConnectionError` 和 `sdk.Timeout` → `network`（1），`sdk.InvalidApiKey` → `auth`（3），`sdk.IoError` → `usage`（2，因为「你指名的文件读不了」是你的错）。无法归因的一律落到 `server`。

shell 里的重试封装：

```bash
attempt=1
until dashscope -o json chat create -m qwen-plus --prompt "hi" > out.json; do
  code=$?
  case $code in
    1) [ $attempt -ge 5 ] && exit 1
       sleep $(( attempt * 2 )); attempt=$(( attempt + 1 )) ;;
    *) exit $code ;;   # 2 / 3 / 4 不会自己好
  esac
done
```

`cli_ci_harness.py` 的场景 2 是同一段逻辑的 Python 版，配一个「第一次限流、第二次放行」的 mock 网关，让你亲眼看它恢复。

### 配置

优先级，从高到低：

```text
命令行 flag  >  -c KEY=VALUE  >  环境变量  >  配置文件  >  内置默认值
```

所以 `-k sk-a -c api_key=sk-b` 用的是 `sk-a`：专用 flag 是更具体的写法，永远赢。

配置文件是 `~/.dashscope/config.toml`。用 `DASHSCOPE_CONFIG` 重定向它——这就是给单个 CI 作业一份独立配置、又不碰任何人 home 目录的办法：

```bash
DASHSCOPE_CONFIG=/tmp/job-42/config.toml dashscope config set default_model qwen-plus
```

顶层键都是 SDK 键，原样传给 `Client(...)`：`api_key`、`protocol`、`base_url`、`workspace`、`region`、`timeout`、`max_retries`、`http_max_connections`、`http_max_keepalive_connections`、`http_keepalive_expiry`。`[cli]` 表里只放两个 CLI 专用默认值：`output` 和 `default_model`。写成 `cli.output` / `cli.default_model` 可以，直接写也行——`config set output json` 知道该放哪。

```toml
api_key = "sk-xxx"
protocol = "native"

[cli]
output = "json"
default_model = "qwen-plus"
```

三个实践上真正要紧的细节：

- `config set api_key` **绝不回显值**，并且密钥落盘的那一刻文件就被 chmod 成 `0600`。终端录制和构建日志都抓不到你刚敲进去的东西。
- `config get api_key` 会打码：前 4 位 + 星号 + 后 2 位（`sk-l*******mo`）。`--reveal` 打印完整值，留给你确实需要核对的那一次。
- `config get` 读的是**文件本身**，`config list` 报告的是**生效值**加上赢的那一层。两者可能不一致——而这个不一致*就是*优先级这一课，`cli_ci_harness.py` 的场景 4 会把它讲明白，而不是让它看起来像个 bug。

`-c` 会被立刻校验，在任何命令上都一样，包括根本不联网的命令：

```bash
dashscope -c defualt_model=qwen-max version
# Error [usage]: unknown configuration key 'defualt_model'; known keys: api_key, base_url, ...
# exit 2
```

这是刻意的。流水线里的一个拼写错误应该在最便宜的命令上就暴露出来，而不是三个阶段之后变成一个「模型莫名其妙不对」的问题。

### 输入约定

每个文本选项都接受三种形态：

| 形态 | 含义 |
| --- | --- |
| `--prompt "text"` | 字面值 |
| `--prompt @path/to/file.txt` | 读文件（会展开 `~`）。`@` 形态能让长 payload 和工具 schema 不进 shell history、不出现在 `ps` 输出里 |
| `--prompt -` | 读 stdin 到 EOF |

三者都没给时，**管道进来的** stdin 会被自动拾取，所以下面这样是可行的：

```bash
cat question.txt | dashscope chat create -m qwen-plus
```

会话来源优先级：`--messages` > `--prompt`（+ `--system`）> 管道 stdin。三者都拿不到东西时，你会得到 exit 2 和 `no input given: pass --prompt, --messages, or pipe text on stdin`。

`--messages` 接受三种形态，并自己判断你给的是哪一种：

```bash
# 一个消息对象的 JSON 数组
dashscope chat create -m qwen-plus \
  --messages '[{"role":"user","content":"你好"},{"role":"assistant","content":"你好！"},{"role":"user","content":"继续"}]'

# JSONL——每行一个消息对象，用循环拼最省事
dashscope chat create -m qwen-plus --messages @turns.jsonl

# 一个带 messages 键的对象，也就是把抓下来的请求体原样回放
dashscope chat create -m qwen-plus --messages @last_request.json
```

JSON 写错会报位置，不会耸肩：`--messages: invalid JSON (line 1 column 23)`。

`--tools` 和 `--parameters` 同样是 JSON，同样接受 `@file`。`--tools` 必须是 **JSON 数组**，给它一个对象是 `usage` 错误（exit 2）。消息里的本地图片和视频路径会原样传给 SDK，由 SDK 自己上传——你不需要预先上传。

### 配方

可直接复制。标 🔑 的需要真实的 `DASHSCOPE_API_KEY`，其余完全离线。

**一次性问答存进变量** 🔑

```bash
answer=$(dashscope -o json chat create -m qwen-plus --prompt "用一句话介绍杭州" \
         | jq -r '.choices[0].message.content')
```

**边到边流式打到终端** 🔑

```bash
dashscope chat create -m qwen-plus --prompt "写一首关于春天的短诗" --stream
```

**长 prompt 走文件，绝不走 argv** 🔑

```bash
dashscope chat create -m qwen-plus --prompt @brief.md --system "你是资深编辑"
```

**工具定义走文件** 🔑

```bash
dashscope chat create -m qwen-plus --prompt @question.txt --tools @tools.json -o json
```

**循环里拼 JSONL 做多轮** 🔑

```bash
: > turns.jsonl
while read -r line; do
  printf '{"role":"user","content":"%s"}\n' "$line" >> turns.jsonl
done < questions.txt
dashscope chat create -m qwen-plus --messages @turns.jsonl
```

**只重试有可能自己好的** 🔑 ——见上面退出码一节的 shell 封装。

**微调并阻塞到尘埃落定** 🔑

```bash
dashscope ft create -m qwen-plus -t "$FILE_ID" --n-epochs 2 --wait --timeout 1800
```

`ft create -m` **不会**回落到 `default_model`。这是有意为之：chat 的默认模型不是微调基座，而静默微调错模型比报个错糟糕得多。

**跟随作业事件** 🔑

```bash
dashscope ft stream "$JOB_ID"
```

**抓取作业的训练日志** 🔑

```bash
dashscope ft logs "$JOB_ID" --offset 1 --lines 500 > train.log
```

**取到生成的视频，然后下载** 🔑

```bash
url=$(dashscope videos generate -m wan2.2-t2v-plus --prompt "海浪拍打礁石" --timeout 1800)
curl -o clip.mp4 "$url"
```

只提交、之后统一收结果的批量写法：

```bash
dashscope -o json videos jobs create -m wan2.2-t2v-plus --prompt @brief.md \
  | jq -r '.task_id' >> task_ids.txt
while read -r id; do dashscope videos jobs get "$id"; done < task_ids.txt
```

**一步提交，另一步阻塞等待** 🔑

```bash
# 第一步——提交并留下 id
dashscope -o json images jobs create -m wan2.2-t2i-flash --prompt @brief.md \
  | jq -r '.task_id' > task_id.txt

# 第二步——可能是另一台 runner，可能是一小时之后
dashscope images jobs wait "$(cat task_id.txt)" --timeout 1800
```

`wait` 收的是一个 **id**，不是作业对象，所以两步之间需要活下来的只有这一个字符串。任务失败或被取消时退出码为 1；`--timeout` 是它无法兑现的值时退出码为 2——`0` 和 `nan` 都算，因为 `nan` 与任何值比较都为假，轮询会永远转下去。`ft wait`、`videos jobs wait`、`embeddings jobs wait`、`rl jobs wait` 和 `audio asr wait` 都是同一个形状。

**语音合成到文件，或直接进管道** 🔑

```bash
dashscope audio tts synthesize "你好，世界" --out hello.mp3
dashscope audio tts synthesize @script.txt --voice longxiaochun | ffplay -i - -autoexit -nodisp
```

不给 `--out`，裸音频字节就写 stdout。给了 `--out`，音频进文件，`wrote N bytes to ...` 那行提示走 stderr——所以两种写法都不会污染重定向。

**转写一段录音并阻塞等结果** 🔑

```bash
dashscope audio asr transcribe "https://example.com/meeting.mp3" --timeout 1800
```

**在实时 socket 上跑一轮** 🔑

```bash
dashscope audio realtime run -m qwen3.8-omni-flash-realtime \
  --audio question.pcm --out reply.wav
```

答案文本走 stdout，输入转写和 `wrote N bytes` 那行提示走 stderr，语音回复进 `--out`。不给 `--out`，回复音频就被**丢掉**——这个命令不会只因为你忘了个参数就往终端里喷二进制。`-o json` 会把整轮变成一行一个事件的 NDJSON 轨迹，这一轮行为不对的时候你要的就是它。

**先建热词表，再拿它去识别** 🔑

```bash
vid=$(dashscope -o json audio vocabularies create \
        --prefix demo --target-model paraformer-realtime-v2 \
        --vocabulary @hotwords.json \
      | jq -r '.vocabulary_id')
dashscope audio asr recognize meeting.pcm -m paraformer-realtime-v2 --vocabulary-id "$vid"
```

注意 `-o json` 的位置：在 `audio` 之前，因为它是全局选项。在这个命令组里面，短选项已经被占掉了——`tts synthesize` 和 `realtime run` 都用 `-o` 表示 `--out`。

**调用百炼应用，并把会话续上** 🔑

```bash
dashscope -o json apps completions create --app-id "$APP_ID" --prompt "帮我查一下订单" \
  | jq -r '.session_id' > session.txt
dashscope apps completions create --app-id "$APP_ID" \
  --prompt "那第二个呢" --session-id "$(cat session.txt)"
```

**用 shell 驱动一个托管 agent** 🔑

```bash
sid=$(dashscope -o json agentstudio sessions create --agent "$AGENT_ID" \
      | jq -r '.id')
dashscope agentstudio sessions events send "$sid" --message "帮我查一下订单"
dashscope agentstudio sessions events stream "$sid" --deltas message
```

`events send` 投递这一轮，`events stream` 跟随这次运行直到落定。它们是两个命令而不是一个，因为流水线通常要投 N 轮再统一收——也因为一条中途断掉的流应该以非零退出，而不是把已经发出去的 send 一起吞掉。

**针对一张工单回答带类型的问题** 🔑

```bash
dashscope decision create --state @ticket.txt --questions @questions.json
```

`--questions` 为每个 id 声明一个 `type`——`choice` 选出一个选项，`noul` 是「是」的概率，`score` 是加权后的档位下标。每个答案回一行并带上置信度，`-o json` 给你 `.answers.<id>.choice` 用来分支。模型不生成散文，所以没有需要从句子里抠出来的东西。`-m` 默认是 `decision-model-preview`，并且**不**读 `default_model`：这是一个决策模型，不是 chat 模型。

**CI：破坏性命令需要 `--yes`** 🔑

```bash
dashscope ft delete "$JOB_ID" --yes
```

没有 `--yes` 又没有终端时，它以 exit 2 拒绝，而不是替你猜。

**按作业隔离配置** 🔑

```bash
export DASHSCOPE_CONFIG="$CI_JOB_DIR/config.toml"
dashscope config set default_model qwen-plus
dashscope config set api_key "$DASHSCOPE_API_KEY"
dashscope chat create --prompt "hi"     # 不传 -m：来自 default_model
```

**干净的日志**——不要颜色：

```bash
export NO_COLOR=1
```

`NO_COLOR=1` 和 `--no-color` 是一回事。Rich 本来就能识别管道，所以这一条主要是给那种「先分配伪终端再把它倒进日志」的 runner 用的。

### 运行示例

两个都离线、确定性，都不需要密钥。在仓库根目录、且检出可导入的情况下：

```bash
PYTHONPATH=. python examples/05_cli/cli_output_and_errors.py
PYTHONPATH=. python examples/05_cli/cli_ci_harness.py
```

（只有包没被 pip 安装时才需要 `PYTHONPATH=.`。装了的话 `python examples/05_cli/cli_output_and_errors.py` 就够了。harness 会为自己的子进程设好这个变量，所以两种情况下都能跑。）

两个都会在最后打印一段小结，并且只有全部检查通过才 exit `0`——`cli_output_and_errors.py` 会断言它跑的每条命令的退出码，`cli_ci_harness.py` 会断言全部七个场景。两者都不读也不写你真实的 `~/.dashscope/config.toml`：harness 把 `DASHSCOPE_CONFIG` 指向一个临时目录，退出时删掉。

要让 harness 的只读场景打真实网关：

```bash
CLI_HARNESS_LIVE=1 DASHSCOPE_API_KEY=sk-xxx python examples/05_cli/cli_ci_harness.py
```

Live 模式跳过所有故障注入场景，并且**绝不**运行 `ft cancel` 或 `ft delete`。只有你同时把 `DASHSCOPE_JOB_ID` 设成一个你自己拥有的作业时，它才会跑 `ft stream`——对真实账户猜一个 id，正是「示例删掉了重要东西」的标准死法。

### 常见注意事项

1. **全局选项不能后置。**`dashscope chat create --prompt "hi" -o json` 得到的是 exit 2，不是格式化好的回答。顶层 `--help` 里的那些选项，位置是相对于 app 的。
2. **`--stream -o yaml` 是 NDJSON，不是 YAML。**每行一个 JSON 对象，和 `-o json` 一样。YAML 解析器读不了它。
3. **破坏性命令在 CI 里会拒绝执行。**没有终端 + 没有 `--yes` = exit 2。你的流水线需要在每一个 `cancel`、`delete`、`scale` 和 `archive` 上加 `--yes`——`ft`、`files`、`embeddings jobs`、`images jobs`、`videos jobs`、`deployments`、`rl`、`audio vocabularies`/`phrases`/`voices`/`voice-design`、`agentstudio agents`/`sessions`——而且必须是显式加，因为重点就在于这个命令不肯替你做决定。
4. **`ft create -m` 永不回落到 `default_model`。**只有 `chat create` 和 `multimodal chat create` 会。源码注释写了原因：chat 的默认模型不是微调基座。
5. **`config get` 和 `config list` 可能互相矛盾。**`get` 读文件，`list` 解析各层。两者不一致时，`list` 的 `source` 列在告诉你是哪一层赢了——通常是你忘了自己 export 过的环境变量。
6. **table 模式下有用的元数据在 stderr 上。**`usage:`、推理文本和 `page x/y, total n` 是刻意不放 stdout 的，这样重定向才干净。想把它们当数据拿，就用 `-o json`，别去捕获 stderr。
7. **`audio tts synthesize` 不带 `--out` 时，往 stdout 写的是裸字节。**这正是 `> hello.mp3` 能用的原因，也是忘了写 `--out` 时终端被二进制刷屏的原因。这条命令没有 table；`-o json` 不会改变音频写到哪里。
8. **`apps completions create --stream` 会同时把服务端切成增量帧。**不切的话，每个事件都带着到目前为止的整段答案，终端上就是同一句话不断变长地重复。确实想要累积形式，就传 `--no-incremental-output`。
9. **`chat create` 已经会自动路由媒体，`multimodal chat create` 依然存在。**当会话必须打到多模态端点、但里面并没有媒体分片时用它——两条命令收同样的参数，渲染方式也一样。
10. **在 `audio` 里面，`-o` 是 `--out`，不是 `--output`。**`dashscope audio tts synthesize hi -o json` 会写一个叫 `json` 的文件然后 exit 0。格式选项是全局的，所以要放在命令组之前：`dashscope -o json audio ...`。这是整个 CLI 里唯一一处把同一个短选项用作另一个含义的地方，而且是有意为之——在这个命令组里，shell 配方真正要伸手去拿的就是 `--out`。
11. **`audio` 下面那几个注册表的分页方式不一致。**`vocabularies`、`voices` 和 `voice-design` 收**从 0 开始**的 `--page-index` 加 `--page-size`；`phrases` 是微调那一套形状，收**从 1 开始**的 `--page` 加 `--size`。这不是 CLI 的疏漏——它们背后的两个服务端本身就不一致，而这几条命令选择按各自服务端的说法来写，而不是发明一个会悄悄吞掉一页的翻译层。
12. **`agentstudio agents update` 不给 `--version` 就不会跑。**服务端在 agent 的当前版本上做乐观并发；漏了它，命令在发出任何调用之前就 exit 2。版本过期是一个 409，CLI 没有为它单独设一档，因此报成 exit 1 的 `Error [server]`——算可重试，但真管用的重试是「把 agent 重读一遍、取它新的 `version`、再 update」，而不是「睡一会儿然后把同样的东西再发一次」。
13. **`agentstudio sessions events stream` 不主动要增量帧时，打印的是整条消息。**table 模式读的是流的 `text_stream`，也就是重连之后能够重建的那一路；`--deltas message`（可重复，还有 `reasoning`）选择接收增量帧，并把命令切到 `text_deltas`。`-o json` 不受影响——两种情况下都是一行一个原始事件。
14. **`--timeout 0` 是错误，不是「查一次就返回」。**每一个 `wait` 命令和每一个 `--wait` 开关都会拒绝不是有限正秒数的超时值——退出码 2，且发生在任何 API 调用之前，所以即使流水线没有配 key，拿到的也是用法错误而不是鉴权错误。`0`、`-1`、`nan` 和 `inf` 全都被拒。其中真正危险的是 `nan`：它与任何值比较都为假，SDK 的截止时间判断永远不会触发，轮询会一直转下去而不是超时退出。想不阻塞地看当前状态，用 `get`。

SDK 侧对应的 mock 网关示例见 [`02_advanced_engineering/local_mock_debugging.py`](./02_advanced_engineering/local_mock_debugging.py)。

## legacy_1x —— 1.x 模块级 API（仅供参考）

以下使用旧的 1.x 风格（`dashscope.Generation`、`dashscope.MultiModalConversation` 等）。内部子目录布局原样保留、内容不做改动。新项目请使用上面的 2.0 分层示例。

**1.x CLI：** 现有 1.x 命令形式与 [`05_cli/`](./05_cli/) 中的 2.0 命令共用 `dashscope` 入口。1.x 专有名称仍由冻结的 1.x 实现执行，共享名称由 2.0 执行并兼容旧参数拼写。

| 路径 | 内容 |
| --- | --- |
| `generation/` | Generation 文本生成：基础、深度思考、工具调用、深度研究、自定义会话 |
| `aio_generation/` | 上述的异步版本 |
| `multimodal_conversation/` | MultiModalConversation 多模态对话：图像、视频、工具调用、OCR、ASR、深度思考 |
| `aio_multimodal_conversation/` | 上述的异步版本 |
| `application.py` | 百炼应用（Application）调用 |
| `assistant_api*.py` | Assistant 智能体：创建 / 更新、文件挂载、端到端对话 |
| `image_generation.py` / `image_synthesis.py` / `aio_image_synthesis.py` | 图像生成（wanx / wan2.x） |
| `video_synthesis.py` / `aio_video_synthesis.py` | 视频生成（wan 系列） |
| `multimodal_embedding.py` | 多模态向量 |
| `text_rerank.py` | 文档重排 |
| `qwen_tts.py` / `qwen_asr.py` | Qwen 语音合成与识别 |
| `tingwu_realtime.py` / `tingwu_usages.py` | 听悟实时与离线任务 |
| `cv_models.py` | CV 模型（人像风格重绘，含本地上传） |
| `set_region_example.py` | 多地域（region）配置 |
