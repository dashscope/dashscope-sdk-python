# DashScope SDK 示例 —— 百炼（Model Studio）生产最佳实践

> [English](README.md) | **中文**

这不是一份"API 调用参考"。在 AI 辅助开发的时代，任何人都能几秒钟生成一行 `call(model="xxx")`——本目录的价值在于**提前、正确地把一个百炼应用推向生产所需的那几百行"脏活代码"写好**：每个进阶文件都是一个微型的生产级 harness 蓝图，带有严格的工程防御，面向一个具体的开发或业务场景。全部示例基于 2.0 资源风格客户端 `dashscope.Client()`（与包内 `dashscope.v2` 一致）；旧版 1.x 示例完整归档在 `legacy_1x/` 下。

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
└── legacy_1x/                1.x 模块级 API 示例（dashscope.Generation 等），仅供参考
```

各层定位（一条从"速查"到"生产蓝图"的进阶路径）：

- **01 原子能力（速查种子）**：每个能力一个文件，最小但可运行。刻意保持简单——它是查询 wire-shape 的权威种子，也是喂给通义灵码 / Cursor 让它们生成正确代码的种子。每个文件末尾有"production springboard（生产化跳板）"页脚，指向加固该能力的 02/03/04 案例。
- **02 进阶工程**：把 SDK 包进真实生产关心的模式里（省钱、抗压、离线调试）。
- **03 Agent 配方**：多步骤、可组合的端到端 agent 流程。
- **04 真实场景**：不是"怎么调某个能力"，而是把一个完整业务端到端建成生产级 harness（护栏、上下文预算、容错、容灾、状态机等生产权衡）；每个案例配有自己的 `README.md`，讲清"为什么这样写"。
- **legacy_1x**：旧的 1.x 模块级风格，子目录布局原样保留，仅供参考；新项目请优先使用上面四层。

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
   python3 examples/01_atomic_capabilities/text_generation/chat.py
   ```

> **离线运行（无需 API key）**：`02_advanced_engineering/local_mock_debugging.py`、`02_advanced_engineering/circuit_breaker_retry.py`、`03_agent_recipes/mcp_server_demo/client_demo.py`、`04_real_world_scenarios/production_ready_rag_harness.py`、`04_real_world_scenarios/realtime_audio_meeting_harness.py` 和 `04_real_world_scenarios/ticket_triage_harness.py` 使用 mock transport / 脚本化接缝 / 子进程——完全离线且输出确定，适合先动手感受调用形态。

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

## legacy_1x —— 1.x 模块级 API（仅供参考）

以下使用旧的 1.x 风格（`dashscope.Generation`、`dashscope.MultiModalConversation` 等）。内部子目录布局原样保留、内容不做改动。新项目请使用上面的 2.0 分层示例。

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
