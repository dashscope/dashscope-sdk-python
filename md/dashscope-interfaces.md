# dashscope 全量接口清单（SDK / 自带 CLI / Agentic）

每个接口一句话说明，含主要输入。分三部分：dashscope SDK（Python）、dashscope 自带 CLI、Agentic（SDK + acli CLI）。

---

# 一、dashscope SDK（Python）

## 文本 / 对话

| 接口 | 说明 |
| --- | --- |
| `Generation.call(model, messages, **kw)` | 文本生成/对话，输入模型名+消息列表，支持 stream、tools、temperature |
| `AioGeneration.call(...)` | `Generation` 的 asyncio 异步版 |
| `Conversation.call(model, prompt, history)` | 老式对话接口，history 为 `History`/`HistoryItem` |
| `CodeGeneration.call(model, messages)` | 代码生成，输入同 messages 格式 |
| `Understanding.call(model, prompt)` | 语义理解，输入模型+单句文本 |
| `Application.call(api_key, app_id, prompt, **kw)` | 百炼应用调用，输入应用 ID+用户提示 |

## 多模态

| 接口 | 说明 |
| --- | --- |
| `MultiModalConversation.call(model, messages)` | 多模态对话，messages 元素含 text/image/audio/video URL |
| `AioMultiModalConversation.call(...)` | 异步版 |
| `ImageSynthesis.call/async_call/fetch/wait(model, prompt, n, size)` | 文生图（通义万相），异步任务四件套 |
| `VideoSynthesis.call/async_call/fetch/wait(model, prompt/img_url)` | 文/图生视频 |
| `MultiModalEmbedding.call(model, input=[...])` | 多模态向量化，input 为 ItemText/ItemImage/ItemAudio 列表 |
| `AioMultiModalEmbedding.call(...)` | 异步版 |

## 向量 / 排序 / 分词

| 接口 | 说明 |
| --- | --- |
| `TextEmbedding.call(model, texts)` | 文本批量向量化，输入文本列表 |
| `BatchTextEmbedding`（call/fetch/wait） | 异步批量 embedding 任务 |
| `TextReRank.call(model, query, documents)` | 语义重排，输入查询+文档列表 |
| `AioTextReRank.call(...)` | 异步版 |
| `Tokenization.call(model, prompt)` | 统计 prompt 的 token 数 |

## 语音

| 接口 | 说明 |
| --- | --- |
| `Recognition.start()/send_audio_frame(bytes)/call(file)/stop()` | Paraformer 实时语音识别（websocket 流式） |
| `Transcription.async_call/fetch/wait(model, file_urls)` | 录音文件批量转写，输入音频 URL 列表 |
| `audio.qwen_asr.QwenTranscription` | Qwen 音频转写 |
| `SpeechSynthesizer.call(model, text, sample_rate)` | sambert/cosyvoice 语音合成 |
| `audio.tts_v2.SpeechSynthesizer.call/streaming_call/streaming_complete/streaming_cancel` | CosyVoice2 流式合成，输入文本流 |
| `HttpSpeechSynthesizer.call(model, text)` | HTTP 一次性合成 |
| `audio.qwen_tts.SpeechSynthesizer` | Qwen3-TTS 合成 |
| `audio.qwen_omni.OmniRealtimeConversation` | Omni 实时多模态对话（websocket 回调式） |

## 微调 / 部署 / 资源

| 接口 | 说明 |
| --- | --- |
| `FineTunes.call/cancel/list/get/delete/stream_events/logs/wait` | 微调任务管理，输入训练/验证文件 ID+基座模型 |
| `Deployments.call/list/get/delete/scale` | 微调模型部署与扩缩容，输入 model_name+capacity |
| `Models.list/get(page_no, page_size)` | 查询可用模型列表/详情 |
| `Files.upload(file_path, purpose)/list/get/delete` | 文件上传（微调用）及管理 |

## Assistants API（百炼智能体）【确定下线：2.0 不再支持】

> ⚠️ Assistants/Threads/Messages/Runs/Steps 整个 API 族将下线，dashscope 2.0 不再迁移支持；托管 Agent 场景由 AgentStudio 承接。

| 接口 | 说明 |
| --- | --- |
| `Assistants.create/retrieve/list/update/delete` | 助手 CRUD，输入 name/model/instructions/tools |
| `Threads.create/retrieve/update/delete/create_and_run` | 会话线程 CRUD |
| `Messages.create/retrieve/list/update/delete` | 线程内消息 CRUD |
| `Runs.create/create_thread_and_run/retrieve/list/submit_tool_outputs/cancel/wait` | 运行调度与工具结果回填 |
| `Steps.list/retrieve` | 查询运行步骤 |

## AgentStudio（托管 Agent，`dashscope.agentstudio`）

独立子 SDK，已是 Client 实例化风格（`Client`/`AsyncClient` + 自有异常体系 + 流式）。

| 接口 | 说明 |
| --- | --- |
| `client.agents.create(name, model, ...)` | 创建托管 Agent |
| `client.sessions.create(agent=agent.id)` | 创建会话 |
| `client.sessions.events.send(session.id, [...])` | 发送事件（`user_message` / `user_tool_result` 等） |
| `client.sessions.events.stream(session.id)` | 流式订阅会话事件（上下文管理器，含 `session_status` 等事件类型） |

## 全局配置

| 接口 | 说明 |
| --- | --- |
| `dashscope.api_key` / `save_api_key()` | 密钥设置与持久化 |
| `base_http_api_url` / `base_websocket_api_url` / `base_compatible_api_url` | 端点配置 |
| `close_shared_aio_session()` | 关闭共享异步 HTTP 会话 |

---

# 二、dashscope 自带 CLI（`dashscope` 控制台脚本）

Typer 实现，全局选项 `-k/--api-key <key>`。

## 生成类

| 命令 | 说明 |
| --- | --- |
| `dashscope generation create --model <m> --prompt <text>` | 文本生成，支持流式与采样参数 |
| `dashscope multimodal_conversation create` | 多模态对话，输入含图片/音频/视频 URL 的消息 |
| `dashscope image_synthesis create` | 文生图（万相），输入 prompt+尺寸/数量 |
| `dashscope image_generation create` | 图片生成（另一组模型） |
| `dashscope video_synthesis create` | 文/图生视频，输入 prompt 或首帧图 |
| `dashscope speech_synthesis create` | 语音合成，输入文本+发音人 |
| `dashscope transcription create` | 录音转写，输入音频 URL |
| `dashscope code_generation create` | 代码生成（hidden 别名 call） |
| `dashscope understanding create` | 语义理解（hidden） |
| `dashscope application create` | 百炼应用调用，输入 app_id+prompt |

## 检索 / 工具类

| 命令 | 说明 |
| --- | --- |
| `dashscope embeddings create` | 文本向量化，输入文本列表 |
| `dashscope multimodal_embedding create` | 多模态向量化 |
| `dashscope rerank create` | 重排，输入 query+文档列表 |
| `dashscope tokenization create` | token 计数，输入 prompt |
| `dashscope models list\|get` | 模型列表/详情 |
| `dashscope files upload\|get\|list\|delete` | 文件上传与管理，upload 输入本地路径+purpose |
| `dashscope oss upload` | 上传本地文件到 OSS 换取可访问 URL |

## 微调 / 部署 / 资源

| 命令 | 说明 |
| --- | --- |
| `dashscope ft create` | 创建微调任务，输入训练/验证文件 ID+基座模型 |
| `dashscope ft get\|list\|stream\|cancel\|delete` | 微调任务查询/列表/事件流/取消/删除 |
| `dashscope deployments create` | 部署微调模型，输入 model_name+capacity |
| `dashscope deployments get\|list\|scale\|delete` | 部署查询/列表/扩缩容/删除 |

（`fine-tunes` 是 `ft` 的隐藏别名）

## Agentic RL 微调（`dashscope rl`）

| 命令 | 说明 |
| --- | --- |
| `dashscope rl register_functions --functions <json>` | 注册 RL 自定义奖励/校验函数 |
| `dashscope rl test_functions` | 测试已注册函数 |
| `dashscope rl upload_data --file <jsonl>` | 上传 RL 训练数据集 |
| `dashscope rl run --model <m> --dataset <id> ...` | 启动 Agentic RL 训练任务 |
| `dashscope rl get\|cancel\|logs` | 任务查询/取消/日志（输入 job_id） |
| `dashscope rl list` | 列出 RL 训练任务 |

（`agentic-rl` 是 `rl` 的隐藏别名）

---

# 三、Agentic（dashscope.acli）

## Agentic SDK（`dashscope.acli.sdk`，包根直接导出）

| 接口 | 说明 |
| --- | --- |
| `load_config(provider, model, api_key, base_url)` | 加载 acli 配置，入参仅补缺省值（workspace/全局配置与环境变量优先级不变） |
| `create_agent(config, system_prompt, session_id)` | 创建可编程 Agent；session_id 指定时会话持久化到 `.acli/session/<id>/history.json` |
| `run_stream(prompt, config, system_prompt, session_id)` | 异步流式跑一轮 agent（含工具调用），yield 文本块 |
| `run_once(prompt, ...)` | 异步单轮执行，返回完整回复文本 |
| `run_once_sync(prompt, ...)` | `run_once` 的同步包装，适合脚本/CI |
| `run_interactive(system_prompt, app_name, default_model, default_provider, api_key, base_url, command, tui, ...)` | 启动交互式 REPL/TUI（acli 的程序化入口） |

## acli CLI — 启动参数

| 参数 | 说明 |
| --- | --- |
| `dashscope [--tui\|--cli\|--dry-run]` | 交互模式选择：TUI / 纯 CLI / 只打印不执行 |
| `-c/--command <cmd>` | 一次性执行命令后退出 |
| `--protocol <p>` | 协议覆盖；`--max-turns N` 限制 agent 轮数 |
| `--version` / `-h` | 版本 / 帮助 |
| `dashscope example ...` | 运行内置示例项目 |

## acli CLI — 会话与上下文

| 命令 | 说明 |
| --- | --- |
| `/session new\|list\|switch\|rename\|fork\|remove\|scene` | 多会话管理；fork 复制会话，scene 管理场景记忆 |
| `/clear` | 清空当前对话历史 |
| `/compress` | LLM 摘要压缩上下文 |
| `/summarize` | 总结当前任务并记录关键步骤 |
| `/history stats\|list\|search\|export\|clear` | 历史统计/全文搜索/导出 |
| `/memory list\|search\|remove\|clear` | 长期记忆条目管理 |
| `/profile list\|search\|add\|remove\|clear` | 用户画像管理 |

## acli CLI — 模型与能力

| 命令 | 说明 |
| --- | --- |
| `/provider` / `/model` | 切换模型服务商/模型 |
| `/capability` | 能力开关目录 |
| `/skill` | 技能管理 |
| `/subagents list\|reload\|enable\|disable\|config` | 子代理管理 |
| `/mcp list\|add\|remove` | MCP 服务连接（stdio 经 config.toml 配置） |
| `/trust` / `/rule` / `/directives` | 权限、规则、指令管理 |

## acli CLI — 输入输出

| 命令 | 说明 |
| --- | --- |
| `/voice on\|off\|model\|silence\|max\|threshold` | 语音输入配置 |
| `/tts on\|off\|speed\|voice\|say\|last` | 语音播报（CosyVoice） |
| `/camera capture\|record` | 拍照/录像输入 |
| `/copy` / `/save [path]` | 复制/保存最近一条回复 |
| `/undo` | 撤销最近一次文件写入/删除（checkpoint） |
| `/theme` | 主题设置；`/json on\|off` JSON 输出 |
| `/log` / `/debug` / `/trace` | 日志与跟踪 |

## acli CLI — 系统与运维

| 命令 | 说明 |
| --- | --- |
| `/cron add\|list\|remove\|pause\|resume` | 定时任务 |
| `/audit recent\|query\|clear` | 审计日志 |
| `/report` | 生成 trace 性能报告 |
| `/setup` | 初始化配置 |
| `/dev` | 模型注册与扩展指南 |
| `/example [restore]` | 示例项目列表/恢复 |
| `/privacy` | 隐私设置 |
| `/info` / `/stats` / `/help` / `/exit` | 运行时信息/统计/帮助/退出 |
