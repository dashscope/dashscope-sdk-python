# DashScope 2.0 SDK 命名评审

对每个接口的**调用语句**与**返回结构体类**做全量抽取，
并对照 `dashscope/v2/` 当前实现逐一核对（方法名、返回类型注解、参数名），评估命名的
一致性与客户友好度。抽取以代码为准（原始依据 `md/dashscope-2.0-sdk-interface.md` 已不在
仓库中）；接口清单见 [`dashscope-2.0-interfaces-mapping.md`](dashscope-2.0-interfaces-mapping.md)，
调用示例见 [`dashscope-2.0-interfaces-examples.md`](dashscope-2.0-interfaces-examples.md)。

核心结论：**`create` 的双重语义（同步推理 / 创建持久资源）不是问题**——OpenAI SDK 同样如此
（`chat.completions.create` 阻塞，`fine_tuning.jobs.create` 不阻塞），资源语义本身已消歧。
真正的问题是：

1. 除 `create` 之外还有 **11 个**表达"发起一次调用"的动词；
2. 返回类型存在 **4 套**互斥的后缀规范，且有 5 处裸 `dict` / `Any` 逃逸；
3. 同样是"跑一个长任务"，**阻塞语义各不相同**。

---

## 一、全量抽取表

| 章节 | 调用语句 | 返回类型 | 语义 | 动词 |
|---|---|---|---|---|
| 0 | `dashscope.chat(model, prompt)` | `ChatCompletion` | 同步推理 | chat |
| 1.1 | `client.chat.completions.create(...)` | `ChatCompletion` / `ChatCompletionChunk` | 同步推理 | **create** |
| 1.4 | `client.apps.call(...)` | `AppCompletion` | 同步推理 | **call** ⚠️ |
| 2.1 | `client.multimodal.chat.create(...)` | `ChatCompletion` | 同步推理 | **create** |
| 2.2 | `client.images.generate(...)` | `Job` | 一步式（阻塞） | **generate** ⚠️ |
| 2.2 | `client.images.jobs.create/fetch/wait` | `Job` | 任务生命周期 | create/fetch/wait |
| 2.3 | `client.videos.generate(...)` | `Job` | 一步式（阻塞） | **generate** ⚠️ |
| 2.3 | `client.videos.jobs.create/fetch/wait` | `Job` | 任务生命周期 | — |
| 2.4 | `client.embeddings.create_multimodal(...)` | `MultiModalEmbeddingResponse` | 同步推理 | **create_*** ⚠️ |
| 3.1 | `client.embeddings.create(...)` | `EmbeddingResponse` | 同步推理 | **create** |
| 3.2 | `client.embeddings.jobs.create/fetch/wait` | `Job` | 任务生命周期 | — |
| 3.3 | `client.rerank.create(...)` | `RerankResponse` | 同步推理 | **create** |
| 3.4 | `client.tokenization.count(...)` | `TokenCount` | 计算 | **count** |
| 4.1 | `client.audio.asr.recognition(...)` | `Recognition`（ctx manager） | 双工会话 | **名词** ⚠️ |
| 4.1 | `rec.send_audio_frame/stop/events` | `None` / `Iterator[dict]` | 会话内 | — |
| 4.2 | `client.audio.asr.transcription.transcribe(urls)` | `Job` | 一步式（阻塞） | **transcribe** ⚠️ |
| 4.2 | `client.audio.asr.transcription.jobs.create/fetch/wait` | `Job` | 任务生命周期 | — |
| 4.3 | `client.audio.tts.synthesize(text)` | `bytes` | 一步式 | **synthesize** ⚠️ |
| 4.3 | `client.audio.tts.stream(text)` | `Iterator[bytes]` | 流式 | **stream** ⚠️ |
| 4.4 | `client.audio.realtime.session(model=)` | `RealtimeSession`（ctx manager） | 双工会话 | **名词** ⚠️ |
| 4.4 | `rt.send_event/recv_event/events/close` | — | 会话内 | — |
| 5.1 | `client.fine_tunes.create(...)` | `FineTune` | 资源创建（不阻塞） | **create** |
| 5.1 | `client.fine_tunes.get/list/wait` | `FineTune` / `FineTuneList` | CRUD | — |
| 5.1 | `client.fine_tunes.cancel/delete` | `FineTuneCancel` / `FineTuneDelete` | CRUD | — |
| 5.1 | `client.fine_tunes.stream_events(job_id)` | `Iterator[FineTuneEvent]` | 流式 | **stream_*** ⚠️ |
| 5.1 | `client.rl.logs(ft.job_id)` | `RlLogs` | 跨资源 ⚠️ | **logs** |
| 5.2 | `client.deployments.create/get/list/delete` | `Deployment` / `DeploymentList` / `DeploymentDelete` | CRUD | — |
| 5.2 | `client.deployments.scale(model, 4)` | `Deployment` | CRUD | **scale** |
| 5.3 | `client.models.list/get` | `ModelList` / `ModelObject` | CRUD | — |
| 5.4 | `client.files.upload(file_path=)` | `FileUploadResponse` | CRUD | **upload** ⚠️ |
| 5.4 | `client.files.list/get/delete` | `FileList` / `FileObject` / `DeletionStatus` | CRUD | — |
| 6 | `client.rl.register_functions(...)` | `Dict[str, List[str]]` | 裸 dict ⚠️ | 动词_名词 |
| 6 | `client.rl.test_functions(...)` | `Any`（1.x 对象） | 类型逃逸 ⚠️ | 动词_名词 |
| 6 | `client.rl.upload_data(...)` | `Dict[str, List[str]]` | 裸 dict ⚠️ | 动词_名词 |
| 6 | `client.rl.run(...)` | `Any`（1.x 对象），**不阻塞** | 语义撞车 ⚠️ | **run** |
| 6 | `client.rl.get/list/cancel/logs` | `FineTune` / `FineTuneList` / `FineTuneCancel` / `RlLogs` | 复用微调 | — |
| 7 | `client.agentstudio.agents/sessions.create` 等 | 1.x 类型 | 懒挂载子 SDK | create |

---

## 二、诊断：方法名与返回类型（按严重度）

### P0-1　`Job.output` 是全 SDK 唯一的裸 dict 出口

`images` / `videos` / `transcription` / `embeddings.jobs` 四类任务共用 `Job`，但字段是
`output: Optional[Dict[str, Any]]`、`usage: Optional[Dict[str, Any]]`。客户必须背下每个服务的
字段名：

- `job.output["results"][0]["url"]`（images）
- `job.output["video_url"]`（videos）
- `job.output["results"][*]["transcription_url"]`（transcription）
- `done.output["url"]`（batch embeddings）

接口文档 §3.2 甚至只能写"字段名以服务返回为准，如 `url` 或 `results.url`"——这就是没有类型化
的直接后果。

同类问题：

| 位置 | 现状 | 对照 |
|---|---|---|
| `AppCompletion.usage` | `Optional[Dict[str, Any]]` | chat / embeddings / rerank / tokenization 的 usage 都是 typed model |
| `FileUploadResponse.data` | `Dict[str, Any]`，取 `up.data["file_id"]` | 同一资源的 `files.list` 返回 typed `FileObject` |
| `rl.register_functions` / `rl.upload_data` | `Dict[str, List[str]]` | — |
| `rl.run` / `rl.test_functions` | `Any`（1.x 原对象） | — |

**建议**：`Job` 保留 `output` 原始字典（服务端字段各异，不宜强类型化整体），但补 typed 视图与
便捷属性：`JobResult`（`url` / `transcription_url` / 其余透传）、`job.results`、`job.image_urls`、
`job.video_url`、`job.result_url`。`FileUploadResponse` 改为直接返回 `FileObject`（`up.file_id`）。

### P0-2　`generate` / `transcribe` 阻塞，`run` / `create` 不阻塞

`images.generate` = `jobs.create` + `jobs.wait`（阻塞到终态），`audio.asr.transcribe` 同理。
但 `rl.run` 文档写"启动完整训练流程"却**立即返回**，随后要 `rl.get(job.job_id).output.status`
自行轮询；`fine_tunes.create` 同样不阻塞。三个"跑长任务"的入口阻塞语义各不相同，是最容易踩的坑。

**建议**确立一条明规则并写进文档首页与每个方法的 docstring：

> `<res>.create()` 永不阻塞；`<res>.wait()` 才阻塞；裸领域动词（`generate` / `transcribe`）
> 是 `create + wait` 的语法糖，只在任务必然终结时提供。

按此规则：`rl.run` → **`rl.jobs.create`**，并补 `rl.jobs.wait(job)`。落地时把 `rl.get/list/cancel/logs`
一并收进 `rl.jobs`（`get` 随之统一为 `fetch`），`rl` 顶层只留 RL 独有的工作流步骤——否则每个任务
动词都有 `rl.x` 与 `rl.jobs.x` 两种写法，正是本章要消灭的歧义。

### P1-1　`apps.call` 是唯一残留的 1.x 动词

`AppCompletion` 这个类名已经与 `ChatCompletion` 平行，方法却叫 `call`。改成
**`client.apps.completions.create(app_id=..., prompt=...)`** 之后，"同步推理一律 `.create()`"
零例外，且类型名与方法名自洽。这是收益最高的一处改名。

### P1-2　`embeddings.create_multimodal` 应挂到 multimodal 命名空间

`_multimodal` 方法后缀不可扩展（下次加 `create_batch`？）。`client.multimodal` 命名空间已存在
（`multimodal.chat`），补 `multimodal.embeddings.create(...)` 即可与之完全对称。

另外 `MultiModalEmbedding`（输出元素）与 `MultiModalEmbeddingItem`（输入元素）名字过近；且输出
元素 `MultiModalEmbedding{index, type, embedding}` 与文本侧 `Embedding{text_index, type, embedding}`
字段几乎相同，**建议合并为一个 `Embedding` 类**、响应合并为 `EmbeddingResponse`。

### P1-3　`chat.completions.create` 与 `multimodal.chat.create` 是同一件事的两个入口

两者返回同一个 `ChatCompletion`，只是 native 端点不同。客户被迫先判断"我这次算不算多模态"。
建议 `chat.completions.create` 在检测到 `content` 为 parts 列表时自动路由到多模态端点，
`multimodal.chat.create` 降级为显式逃生口。

**落地**：端点常量从资源层移到协议层（`ChatProtocol.MULTIMODAL_PATH`，仅 native 适配器有值，
兼容协议在自己的 chat 路径上翻译 parts），`chat.completions.create` 依据
`types.messages.has_media_parts()` 自动路由；纯文本 parts 列表仍走文本端点。
`multimodal.chat.create` 保留为逃生口（强制走多模态端点，覆盖 qwen-vl 纯文本提问这类场景），
两者共用 `chat/completions.py` 里的 `_build_request` / `_send` / `_asend`，签名完全一致。

### P1-4　delete / cancel 结果有 4 个类名，实际只有 2 种形状

| 类 | 形状 | 用于 |
|---|---|---|
| `DeletionStatus` | `{request_id, deleted: bool}` | `files.delete` |
| `FineTuneDelete` | `{request_id, output: StatusResult}` | `fine_tunes.delete` |
| `FineTuneCancel` | `{request_id, output: StatusResult}` | `fine_tunes.cancel`、`rl.cancel` |
| `DeploymentDelete` | `{request_id, output: StatusResult}` | `deployments.delete` |

后三个结构完全一样。**建议**删掉 `FineTuneDelete` / `FineTuneCancel` / `DeploymentDelete`，
统一复用 `StatusResult`（改名 `JobStatus` 更达意）。

### P1-5　同一个"透传字典"有 6 个名字

| 参数名 | 出现位置 | 含义 |
|---|---|---|
| `parameters` | images / videos / `jobs.create` / asr | 服务端 `parameters` 段透传 |
| `extra_parameters` | apps / embeddings / rerank / tokenization | **同上** |
| `extra_input` | images / videos | 服务端 `input` 段透传 |
| `hyper_parameters` | fine_tunes | 服务端字段名 |
| `rag_options` | apps | 服务端字段名 |
| `protocol_options` | `Client` | SDK 自有，协议适配开关 |

前两个是同一个东西，**必须统一为 `parameters`**。其余四个语义各异，可保留。

### P1-6　`file_urls` vs `url`

`audio.asr.*.create(file_urls=[...])` 与 `embeddings.jobs.create(url=...)` 都是"待处理文件 URL"，
一复数一单数。

**修正**：核对 1.x 实现（`dashscope/embeddings/batch_text_embedding.py`，请求体为
`input = {"url": url}`）后确认，批量向量化端点**只接受一个文件**，原建议的 `file_urls: List[str]`
会造出一个服务端无法兑现的复数签名。落地为 **`file_url: str`**，并在 docstring 里点明与
`asr.transcribe(file_urls=)` 的差异。规则因此改为：单复数必须跟随端点的真实入参个数，
不做表面统一。

### P1-7　微调训练日志挂在 `rl` 资源面上

接口文档 §5.1 明写"训练日志（对应 1.x `FineTunes.logs`，2.0 移到 rl 资源面）"，于是纯微调用户
要写 `client.rl.logs(ft.job_id)`——完全无法发现。既然 `rl.get/list/cancel` 已复用 `FineTune*`
模型说明后端同源，应改为 `client.fine_tunes.logs(job_id)`；`RlLogs` 随之改名 `JobLogs`。

**落地**：`fine_tunes.logs(job_id)` 为唯一实现，`RlLogs` → `JobLogs`（字段 `logs` → `content`）。
不留 `rl.logs` alias——RL 用户经 P0-2 引入的 `rl.jobs.logs` 转发到同一实现，两条路径都是
资源树的正常形态，而非新旧并存的过渡写法。

### P2-1　会话方法用名词

`audio.asr.recognition(...)`、`audio.realtime.session(...)` 都是名词当方法名。
建议 `audio.asr.recognize(...)`、`audio.realtime.connect(...)`（后者与 OpenAI Realtime
`client.realtime.connect` 一致）。

### P2-2　层级深度 3~6 层不齐

| 深度 | 例 |
|---|---|
| 3 | `embeddings.create` |
| 4 | `multimodal.chat.create`、`audio.tts.synthesize`、`images.jobs.create` |
| 6 | `audio.asr.transcription.jobs.create` |

最深那条里 `transcription` 是冗余层（`asr` 已含转写义）。建议拍平为 `audio.asr.transcribe(...)`
+ `audio.asr.jobs.create/fetch/wait` + `audio.asr.recognize(...)`，全部收敛到 4~5 层。

### P2-3　`*Object` / `*Output` 两套 payload 后缀，且扁平化程度不一

- `ModelObject`、`FileObject`（models / files）
- `AppOutput`、`FineTuneOutput`、`DeploymentOutput`（apps / fine_tunes / deployments）——1.x
  `response.output` 信封遗留

导致取值深度不一：chat / embeddings / rerank / models 扁平（`resp.choices[0]`、`m.model`），
apps / fine_tunes / deployments 要穿 `output`（`dep.output.deployed_model`）。且便捷属性覆盖不全：
`FineTune.job_id` ✓、`AppCompletion.text` ✓、`Deployment` ✗。

**建议**：要么全部扁平化，要么给 `Deployment` 补 `dep.deployed_model` / `dep.status`，与另两个
对齐。`ModelObject` → `Model`。

### P2-4　必填主输入的位置 / 关键字风格不统一

允许位置传参：`tts.synthesize("你好", ...)`、`transcribe(urls, ...)`、`dashscope.chat(model, prompt)`、
`deployments.scale(model, 4)`。
强制 keyword-only：`images.generate(*, model, prompt)`、`videos.generate(*, ...)` 等。

建议除顶层快捷入口 `dashscope.chat()` 外统一 keyword-only（IDE 补全与重构更安全），
`scale(model, capacity=4)`。

### P2-5　流式有 4 种表达

| 形态 | 位置 | 评价 |
|---|---|---|
| `create(stream=True)` → `Iterator[Chunk]` | chat / multimodal | OpenAI 惯例，保留 |
| `synthesize()` → `bytes` vs `stream()` → `Iterator[bytes]` | audio.tts | 返回类型不同，不宜用 kwarg 合并，保留 |
| `stream_events(job_id)` → `Iterator[FineTuneEvent]` | fine_tunes | 唯一的 `stream_*` 前缀方法 |
| `rec.events()` / `rt.events()` → `Iterator[dict]` | websocket 会话 | 服务端事件 schema，dict 可接受 |

各有理由，但需在文档同一处集中说明选用规则，否则客户会以为漏了什么。

---

## 三、诊断：入参（按严重度）

以下基于对 `dashscope/v2/resources/**` 全部公开方法签名的逐条抽取（共 44 个方法）。

### P0-A　`timeout` 一词三义，且在相邻两层含义相反

| 签名 | `timeout` 的实际含义 |
|---|---|
| `Jobs.fetch(task_id, *, timeout=None)` | 单次 HTTP 请求超时 |
| `Jobs.create(*, ..., timeout=None)` | 单次 HTTP 请求超时 |
| `Jobs.wait(job, *, timeout=86400)` | **等待终态的总预算**（默认 24h） |
| `Images.generate(*, timeout=None, wait_timeout=86400)` | `timeout` = 提交那次 HTTP；`wait_timeout` = 轮询 |

于是 `client.images.generate(..., timeout=600)`——客户的直觉是"最多等 10 分钟"，实际语义是
"提交请求的 HTTP 超时 10 分钟"，而等待仍按 24h 走。同一个方法里 `timeout` 与 `wait_timeout`
并存，且 `wait(timeout=600)` 在下一层又变成了等待预算。**这是会直接导致误用的命名。**

建议：阻塞式一步法（`generate` / `transcribe`）中 `timeout` 表示整体等待预算，提交超时改名
`submit_timeout`（或走 `client.with_options(timeout=)`）；`Jobs.wait` 保持 `timeout`（上下文
已明确是等待）。

### P0-B　`task_id` vs `job_id`：两套任务主键

| 资源 | 主键属性 | 传参名 |
|---|---|---|
| images / videos / transcription / batch-embedding | `Job.task_id` | `jobs.fetch(task_id)` |
| fine_tunes / rl | `FineTune.job_id`（→ `output.job_id`） | `get(job_id)` / `cancel(job_id)` / `logs(job_id)` |

同一个 SDK 里"任务标识"有两个名字，客户从 images 走到 fine_tunes 必须重学一遍。
建议：`Job` 增 `job_id` 别名属性、`FineTune` 增 `task_id` 别名属性，文档统一推荐 `job_id`
（`task_id` 是服务端字段名，不宜直接改）。

### P1-C　分页参数 `page` vs `page_no`，默认值也不齐

| 方法 | 页码参数 | `page_size` 默认 |
|---|---|---|
| `files.list` | **`page`** ⚠️ | 10 |
| `deployments.list` | `page_no` | 10 |
| `fine_tunes.list` | `page_no` | 10 |
| `models.list` | `page_no` | **20** ⚠️ |
| `rl.jobs.list` | `page_no` | 10 |

`files.list(page=)` 是孤例，应统一为 `page_no`。`models.list` 的默认 20 若确为服务端默认，
需在 docstring 标注原因，否则对齐为 10。

### P1-D　资源标识符参数名有 7 种

`task_id` / `job_id` / `file_id` / `deployed_model` / `model_name` / `app_id` / `instance_id`。
其中两处需修：

- `models.get(model_name)` → **`model=`**：与 `ModelObject.model` 字段名、以及全 SDK 其余 40 处
  `model=` 都不一致。
- `rl.test_functions(instance_id, functype, input_data)`：全 SDK 唯一"三个连续必填位置参数"的
  方法（其余都是 keyword-only），且 `functype` 拼写不规范 → `func_type`，并改 keyword-only。

`file_id` / `app_id` / `deployed_model` 与服务端字段一致，保留。

### P1-E　`**kwargs` 转发吞掉了下一层的完整签名

| 位置 | 现状 | 下一层实际签名 |
|---|---|---|
| `Asr.recognition(**kwargs)` | 无任何可见参数 | `Recognition.__init__(*, model, format, sample_rate, parameters, ws_url, timeout)` |
| `Realtime.session(**kwargs)` | 同上 | `RealtimeSession.__init__(...)` |
| `ImageJobs.fetch(task_id, **kwargs)` | 只可见 `task_id` | `Jobs.fetch(task_id, *, timeout)` |
| `ImageJobs.wait(job, **kwargs)` | 只可见 `job` | `Jobs.wait(job, *, poll_interval, max_interval, timeout)` |
| `VideoJobs` / `EmbeddingJobs` / `TranscriptionJobs` 的 `fetch` / `wait` | 同上 | 同上 |

功能上是通的（`**kwargs` 会转发），但 IDE 补全、类型检查、`help()`、API 文档自动生成全部失效。
文档开头承诺"所有资源的每次调用都支持 `timeout=` 关键字参数"，而客户在工具层面**看不到**这个
参数——这正是 `**kwargs` 的代价。建议显式写出签名再转发。

### P1-F　`rl.*` 4 个方法确实没有 `timeout=`

`register_functions` / `test_functions` / `upload_data` / `run` 既无 `timeout` 也无 `**kwargs`，
与文档开头的全局承诺直接矛盾（这几个走 1.x 实现）。→ 补上，或修正文档表述为"除 `rl` 的
1.x 透传方法外"。

### P1-G　`apps.call` 22 个参数：3 个互斥输入 + 3 个透传字典

- 输入入口：`prompt` / `messages` / `history` 三者并存且**无互斥校验**（文档只说"prompt 与
  messages 二选一"）。`history` 是 1.x 遗留，与 §1.2"2.0 无状态、自行维护 messages"的设计
  原则冲突。
- 透传字典：`biz_params` / `rag_options` / `extra_parameters` 三个（其中 `extra_parameters`
  应并入 `parameters`，见 P1-5）。

→ 补互斥校验与清晰报错；`history` 标 deprecated 或移除；参数按"高频在前、低频收进
`parameters`"重排。

**落地**：`history` 直接移除——2.0 接口文档里从未出现它，1.x 的形状是 `{user, bot}` 而非
`role/content`，且与 §1.2"2.0 无状态、自行维护 messages"冲突。移除后只剩 `prompt` /
`messages`，两者组合语义明确（同时给出时 `prompt` 追加为最新一条 user 消息），
`_validate` 继续保证"二者至少给一个"。`extra_parameters` 已并入 `parameters`（见 P1-5）。

`biz_params` **不改名**：它是百炼应用侧 flow/plugin 入参的服务端公开字段，且落在 `input`
而非 `parameters` 段，改叫 `parameters` 会指向另一个位置。改为在 `create` docstring 中
写明它与 `parameters` 的分工。

### P1-H　流式返回类型不对称

| 入口 | `stream=True` 返回 | 有无便捷能力 |
|---|---|---|
| `chat.completions.create` / `multimodal.chat.create` | `ChatStream` | `text_stream` / `get_final_message()` / `close()` |
| `apps.call` | 裸 `Iterator[AppCompletion]` | 无 |

同样是流式，chat 给了包装类，apps 给裸迭代器。建议 apps 返回 `AppStream`（至少带
`text_stream`）。

### P2-I　`size="1024*1024"` 不接受 `1024x1024`

服务端用 `*` 分隔，OpenAI 用 `x`。从 OpenAI 迁移的客户会写 `size="1024x1024"`，且 SDK 不校验，
直接吃一个 400。→ 两种写法都接受并归一化。

### P2-J　孤例参数

| 参数 | 出现位置 | 问题 |
|---|---|---|
| `extra_headers` | 仅 `embeddings.create_multimodal` | 其余 40+ 方法都没有；要么全线补，要么删掉走 `with_options`。**已移除**：它唯一的真实用途是让服务端解析 `oss://` 私桶地址，现在 SDK 检测到 `oss://` 入参就自动带上 `X-DashScope-OssResourceResolve: enable`（与 1.x 行为一致），客户不必再知道这个头 |
| `connector` | `tts.synthesize/stream`、`Recognition` | 无类型注解，文档未提及。**已补**：`transport.ws.Connector = Callable[[str, Dict[str, str]], Any]` 公开导出，三处签名统一注解 |
| `sleep: Callable[[float], Any]` | `Jobs.wait` / `FineTunes.wait` | 测试注入钩子暴露在公开签名 → 改 `_sleep` 或经 `with_options`。**已改** `_sleep`（全线 7 个资源 + 6 个测试文件），docstring 点明它是替换轮询间隔的测试缝 |

### P2-K　`input` 的接受度不一致

- `embeddings.create(input: Union[str, List[str]])` ✓ 接受单个
- `chat.completions.create(messages: Union[str, List[MessageInput]])` ✓ 接受单个
- `embeddings.create_multimodal(input: List[...])` ✗ 必须列表

→ `create_multimodal` 也应接受单个 item / dict。

### P2-L　低频、不透明的服务端参数散在签名上

`res_level` / `enable_fusion` / `max_video_frames` / `fps` / `output_type` / `instruct` /
`doc_tag_codes` / `doc_reference_type` / `has_thoughts` / `memory_id` / `image_list` /
`file_list` / `biz_params`。

`res_level` 这类缩写没有任何自解释性。签名越长，`model=` / `prompt=` 这些高频参数在 IDE 补全
列表里越靠后、越难发现。建议：高频参数留在签名上，低频的收进 `parameters=`，并在 docstring
给出常用键清单。

---

## 四、诊断：输出字段（按严重度）

以下基于对 `dashscope/v2/**` 全部 pydantic 模型（38 个）字段的逐条抽取。
**字段名比类名更容易产生歧义**——类名客户只写一次，字段名要写在每一处取值代码里。

### P0-C　`gmt_create` / `gmt_modified`：阿里内部 DB 命名泄漏到公开 SDK

`DeploymentOutput` 的时间字段叫 `gmt_create` / `gmt_modified`。这是阿里内部数据库建表规范，
对外部客户完全不可理解（`gmt` 会被读成时区），且与同一 SDK 的其他时间字段全不一致：

| 模型 | 时间字段 |
|---|---|
| `DeploymentOutput` | **`gmt_create` / `gmt_modified`** ⚠️ |
| `FineTuneOutput` | `create_time` / `end_time` |
| `Job` | `submitted_time` / `scheduled_time` / `end_time` |
| `FileObject` | `upload_time` |

附带问题：`Job.submitted_time` 是从服务端 `submit_time` **主动改名**而来
（`from_payload` 里 `submitted_time=output.get("submit_time")`）。客户对着服务端文档 / 抓包
调试时会发现字段名对不上，这类"SDK 自作主张的改名"要么别做，要么在 docstring 标注。

建议全线统一为 `created_at` / `updated_at` / `submitted_at` / `finished_at`（ISO 8601 生态惯例，
OpenAI / Anthropic / Stripe 均如此），或退一步统一为 `create_time` / `modify_time` / `end_time`。

### P0-D　`finetuned_output`：产出与消费两端叫不同名字

微调产出的模型版本 ID，在产出侧叫 `FineTuneOutput.finetuned_output`，在消费侧叫
`deployments.create(version=...)`：

```python
done = client.fine_tunes.wait(ft)
version = done.output.finetuned_output          # "ft-20260827-xxxx"
dep = client.deployments.create(model="qwen-turbo", version=version)
```

`finetuned_output` 这个名字无法让人知道它是一个**模型版本标识**（"output" 在本 SDK 里到处是
响应信封的意思，见 P2-3）。同一个值两个名字，是客户必须靠读文档才能跨过的坎。

建议：`finetuned_output` → **`finetuned_model`**，`deployments.create(version=)` →
**`model_version=`**，两端对齐。同时 `DeploymentOutput.model_name` → **`model`**（与全 SDK
其余 40 处 `model` 一致；该模型里已有 `base_model`，`model_name` 反而更像"基座名"，歧义大）。

### P1-I　同一个 `usage` 字段有 4 个类 + 4 处退化成裸 dict / Any

| 模型 | `usage` 类型 | 字段 |
|---|---|---|
| `ChatCompletion` / `ChatCompletionChunk` | `Usage` | input_tokens / output_tokens / total_tokens |
| `EmbeddingResponse` | `EmbeddingUsage` | total_tokens |
| `RerankResponse` | `RerankUsage` | total_tokens |
| `TokenCount` | `TokenCountUsage` | input_tokens |
| `AppCompletion` | **`Dict[str, Any]`** ⚠️ | — |
| `MultiModalEmbeddingResponse` | **`Dict[str, Any]`** ⚠️ | — |
| `Job` | **`Dict[str, Any]`** ⚠️ | — |
| `FineTuneOutput` | **`Any`** ⚠️ | — |

4 个结构几乎相同的 Usage 类，4 处完全无类型。`EmbeddingUsage{total_tokens}` 与
`RerankUsage{total_tokens}` 是**逐字节相同的两个类**。

建议合并为单一 `Usage`（三个 token 字段都给默认值），全 SDK 统一使用；`AppCompletion.usage`
文档里明确是 `{"models": [...]}` 结构，应有独立的 `AppUsage{models: List[ModelUsage]}`。

### P1-J　"总条数"两个名字：`total` vs `total_count`

| 模型 | 字段 |
|---|---|
| `FileList` | `total_count` |
| `ModelList` | `total_count` |
| `DeploymentListOutput` | **`total`** ⚠️ |
| `FineTuneListOutput` | **`total`** ⚠️ |

同为分页列表的总条数，两个名字。统一为 `total_count`。

### P1-K　"序号"两个名字：`text_index` vs `index`

`Embedding.text_index`（文本向量）vs `MultiModalEmbedding.index`（多模态向量）——**同一个
`embeddings` 资源面下**，客户写 `emb.text_index` 和 `item.index`。文档 §3.1 与 §2.4 就分别
示范了这两种取法。其余 `Choice.index` / `StreamChoice.index` / `RerankResult.index` 都叫 `index`。

统一为 `index`（`text_index` 是服务端字段名，用 alias 承接即可）。这也印证 P1-2 的结论：
`Embedding` 与 `MultiModalEmbedding` 本就该是同一个类。

### P1-L　list 响应两种形状：扁平 vs 套 `output`

| 扁平（字段在顶层） | 套 `output` 信封 |
|---|---|
| `ChatCompletion.choices` | `AppCompletion.output.text` |
| `EmbeddingResponse.embeddings` | `FineTune.output.job_id` |
| `RerankResponse.results` | `Deployment.output.deployed_model` |
| `ModelList.models` / `.page_no` | `FineTuneList.output.jobs` / `.output.page_no` |
| `FileList.files` / `.page_no` | `DeploymentList.output.deployments` / `.output.page_no` |
| `TokenCount.usage` | `FileUploadResponse.data["file_id"]` |
| `RlLogs.logs` | `Job.output["results"]` |

**`FileList` 扁平而 `FineTuneList` / `DeploymentList` 套信封**——同样是 `*.list()` 的返回值，
取值深度差一层，这是最刺眼的对比。便捷属性也只补了一半：`FineTune.job_id` ✓、
`AppCompletion.text` ✓、`TokenCount.input_tokens` ✓、`ChatCompletion.output_text` ✓，
而 `Deployment` ✗、`FineTuneList` ✗、`DeploymentList` ✗、`FileUploadResponse` ✗。

建议二选一并全线执行：
- **方案 A（推荐）**：全部扁平化，`Deployment.deployed_model`、`FineTuneList.jobs`、
  `FileUploadResponse` 直接返回 `FileObject`。与服务端信封的映射放在 `_parse` 里。
- **方案 B**：保留 `output` 但给每个模型补齐便捷属性，做到"穿不穿信封都能取到"。

### P1-M　业务错误字段 `code` / `message` / `is_error` 只覆盖 3 / 38 个模型

| 有 | 无 |
|---|---|
| `ChatCompletion`（code / message / is_error） | `EmbeddingResponse`、`RerankResponse`、`TokenCount` |
| `AppCompletion`（code / message / is_error） | `ModelList`、`FileList`、`FileObject`、`RlLogs` |
| `Job`（code / message，**缺 is_error**） | `FineTune`、`Deployment`、`FineTuneList`、`DeploymentList` |

DashScope 的特点是 HTTP 200 也可能带业务错误码。现在客户在 chat / apps 上能看 `is_error`，
在 fine_tunes / deployments 上只能靠 `output.status == "FAILED"` 自己判断，在 embeddings /
rerank 上**完全无从判断**。`Job` 有 `code`/`message` 却没有 `is_error`，与另两个模型不齐。

建议抽一个 `ResponseBase`（`request_id` + `code` + `message` + `is_error`）让所有顶层响应模型
继承，`raise_on_error=False` 时行为全线一致。

### P2-M　其余字段级问题

| 字段 | 问题 | 建议 |
|---|---|---|
| `ChatCompletion.status_code` | 全 SDK 孤例，其他模型都没有 HTTP 状态码 | 移入 `ResponseBase` 或删除 |
| `ModelObject.opensource` | 非标准拼写，语义模糊（开源模型？开源许可？） | `open_source` |
| `RlLogs.logs: str` | 类名与字段名重复（`rl.logs(id).logs`），且看不出是 str 还是 list | `JobLogs.content` |
| `FineTuneEvent.output: Any` | 完全无类型 | 至少 `Dict[str, Any]`，理想是 typed event |
| `AppOutput.doc_references: List[Dict]` | RAG 引用来源是高频消费字段，却是裸 dict 列表 | typed `DocReference` |
| `AppOutput.thoughts: List[Dict]` | 同上 | typed `Thought` |
| `Job.task_id: str`（REQUIRED） | 类型上必填，但 `from_payload` 用 `or ""` 兜底，语义可空 | 改 `Optional[str]` 或真必填 |
| `Job.output` / `FileUploadResponse.data` | 裸 dict（见 P0-1） | typed |
| `MultiModalEmbeddingItem.factor` | 修饰项排在 4 个模态字段之前，读起来像第 5 个模态 | 移到最后 |
| `ImagePart.url` vs 文档示范的 `{"image": url}` | typed part 与 1.x dict part 两套表示法，文档只示范 dict 形式（而 embeddings 侧反而示范 typed） | 文档统一示范 typed，dict 作为兼容写法 |

---

## 五、建议规范

### 5.1 动词规范（收敛到 8 个）

| 语义 | 动词 | 阻塞 | 适用 |
|---|---|---|---|
| 同步推理 | `create` | 是 | chat / multimodal.chat / multimodal.embeddings / embeddings / rerank / **apps.completions** |
| 长任务提交 | `jobs.create` | 否 | images / videos / audio.asr / embeddings / fine_tunes / rl |
| 长任务查询·等待 | `jobs.fetch` / `jobs.wait` | 否 / 是 | 同上（现状已完全一致 ✓） |
| 一步式语法糖 | `generate` / `transcribe` | 是 | images / videos / audio.asr（= create + wait，需明写） |
| 持久资源 CRUD | `get` / `list` / `delete` / `cancel` / `scale` | 否 | fine_tunes / deployments / models / files（现状已一致 ✓） |
| 双工会话 | `recognize` / `connect` | ctx manager | audio.asr / audio.realtime |
| 纯计算·文件 | `count` / `upload` | 是 | tokenization / files |
| 流式 | `stream=True` 或 `.stream()` / `.events()` | — | 见 P2-5 |

### 5.2 入参规范

| 类别 | 规范 | 说明 |
|---|---|---|
| 模型 | 一律 `model: str` | 不允许 `model_name` 之类变体 |
| 任务主键 | 对外统一 `job_id`，服务端字段 `task_id` 保留为别名 | 见 P0-B |
| HTTP 超时 | `timeout: Optional[float]` | 单次请求；所有公开方法都必须有，且**显式写出**不用 `**kwargs`。唯一例外：`rl` 的 4 个 1.x 透传方法（见 P1-F） |
| 轮询预算 | `wait` 上下文里叫 `timeout`；阻塞式一步法里 `timeout` = 总预算，提交超时叫 `submit_timeout` | 见 P0-A |
| 服务端透传 | `parameters=`（parameters 段）/ `extra_input=`（input 段） | 消灭 `extra_parameters` |
| 分页 | `page_no` / `page_size`，默认值全线一致 | 消灭 `page=` |
| 文件 URL | `file_urls: List[str]`（多文件端点）/ `file_url: str`（单文件端点） | 消灭裸 `url=`；单复数跟随端点真实入参个数，见 P1-6 |
| 必填主输入 | keyword-only（顶层 `dashscope.chat()` 例外） | 见 P2-4 |
| 低频服务端参数 | 收进 `parameters=`，docstring 列常用键 | 见 P2-L |
| 测试注入钩子 | 不进公开签名（`_` 前缀或 `with_options`） | 见 P2-J 的 `sleep` |

### 5.3 输出字段规范

| 类别 | 规范 | 说明 |
|---|---|---|
| 时间戳 | `created_at` / `updated_at` / `submitted_at` / `finished_at` | 消灭 `gmt_*`、`*_time` 混用，见 P0-C |
| 用量 | 单一 `Usage` 类，全 SDK 复用 | 消灭 4 个同构 Usage 类与 4 处裸 dict，见 P1-I |
| 分页 | `page_no` / `page_size` / `total_count` | 消灭 `total`，见 P1-J |
| 序号 | 一律 `index` | 消灭 `text_index`，见 P1-K |
| 模型标识 | 一律 `model`；微调产物 `finetuned_model`；部署产物 `deployed_model` | 消灭 `model_name` / `finetuned_output` / `version`，见 P0-D |
| 任务标识 | 对外 `job_id`，`task_id` 为服务端别名 | 见 P0-B |
| 错误字段 | 所有顶层响应继承 `ResponseBase`（`request_id` / `code` / `message` / `is_error`） | 见 P1-M |
| 响应形状 | 一律扁平，服务端 `output` / `data` 信封在 `_parse` 里拆掉 | 见 P1-L 方案 A |
| 禁止 | 公开模型上不得出现裸 `Dict[str, Any]` / `Any` 字段（服务端原始 `Job.output` 除外，且必须另给 typed 视图） | 见 P0-1 |
| 禁止 | 不得出现阿里内部命名习惯（`gmt_*`、`is_deleted`、`biz_*`） | `biz_params` 见 P1-G |

---

## 六、改名清单

**加粗为破坏性变更**（需同步改接口文档与测试）。

> 前提：`dashscope.__version__` 当前为 `1.27.1`，2.0 尚未发布（只存在于 `V2` 分支），
> 因此**直接改名，不保留兼容 alias、不加 deprecation 垫片**。清单中少数标注"保留 alias"的
> 条目，指的是服务端字段名与 SDK 对外名的双向别名属性（如 `Job.job_id` ↔ `Job.task_id`），
> 而非新旧 API 的过渡兼容。

### 6.1 方法名 / 资源树

| 优先级 | 现状 | 目标 |
|---|---|---|
| P0 | `rl.run` 不阻塞，与 `generate`/`transcribe` 语义撞车 | **`rl.jobs.create`** + 新增 `rl.jobs.wait`；`rl.get/list/cancel/logs` 一并收进 **`rl.jobs`**（`get` → `fetch`） |
| P1 | `apps.call` | **`apps.completions.create`** |
| P1 | `embeddings.create_multimodal` | **`multimodal.embeddings.create`** |
| P1 | `chat.completions.create` 不识别 parts | parts 自动路由到多模态端点 |
| P1 | `rl.logs` 承载微调日志 | **`fine_tunes.logs`**（`rl.jobs.logs` 转发，不留 alias） |
| P2 | `audio.asr.recognition` | `recognize` |
| P2 | `audio.realtime.session` | `connect` |
| P2 | `audio.asr.transcription.*`（6 层） | 拍平到 `audio.asr.*`（4~5 层） |
| P2 | `rl.register_functions` / `test_functions` / `upload_data` | `rl.functions.register` / `.test` / `rl.data.upload` |
| P2 | `fine_tunes.stream_events` 与 agentstudio `sessions.events.stream` 两种形态 | 保留但在文档集中说明（见 P2-5） |

### 6.2 返回类型

| 优先级 | 现状 | 目标 |
|---|---|---|
| P0 | `Job.output` / `Job.usage` 裸 dict | 补 `JobResult` + `job.results` / `job.image_urls` / `job.video_url` / `job.result_url` |
| P0 | `rl.run` → `Any`（1.x 对象） | **`rl.jobs.create` → `FineTune`**：1.x dataclass 经 `_to_job` 适配为 v2 模型 |
| P0 | `rl.test_functions` → `Any` | **不改类型**：返回值就是客户自己 rollout/reward 函数的返回值，任何标注都是假的。以 docstring 明写这一事实 |
| P1 | `rl.register_functions` / `upload_data` → `Dict[str, List[str]]` | typed model |
| P1 | `FileUploadResponse.data` 裸 dict | 改返回 **`FileObject`**（`up.file_id`） |
| P1 | `FineTuneDelete` / `FineTuneCancel` / `DeploymentDelete` | 统一 **`JobStatus`**（原 `StatusResult`） |
| P1 | `MultiModalEmbedding` / `MultiModalEmbeddingResponse` | 合并入 **`Embedding` / `EmbeddingResponse`** |
| P1 | `AppCompletion.usage` 裸 dict | typed `Usage` |
| P1 | `apps.call(stream=True)` → 裸 `Iterator` | **`AppStream`**（带 `text_stream`），与 `ChatStream` 对齐 |
| P1 | `RlLogs` | **`JobLogs`** |
| P2 | `ModelObject` | `Model` |
| P2 | `Deployment` 无便捷属性 | 补 `dep.deployed_model` / `dep.status` |
| P2 | `MultiModal*` 大小写与 `multimodal` 命名空间不一致 | 统一 `Multimodal*` |

### 6.3 入参

| 优先级 | 现状 | 目标 |
|---|---|---|
| P0 | `generate`/`transcribe` 的 `timeout` = HTTP 超时、`wait_timeout` = 轮询 | `timeout` = 总预算，提交超时改 **`submit_timeout`** |
| P0 | `Job.task_id` vs `FineTune.job_id` | 互为别名属性（服务端字段名映射，非兼容垫片），文档统一推荐 `job_id` |
| P1 | `files.list(page=)` | **`page_no=`** |
| P1 | `models.list(page_size=20)` vs 其余 10 | 对齐或 docstring 说明 |
| P1 | `models.get(model_name=)` | **`model=`** |
| P1 | `rl.test_functions(instance_id, functype, input_data)` 位置参数 | keyword-only，`functype` → **`func_type`** |
| P1 | `Asr.recognition(**kwargs)` / `Realtime.session(**kwargs)` / 4 个 `jobs.fetch`/`wait(**kwargs)` | **显式写出签名**再转发 |
| P1 | `rl.register_functions` / `test_functions` / `upload_data` / `jobs.create` 无 `timeout=` | **不补**：1.x `AgenticRL` 本身不接受 timeout，补上就是静默失效的假参数。改为在 `reinforcement_learning` 模块 docstring 中显式声明例外，文档同步修正全局承诺 |
| P1 | `extra_parameters`（apps / embeddings / rerank / tokenization / chat） | **`parameters`** |
| P1 | `embeddings.jobs.create(url=)` | **`file_url=`**（单数，见 P1-6 修正） |
| P1 | `apps.call` 的 `prompt`/`messages`/`history` 无互斥校验 | **移除 `history`**（2.0 文档从未出现、1.x 形状是 `{user, bot}`、与无状态原则冲突）；剩下 `prompt`/`messages` 组合语义明确并保留"至少给一个"校验，见 P1-G |
| P2 | `videos.generate(img_url=)` | **已改** `image_url=`（内部仍写 `img_url` 上线） |
| P2 | `images.generate(size="1024*1024")` 不接受 `1024x1024` | **已改**：两种都接受，`_normalize_size` 归一化为服务端写法 |
| P2 | `extra_headers` 仅 `create_multimodal` 有 | **已移除**，改为检测到 `oss://` 入参自动带 `X-DashScope-OssResourceResolve`，见 P2-J |
| P2 | `connector` 无类型注解、未文档化 | **已补** `transport.ws.Connector` 公开别名并统一注解 |
| P2 | `Jobs.wait(sleep=...)` 测试钩子入公开签名 | **已改** `_sleep` |
| P2 | `create_multimodal(input=)` 不接受单个 item | **已改**：单个 dict / `EmbeddingItem` 均可 |
| P2 | `deployments.scale(model, 4)` 位置参数 | **已改** `scale(model, capacity=4)` keyword-only |
| P2 | `res_level` 等低频不透明参数散在签名 | **不收进 `parameters=`**：`res_level` / `max_video_frames` / `fps` 都是 1.x 已文档化的独立入参，且 `res_level` 就是服务端字段名，改名只会制造新的对不上。改为在 `multimodal.embeddings.create` docstring 中说明"仅快照模型可用"，其余未列名参数一律走 `parameters=` |

### 6.4 输出字段

| 优先级 | 现状 | 目标 |
|---|---|---|
| P0 | `DeploymentOutput.gmt_create` / `gmt_modified` | **`created_at` / `updated_at`** |
| P0 | `FineTuneOutput.create_time` / `end_time`、`Job.submitted_time` / `scheduled_time` / `end_time`、`FileObject.upload_time` | 统一 **`created_at` / `submitted_at` / `scheduled_at` / `finished_at` / `uploaded_at`** |
| P0 | `FineTuneOutput.finetuned_output` | **`finetuned_model`** |
| P0 | `Job.output` / `Job.usage` / `AppCompletion.usage` / `FineTuneOutput.usage` 裸 dict / Any | typed（见 6.2） |
| P1 | `deployments.create(version=)` | **`model_version=`**（与 `finetuned_model` 对齐） |
| P1 | `DeploymentOutput.model_name` | **`model`** |
| P1 | `Usage` / `EmbeddingUsage` / `RerankUsage` / `TokenCountUsage` 四个同构类 | 合并为单一 **`Usage`** |
| P1 | `DeploymentListOutput.total` / `FineTuneListOutput.total` | **`total_count`** |
| P1 | `Embedding.text_index` | **`index`** |
| P1 | `code` / `message` / `is_error` 仅 3 个模型有 | **已完成**：抽出 `ResponseBase`，全部顶层响应继承。`is_error` 改由 `ResponseBase` 的 after-validator 从非空 `code` **自动推导**（与 `errors.response_error` 同一判据），修掉了"非抛异常模式下除 chat/apps/fine_tunes 外全都漏标 `is_error`"的缺陷——不再有资源能忘记设置它。载荷是列表或 `data` 下单条、`flatten_envelope` 无从合并的地方，新增 `response_envelope(body)` 单独投递信封字段。三个例外（载荷里没有 HTTP 信封）：`FineTuneEvent`（SSE 帧，自带 `is_error`）、`RegisteredFunctions` / `UploadedData`（透传 1.x 返回值的记录）。规范落进 `md/dashscope-2.0-design.md` §2.6 |
| P1 | `AppCompletion.output.*` / `FineTune.output.*` / `Deployment.output.*` / `*List.output.*` / `FileUploadResponse.data` 信封 | **已完成全面扁平化**（P1-L 方案 A）。`File` / `Model` / `Deployment` / `FineTune` 四个条目类型同时改为继承 `ResponseBase`；files、models 两处解析器原先从 `data` / `output` 下的单条建模、信封到不了模型上，已用 `response_envelope()` 逐个穿进 `_parse_upload` / `_parse_list` / `_parse_file` / `_parse_delete` 与 `models.get`（含 async） |
| P2 | `ChatCompletion.status_code` 孤例 | **已移入** `ResponseBase`，全部顶层响应共用 |
| P2 | `ModelObject.opensource` | **已改** `Model.open_source`（服务端 `opensource` 保留为 validation alias） |
| P2 | `RlLogs.logs` | **已改** `JobLogs.content` |
| P2 | `FineTuneEvent.output: Any` | **已改** `Union[Dict[str, Any], str, None]`（SSE 帧不保证是 JSON） |
| P2 | `AppOutput.doc_references` / `thoughts` 裸 dict 列表 | **已改**：扁平化后为 `AppCompletion.doc_references: List[DocReference]` / `thoughts: List[Thought]` |
| P2 | `Job.task_id: str` 声明必填但实际可空 | **已对齐**：`task_id: str = ""`，与 `job_id` 属性的 `str` 返回类型一致 |
| P2 | `MultiModalEmbeddingItem.factor` 排在模态字段之前 | **已改**：类名 `EmbeddingItem`，`factor` 移到四个模态字段之后 |
| P2 | `Apps.call(biz_params=)` 内部命名 | **不改名，明确文档化**：它是百炼 flow/plugin 入参的服务端公开字段，且落在 `input` 段而非 `parameters` 段；`create` docstring 已写明两者分工，见 P1-G |

### 6.5 异常类名

**这一类名字不再由 SDK 决定。** 异常类名与 HTTP 状态一律从上游生成的错误目录 `dashscope/v2/_error_registry.py`（`PUBLIC_ERRORS` 16 条）派生：每条目录项的 `error_code` 就是类名、`status_code` 就是它的状态码。所以本节记录的不是"改成了什么名"，而是**哪些名从 SDK 手写变成目录派生**，以及派生之后暴露出的缺类。规范正文见 `md/dashscope-2.0-design.md` §2.7，逐条决策与验证数据见 `md/dashscope-2.0-errors-registry-plan.md`。

| 优先级 | 现状 | 目标 |
|---|---|---|
| P0 | 状态类的类名与 `status_code` 在 `errors.py` 里手写字面量，`_STATUS_MAP` 只有 5 项（400 / 401 / 403 / 404 / 429） | **已改**：9 个类全部从 `PUBLIC_ERRORS` 派生（`_CATALOG_STATUS` / `_STATUS_MAP`），目录改状态码 SDK 自动跟随，无需改代码 |
| P0 | 503 / 504 无类，与 500 一并落 `InternalServerError`（`ServiceUnavailable` / `GatewayTimeout` 两个线缆码也被压平） | **已补** `ServiceUnavailableError(503)` / `GatewayTimeoutError(504)`；目录里它们本就是两条独立项，压平是 SDK 侧丢信息 |
| P0 | 413 无类，落裸 `APIStatusError` | **已补** `RequestTooLargeError(413)` |
| P0 | 目录未收录的状态码走 `>= 500` 猜测，418 / 502 / 507 一律变成 `InternalServerError` | **已改**：落 `APIStatusError` 本身，**不猜**——猜出来的类会让 `except InternalServerError` 抓到并非 500 的失败，且实例 `status_code`（502）与类声明的（500）矛盾。重试判定不受影响：`transport/_retry.py` 按原始状态码决定，不看异常类 |
| P1 | 13 个 `ErrorCode` 线缆码只有 11 个进 `_CODE_MAP`；漏掉的 `InvalidParameter` 在业务层（HTTP 200）落基类 `DashScopeError`、`Model.NotExist` 完全无映射 | **已补** `_WIRE_TO_PUBLIC` crosswalk 覆盖全部 13 项，业务层与 HTTP 层现在得到同一个类 |
| P1 | `ContentFilterError(DashScopeError)` 与其余状态类不同族，`except BadRequestError` 抓不到审核失败 | **已改** `ContentFilterError(BadRequestError)`：目录把 `content_policy_violation` 命名为 `BadRequestError`，细化只能用子类化，不能引入目录外的状态 |
| P1 | `rl` 直通 1.x，失败抛裸 1.x 异常（`common/errors.py` 共 19 个类），`except DashScopeError` 抓不到 | **已补** `SdkError`：对应目录 `SDK_ERRORS`（32 条）的客户端侧错误，`code` = 目录名、`solution` = 目录随附的修复建议；挂载层 8 个直通方法（sync + async）全部经 `error_from_sdk_exception` 翻译 |
| P2 | `APIStatusError` 读作"非 2xx 响应" | **已改语义**：读作"**网关声明的失败**"——业务层（HTTP 200 带 `code`）造出的实例也是它，实例的 `status_code` 说明是哪一层 |
| P2 | `SdkError` 是否该配一个状态码 | **不给，`status_code is None`**：目录的 `SdkErrorDef` 没有 `external` 字段指回 `PublicError`，标准本身就没给 SDK 侧错误任何公开投影，凭空造一个等于给标准里没有的东西加含义 |
| P2 | `agentstudio` 直通 1.x 异常 | **未改**，仍裸奔：其挂载层直接返回 1.x client 对象，异常在 SDK 边界外抛出，需要透明代理才能拦截，挂在 M3 任务 3.12 |

**没有引入任何 SDK 自造名。** 新增的 4 个类逐个对应目录里已有的条目：`RequestTooLargeError` / `ServiceUnavailableError` / `GatewayTimeoutError` 是 `PUBLIC_ERRORS` 的 `error_code`，`SdkError` 是 `SDK_ERRORS` 所在的客户端子图。公开异常类 11 → 15，`APIStatusError` 的直接子类 6 → 9 且互为扁平兄弟。唯一带 SDK 判断的是 crosswalk：它把百炼历史线缆码对到目录 key 上，属"选哪条目录项"，不属命名——类永远由目录项给出，`test_wire_code_never_leaves_the_catalogued_class` 对每一行钉住这一点。
