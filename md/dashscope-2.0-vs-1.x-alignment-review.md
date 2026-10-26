# DashScope SDK 2.0 对 1.x 接口/功能对齐 — 深度 Review

> 评审范围:`dashscope/v2/`(2.0)对照 `dashscope/`(1.x: aigc / audio / embeddings / rerank / tokenizers / app / assistants / threads / finetune / files / models / utils)。
> 方法:按 6 个功能域逐文件比对公开方法、参数、流式语义、响应字段。
> 结论先行:**v2 是一次有意的重设计,不是 1:1 移植**。管理面与批量任务基本对齐甚至更优;推理面(chat / multimodal / image / video / audio)的多处功能回退已按 P0/P1/P2 修复(commit `c9604e2` / `d8ac331` / `b69b585`),TingWu 能力域已补 `client.tingwu`(`82e15d5`)。CodeGeneration、nlp Understanding、MultiModalDialog、Conversation 已决策 **hold,v2 先不支持**;音频侧 HTTP-TTS / omni 类型化助手 / 实时翻译 / 词表声纹管理经 2026-09 产品页核查确认存活,单独立章 **§7 语音模块专项(需立项,专门做)**。所有不同步到 v2 的能力在 **§6 整体标记**。按"能否无损迁移"衡量,**决策范围内的主干场景已可平移**。
>
> **范围说明**:1.x 的 Assistants / Threads / Runs 子系统正在下线,**2.0 有意不包含**,不计为缺口(详见 §3 P2-0)。`dashscope` CLI 在 2.0 是**后续再加**的计划项,同样不计缺口(详见 §3 P2-7)。

---

## 0. 对齐计分卡

| 领域 | 对齐度 | 主要缺口 |
|---|---|---|
| files / models / fine_tunes / deployments | 🟢 对齐+ | ~~deployments 缺 `stream_events`;finetune/deployment 响应少 `creator/modifier/group`~~ ✅ `b69b585` |
| apps(百炼 Application) | 🟢 对齐+ | ~~`workflow_message`、`usage.models[]` 未建模~~ ✅ `b69b585` |
| 批量转写 `asr.transcribe` | 🟢 对齐 | — |
| chat / multimodal.chat | 🟢 对齐 | ~~本地文件上传、per-call 鉴权、命名参数、累积流、`search_info`~~ ✅ `c9604e2` / `d8ac331` |
| embeddings / rerank | 🟢 对齐 | ~~batch 缺 `cancel/list`;本地 tokenizer 缺失;multimodal 本地文件上传缺失~~ ✅ `c9604e2` / `d8ac331` / `b69b585` |
| tokenization | 🟢 对齐 | ~~`token_ids/tokens` 未类型化;无本地 tiktoken~~ ✅ `d8ac331` / `b69b585`(v2 再导出 1.x `get_tokenizer`) |
| images / videos | 🟢 对齐 | ~~缺 `cancel/list`、本地文件上传、`function` 字段~~ ✅ `c9604e2` / `d8ac331` / `b69b585`;视频 `extend_prompt` 默认值差维持 v2 语义(已标注:迁移指南 §3.3) |
| 实时音频 asr.recognize / tts / realtime | 🟡 部分 | 增量 TTS / `recognize_file` / `RecognitionResult` / callback 适配器 ✅ `b69b585`;omni 类型化助手 ✅(M2)、实时翻译 ✅(M3 `client.audio.translation`)、HTTP-TTS ✅(M4 `client.audio.tts.http`)、管理面 ✅(M5 `client.audio.vocabularies/phrases/voices/voice_design`) |
| reinforcement_learning | 🟢 对齐 | ~~base_url 不一致;缺 function-instance 日志/删除;`jobs.delete` 缺失~~ ✅ `b69b585`(按 v2 默认 endpoint,见 P2-2) |
| 认证(save_api_key 密钥文件) | 🟢 对齐 | ~~v2 不读 `~/.dashscope/api_key`~~ ✅ `b69b585`(key 文件为最低优先级) |
| **TingWu(听悟)** | 🟢 **已补** | ✅ `82e15d5`:`client.tingwu`(离线 call + realtime 会话,wire 形状与 1.x 一致) |
| **nlp Understanding / MultiModalDialog / Conversation** | ⚪ **hold** | 已决策:v2 先不支持(见 §1 P0-4),存量用户继续用 1.x |
| **CodeGeneration** | ⚪ **hold** | 已决策:v2 先不支持(见 §1 P0-3) |
| **Assistants / Threads / Runs** | ⚪ 不在范围 | 1.x 下线中,2.0 有意不包含(非缺口) |
| **`dashscope` CLI** | ⚪ 后续再加 | 2.0 计划项,非缺口(见 §3 P2-7) |

---

## 1. 系统性回退(P0,迁移阻塞级)

### P0-1 本地文件上传 / OSS 预处理全面移除

> **状态**:✅ 已修复(commit `c9604e2`)。v2 复用 1.x `oss_utils.check_and_upload_local`(`v2/resources/_oss.py`),chat(native)/images/videos/embeddings 均先上传本地路径再置 `X-DashScope-OssResourceResolve: enable`。

**现象**:1.x 在 multimodal chat、multimodal embeddings、image、video 中会把本地路径自动上传 OSS,并置请求头 `X-DashScope-OssResourceResolve: enable`;v2 只接受已有 `oss://` / HTTP(S) URL。传本地文件的 qwen-vl / 图生图 / 视频用户会直接失败或拿到不可解析的入参。

**1.x 参照**
- 工具:`dashscope/utils/oss_utils.py` — `check_and_upload_local:174`、`preprocess_message_element:280`、`upload_file:149`、`OssUtils.upload:34` / `get_upload_certificate:112`。
- 调用:`dashscope/aigc/multimodal_conversation.py:151` `_preprocess_messages`(:218/:480),命中上传后 `headers["X-DashScope-OssResourceResolve"]="enable"`(:154/:419)。
- 图像/视频:`dashscope/aigc/image_synthesis.py:287-343`、`video_synthesis.py:293-321`。

**v2 现状**
- `dashscope/v2/protocols/native.py:22` `_translate_part` 仅读 `part["url"]`,无上传分支。
- `dashscope/v2/types/messages.py` 媒体 part 仅 `url`/`frames`。
- `dashscope/v2/resources/multimodal/embeddings.py:71` `_uses_oss` 只在值已以 `oss://` 开头时置 resolve 头,**不上传**。
- images/videos 无任何 `check_and_upload_local` 调用。

**改法**
1. v2 直接复用 1.x 的 `dashscope/utils/oss_utils.py`(纯工具,不引入循环依赖)。
2. 在 `native.py:_translate_part` 增加本地路径分支:当 `part["url"]` 是 `file://` / 本地存在路径时,调用 `check_and_upload_local(model, url, api_key)` 换成 `oss://`,并标记"本次有上传"。
   - 因 `_translate_part` 是纯函数且无 api_key,需把上传提到 `build_chat_request`(native.py:59)层:在 `_translate_messages` 前对 messages 跑一遍 `preprocess_message_element`,并把 `has_upload` 透传到 `ChatRequest.headers`,命中时加 `X-DashScope-OssResourceResolve: enable`。
   - api_key 来源:`client.config.api_key`(protocol 目前拿不到 client,需在 `completions._build_request` 处把 config/api_key 传入,或让 protocol 持有一个 uploader 回调)。最小改动:在 `resources/chat/completions.py:_build_request`(completions.py:64)里先对 `messages` 做上传预处理,再交给 protocol。
3. multimodal embeddings:把 `multimodal/embeddings.py:_contents`(:47)里每个 item 的 `image/audio/video` 值过一遍 `check_and_upload_local`,`_uses_oss` 改为"上传过或本就是 oss://"即置 resolve 头。
4. images/videos:在 `images.py` 构造 body 前,对 `image_urls` / `extra_input` / `media` 里的本地路径调用 `check_and_upload_local`(`videos.py` 同理对 `image_url`/`media`)。
5. 上传需要 api_key,统一从 `self._client.config.api_key` 取。

**验收**:本地 png/wav/mp4 路径 → 抓包确认请求体已是 `oss://` 且带 `X-DashScope-OssResourceResolve: enable`。

---

### P0-2 per-call `api_key` / `workspace` / `headers` 全部取消

> **状态**:✅ 已修复(commit `c9604e2`,采用方案 A)。transport 四个方法新增 `api_key=`/`workspace=` 形参并覆盖默认头;全部资源方法(含 ws 侧 audio)透传。

**现象**:1.x 几乎每个方法都能逐调用覆盖 `api_key=` / `workspace=`(多租户、子账号),chat 还能注入 `plugins` 头;v2 只有 client 级配置。多租户调用方与 per-call header 注入场景无法平移。

**1.x 参照**:`generation.py:49` `call(..., workspace, api_key, ...)`;`generation.py:186-192` 注入 `X-DashScope-Plugin`;各资源 `call` 均带 `api_key/workspace`。

**v2 现状**
- `dashscope/v2/client.py:53` 仅 client 级;`with_options`(client.py:152)可派生新 client(重量级)。
- `dashscope/v2/transport/http.py`:`_build_headers:48` 只在 transport `__init__`(:114-126)用 client 级 api_key/workspace;`request:136` / `open_stream` 已支持 per-call `headers=` 与 `base_url=`,但**没有 `api_key=` / `workspace=` 形参**。

**改法**(二选一,推荐 A)
- **A. 传输层加 per-call 鉴权形参**:给 `http.py` 的 `request` / `open_stream`(sync+async 共 4 个)增加 `api_key: Optional[str]=None`、`workspace: Optional[str]=None`,在内部用 `_build_headers(api_key or self._api_key, workspace or self._workspace, ...)` 覆盖默认头,再经 `_with_sdk_headers` 合并(注意大小写不敏感,已实现 http.py:71)。资源方法签名透传 `api_key/workspace` 到 transport。ws 侧在 `transport/ws.py:WsSession.__init__` 同样接受覆盖。
- **B. 仅在资源层透传 headers**:资源方法接受 `extra_headers: Mapping`,直接作为 `headers=` 传给 transport(caller 头已优先且大小写不敏感)。改动小,但把鉴权拼串责任丢给用户。

> 注:`plugins` 头属 B 即可覆盖;若要保留 1.x 的 `plugins=` 具名 kwarg,在 chat create 增加 `plugins` 形参并映射为 `X-DashScope-Plugin` 头。

**验收**:同一 client 下两次调用传不同 `api_key`,抓包 `Authorization` 不同;传 `workspace` 时 `X-DashScope-WorkSpace` 覆盖 client 默认。

---

### P0-3 CodeGeneration 缺失(需决策)

> **状态**:⚪ **已决策:hold,v2 先不支持**(2026-09 评审拍板)。文档核查亦显示官方文档无独立页面,代码生成场景走通用 chat API。存量用户继续用 1.x;后续若后端确认仍在线且有必要,再单独立项。

**1.x 参照**:`dashscope/aigc/code_generation.py:52/72`(scene 化代码生成)。
**v2 现状**:无资源。
**改法**:确认该 API 是否仍在线。若在线,作为 `chat.completions` 的一个 scene 入口或独立 `client.code_generation` 资源补;若已下线(同 Assistants/Threads/Runs),在迁移文档标注废弃即可,不计缺口。**先与后端确认存废再动手。**

> **文档核查(2026-09)**:官方文档主入口([DashScope API 参考](https://help.aliyun.com/zh/model-studio/qwen-api-via-dashscope))及站内检索均无 CodeGeneration 独立页面,代码生成场景走通用 chat/文本生成 API。**倾向"不再作为独立 scene 主推",仍以后端确认为准。**

> Assistants / Threads / Runs 已确认 1.x 下线中、2.0 有意不包含,见 §3 P2-0,不再作为 P0 阻塞项。

---

### P0-4 整个能力域在 v2 无对应(需决策)

**现象**:除 CodeGeneration(P0-3)外,以下 1.x 能力域在 v2 整体无对应资源,迁移这些能力的用户没有落点:

- **nlp Understanding**:`dashscope/nlp/understanding.py`(百炼知识库问答/理解)。
- **TingWu / TingWuRealtime**:`dashscope/multimodal/tingwu/`(听悟会议转写,含实时)。
- **MultiModalDialog**:`dashscope/aigc/multimodal_dialog.py`(多模态对话,带历史管理)。
- **Conversation / History**:`dashscope/conversation/`(对话状态托管)。

**改法**:与后端逐一确认**存废/归属**(随 1.x 下线?并入百炼 apps?独立立项?),再决定 v2 是补资源、委派 1.x 还是在迁移文档标注下线。**先确认再动手**;确认前这些域按 🔴 缺失计(见计分卡)。

> **状态**:✅ TingWu 已补(commit `82e15d5`)。新增 `client.tingwu`:`TingWu.call`(离线,POST `services/aigc/multimodal-generation/generation`,返回类型化 `TingWuResult`)与 `TingWu.realtime`(duplex ws 会话,`task_group=aigc`/`task=multimodal-generation`/`function=generation`,start/stop wire 形状与 1.x 一致;stop 的 `appId` 修正为真实值——1.x `tingwu_realtime.py` 因 `_Request.app_id` 从未赋值而恒发 `None`)。`WsSession.finish_task` 增加可选 `input` 载荷。**Understanding/MultiModalDialog/Conversation 已决策:hold,v2 先不支持**(2026-09 评审拍板),存量用户继续用 1.x,后续视后端归属结论再立项。
>
> **文档核查(2026-09)**:
> - **TingWu 确认存活**:官方文档有专页且 2026-09 仍在更新([服务端 Python SDK](https://help.aliyun.com/zh/model-studio/tingwu-sdk-python) / [realtime](https://help.aliyun.com/zh/model-studio/tingwu-sdk-python-realtime)),示例直接用 1.x `dashscope.multimodal.tingwu.tingwu.TingWu`(经 `multimodal-generation/generation` 端点跑 `createTask/getTask` 离线任务)。→ 从"待确认"升级为**"确认保留,v2 缺口坐实"**,建议立项补 `client.tingwu`(可复用 v2 jobs/transport,wire 形状与 1.x 一致)。
> - **Understanding / MultiModalDialog / Conversation 文档无独立页面**:站内检索无对应 API 文档;多轮对话改以 messages 数组教学([multi-round-conversation](https://help.aliyun.com/zh/model-studio/multi-round-conversation)),多模态交互由 omni 实时([multimodal-interaction-protocol](https://help.aliyun.com/zh/model-studio/multimodal-interaction-protocol))承接。**倾向"不再主推/已并入其他面"**,仍以后端确认为准。

---

## 2. 功能回退(P1)

### P1-1 高频命名参数降级为无类型 `parameters=` 透传

> **状态**:✅ 已修复(commit `d8ac331`)。chat 9 个具名参数(top_k/tool_choice/n/logprobs/top_logprobs/thinking_budget/enable_search/search_options/parallel_tool_calls)在三个协议落地:anthropic 对无对应项显式拒绝、`top_k` 映射为 Anthropic 原生字段;images/videos/audio 具名参数同步补齐,具名优先于 `parameters=` 兜底。

**现象**:`top_k / tool_choice / n / logprobs / top_logprobs / thinking_budget / enable_search / search_options / parallel_tool_calls`(chat),`seed/style/duration/resolution/ratio/prompt_extend/watermark`(image/video),`volume/rate/pitch/vocabulary_id/language_hints`(audio)等在 v2 不再是具名 kwarg,可发现性与类型校验下降;且 **openai 协议 strict 模式会拒绝未知 key**(`dashscope/v2/protocols/openai.py:137-143`)。

**v2 现状**:`native.py:59` `build_chat_request` 形参只到 `tools`;其余靠 `extra_parameters`(native.py:95)。chat create 形参见 `completions.py:383`。

**改法**
1. 在 `completions.py` 的 `create`(sync :383 / async :477 / 各 overload)与 `native.py:build_chat_request`(:59)、`openai.py` 适配器同步增加具名形参:`top_k / tool_choice / n / logprobs / top_logprobs / thinking_budget / enable_search / search_options / parallel_tool_calls`,在 native 的 `for key,value in (...)`(native.py:81)元组里登记。
2. openai 协议侧:把这些映射到 OpenAI 字段名;strict 白名单(openai.py:137)相应放行。
3. image/video:把 `seed/size/style/negative_prompt/duration/resolution` 等提升为 `images.generate` / `videos.generate` 具名 kwarg(当前在 `images.py` 走 `parameters`),保留 `parameters=` 兜底。
4. audio:`tts.synthesize`(tts.py:45)增加 `volume/rate/pitch/language_hints` 具名;asr.recognize(asr.py:255)增加 `vocabulary_id/phrases/disfluency_removal_enabled/diarization_enabled` 等,内部按 1.x 形状拼 `resources=[{resource_id,resource_type:"asr_phrase"}]`(参照 `recognition.py:341`、`transcription.py:245`)。

**验收**:具名传参等价于旧的 `parameters={...}`;openai 协议不再因这些 key 报错。

---

### P1-2 chat 流式:无累积模式 + text_stream 丢 reasoning/tool_call

> **状态**:✅ 已修复(commit `d8ac331`)。create 新增 `incremental_output`;native 侧沿用 1.x `ParamUtil.should_modify_incremental_output` 语义——常规模型 wire 保持增量、SDK 端合并为累积快照,tts/omni/qwen-deep-research 透传服务端;openai/anthropic 恒客户端合并。`ChatStream`/`AsyncChatStream` 新增 `reasoning_stream` / `tool_call_stream`。注:1.x 的 merge 实现位于 `dashscope/utils/message_utils.py`(含 n>1 分包/usage 按 index 聚合),v2 在 `completions.py` 以类型化 per-choice slot 重新实现,未逐行移植 n>1 的 finish_reason 隐藏逻辑。

**现象**:v2 native 恒置 `incremental_output=True`(native.py:79-80),无累积快照模式;1.x 的 incremental→cumulative 客户端 merge(generation.py:204-243)无对应。`ChatStream.text_stream` 只产出 `delta.content`,**丢弃 `reasoning_content` 与 tool_call 增量**(completions.py:114-119)。

**改法**
1. `text_stream`(completions.py:114)保持只出正文;新增 `reasoning_stream` 与 `tool_call_stream`(或在 chunk 上保留 delta 即可,文档说明)。`get_final_message`(completions.py:180 聚合)已含 reasoning/tool_calls,补齐 stream 侧便捷访问。
2. 累积模式:给 chat create 增加 `incremental_output: Optional[bool]=None`,透传到 native(默认仍 True);若用户传 False,提供一个 `accumulate` 包装器在 SDK 侧把增量合并成累积快照(对齐 1.x merge 行为),避免依赖服务端。

**验收**:`for t in stream.text_stream` 与 1.x 一致;`incremental_output=False` 时每个 chunk 是到目前为止的全量文本。

---

### P1-3 image/video/batch-embedding 缺 `cancel` / `list`

> **状态**:✅ 已修复(commit `d8ac331`)。`Jobs`/`AsyncJobs` 新增 `cancel`(POST /tasks/{id}/cancel)与 `list`(GET /tasks,`JobList` 分页模型),images/videos/embeddings 的 jobs 门面全部透传。

**现象**:1.x 任务类资源都有 cancel/list;v2 `_jobs.py` 只有 create/fetch/wait。

**1.x 参照**:`image_synthesis.py:544 cancel` / `:565 list`;`video_synthesis.py:564/585`;`batch_text_embedding.py:167 cancel` / `:188 list`。

**v2 现状**:`dashscope/v2/resources/_jobs.py:163 Jobs`(:182 create / :205 fetch / :221 wait),`AsyncJobs:269`。无 cancel/list。

**改法**
1. 在 `Jobs` / `AsyncJobs` 增加:
   - `cancel(task_id, *, timeout=None)`:1.x 走 `PUT/POST /tasks/{task_id}/cancel`(以 `image_synthesis.cancel` 实际 method/path 为准),复用 `self._client.transport.request(..., sdk_module=self._module)`。
   - `list(*, page_no, page_size, status=None, ...)`:`GET /services/{service}/... ` 或 1.x list 对应 path,返回任务列表类型。
2. images/videos/embeddings 的包装类(`ImageJobs`/`VideoJobs`/`EmbeddingJobs` 及 async)透传 `cancel/list`。
3. 新增响应模型 `JobList`(pydantic),对齐 1.x 分页字段。

**验收**:`client.images.jobs.cancel(task_id)` / `.list()` 可用;与 1.x 返回字段一致。

---

### P1-4 chat 响应丢 `output.search_info` / `thoughts` / 原始 output 透传

> **状态**:✅ 已修复(commit `d8ac331`)。`ChatCompletion.search_info` 类型化(`SearchInfo.search_results[]`,extra="allow" 兜底未建模字段)。`thoughts` 确认 chat 侧无返回,不建模。

**现象**:`parse_completion`(`dashscope/v2/types/completions.py:122`)只保留 `choices/usage/envelope`,`output.search_info`(web 搜索引用)、`output.thoughts`、顶层 `finish_reason`/`text` 被丢弃 → `enable_search` 结果在 v2 不可观测。

**改法**
1. `types/completions.py` 的 `ChatCompletion` 增加 `search_info: Optional[SearchInfo]=None`(建模 `search_results[].{title,url,site_name,index}`),`parse_completion` 从 `output.search_info` 提取。
2. 保留一个 `raw_output: Optional[Dict]=None`(或 `model_extra`)承接未建模字段,避免信息硬丢。
3. `thoughts` 若 chat 侧确有返回,按 apps 的 `Thought`(apps.py:49)同构建模。

**验收**:`enable_search=True` 时 `resp.search_info.search_results` 非空。

---

### P1-5 本地 tokenizer(get_tokenizer / tiktoken)缺失 + tokenization 丢 token_ids

> **状态**:✅ 已修复。`TokenCount.token_ids`/`tokens` 于 commit `d8ac331` 补齐;本地 tokenizer 入口于 commit `b69b585` 落地——`dashscope.v2` 直接再导出 1.x `get_tokenizer`/`list_tokenizers`(委派 `dashscope/tokenizers`,不重写 tiktoken),与包根既有导出指向同一对象。

**现象**:1.x `get_tokenizer(model)` 返回离线 tiktoken 对象(`encode/decode`);v2 无本地 tokenizer,`tokenization.count` 是服务端往返且只返回计数。

**1.x 参照**:`dashscope/tokenizers/tokenizer.py:17 get_tokenizer` / `:43 list_tokenizers`;`qwen_tokenizer.py:103 encode` / `:156 decode`;`tokenization.py:44 Tokenization.call`(返回 `output.token_ids/tokens`)。

**v2 现状**:`dashscope/v2/resources/tokenization.py:32 _parse` 只映射 `usage`+envelope,丢 `token_ids/tokens`;`count:70`。

**改法**
1. `tokenization.py:_parse`(:32)补 `token_ids`/`tokens` 字段;`TokenCount`(:24)增加 `token_ids: Optional[List[int]]`、`tokens: Optional[List[str]]`。
2. 本地 tokenizer:决策是否在 v2 暴露 `client.tokenizers.get(model)`。若要,直接复用 1.x `dashscope/tokenizers`(同 P0-3 委派思路),不重写 tiktoken 逻辑。

**验收**:`count` 返回 token_ids;`get_tokenizer` 入口在 v2 可达(或文档指明用 1.x)。

---

### P1-6 `save_api_key` 落盘的 key 文件 v2 不读取

> **状态**:✅ 已修复(commit `b69b585`)。`resolve_config` 的 `_pick` 链在 `config.toml` 之后补 key 文件级(`DASHSCOPE_API_KEY_FILE_PATH` → `~/.dashscope/api_key`,文件缺失回退 None 不抛错);`dashscope.v2` 再导出 1.x `save_api_key`,与包根同一对象。

**现象**:1.x 支持 `dashscope.save_api_key(...)` 把 key 写入 `~/.dashscope/api_key`(或 `DASHSCOPE_API_KEY_FILE_PATH` 指定路径),之后免配读取(`dashscope/common/api_key.py:45 save_api_key` / `:15 get_default_api_key`)。v2 的 key 解析(`v2/_config.py:542-549 resolve_config`、`Config.api_key` :279)只看 显式参数 → `DASHSCOPE_API_KEY` 环境变量 → `config.toml`,**不读 key 文件** → 走 `save_api_key` 流程的存量用户迁到 v2 直接丢鉴权。

**改法**
1. `resolve_config` 的 `_pick` 链在 `config.toml` 之后补两级:`DASHSCOPE_API_KEY_FILE_PATH` 环境变量指定的文件、`~/.dashscope/api_key` 默认文件(读取对齐 1.x `get_default_api_key` 的优先级)。
2. v2 暴露 `save_api_key(api_key, api_key_file_path=None)`(顶层 `dashscope.v2` 或 `client` 静态方法),实现直接复用/委派 1.x `common/api_key.py:45`,不重写。

**验收**:无显式 key、无环境变量、仅有 key 文件时 v2 Client 可初始化;`save_api_key` 入口在 v2 可达。

---

### P1-7 video `extend_prompt` 默认值与 image/video `function` 字段

> **状态**:✅ `function` 已修复(commit `b69b585`)。`images.generate`/`videos.generate` 及各自 jobs(sync+async)新增具名 `function: Optional[str]=None`,写入 `inputs` 且具名优先于 `extra_input`;images 两个 input 构造器(legacy + multimodal)均覆盖。`extend_prompt` 按改法 1 维持 v2 `prompt_extend=None` 语义,默认值差已在迁移指南 §3.3 标注。

**现象**
1. 1.x video 的 `extend_prompt: bool = True`(`video_synthesis.py:63` / `:199`,`prompt_extend` 的 deprecated 别名,**默认开**,写入 `inputs["extend_prompt"]` :213);v2 `videos.generate` 的 `prompt_extend` 默认 `None` 且只进 `parameters`。显式行为不同,逐参数对齐时需注意默认值差。
2. 1.x image/video 有 `function: str = None` 形参(`image_synthesis.py:46` → `inputs["function"]` :312-313;`video_synthesis.py` → `inputs["function"]` :219);v2 `images.generate`/`videos.generate` 无此具名参数(P1-1 只恢复了 parameters 侧字段,`function` 属 inputs 侧)。

**改法**
1. video:维持 v2 `prompt_extend=None`(不擅自引入 1.x 的默认 True),但在迁移文档标注默认值差;`extend_prompt` 别名不再恢复(1.x 已 deprecated)。
2. image/video:`generate`/`jobs.create`(sync+async)增加具名 `function: Optional[str]=None`,写入 `inputs`(非 `parameters`),具名优先于 `extra_input`。

**验收**:`images.generate(function="...")` / `videos.generate(function="...")` 与 1.x inputs 形状一致。

---

## 3. 一致性 / 补全(P2)

### P2-0 Assistants / Threads / Runs — 有意不在 2.0 范围(非缺口)

> **状态**:✅ 已在迁移指南 §5 显式声明下线(`md/dashscope-2.0-migration-guide.md`)。

1.x 的 Assistant 运行时(`dashscope/assistants/`、`dashscope/threads/`、`threads/runs/`)正在下线,**2.0 有意不实现**。这是产品决策,不计为对齐缺口,也不需要委派/重写。

**唯一动作**:在迁移文档显式声明"Assistants/Threads/Runs 不在 2.0 范围,随 1.x 下线",避免用户误以为是遗漏。CodeGeneration(P0-3)若同属下线范畴,一并标注。

### P2-1 deployments 缺 `stream_events` / 通用 `put`

> **状态**:✅ 已修复(commit `b69b585`)。SSE 实现抽出为共享模块 `v2/resources/_sse.py`(fine_tunes 同步重构复用);`Deployments.stream_events`/`AsyncDeployments.stream_events`(GET `/deployments/{id}/stream`)落地,raise/yield 契约与 fine_tunes 一致;新增类型化 `DeploymentEvent`。通用 `put` 未补(1.x `PutMixin` 无实际调用方,`scale` 已覆盖唯一在用场景)。

1.x `dashscope/finetune/deployments.py:20` 继承 `StreamEventMixin`/`PutMixin`;v2 `deployments.py:119` 只有固定 `scale`(:188)。
**改法**:把 `fine_tunes.py` 已有的 `stream_events`(fine_tunes.py:626 SSE 实现)抽出复用到 `Deployments.stream_events(deployed_model)`;path 用 `/deployments/{id}/stream`(对齐 1.x)。

### P2-2 RL base_url 不一致 + 缺 function-instance 操作 + 缺 `jobs.delete`

> **状态**:✅ 已修复(commit `b69b585`)。**决策:RL 走 v2 默认 endpoint**——1.x 把每个 RL 动词强转 `DASHSCOPE_HTTP_BASE_URL` 属 1.x 配置问题,不在 v2 复刻(评审会上拍板,替代原"host 对齐"改法)。已补:`RlJobs.delete`、三个 function-instance 方法(`query_function_instance_logs`/`query_all_function_instance_logs`/`delete_function_instance`,委派 1.x `AgenticRL`)、`test_functions` 的 `pull_logs/log_page_size/start_time/end_time`。

1.x `dashscope/finetune/agentic_rl.py:353-438` 的 `cancel/list/get/delete/logs` 强制 `DASHSCOPE_HTTP_BASE_URL`;v2 `reinforcement_learning.py:199 RlJobs`(:239-292)委派 `FineTunes` 走 v2 标准 endpoint → **host 不等价时行为偏**。另缺 `query_function_instance_logs`/`query_all_function_instance_logs`/`delete_function_instance`(agentic_rl.py:513/560/607)与 `jobs.delete`(1.x agentic_rl.py:400)。
**改法**:确认 RL 管理面正确 host;若确需 `DASHSCOPE_HTTP_BASE_URL`,在 `RlJobs` 的 fetch/cancel/list/logs 调用里传 `base_url=`(transport 已支持 per-request base_url,http.py:92)。补 `RlJobs.delete` 与三个 function-instance 方法(委派 1.x `AgenticRL` 对应方法)。`test_functions`(rl.py:331)补 `pull_logs/log_page_size/start_time/end_time`。

### P2-3 任务等待语义翻转(需文档化 + 可选对齐)

> **状态**:✅ `raise_on_error=False` 分支已修复(commit `b69b585`)。`Jobs.wait`/`AsyncJobs.wait` 在 compat 模式下对 `FAILED/CANCELED` 终态直接返回 Job 而非抛错,且不再空轮询。`DEFAULT_WAIT_TIMEOUT`(86400 vs 1.x 无限)维持 v2 语义,已在迁移指南 §3.2 标注。

v2 `Jobs.wait` 默认 `timeout=86400`(1.x 为无限 `-1`),`FAILED/CANCELED` 在 `raise_on_error` 下抛异常而非返回(_jobs.py:73/250/260)。
**改法**:`DEFAULT_WAIT_TIMEOUT` 是否改回"无限"由产品决策;至少在迁移文档显式说明,并保证 `raise_on_error=False` 时返回终态 Job 而非抛错(确认 `_parse`/`_check_canceled` 在 False 分支返回对象)。

### P2-4 响应字段补全

> **状态**:✅ 已修复(commit `b69b585`)。`AppCompletion.workflow_message`(嵌套 `WorkflowMessage`)与 `AppUsage.models[]`(`AppModelUsage`)落地;`FineTune` 补 `user_identity/modifier/creator/group`;`Deployment` 补 `modifier/creator`。

- apps:`AppCompletion`(apps.py:80)补 `workflow_message`(1.x `application_response.py:120-166`)与 `usage.models[]` 每模型 token 明细类型化(当前仅 `extra="allow"`)。
- fine_tunes:`FineTune`(fine_tunes.py:86)补 `user_identity/modifier/creator/group`(1.x `customize_types.py:47-50`)。
- deployments:`Deployment`(deployments.py:45)补 `modifier/creator`(customize_types.py:143-144)。

### P2-5 AgentStudio 委派不走 v2 栈(说明性)
`client.py:115 agentstudio` 委派 1.x `agentstudio.Client`,只转发 `api_key/workspace/region`,**不转发 base_url/timeout/http_client**,且不经过 v2 transport/header/error 栈(刚统一的 `x-dashscope-sdk-*` 头对其不生效)。属有意复用,但需文档说明;若要统一标识,可在委派构造时把 v2 的 sdk 头/超时透传给 1.x agentstudio 的 transport。

### P2-6 音频实时面补全(较大,单列)

> **状态**:✅ 前三项已修复(commit `b69b585`),立项项不变。已落地:callback 适配层 `v2/resources/audio/_callbacks.py:run_with_callback`(鸭子类型同时服务 asr/tts 两种 1.x callback 形状);增量 TTS `client.audio.tts.connect`→`SynthesisSession.send_text/stop/audio_chunks` + `enable_ssml`(联动 `text_type` PlainText/SSML),`tts.stream` 重构复用之;本地文件 ASR `asr.recognize_file(path)`(12800 字节分块,对齐 1.x);类型化结果 `RecognitionResult.get_sentence/is_sentence_end/get_usage` + `Recognition.results()`(过滤心跳与非 result 事件)。

- **callback 适配层**:1.x push 回调(`RecognitionCallback` recognition.py:140、`ResultCallback` tts_v2:69/90、omni/qwen-tts callback)无适配。建议提供 `events()`→callback 的可选适配器,降低迁移成本。
- **增量 TTS**:`tts.stream`(tts.py:73)当前一次性 `continue_task({"text":text})`+`finish_task`(tts.py:105),且硬编码 `text_type:"PlainText"`、无 SSML。补 `streaming_call(text)`/`streaming_complete()` 等价能力与 `enable_ssml`(1.x tts_v2:627/657/885)。
- **本地文件 ASR**:1.x `Recognition.call(file)` 自动分块(recognition.py:444);v2 需手动 `send_audio_frame`。补 `recognize_file(path)` 便捷方法。
- **缺失能力(需立项)**:实时翻译 `TranslationRecognizerRealtime`(translation_recognizer.py:288)+`TranslationRecognizerChat`(:807)、omni 音视频类型化助手(`update_session/append_audio/append_video/create_response/commit`,omni_realtime.py:399)。当前 v2 仅 `realtime.connect` 裸事件可达。(注:omni 类型化助手 M2、实时翻译 M3、HTTP 合成 M4、词表/热词/声纹管理 M5、连接复用 M6 均已落地,见 §7.4。)
- **类型化结果**:`asr.recognize` 的 `events()` 产裸 dict;补 `RecognitionResult.get_sentence/is_sentence_end/get_usage`(recognition.py:99-125)等价类型。

### P2-7 `dashscope` CLI 暂未覆盖 v2(计划项)

> **状态**:✅ 已在迁移指南 §5 提示"v2 无 CLI,命令行场景继续用 1.x CLI"(`md/dashscope-2.0-migration-guide.md`);v2 版 CLI 仍后续再加。

v2 暂不提供 CLI;现有 `dashscope` 命令行(acli,copy 自 agentic CLI,保持自包含、不依赖 dashscope 包)仍走 1.x。**v2 版 CLI 后续再加**,届时再评估入口命名与共用层。当前动作仅:在迁移文档提示"v2 无 CLI,命令行场景继续用 1.x CLI"。

---

## 4. v2 的净增益(对齐时勿回退)

- 全资源 **async 孪生**(1.x 多为 sync 或零散 aio)。
- **pydantic 强类型** + `extra="allow"` 兜底,响应可点出字段。
- 统一 **`raise_on_error`** 与 `is_error/code/message/status_code` 信封(`types/common.py:24-49`)。
- files `NotFoundError` 精化(files.py:159)、models `provider` 过滤(models.py:99)、deployments MU 字段 `deploy_spec/billing_method`(deployments.py:111)、`AppStream.text_stream`(apps.py:260)。
- multimodal embedding 新增 `video` item;`with_options`(client.py:152)派生配置。
- 统一 SDK 标识头 `x-dashscope-sdk-client`/`session-id`(本次已修)。

---

## 5. 落地优先级建议

| 优先级 | 项 | 估时倾向 |
|---|---|---|
| P0 | 本地文件上传(P0-1)、per-call 鉴权(P0-2) | ✅ `c9604e2` |
| P0 | CodeGeneration(P0-3) | ⚪ 已决策:hold,v2 先不支持 |
| P0 | 能力域(P0-4):TingWu ✅ `82e15d5`;Understanding/MultiModalDialog/Conversation | ⚪ 已决策:hold,v2 先不支持 |
| P1 | 命名参数(P1-1)、流式累积+reasoning(P1-2)、cancel/list(P1-3)、search_info(P1-4)、token_ids(P1-5)、key 文件(P1-6)、extend_prompt/function(P1-7) | ✅ `d8ac331` / `b69b585` |
| P2 | deployments.stream_events、RL endpoint/补全、等待语义、响应字段补全、音频实时面 | ✅ `b69b585`(音频立项项除外) |
| P2 | v2 CLI(P2-7) | 后续再加;✅ 迁移指南 §5 已提示 |
| 专项 | 语音模块专项(§7):ws 协议补齐与 omni 助手(`8eaaf81`)、M3 翻译门面、M4 HTTP 合成、M5 管理面、M6 连接复用(`WsConnection` + 全模型 HTTP keep-alive settings) | ✅ 全部落地;修复计划见 §7.4 |
| — | Assistants/Threads/Runs(P2-0) | 不在 2.0 范围;✅ 迁移指南 §5 已标注下线 |

> 建议每个 P0/P1 项各开一个 issue,附本文对应小节的 file:line 作为实施清单起点;改动均落在 `dashscope/v2/`,复用 1.x 工具(oss_utils / tokenizers)而非重写,以保持物理隔离同时避免功能回退。

---

## 6. 不同步到 v2 的能力(整体标记)

以下能力 **v2 不提供**,存量用户继续用 1.x;均已拍板或落地,不再视为缺口。迁移口径统一见 [`dashscope-2.0-migration-guide.md`](dashscope-2.0-migration-guide.md) §5。

| 能力 | 1.x 位置 | 处置 | 产品页现状(2026-09 核查) |
|---|---|---|---|
| Assistants / Threads / Runs | `dashscope/assistants/`、`threads/` | ⚪ 1.x 下线中,2.0 有意不包含(§3 P2-0) | — |
| CodeGeneration | `aigc/code_generation.py` | ⚪ hold,v2 先不支持(§1 P0-3) | 文档无独立页面 |
| nlp Understanding | `nlp/understanding.py` | ⚪ hold,v2 先不支持(§1 P0-4) | 文档无独立页面 |
| MultiModalDialog | `aigc/multimodal_dialog.py` | ⚪ hold,v2 先不支持(§1 P0-4) | 文档无独立页面 |
| Conversation / History | `conversation/` | ⚪ hold,v2 先不支持(§1 P0-4) | 文档无独立页面,多轮对话以 messages 数组教学 |
| HTTP-TTS(sambert 等) | `audio/http_tts/http_speech_synthesizer.py:114 HttpSpeechSynthesizer` | ✅ M4 已落地(`client.audio.tts.http`,见 §7.4) | [专页存活](https://help.aliyun.com/zh/model-studio/sambert-python-sdk),但示例已统一走 ws 版 `SpeechSynthesizer`,HTTP 形态不再主推 |
| qwen-tts HTTP | `audio/qwen_tts/speech_synthesizer.py:11` | ✅ M4 已落地(同上,按模型名前缀路由) | 存活且有独立页([模型信息](https://help.aliyun.com/zh/model-studio/model-qwen-tts) / [Qwen-TTS API](https://www.alibabacloud.com/help/zh/model-studio/qwen-tts-api)) |
| 实时翻译(gummy) | `audio/asr/translation_recognizer.py:288/807` | ✅ M3 已落地(`client.audio.translation`,见 §7.4) | [专页存活](https://help.aliyun.com/zh/model-studio/real-time-python-sdk),示例用 `TranslationRecognizerRealtime` |
| 定制词表 / 热词 | `audio/asr/vocabulary.py:31 VocabularyService`、`asr_phrase_manager.py:14 AsrPhraseManager` | ✅ M5 已落地(`client.audio.vocabularies` / `client.audio.phrases`,见 §7.4) | 存活([paraformer 热词](https://help.aliyun.com/zh/model-studio/paraformer-asr-phrase-manager) / [定制热词](https://help.aliyun.com/zh/model-studio/custom-hot-words/)) |
| 声音复刻(声纹注册) | `audio/tts_v2/enrollment.py:31 VoiceEnrollmentService` | ✅ M5 已落地(`client.audio.voices`;同端点的 qwen 声音设计一并建模为 `client.audio.voice_design`,1.x 无对应类) | [专页存活](https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide),但示例以 `tts_v2.SpeechSynthesizer` + 直接 HTTP API 创建音色,未用该类 |
| omni 类型化助手 | `audio/qwen_omni/omni_realtime.py:399`(update_session/append_audio/create_response 等) | ✅ M2 已落地(`client.audio.realtime` omni 类型化助手,`8eaaf81`,见 §7.4) | 存活且主推([Qwen-Omni-Realtime](https://help.aliyun.com/zh/model-studio/realtime) / [Realtime API 概述](https://help.aliyun.com/zh/model-studio/realtime-api-overview)) |
| `dashscope` CLI(v2 版) | acli(自包含,走 1.x) | ⚪ 后续再加(§3 P2-7) | — |

> 上表各项均已拍板或落地:hold 项存量用户继续用 1.x,CLI 后续再加;语音相关项已全部落地(§7)。

---

## 7. 语音模块专项(需立项,专门做)

> 依据语雀专项文档([语音模块专项](https://aliyuque.antfin.com/songsong.sss/tcd79h/vi8tshautdo2k5iy))成稿,正文已由专项负责人提供(2026-09-22),接口对齐清单与严重问题以语雀为准,本章已按 v2 最新代码校正状态。
>
> 定位:本章各项**不属于对齐修复**,而是 v2 语音面的独立建设,按一个专项(或拆子项)立项推进,不挤占 P0-P2 的修复节奏。

### 7.1 专项范围(语雀界定)

- **语音识别**:qwen-audio(fun-asr/paraformer)系列实时识别、funasr-flash 等离线识别、录音文件识别。
- **语音合成**:qwen-audio、CosyVoice 等。
- **声音复刻 / 声音设计 / 热词词表**:均基于 HTTP 协议,可替换通用接口或直接用 http 请求。
- **Omni**:omni / 翻译 / realtime。

### 7.2 严重问题(专项首要,必须解决)

1. **ASR 缺 `update_context`(必须)** — ✅ 已解决(`8eaaf81` 合入 `dev/v2-audio-streaming-completion`,`d1de92f`):`Recognition.update_context(context)` 走 `continue-task` + `{"input": context}` wire,与 1.x `audio/asr/recognition.py:632` 对齐;非 dict 输入 `TypeError`。
2. **TTS 流式上传/控制协议缺失(必须)** — ✅ 已解决(`8eaaf81`,`c570d20`/`24bb67b`):`SynthesisSession` 显式生命周期 `start/stop/cancel/close`,`flush()`(continue-task `{"input":{"flush":True}}`,已经真实服务验证,无需 1.x 的重述 model/task 字段)、`cancel()`(finish-task `input.directive=cancel`,ws 层 `_cancel_sent` 幂等);命名按评审 no-alias 前提(naming-review §六:直接改名、不留兼容 alias)收敛为 canonical 名(曾短暂保留的 1.x 别名 `streaming_*` 与 `TtsStream`/`start_stream` 已移除,同一动作只留一个名字)。`async_streaming_complete` 不做(v2 实时面 sync-only 设计)。
3. **连接复用缺失(必须)** — ✅ 已解决(M6 落地):
   - **ws 连接复用**:`WsConnection` + `WsTask`(transport/ws_connection.py)——一条物理连接 + 后台读线程按 `header.task_id` 解复用,支持并发多任务与顺序复用(免再握手);`client.ws_connection()` 工厂继承 client 配置,asr/tts/translation/tingwu 入口均接受 `connection=`。协议约束:二进制帧无 task_id,单连接二进制上行/下行各单 owner(TTS `start()` 自动 claim;并发 TTS 需多条连接),纯 JSON 任务完全并发;realtime 族协议无 task_id、会话结束即关连接,不可复用(非 SDK 限制,loopback 测试固化;压首开延迟走预热)。
   - **HTTP 连接复用(所有模型)**:每 `Client` 一个 httpx keep-alive 池(1.x 每次调用新建 `requests.Session`),所有资源默认复用;池参数开放 settings/env/config.toml(`http_max_connections` / `http_max_keepalive_connections` / `http_keepalive_expiry`);`with_options` 未改连接级选项时共享池。`tests/v2/test_http_pool.py` 本地服务器证明同一 TCP 连接被复用、keepalive=0 端到端生效。背景(2026-09-23 评审):1.x 大量用户以连接复用模式接入(主要 Java 侧),语音实时交互对延时极敏感——连接复用是硬需求而非优化项。

### 7.2.1 `dev/v2-audio-streaming-completion` 合入修复清单

> 2026-09-23 以 merge `8eaaf81` 合入(tag `v2-audio-m1` / `v2-audio-m2`)。分支经 4 个真实模型线下手工验证(未纳入 CI,模型名以专项验证记录为准):qwen-audio-3.1-asr-flash-streaming、qwen-audio-3.0-tts-flash(voice longanfengyu)、qwen3.5-omni-flash-realtime、qwen-audio-3.0-realtime-flash。合入策略:架构以分支为准,HEAD 独有能力逐项 graft(module 标识头、per-call `api_key`/`workspace`、`RecognitionResult`/`results()`/`recognize_file`、合成具名参数 + `enable_ssml` 联动、jobs `headers` 参数)。

| 修复 | 分支 commit | 对应条目 |
|---|---|---|
| `Recognition.update_context(context)`(continue-task + `{"input": context}`) | `d1de92f` | §7.2-1 |
| `SynthesisSession` 显式生命周期 `start/stop/cancel/close` + `flush()` + `cancel()`(ws `_cancel_sent` 幂等) | `c570d20` / `24bb67b` | §7.2-2 |
| ws 严格 framing/生命周期:task-failed 经 header `error_code`/`error_message` 走错误目录、EOF → `APIConnectionError`、finish/cancel/close 幂等 | `2c9a0b2` | §7.2 基础 |
| `RealtimeSession` 具名方法全集 + `session_id`(omni 类型化助手) | `ab3ec1c` | §7.3 Omni 表(M2) |
| livetranslate / qwen-audio 协议族(wire_model 前缀路由;`_incremental_session` / `_nested_translation_session`) | `a147465` / `68f6afe` | §7.4 M3 前置 |
| asr / tts / realtime SDK 使用指南(`dashscope-2.0-{asr,tts,realtime}-sdk.md`) | `cca6dfd` | 文档 |
| loopback ws 集成测试基座(`_LocalWebSocketServer`,aiohttp 真实回环;单测 mock `FakeWebSocket`/`_sent_frames` 早于本分支,由 `52f87e7` 引入、分支内各 commit 复用扩展) | `3adc9f5` | 测试 |

合入后仍开放:`async_streaming_complete`(不做,v2 实时面 sync-only);§7.2-3 连接复用随后于 M6 落地(见 §7.4)。合入后的命名收敛(移除 `streaming_*` 别名 / `TtsStream` / `start_stream`,`update_context(input)` → `(context)`)见 §7.2-2 与 commit `5f70c02`。M3 gummy 门面(`client.audio.translation`)、M4 HTTP 合成(`client.audio.tts.http`)、M5 管理面(`client.audio.vocabularies/phrases/voices/voice_design`)、M6 连接复用(`WsConnection` + HTTP keep-alive settings)均已于本阶段落地,见 §7.4。

### 7.3 接口对齐清单(语雀表格,按 v2 现状校正)

**Transcription 离线转写** — ✅ 已对齐:`call→asr.transcribe`、`async_call→asr.jobs.create`、`fetch→asr.jobs.fetch`、`wait→asr.jobs.wait`。

**Recognition 实时转写**

| 线上 SDK | v2 入口 | 状态 |
|---|---|---|
| `Recognition.start` | `asr.recognize(...)` + with(`start()`) | ✅ |
| `send_audio_frame` / `stop` / 取结果 | `send_audio_frame` / `stop` / `events()` | ✅ |
| `Recognition.call(file)` | `asr.recognize_file(path)` | ✅ 非必须,已补(`b69b585`) |
| **`Recognition.update_context`** | `Recognition.update_context(context)` | ✅(`8eaaf81`/`d1de92f`,见 7.2-1) |
| `get_first_package_delay` / `get_last_package_delay` | `first_package_delay` / `last_package_delay` 属性 | ✅(`8eaaf81`/`d1de92f`) |
| `get_last_request_id` | `Recognition.request_id` 属性 | ✅(`8eaaf81`/`d1de92f`) |
| `RecognitionCallback` | 主动迭代 `events()`;`_callbacks.run_with_callback` 适配 | 🔀 |
| `RecognitionResult` | `events()` 裸 dict;`results()` 类型化 | ✅(`b69b585`) |
| 热词 | `phrase_id`(run-task 顶层 `resources`,与 1.x 同位) | ✅(`8eaaf81`;原 `phrases` 参数废弃) |

**语音合成**

| 线上 SDK | v2 入口 | 状态 |
|---|---|---|
| `tts_v2.SpeechSynthesizer.call` | `audio.tts.synthesize(...)` | ⚠️ 一次性合成近似对应 |
| `streaming_call` | `SynthesisSession.send_text` | ✅(`8eaaf81`) |
| **`streaming_flush`** | `SynthesisSession.flush` | ✅(`8eaaf81`/`c570d20`,见 7.2-2) |
| `streaming_complete` | `stop()` + `audio_chunks()` | ✅(`8eaaf81`) |
| **`async_streaming_complete`** | 无 | ❌ 不做(v2 实时面 sync-only) |
| **`streaming_cancel`** | `SynthesisSession.cancel` | ✅(`8eaaf81`/`c570d20`,见 7.2-2) |
| `close`(长会话管理) | `SynthesisSession` 显式 `start/stop/close` | ✅(`8eaaf81`/`24bb67b`) |
| `HttpSpeechSynthesizer.call(stream=False/True)` | `audio.tts.http.synthesize/stream(...)`(真 HTTP,非 ws) | ✅(M4;`HttpSynthesis`/`HttpSynthesisChunk` 类型化音频 URL/ID/过期时间,见 §7.4) |

**Omni**

| 线上 SDK | v2 入口 | 状态 |
|---|---|---|
| `OmniRealtimeConversation.connect` | `audio.realtime.connect(...)` + with | ✅ |
| `send_raw` | `session.send_event(...)`(dict,非 JSON 串) | ⚠️ |
| 接收事件 | `recv_event()` / `events()` | ✅ |
| `update_session`/`create_item`/`append_audio`/`append_video`/`commit`/`clear_appended_audio`/`create_response`/`cancel_response` | `RealtimeSession` 同名具名方法(另含 `retrieve_item`/`delete_item`) | ✅(`8eaaf81`/`ab3ec1c`) |
| `end_session` | `RealtimeSession.end_session()` | ✅(`8eaaf81`/`ab3ec1c`) |
| `get_session_id` / response ID / 延迟指标 | `session_id` 属性;指标见 realtime 事件跟踪 | ✅/❔(`8eaaf81`) |

**实时翻译** — livetranslate 协议变体(qwen3-livetranslate/qwen3.5/qwen3.8)已内建于 `audio.realtime.connect` 会话层(`8eaaf81`/`a147465`,含嵌套翻译 session 组装);gummy 族(gummy-realtime-v1 / gummy-chat-v1)由 `client.audio.translation` 子资源承载(M3 已落地,见 §7.4):与 `asr.recognize` 同 wire(task_group=audio / task=asr / function=recognition,与 1.x `translation_recognizer.py:451/987` 逐帧一致),翻译参数具名化(`source_language`/`transcription_enabled`/`translation_enabled`/`translation_target_languages`/`max_end_silence`),`results()` 产双语类型 `TranslationResult`(`get_transcription`/`get_translations`/`is_sentence_end`),`recognize_file(path)` 对应 1.x `call(file)`;chat 形态不单独建模(1.x Chat 仅差"句终停发"的客户端行为,wire 相同,以 `is_sentence_end()` 自流控)。

**已验证模型**(分支真实跑通,2026-09-23):`qwen-audio-3.1-asr-flash-streaming`、`qwen-audio-3.0-tts-flash`(voice: longanfengyu)、`qwen3.5-omni-flash-realtime`、`qwen-audio-3.0-realtime-flash`。

**管理面(HTTP)** — ✅ M5 已落地:定制词表 `client.audio.vocabularies`(1.x `VocabularyService`,POST `services/audio/asr/customization`)、热词 `client.audio.phrases`(1.x `AsrPhraseManager`,编译任务:`create/update` POST `/fine-tunes`,`get/list/delete` 走 `/fine-tunes/outputs`)、声音复刻 `client.audio.voices`(1.x `VoiceEnrollmentService`,POST `services/audio/tts/customization`)、声音设计 `client.audio.voice_design`(官方 REST,与复刻同端点,`model=qwen-voice-design`,action 词 `create/list/query/delete`)。1.x 五动词统一为 `create/list/get/update/delete`(`query_*` → `get`);1.x 服务内 3 次重试改由 transport 重试循环承担;1.x `list_phrases(page=)` 实参被 `**kwargs` 丢弃的缺陷不复刻(v2 发 `page_no`)。

### 7.4 修复计划

严重问题(7.2)优先于能力补齐,分阶段交付,每阶段独立 commit、独立可发版。
**进度更新(2026-09-23)**:`dev/v2-audio-streaming-completion` 分支(9 commits,四模型真实验证)经 `8eaaf81` 合入,M1(除连接复用)/M2 已落地,M3 协议层随合入落地、gummy 门面(`client.audio.translation`)本阶段补齐;M4 HTTP 合成(`client.audio.tts.http`)、M5 管理面(`client.audio.vocabularies/phrases/voices/voice_design`)、M6 连接复用(`WsConnection` + 全模型 HTTP keep-alive settings)均已落地,语音专项 §7 全部关闭。

| 阶段 | 范围 | 状态 | 交付物 | 验收 |
|---|---|---|---|---|
| M1 ws 协议能力补齐 | 严重问题 1-2 | ✅ `8eaaf81`(tag `v2-audio-m1`) | `Recognition.update_context`;`SynthesisSession` 显式生命周期 + `flush/cancel`;ws 严格 framing/生命周期(`2c9a0b2`) | 真实模型验证(asr-flash-streaming / tts-flash)+ loopback ws 单测(`3adc9f5`) |
| M2 omni 类型化助手 | 7.3 Omni 行 | ✅ `8eaaf81`(tag `v2-audio-m2`) | `RealtimeSession` 具名方法全集(update_session/append_audio/append_video/commit/clear_appended_audio/create_item/retrieve_item/delete_item/create_response/cancel_response/end_session)+ `session_id`(`ab3ec1c`) | omni-flash-realtime / realtime-flash 真实验证 + `test_audio_omni_session.py`(`68f6afe`) |
| M3 实时翻译 | livetranslate / gummy | ✅ | livetranslate 三变体内建 `realtime.connect`(`8eaaf81`/`a147465`);gummy 门面 `client.audio.translation`:`TranslationSession`(复用 asr wire)+ 具名翻译参数 + 双语 `TranslationResult` + `recognize_file`;chat 形态评估结论:不单独建模(wire 相同,句终停发由调用方经 `is_sentence_end()` 表达) | [Gummy 页](https://help.aliyun.com/zh/model-studio/real-time-python-sdk)流程平移至 `md/dashscope-2.0-translation-sdk.md`;`test_audio_translation.py` wire 逐帧对齐 |
| M4 HTTP 合成 | `http_tts:114` + `qwen_tts:11`(合并) | ✅ | `client.audio.tts.http`(真 HTTP POST,非 ws):`synthesize` 一次性 → `HttpSynthesis`(audio_url/audio_id/expires_at/usage/request_id 类型化);`stream` SSE 流式 → `HttpSynthesisChunk`(audio 帧 base64 解码、sentence、is_final,末帧带整段音频 URL 引用);双模型族按模型名前缀路由(`qwen*` → aigc/multimodal-generation/generation,其余 → audio/tts/SpeechSynthesizer),具名参数 `voice/format/sample_rate/language_type` + `parameters=`/`extra_input=` 透传(具名优先) | sambert 与 qwen-tts 两个官方示例平移至 `md/dashscope-2.0-tts-sdk.md`;`test_audio_tts_http.py` MockTransport 6 例(端点路由/参数合并/SSE 分片/错误帧/错误状态) |
| M5 管理面 | 词表/热词/声音复刻/声音设计 | ✅ | `client.audio.vocabularies`(1.x `VocabularyService`,`services/audio/asr/customization`)/ `phrases`(1.x `AsrPhraseManager`,`/fine-tunes` + `/fine-tunes/outputs` 编译任务)/ `voices`(1.x `VoiceEnrollmentService`,`services/audio/tts/customization`)/ `voice_design`(官方 REST,同复刻端点,`qwen-voice-design`);五动词统一 `create/list/get/update/delete`,类型化返回(Vocabulary/Voice/DesignedVoice/Phrase + 各自 List + `JobStatus`) | 1.x 五动词 × 各服务 wire 逐项断言;官方四族示例平移至 `md/dashscope-2.0-audio-management-sdk.md`;`test_audio_management.py` MockTransport 8 例 |
| M6 连接复用(上调,见 7.2-3) | ws 复用 + 全模型 HTTP keep-alive | ✅ | ws:`WsConnection`+`WsTask`(读线程按 `header.task_id` 解复用,并发多任务 + 顺序复用免握手;二进制帧无 task_id → 单连接上行/下行各单 owner,纯 JSON 全并发),`client.ws_connection()` 工厂,asr/tts/translation/tingwu 接受 `connection=`;realtime 族验证结论:协议无 task_id、会话结束即关连接,不可复用(建议预热),loopback 测试固化;HTTP:每 `Client` 一个 keep-alive 池,池参数开放 settings/env/config.toml(`http_max_connections`/`http_max_keepalive_connections`/`http_keepalive_expiry`) | `test_ws_connection.py` 12 例(并发解复用/顺序复用单握手/二进制 owner/错误广播)+ `test_audio_connection.py` 资源集成 7 例;`test_http_pool.py` 本地服务器证明同一 TCP 连接被复用、keepalive=0 端到端生效;指南 `md/dashscope-2.0-connection-reuse-sdk.md` |

**节奏约定**
- 每阶段 = 1 个 commit(与 P 级修复同样约定),完成后 §6 对应行与 7.3 清单回填 ✅ 与 commit。
- M6 的 ws 复用经设计评审后落地(评审结论:并发 + 顺序复用全量 demux、realtime 作为验证项、HTTP 池参数开放 settings);HTTP keep-alive 为 transport 默认行为,随 M6 同批落地。
- 背景(2026-09-23 评审):1.x 大量用户以连接复用模式接入(主要 Java 侧),语音实时交互对延时敏感——连接复用是硬需求而非优化项。
- 范围与优先级以语雀专项文档为准;本表随专项推进滚动更新。
