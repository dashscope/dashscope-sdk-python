# DashScope Python SDK 2.0.3 接口与迁移指南

本文面向 Python SDK 2.0.3，汇总当前包结构、1.x → 2.0 接口映射、核心调用示例与
迁移注意事项。示例基于当前实现核对；完整参数与分领域说明见文末相关文档。

## 1. 安装与最小示例

Python 要求：**3.9 及以上**。

```bash
pip install -U dashscope
export DASHSCOPE_API_KEY="sk-xxx"
# 推荐配置 workspace；也可在 Client(workspace="ws-xxx") 中显式传入
export DASHSCOPE_WORKSPACE="ws-xxx"
```

```python
import dashscope

with dashscope.Client() as client:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages="用一句话介绍杭州",
    )
    print(response.choices[0].message.content)
    print(response.usage.total_tokens, response.request_id)
```

异步客户端具有同名资源与方法：

```python
import dashscope

async with dashscope.AsyncClient() as client:
    response = await client.chat.completions.create(
        model="qwen-plus",
        messages="你好",
    )
    print(response.output_text)
```

只需一次文本调用时，可以使用快捷入口：

```python
response = dashscope.chat("qwen-plus", "给我讲个笑话")
print(response.output_text)
```

## 2. 2.0.3 包结构与兼容边界

### 2.1 当前规范入口

2.0 实现直接位于 `dashscope` 根包，**不再存在 `dashscope.v2` 命名空间**：

```python
import dashscope
from dashscope import AsyncClient, Client, Config, Settings
from dashscope import DashScopeError, ErrorCode, WsConnection, WsTask
from dashscope import errors, settings
from dashscope.tokenizers import get_tokenizer, list_tokenizers
```

资源树挂在 `Client` / `AsyncClient` 上，主要入口如下：

| 领域 | 当前入口 |
| --- | --- |
| 文本对话 | `client.chat.completions` |
| 百炼应用 | `client.apps.completions` |
| 多模态对话与向量 | `client.multimodal.chat` / `client.multimodal.embeddings` |
| 图像与视频 | `client.images` / `client.videos`，均含 `jobs` |
| 文本向量与批量向量 | `client.embeddings` / `client.embeddings.jobs` |
| 排序与分词 | `client.rerank` / `client.tokenization` |
| 语音 | `client.audio.asr` / `tts` / `realtime` / `translation` |
| 语音管理面 | `client.audio.vocabularies` / `phrases` / `voices` / `voice_design` |
| 听悟 | `client.tingwu` |
| 微调与部署 | `client.fine_tunes` / `client.deployments` |
| 文件与模型 | `client.files` / `client.models` |
| Agentic RL | `client.rl`，长名别名 `client.reinforcement_learning` |
| AgentStudio | `client.agentstudio` |

### 2.2 1.x 与 2.0 的目录分工

- 根包保留 `Generation`、`ImageSynthesis`、`FineTunes`、`Deployments` 等既有
  1.x 公开导出，存量代码可继续使用。
- 1.x 深层实现统一位于 `dashscope.legacy_1x`；旧的 `dashscope.aigc.*`、
  `dashscope.files`、`dashscope.models`、`dashscope.client.base_api` 等 deep import
  通过懒加载别名继续可用，存量代码无需修改。
- AgentStudio 与 FineTune/Agentic RL 的物理实现归入 `dashscope.applications`；
  公开 `dashscope.agentstudio`、`dashscope.finetune` 导入路径通过别名保持不变。
- 2.0 Agentic RL 公开入口仍为 `dashscope.finetune.agentic_rl`，函数组件与运行框架
  位于 `dashscope.finetune.reinforcement`。
- 1.x Agentic RL 管理面位于 `dashscope.legacy_1x.finetune.agentic_rl`。
- 不提供 `dashscope.reinforcement` 或 `dashscope.legacy_2x` 兼容命名空间。
- `dashscope` 命令行入口仍由 1.x CLI 提供；2.0 以 `Client` 资源树为编程入口。

直接使用 2.0 Agentic RL 类型时：

```python
from dashscope.finetune.agentic_rl import AgenticRL, SubmitResult
from dashscope.finetune.reinforcement import (
    Dataset,
    TrainingDataset,
    ValidationDataset,
)
```

2.0 `AgenticRL.submit_job()` 返回 `SubmitResult`。1.x wrapper 的同名方法返回
`FineTune`，并额外提供 `cancel/list/get/delete/logs`；两者不能仅因类名相同就
混用返回契约。常规业务代码优先使用 `client.rl`，无需直接管理这些实现类。

## 3. 1.x → 2.0 接口映射

状态说明：**迁移**表示有直接对应入口；**收敛**表示多个旧入口合并为同一资源。

### 3.1 文本、应用与多模态

| 1.x | 2.0.3 | 说明 |
| --- | --- | --- |
| `Generation.call` | `client.chat.completions.create` | 迁移 |
| `AioGeneration.call` | `await aclient.chat.completions.create` | 迁移 |
| `Conversation.call` | `client.chat.completions.create` | 收敛；调用方维护 messages |
| `CodeGeneration.call` | `client.chat.completions.create` | 收敛；通过 model 区分 |
| `Understanding.call` | `client.chat.completions.create` | 收敛；文本理解走对话端点 |
| `Application.call` | `client.apps.completions.create` | 迁移 |
| `MultiModalConversation.call` | `client.chat.completions.create` 或 `client.multimodal.chat.create` | 媒体 part 自动路由；后者为显式入口 |
| `ImageSynthesis.call/async_call/fetch/wait` | `client.images.generate` / `client.images.jobs.*` | 一步到位或分步控制 |
| `ImageGeneration.call/async_call/fetch/wait` | `client.images.generate` / `client.images.jobs.*` | 按模型名自动路由协议 |
| `VideoSynthesis.call/async_call/fetch/wait` | `client.videos.generate` / `client.videos.jobs.*` | 返回统一 `Job` |
| `MultiModalEmbedding.call` | `client.multimodal.embeddings.create` | 文本、图片、音频、视频向量 |

### 3.2 向量、排序与分词

| 1.x | 2.0.3 |
| --- | --- |
| `TextEmbedding.call` | `client.embeddings.create` |
| `BatchTextEmbedding.call/fetch/wait` | `client.embeddings.jobs.create/fetch/wait` |
| `TextReRank.call` / `AioTextReRank.call` | `client.rerank.create` / `await aclient.rerank.create` |
| `Tokenization.call` | `client.tokenization.count` |
| 本地 tokenizer | `dashscope.tokenizers.get_tokenizer` / `list_tokenizers` |

### 3.3 语音与听悟

| 1.x | 2.0.3 |
| --- | --- |
| `Recognition.start/send_audio_frame/stop` | `client.audio.asr.recognize(...)` 上下文管理器 |
| `Transcription.async_call/fetch/wait` | `client.audio.asr.transcribe` / `client.audio.asr.jobs.*` |
| `QwenTranscription` | 同上，通过 `model=` 区分 |
| `SpeechSynthesizer.call` | `client.audio.tts.synthesize/stream` |
| `tts_v2.SpeechSynthesizer` | `client.audio.tts.stream/connect` |
| `HttpSpeechSynthesizer.call` | `client.audio.tts.http.synthesize/stream` |
| `TranslationRecognizerRealtime` | `client.audio.translation.recognize(...)` |
| `OmniRealtimeConversation` | `client.audio.realtime.connect(...)` |
| `VocabularyService` | `client.audio.vocabularies` |
| `AsrPhraseManager` | `client.audio.phrases` |
| `VoiceEnrollmentService` | `client.audio.voices` |
| `TingWu.call` / 实时理解 | `client.tingwu.call` / `client.tingwu.realtime` |

2.0 音频 API 当前仍按 preview 管理，首次访问 `client.audio` 会发出
`FutureWarning`；升级时应核对音频专题文档中的协议与生命周期约束。

### 3.4 微调、部署、资源与 Agent

| 1.x | 2.0.3 |
| --- | --- |
| `FineTunes.call` | `client.fine_tunes.create` |
| `FineTunes.cancel/list/get/delete` | `client.fine_tunes.cancel/list/get/delete` |
| `FineTunes.stream_events/wait/logs` | `client.fine_tunes.stream_events/wait/logs` |
| `Deployments.call/list/get/delete/scale` | `client.deployments.create/list/get/delete/scale` |
| `Models.list/get` | `client.models.list/get` |
| `Files.upload/list/get/delete` | `client.files.upload/list/get/delete` |
| CLI `dashscope rl ...` | `client.rl.*` + `client.rl.jobs.*` |
| AgentStudio 子 SDK | `client.agentstudio.*` 懒挂载 |

Assistants / Threads / Messages / Runs / Steps 不属于 2.0 资源面。新托管 Agent
场景使用 `client.agentstudio`；不要为了迁移旧对话状态而自行模拟这些资源。

## 4. 客户端配置与协议

### 4.1 构造参数与配置优先级

```python
client = dashscope.Client(
    api_key="sk-xxx",
    workspace="ws-xxx",
    region="cn-beijing",
    protocol="native",
    timeout=60.0,
    max_retries=2,
    raise_on_error=True,
    config_path="~/.dashscope/config.toml",
    protocol_options={"strict": False},
    http_client=None,
)
```

配置优先级：**显式参数 > `dashscope.settings` 赋值 > 环境变量 > config.toml >
默认值**。不显式传 `api_key` 时读取 `DASHSCOPE_API_KEY`。建议配置 `workspace`
以使用专属网关；未配置 workspace 且未显式给 `base_url` 时，SDK 回退共享网关并
发出一次 `UserWarning`。

`region` 默认 `cn-beijing`，支持 `cn-beijing`、`ap-southeast-1`、
`cn-hongkong`、`us-east-1`、`eu-central-1`、`ap-northeast-1`。自建或私有网关
可显式传 `base_url`。

运行时可使用全局配置门面，或派生只改部分选项的客户端：

```python
dashscope.settings.timeout = 30.0
dashscope.settings.max_retries = 3

strict = client.with_options(protocol="openai", timeout=10.0)
```

`with_options()` 在未改变连接级选项时共享原客户端连接池，派生实例不应比原实例
活得更久。普通请求上的 `timeout=` 是单次 HTTP 请求超时；`images.generate`、
`videos.generate`、`audio.asr.transcribe` 和各类 `wait` 上的 `timeout=` 是等待
任务终态的总预算。

### 4.2 三种对话协议

`protocol` 只改变 `chat.completions` / `multimodal.chat` 的线缆格式，调用方式和
统一返回类型不变：

| 取值 | 端点形态 | 说明 |
| --- | --- | --- |
| `native` | DashScope `/services/...` | 默认，支持完整 DashScope 参数 |
| `openai` | `/chat/completions` | 对接百炼 compatible-mode 或 OpenAI 兼容网关 |
| `anthropic` | `/v1/messages` | system 独立、工具 schema 转换 |

```python
openai_client = dashscope.Client(protocol="openai")
response = openai_client.chat.completions.create(
    model="qwen-plus",
    messages="你好",
)
```

`openai` 协议默认透传非 OpenAI 原生参数；`protocol_options={"strict": True}`
可改为拒绝未知参数。`anthropic` 对无对应字段的参数直接报 400，避免参数被静默
丢弃；该协议下 `max_tokens` 缺省自动补 `4096`，`stop` 映射为 `stop_sequences`。

## 5. 核心调用示例

以下示例复用已构造的 `client` / `aclient`。

### 5.1 对话、工具与结构化输出

```python
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "北京今天天气怎么样？"},
    ],
    temperature=0.7,
    max_tokens=512,
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
print(response.choices[0].message.tool_calls)

structured = client.chat.completions.create(
    model="qwen-plus",
    messages="给我一个用户 JSON",
    response_format={"type": "json_object"},
)
print(structured.output_text)
```

签名之外的模型参数通过 `parameters=` 传递。多轮对话由调用方维护 messages 列表，
2.0 chat 本身不保存历史。

### 5.2 增量流式输出

```python
stream = client.chat.completions.create(
    model="qwen-plus",
    messages="写一首短诗",
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

`ChatStream.text_stream` 只迭代文本，`get_final_message()` 聚合完整 message（含
工具调用），`headers` 提供响应头。与 1.x 默认重放全量文本不同，2.0 chat 每帧
都是增量。异步流式使用 `await aclient.chat.completions.create(..., stream=True)`
和 `async for`。

### 5.3 百炼应用

```python
response = client.apps.completions.create(
    app_id="your-app-id",
    prompt="帮我写一份周报大纲",
)
print(response.text)

for partial in client.apps.completions.create(
    app_id="your-app-id",
    prompt="讲个故事",
    stream=True,
    incremental_output=True,
):
    print(partial.text, end="")
```

`prompt` 与 `messages` 至少传一个；同时给出时，`prompt` 作为最新 user 消息追加。
`session_id` 用于服务端会话状态，`biz_params` 用于应用流程或插件入参。

### 5.4 多模态对话与向量

```python
response = client.chat.completions.create(
    model="qwen-vl-plus",
    messages=[{
        "role": "user",
        "content": [
            {"image": "https://example.com/cat.jpg"},
            {"text": "这张图里有什么？"},
        ],
    }],
)
print(response.output_text)

embeddings = client.multimodal.embeddings.create(
    model="multimodal-embedding-v1",
    input=[
        {"text": "一只猫"},
        {"image": "https://example.com/cat.jpg"},
    ],
)
for item in embeddings.embeddings:
    print(item.index, item.type, len(item.embedding))
```

只要 messages 含 image / audio / video part，`chat.completions.create` 就自动走
多模态端点。`client.multimodal.chat.create` 是始终使用该端点的显式入口。类型化
向量输入可从 `dashscope.resources.multimodal.embeddings` 导入 `EmbeddingItem`。

### 5.5 图像、视频与统一 Job

```python
image_job = client.images.generate(
    model="wanx2.1-t2i-turbo",
    prompt="一只在月球上弹吉他的猫，水彩风格",
    size="1024*1024",
)
print(image_job.image_urls)

video_job = client.videos.generate(
    model="wan2.7-t2v",
    prompt="一只小猫在草地上奔跑",
    size="1280*720",
)
print(video_job.video_url)

# 分步提交与轮询；images / videos / embeddings / asr 的 jobs 语义一致
job = client.images.jobs.create(
    model="wanx2.1-t2i-turbo",
    prompt="一只猫",
)
snapshot = client.images.jobs.fetch(job.task_id)
done = client.images.jobs.wait(job, timeout=600.0)
```

`generate` 是“提交并等待”，`jobs.create` 只提交，`wait` 使用退避轮询。统一
`Job` 常用字段为 `task_id` / `status` / `results` / `result_url` / `image_urls` /
`video_url`；只有服务相关的原始载荷保留在 `Job.output`。任务失败或被取消时，
默认抛出 `DashScopeError` 异常族。

### 5.6 文本向量、排序与分词

```python
embedded = client.embeddings.create(
    model="text-embedding-v3",
    input=["今天天气不错", "明天会下雨吗"],
    dimension=1024,
    text_type="document",
)
print([len(item.embedding) for item in embedded.embeddings])

ranked = client.rerank.create(
    model="gte-rerank-v2",
    query="什么是量子计算？",
    documents=["量子计算利用量子力学原理。", "今天超市鸡蛋打折。"],
    top_n=1,
    return_documents=True,
)
print(ranked.results[0].index, ranked.results[0].relevance_score)

count = client.tokenization.count(model="qwen-plus", prompt="你好，世界")
print(count.input_tokens)
```

批量向量使用 `client.embeddings.jobs.create(model=..., file_url=...)`，完成后从
`job.result_url` 读取结果文件。本地分词不发送请求：

```python
from dashscope.tokenizers import get_tokenizer

tokenizer = get_tokenizer("qwen-plus")
token_ids = tokenizer.encode("你好，世界")
print(tokenizer.decode(token_ids))
```

### 5.7 语音与实时会话

```python
# 文件转写：提交并等待
job = client.audio.asr.transcribe(
    ["https://example.com/meeting.wav"],
    model="paraformer-v2",
)
print([item.transcription_url for item in job.results])

# 一次性语音合成
audio = client.audio.tts.synthesize(
    "你好，欢迎收听今天的节目。",
    model="cosyvoice-v1",
    voice="longxiaochun",
    format="mp3",
)
with open("out.mp3", "wb") as file:
    file.write(audio)

# 流式语音合成
with open("stream.mp3", "wb") as file:
    for chunk in client.audio.tts.stream(
        "这是一段较长的文本。",
        model="cosyvoice-v1",
        voice="longxiaochun",
    ):
        file.write(chunk)
```

实时 ASR、双向流 TTS、Realtime 与实时翻译采用 websocket 会话，不是普通 HTTP
流式响应。它们通过上下文管理器管理生命周期，并用 `events()` / `results()` 迭代
服务端事件。发送与接收通常需要并发，完整示例见文末语音专题文档。

支持共享连接的会话可复用 `client.ws_connection()`；共享连接由调用方关闭。实时
协议是否支持复用取决于任务帧是否携带 `task_id`，不要把 Realtime 会话强行放入
共享连接。

### 5.8 文件、微调与部署

```python
uploaded = client.files.upload(
    file_path="./train.jsonl",
    purpose="fine-tune",
)

fine_tune = client.fine_tunes.create(
    model="qwen-turbo",
    training_file_ids=[uploaded.file_id],
    hyper_parameters={"n_epochs": 2, "learning_rate": "1e-5"},
)
finished = client.fine_tunes.wait(fine_tune, timeout=7200)
print(finished.finetuned_model)

for event in client.fine_tunes.stream_events(fine_tune.job_id):
    print(event.output)

# 部署后使用 deployed_model 作为 get / scale / delete 的定位键
deployment = client.deployments.create(model="qwen-turbo", capacity=2)
client.deployments.scale(deployment.deployed_model, capacity=4)
print(client.deployments.get(deployment.deployed_model).status)
```

2.0 响应信封已扁平化：直接读取 `fine_tune.status`、`uploaded.file_id`，不再经过
`.output.status` 或 `.data["file_id"]`。顶层响应共享 `request_id` / `code` /
`message` / `is_error` / `status_code` / `headers` 字段。

### 5.9 Agentic RL

```python
registered = client.rl.register_functions(
    rollout_classpaths=["my_project.rollout:MyRollout"],
    reward_classpaths=["my_project.reward:MyReward"],
)

client.rl.test_functions(
    instance_id=registered.rollout_instance_ids[0],
    func_type="rollout",
    input_data={"query": "1+1=?"},
)

uploaded = client.rl.upload_data(
    training_files=["./rl_train.jsonl"],
    validation_files=["./rl_val.jsonl"],
)

job = client.rl.jobs.create(
    config_path="./rl_config.yaml",
    job_name="my-rl-job",
)
print(client.rl.jobs.fetch(job.job_id).status)
done = client.rl.jobs.wait(job)
```

`client.rl.*` 负责函数注册、测试和数据上传，`client.rl.jobs.*` 负责训练任务生命周期。
`jobs.create/register_functions/test_functions/upload_data` 以及函数实例管理方法
复用底层工作流，不接受 `timeout=`；其余 `jobs` 方法支持 `timeout=`。整个
`client.rl` 的错误统一由 `DashScopeError` 覆盖，本地工作流失败通常表现为
`SdkError`。

### 5.10 AgentStudio

```python
from dashscope.agentstudio import user_message

studio = client.agentstudio
agent = studio.agents.create(
    name="demo",
    model="qwen-plus",
    system_prompt="你是助手",
)
session = studio.sessions.create(agent=agent.id, title="第一次对话")
studio.sessions.events.send(session.id, [user_message("你好")])

with studio.sessions.events.stream(session.id) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

AgentStudio 在首次访问 `client.agentstudio` 时构造，包含 `agents`、`sessions`、
`events`、`files`、`skills`、`vaults`、`environments`、`deployments` 等完整子
资源。其返回模型沿用 AgentStudio 自身类型体系，不套用 2.0 的响应扁平化；通过
`client.agentstudio` 调用时，异常统一翻译为 `dashscope.errors` 异常族。关闭主
Client 会连带关闭已挂载的 AgentStudio Client。

## 6. 错误、重试与流式失败

### 6.1 统一异常族

```python
from dashscope.errors import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    ContentFilterError,
    DashScopeError,
    NotFoundError,
    RateLimitError,
    SdkError,
)

try:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages="你好",
    )
except ContentFilterError as error:
    print("内容审核拦截", error.request_id)
except BadRequestError as error:
    print("请求参数错误", error.code, error.message)
except RateLimitError as error:
    print("限流", error.status_code, error.request_id)
except DashScopeError as error:
    print(type(error).__name__, error.code, error.message, error.request_id)
```

九个 HTTP 状态类为 `BadRequestError`（400）、`AuthenticationError`（401）、
`PermissionDeniedError`（403）、`NotFoundError`（404）、
`RequestTooLargeError`（413）、`RateLimitError`（429）、
`InternalServerError`（500）、`ServiceUnavailableError`（503）和
`GatewayTimeoutError`（504）。`ContentFilterError` 是 `BadRequestError` 子类，
单独捕获时必须放在前面。

异常常用字段：`message` / `code` / `status_code` / `request_id` / `body` /
`headers`。连接错误、超时和 SDK 本地错误没有 HTTP 响应，`status_code` 为
`None`；`SdkError` 另提供 `solution` / `allow_retry`。HTTP 非 2xx 与 HTTP 200
但 body 带业务错误码的情况使用同一异常族。

### 6.2 自动重试与返回值模式

默认 `max_retries=2`，传输层按 `Retry-After` 或指数退避加抖动重试连接错误及
408 / 409 / 429 / 500 / 502 / 503 / 504。认证、权限、资源不存在和参数错误不应
盲目做应用层重复重试。

若需要 1.x 风格的返回值自查：

```python
lenient = client.with_options(raise_on_error=False)
response = lenient.chat.completions.create(model="qwen-plus", messages="你好")
if response.is_error:
    print(response.code, response.message, response.request_id)
```

`raise_on_error=False` 不会吞掉连接失败和超时，因为没有响应对象可返回。流式
业务错误在返回值模式下产出一个 `is_error=True` 的对象后结束迭代。

## 7. 全局配置与运维

### 7.1 常用环境变量

| 环境变量 | 作用 | 默认 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | API Key | — |
| `DASHSCOPE_API_KEY_FILE_PATH` | API Key 文件读取路径 | — |
| `DASHSCOPE_WORKSPACE` | workspace id | — |
| `DASHSCOPE_REGION` | 地域 | `cn-beijing` |
| `DASHSCOPE_BASE_URL` | 自定义 HTTP 端点 | — |
| `DASHSCOPE_PROTOCOL` | `native` / `openai` / `anthropic` | `native` |
| `DASHSCOPE_TIMEOUT` | 请求超时（秒） | — |
| `DASHSCOPE_MAX_RETRIES` | 自动重试次数 | `2` |
| `DASHSCOPE_CONFIG` | config.toml 路径 | `~/.dashscope/config.toml` |
| `DASHSCOPE_LOG` | 当前客户端日志级别 | `warning` |
| `DASHSCOPE_WEBSOCKET_BASE_URL` | ASR/TTS 等 websocket 端点覆盖 | 跟随配置 |
| `DASHSCOPE_AGENTSTUDIO_URL` / `AGENTSTUDIO_URL` | AgentStudio 自定义端点 | — |

HTTP 连接池可通过 `DASHSCOPE_HTTP_MAX_CONNECTIONS`、
`DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS`、
`DASHSCOPE_HTTP_KEEPALIVE_EXPIRY` 或对应 settings 字段调整。长驻并复用 Client
才能充分利用 keep-alive。

### 7.2 日志与 API Key 文件

当前客户端使用独立的 `dashscope._logging` logger 层级，配置它不会改变 1.x 根
logger 的 level、handlers 或 filters：

```python
dashscope.settings.log_level = "DEBUG"
```

根 `save_api_key(api_key, api_key_file_path=None)` 来自 `dashscope._config`，默认
写入 `~/.dashscope/api_key`。它使用 UTF-8、展开 `~`，并为自定义路径创建缺失的
父目录；`DASHSCOPE_API_KEY_FILE_PATH` 只影响读取，不改变默认写入位置。

## 8. 升级检查清单

1. 将所有新代码的 `dashscope.v2...` 导入改为根 `dashscope...`；不要保留已删除的
   `dashscope.v2` 兼容假设。
2. 新代码使用 `Client` / `AsyncClient`，不要继续依赖模块级 `api_key` 和
   `.call()` 类方法组织新调用。
3. 将响应检查从“先判断 `status_code`”调整为异常处理；若暂时保留返回值模式，
   必须检查 `is_error`。
4. 将 chat 流式消费改为增量 delta；应用流式仍按 `incremental_output` 判断帧语义。
5. 将图像、视频、批量向量、转写等长任务统一到 `create/fetch/wait`，并为等待
   操作设置符合业务预算的 `timeout`。
6. 检查多轮对话是否改为由应用维护 messages；不要依赖 1.x `Conversation` history。
7. 旧 deep import 可继续通过别名使用；新代码可按公开入口导入。AgentStudio 与
   FineTune/Agentic RL 物理实现位于 `dashscope.applications`，原公开路径不变；
   不存在 `dashscope.reinforcement`。
8. Agentic RL 不要混淆 2.0 `SubmitResult` 与 1.x `FineTune` 返回契约；优先迁移到
   `client.rl`。
9. 视频调用若依赖 1.x `extend_prompt=True` 的旧默认行为，迁移后显式传
   `prompt_extend=True`。
10. 使用 `client.close()` / `await aclient.aclose()` 或上下文管理器释放连接；
    音频 preview、AgentStudio 与 RL 分别按专题文档复核生命周期和参数约束。

## 9. 相关文档

- [2.0 接口映射核对表](dashscope-2.0-interfaces-mapping.md)
- [2.0 接口调用示例集](dashscope-2.0-interfaces-examples.md)
- [1.x → 2.0 迁移指南](dashscope-2.0-migration-guide.md)
- [2.0 设计规范](dashscope-2.0-design.md)
- [ASR：实时识别与文件转写](dashscope-2.0-asr-sdk.md)
- [TTS：一次性、流式与双向流合成](dashscope-2.0-tts-sdk.md)
- [Realtime：Omni、Qwen-Audio 与 Livetranslate](dashscope-2.0-realtime-sdk.md)
- [实时语音翻译](dashscope-2.0-translation-sdk.md)
- [语音管理面](dashscope-2.0-audio-management-sdk.md)
- [websocket 连接复用](dashscope-2.0-connection-reuse-sdk.md)
