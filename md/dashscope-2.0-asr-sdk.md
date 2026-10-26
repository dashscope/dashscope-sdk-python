# 语音识别（ASR）Python SDK

通过 `client.audio.asr`，您可以使用 WebSocket 实时识别音频流，或通过 HTTP 提交录音文件转写任务。本文对应当前仓库的 SDK 实现；模型能力、音频限制及服务端字段以相应模型协议为准。

| 场景 | 同步客户端 `Client` | 异步客户端 `AsyncClient` |
| --- | --- | --- |
| 实时语音识别 | `audio.asr.recognize()` | 不支持 |
| 录音文件转写 | `audio.asr.transcribe()`、`audio.asr.jobs` | 支持相同批量接口，需使用 `await` |

## 前期准备

1. 准备 API Key，并通过环境变量 `DASHSCOPE_API_KEY` 配置，避免将密钥写入代码。
2. 建议通过 `DASHSCOPE_WORKSPACE` 配置工作空间，并确保 API Key、工作空间和模型所在地域匹配。地域默认是 `cn-beijing`，可通过 `DASHSCOPE_REGION` 或 `Client(region=...)` 设置。不配置工作空间且未指定端点时，当前 SDK 会回退到共享网关并发出警告。
3. 在本仓库根目录执行源码安装：

   ```bash
   python -m pip install -e .
   ```

4. 实时示例需要当前目录下的 `input.pcm`：有限长度、无文件头、16 kHz、16 bit、单声道的原始 PCM 音频。文件转写示例需要设置 `AUDIO_FILE_URL`，其值为服务端可访问的音频文件 URL。

客户端使用公开入口 `from dashscope import Client, AsyncClient`。也可以在创建客户端时显式传入 `api_key`、`workspace`、`region`；本文示例从环境配置读取凭据，不调用麦克风或下载音频。

## 快速开始

### 实时语音识别

以下代码可保存为 `asr_example.py`，在音频文件所在目录运行 `python asr_example.py`。示例使用 `fun-asr-realtime`，请先确认目标地域可用且已具备访问权限。

发送和接收分别运行在两个线程中，逐块读取文件、逐事件处理结果，不在内存中收集整段音频或全部事件。

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dashscope import Client


def send_audio(session, audio_path):
    with open(audio_path, "rb") as audio:
        while True:
            frame = audio.read(3200)
            if not frame:
                break
            session.send_audio_frame(frame)
            time.sleep(0.1)
    session.stop()


def receive_events(session):
    for event in session.events():
        payload = event.get("payload") or {}
        sentence = (payload.get("output") or {}).get("sentence") or {}
        if sentence.get("text") and not sentence.get("heartbeat"):
            label = "最终" if sentence.get("sentence_end") else "中间"
            print(label, sentence["text"], flush=True)


def main(audio_path):
    with Client() as client:
        session = client.audio.asr.recognize(
            model="fun-asr-realtime",
            format="pcm",
            sample_rate=16000,
            timeout=30.0,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                session.start()
                print("task_id:", session.session.task_id)
                futures = [
                    executor.submit(receive_events, session),
                    executor.submit(send_audio, session, audio_path),
                ]
                for future in as_completed(futures):
                    future.result()
            finally:
                session.close()
        print("request_id:", session.request_id)
        print("usage:", session.usage)
        print("first_package_delay(ms):", session.first_package_delay)
        print("last_package_delay(ms):", session.last_package_delay)


if __name__ == "__main__":
    main("input.pcm")
```

- `recognize()` 只创建 `Recognition` 对象；`start()` 才建立连接、发送 `run-task`，并在内部读取和等待 `task-started`。启动成功后才能发送音频。
- 示例每次发送 3200 字节并暂停 0.1 秒，对应上述 PCM 格式约 100 毫秒的音频。这是模拟实时输入的建议节奏，不是 SDK 强制的分块大小或发送间隔；生产环境需遵守模型的服务端限制。
- `stop()` 仅结束输入。接收线程继续读取直到 `task-finished`，以获得最后的识别结果和统计信息。
- `Future.result()` 将工作线程异常传回主线程。`session.close()` 的 `finally` 必须位于线程池上下文内部，确保发送失败时先关闭连接、解除接收阻塞，再等待线程退出。不要只在退出线程池后关闭会话。

### 录音文件转写

`transcribe()` 提交任务并等待终态，返回 `Job`，不是转写正文。

```python
import os

from dashscope import Client


with Client() as client:
    job = client.audio.asr.transcribe(
        file_urls=[os.environ['AUDIO_FILE_URL']],
        model="paraformer-v2",
        timeout=86400,
        submit_timeout=30,
    )
    print("task_id:", job.task_id, "status:", job.status)
    for result in job.results:
        print("transcription_url:", result.transcription_url)
```

请根据 `job.output` 和各文件结果判断转写情况。`transcription_url` 是服务端返回的结果地址，SDK 不自动下载或解析该地址中的转写文件。

### 异步录音文件转写

`AsyncClient` 的批量转写接口使用相同参数和 `Job` 返回结构；实时 `recognize()` 不提供异步版本。

```python
import asyncio
import os

from dashscope import AsyncClient


async def main():
    async with AsyncClient() as client:
        job = await client.audio.asr.transcribe(
            file_urls=[os.environ['AUDIO_FILE_URL']],
            model="paraformer-v2",
        )
        for result in job.results:
            print(result.transcription_url)


if __name__ == "__main__":
    asyncio.run(main())
```

## 请求参数

### 实时识别：`client.audio.asr.recognize()`

以下参数均为关键字参数。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `model` | `str` | 必填，无默认值。实时识别模型名称，例如 `fun-asr-realtime`；可用性取决于地域及服务权限。 |
| `format` | `str` | 可选，默认 `"pcm"`。输入音频格式，必须与实际数据一致；支持范围由模型决定。SDK 不转码。 |
| `sample_rate` | `int` | 可选，默认 `16000`，单位 Hz。输入音频采样率，SDK 不重采样。 |
| `parameters` | `dict` | 可选，默认 `None`。模型参数透传到 `run-task.payload.parameters`；其中同名键会覆盖由 `format`、`sample_rate` 形参生成的值，包括显式传入的值。 |
| `input` | `dict` | 可选，默认 `None`，发送时为 `{}`。任务启动时的初始上下文，透传到 `run-task.payload.input`，结构由模型协议定义。 |
| `phrase_id` | `str` | 可选，默认 `None`。已有热词资源 ID，转换为 `resources=[{"resource_id": phrase_id, "resource_type": "asr_phrase"}]`，不是 `parameters` 中的字段。资源须与模型匹配。 |
| `ws_url` | `str` | 可选，默认 `None`。完整 WebSocket 端点；未指定时采用全局 WebSocket 端点配置，否则按工作空间和地域选择默认端点。 |
| `connector` | `Callable` | 可选，默认 `None`，使用 SDK 默认连接器。自定义连接器按 `connector(url, headers)` 调用，返回兼容 `send`、`recv`、`settimeout`、`close` 的连接对象；自行负责连接建立阶段的超时。 |
| `timeout` | `float` | 可选，默认 `None`，继承客户端超时；客户端未另行配置时为 `300.0` 秒。值必须大于 0；默认连接器将其用于连接建立，并设置连接的读写超时，不是整个识别任务的总时限。 |
| `headers` | `Mapping[str, str]` | 可选，默认 `None`。附加握手请求头。SDK 会根据客户端配置写入认证和工作空间请求头，不应以此替代凭据配置。 |

`DASHSCOPE_WEBSOCKET_BASE_URL` 可配置 WebSocket 端点，调用级 `ws_url` 优先。HTTP 的 `Client(base_url=...)` 不会自动成为实时识别的 WebSocket 地址。

### 文件转写：`client.audio.asr.transcribe()`

`file_urls` 可按位置或关键字传入，其余均为关键字参数。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `file_urls` | `list[str]` | 必填，无默认值，不能为空。服务端可访问的音频文件 URL 列表，不是本地路径；文件数量、格式及大小限制由模型决定。 |
| `model` | `str` | 可选，默认 `"paraformer-v2"`。录音文件转写模型名称。 |
| `parameters` | `dict` | 可选，默认 `None`。服务端转写参数，原样传递；字段和默认行为以所选模型为准。 |
| `phrase_id` | `str` | 可选，默认 `None`。已有热词资源 ID，与实时接口相同，映射为 `asr_phrase` 类型的 `resources`。 |
| `timeout` | `float` | 可选，默认 `86400.0` 秒，即 24 小时。提交成功后等待任务终态的预算，不是提交请求超时。 |
| `submit_timeout` | `float` | 可选，默认 `None`，继承客户端超时。仅用于提交任务这一次 HTTP 请求，单位秒。 |
| `poll_interval` | `float` | 可选，默认 `1.0` 秒。初始轮询间隔，随后按 2 倍递增；建议使用正数。 |
| `max_interval` | `float` | 可选，默认 `8.0` 秒。后续轮询间隔上限，建议不小于 `poll_interval`。 |

### 分步任务接口：`client.audio.asr.jobs`

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `jobs.create`：`model` | `str` | 必填关键字参数，无默认值；与 `transcribe()` 的默认模型行为不同。 |
| `jobs.create`：`file_urls` | `list[str]` | 必填关键字参数，无默认值，不能为空。 |
| `jobs.create`：`parameters`、`phrase_id` | `dict`、`str` | 均可选，默认 `None`，含义与 `transcribe()` 相同。 |
| `jobs.create`：`timeout` | `float` | 可选关键字参数，默认 `None`，继承客户端超时；仅为提交请求超时，单位秒。 |
| `jobs.fetch`：`task_id` | `str` | 必填，无默认值。要查询的任务 ID。 |
| `jobs.fetch`：`timeout` | `float` | 可选关键字参数，默认 `None`，继承客户端超时；仅为本次查询请求超时，单位秒。 |
| `jobs.wait`：`job` | `Job` 或 `str` | 必填，无默认值。任务快照或任务 ID。 |
| `jobs.wait`：`timeout` | `float` | 可选关键字参数，默认 `86400.0` 秒。等待终态的预算。 |
| `jobs.wait`：`poll_interval`、`max_interval` | `float` | 可选关键字参数，默认分别为 `1.0`、`8.0` 秒，含义与 `transcribe()` 相同。 |

等待预算不会替换单次 HTTP 查询超时，也不是可中断在途请求的硬截止时间；查询仍使用客户端超时，因此实际返回时间可能超过等待预算。

## 关键接口

### 实时会话生命周期

`Recognition` 的推荐调用顺序为：创建对象 → `start()` → 并发发送音频和读取事件 → `stop()` → 读到 `task-finished` → `close()`。

| 接口 | 说明 |
| --- | --- |
| `start()` | 新建 WebSocket 连接并等待 `task-started`，成功后返回当前 `Recognition`。已打开时再次调用会抛出 `RuntimeError`。 |
| `send_audio_frame(data)` | 将音频作为 WebSocket 二进制帧发送，不是 Base64 或 JSON；通常传入 `bytes`，也接受 `bytearray`、`memoryview`。 |
| `update_context(context)` | `context` 必须是 `dict`。发送 `continue-task`，其 `payload.input` 就是该字典，用于后续音频的上下文，不会重启任务。 |
| `stop()` | 发送 `finish-task` 结束输入，不等待最终结果、不关闭连接；结束输入后不能再发送音频或更新上下文。 |
| `events()` | 阻塞读取并逐条产出原始事件字典，包含最终的 `task-finished`。请由一个接收线程持续消费，不要并发创建多个读取者。 |
| `close()` | 关闭底层连接，可重复调用；不会代替正常的结束输入及读取最终结果流程。 |

`with client.audio.asr.recognize(...) as session:` 会自动执行 `start()`，退出时执行 `close()`，但不会自动发送音频、调用 `stop()` 或读完事件。使用线程池时仍应采用快速开始中的内部 `finally` 关闭方式，避免错误退出时等待读线程。

如模型支持文本上下文，可在启动时传入 `input={"context": {"text": "上一轮内容"}}`，或在会话启动后调用 `session.update_context({"context": {"text": "更新后的内容"}})`。更新应在发送相应后续音频之前完成；具体内容约束以模型协议为准。

`close()` 后可对同一个 `Recognition` 再次调用 `start()`。这会创建新连接、新任务并重置本次统计，不是原任务续传，也不复用 WebSocket 连接池。请先等待旧的收发线程退出，再重新启动；`Client.close()` 不替代实时会话的显式关闭。

### 提交、查询与等待转写任务

- `jobs.create(...)`：提交并返回任务快照，不等待转写完成。
- `jobs.fetch(task_id, ...)`：查询一次，返回新的任务快照，不持续轮询。
- `jobs.wait(job_or_task_id, ...)`：轮询直到终态，返回终态快照或抛出异常。

```python
import os

from dashscope import Client


with Client() as client:
    job = client.audio.asr.jobs.create(
        model="paraformer-v2",
        file_urls=[os.environ['AUDIO_FILE_URL']],
        timeout=30,
    )
    print("task_id:", job.task_id)
    job = client.audio.asr.jobs.fetch(job.task_id, timeout=30)
    job = client.audio.asr.jobs.wait(job, timeout=86400)
    print(job.status, job.succeeded)
```

异步客户端对应 `await client.audio.asr.jobs.create(...)`、`await client.audio.asr.jobs.fetch(...)` 和 `await client.audio.asr.jobs.wait(...)`。这里的服务端异步任务与 Python 异步调用是两个概念：同步 `Client` 同样可以提交服务端任务。

## 返回事件与属性

### 实时事件

`events()` 返回原始 `dict`，通过 `event["header"]["event"]` 区分事件，而不是通过事件对象的属性访问。

| 事件 | SDK 行为 |
| --- | --- |
| `task-started` | 由 `start()` 内部读取，启动返回时保存在 `last_response` 中；不会再次由 `events()` 产出。 |
| `result-generated` | 由 `events()` 产出；识别结果通常位于 `payload.output.sentence`，也可能是心跳或空文本事件。 |
| `task-finished` | 由 `events()` 产出，随后迭代结束；发送 `stop()` 后仍须读取此事件。 |
| `task-failed` | 转换为 SDK 业务异常并关闭连接，不作为普通事件产出。 |

按模型返回情况，`payload.output.sentence` 中可包含 `text`、`sentence_end`、`heartbeat` 等字段。中间文本可能被后续结果修订，不应简单拼接所有 `text`；心跳和空文本事件仍会产出，应用可按需忽略。完整字段见相关服务端事件文档。

### `Recognition` 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `last_response` | `dict` 或 `None` | 最近一次成功读取的原始事件。初始为 `None`；启动后为握手事件，随后在消费 `events()` 时更新。 |
| `usage` | `dict` 或 `None` | 最近一次读取到的非 `None` 的 `payload.usage`；后续事件缺失该字段或值为 `None` 时保留已有值，不自动累计，也不保证只在最后一条事件出现。 |
| `first_package_delay` | `float` 或 `None` | 单位毫秒。从第一次调用 `send_audio_frame()` 到读取首个非心跳且文本非空的识别结果的耗时；尚未满足条件时为 `None`。 |
| `last_package_delay` | `float` 或 `None` | 单位毫秒。从第一次调用 `stop()` 到读取 `task-finished` 的耗时；未调用 `stop()` 或未读到终态时为 `None`。 |
| `request_id` | `str` 或 `None` | 当前记录的请求标识，从已成功读取事件更新；缺少服务端 `request_id` 时可回退到任务 ID。异常排查时优先查看异常携带的请求标识。 |
| `session.task_id` | `str` | 底层会话的任务 ID。须在启动后、关闭前读取；未打开时访问 `Recognition.session` 会抛出 `RuntimeError`。 |

统计是客户端**实际读取事件时**更新的，不是后台自动刷新；延迟包含网络、线程调度及消费滞后，不能等同于纯服务端推理耗时。`start()` 读取的握手只更新启动响应和请求标识，不通过 `events()` 的用量及延迟统计逻辑。关闭后已保存的统计仍可读取，再次启动会重置。

### 文件转写 `Job`

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `str` 或 `None` | 本次提交或查询响应的请求标识，不是任务 ID。 |
| `task_id` | `str` | 服务端任务 ID，可用于后续查询和等待。 |
| `status` | `str` | 任务状态，包括 `PENDING`、`RUNNING`、`SUCCEEDED`、`FAILED`、`CANCELED`。 |
| `output` | `dict` 或 `None` | 原始任务输出，保留服务端结果和每个文件的详情。 |
| `results` | `list[JobResult]` | `output.results` 的结构化视图；没有结果时为空列表。 |
| `results[i].transcription_url` | `str` 或 `None` | 第 i 个结果的转写文件地址；该字段属于结果项，不是 `Job` 顶层字段。 |
| `usage` | `Usage` 或 `None` | 服务端用量对象，保留服务特有字段；不存在时为 `None`。 |
| `finished` | `bool` | 状态为 `SUCCEEDED`、`FAILED` 或 `CANCELED` 时为 `True`，不代表成功。 |
| `succeeded` | `bool` | 仅在状态为 `SUCCEEDED` 时为 `True`。 |

`Job` 是一次响应的快照，不会自行更新，也不包含 SDK 已下载的转写正文。请接收 `fetch()`、`wait()` 返回的新对象，并逐项检查结果；不要把缺少 `transcription_url` 的条目当作已取得转写内容。

## 异常处理

| 情况 | 行为与处理建议 |
| --- | --- |
| WebSocket 提前结束 | 底层 `recv()` 在 `task-finished` 前返回 `None` 或空字符串时，SDK 抛出 `APIConnectionError`，不能当作正常识别完成。 |
| 服务端 `task-failed` | SDK 根据业务错误码抛出 `DashScopeError` 或其业务子类，如 `AuthenticationError`、`BadRequestError`、`RateLimitError`；检查异常的 `code`、`message`、`request_id`。 |
| WebSocket 原生错误 | 连接器建立连接、底层 `send()`、`recv()` 抛出的异常通常原样传播，例如 `websocket.WebSocketTimeoutException`、其他 WebSocket 异常或 `OSError`；不会全部自动转换成 `APIConnectionError` 或 `APITimeoutError`。 |
| 协议错误 | 缺少事件头或握手事件不符等情况可能抛出 `WsEventError`（属于 `DashScopeError`）；不要继续使用失败的会话。 |
| 本地参数或状态错误 | 空 `file_urls`、非字典上下文、未启动就发送、结束输入后继续发送等，可能分别触发 `ValueError`、`TypeError` 或 `RuntimeError`。 |
| HTTP 或转写业务失败 | 默认 `raise_on_error=True` 时，HTTP 错误、响应业务错误码以及解析到 `FAILED` 任务状态会抛出 SDK 异常；`wait()` 遇到 `CANCELED` 也会抛出 `DashScopeError`。 |
| 等待预算耗尽 | `transcribe()` 或 `jobs.wait()` 抛出 `DashScopeError`；这不表示服务端任务已取消。保留任务 ID 后可再次 `fetch()` 或 `wait()`。 |

- 在收发线程外读取 `Future.result()` 并确保 `finally` 关闭会话。仅捕获 `DashScopeError` 无法覆盖全部 WebSocket 原生异常；不要静默忽略线程错误。
- `Client(raise_on_error=False)` 可用于检查 HTTP 返回对象中的 `code`、`message`、`is_error` 和任务状态，但不会关闭 WebSocket 的失败事件异常，也不会抑制本地错误或等待超时。
- `max_retries` 不提供 WebSocket 自动重连或音频重放。实时任务断开后，由应用决定是否新建任务及重新发送可重放的音频，避免重复结果；不要宣称可以续传原任务。
- 排查时记录异常类型、请求 ID 和任务 ID；不要记录 API Key 或未经脱敏的完整音频、上下文及转写正文。

## 常见问题

**为什么 `stop()` 返回了，却没有最终识别结果？**

`stop()` 只通知服务端输入结束。必须继续消费 `events()` 到 `task-finished`，不要立即关闭连接；握手事件则已由 `start()` 读取，不要额外等待一次 `task-started`。

**可以把 WAV 文件直接作为 PCM 发送吗？**

不能仅修改扩展名。快速开始要求无文件头的原始 PCM；实际格式、采样率和声道必须匹配模型要求。SDK 不负责去除 WAV 文件头、转码或重采样。

**为什么用量或延迟仍为 `None`？**

可能尚未消费包含用量的事件、没有非心跳的有效文本，或还未在 `stop()` 后读到 `task-finished`。这些值不是服务端必定返回的完整统计，不能用空值判定任务成功或失败。

**转写返回的是 URL，怎样获取文本？**

检查任务及各文件结果后，由应用读取 `job.results` 中的 `transcription_url`，再自行下载和解析转写文件；本接口不会自动下载正文。

**能用 `AsyncClient` 做实时识别，或在断线后恢复旧任务吗？**

当前均不支持。`AsyncClient` 仅提供本文的批量转写接口；实时识别使用同步会话并发收发。关闭后重新启动是新任务，不是旧任务恢复；但多条识别可以复用同一条连接，见下节。

## 连接复用

`recognize(..., connection=conn)` 把识别任务跑在共享的
`WsConnection`（`client.ws_connection()` 创建）上：顺序任务免再次握手，
多路 JSON 任务可并发；二进制上行每条连接单 owner（一路活跃识别）。
会话 `close()` 只摘离任务、不关共享连接。完整约束与示例见
[连接复用指南](dashscope-2.0-connection-reuse-sdk.md)。

## 相关文档

- [Fun-ASR 客户端事件](https://docs.bailian.console.aliyun.com/zh/model-studio/fun-asr-client-events)
- [Fun-ASR 服务端事件](https://docs.bailian.console.aliyun.com/zh/model-studio/fun-asr-server-events)
- [连接复用指南](dashscope-2.0-connection-reuse-sdk.md)
- [语音合成 Python SDK](dashscope-2.0-tts-sdk.md)
- [实时多模态 Python SDK](dashscope-2.0-realtime-sdk.md)
