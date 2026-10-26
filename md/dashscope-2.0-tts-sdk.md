# DashScope 2.0 Python SDK：语音合成（TTS）

通过 `client.audio.tts` 将文本转换为语音，支持一次性合成、流式音频输出及双向流式交互。本文对应当前仓库实现，仅支持同步 `Client`；`AsyncClient.audio.tts` 尚未实现。

当前入口使用 CosyVoice 的 `SpeechSynthesizer` WebSocket 协议；不能假设仅更换模型名就能调用采用不同协议的所有 TTS 模型。

## 前期准备

1. 准备已开通目标模型权限的 API Key，确认模型与音色在所选工作空间及地域可用。
2. 在本仓库根目录安装源码版本，不以未经验证的已发布版本号作为前提：

   ```bash
   python -m pip install -e .
   ```

3. 在运行环境中安全配置以下环境变量，勿将密钥写入代码或提交到版本库。以下示例均通过 `Client()` 读取配置。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | `str` | API Key，用于服务鉴权。 |
| `DASHSCOPE_WORKSPACE` | `str` | 工作空间 ID，建议配置，用于工作空间网关路由及请求头。 |

可通过 `Client(region=...)` 指定与模型服务一致的地域。本文示例调用真实服务并可能产生费用，请在完成配置后按需运行。

## 快速开始

### 1. 一次性合成

`synthesize()` 接收完整文本，等待音频全部返回后将各块拼接为 `bytes`。适合较短文本；结果整体保存在内存中。

```python
from dashscope import Client

with Client() as client:
    audio = client.audio.tts.synthesize(
        "你好，欢迎使用语音合成服务。",
        model="cosyvoice-v1",
        voice="longxiaochun",
        format="mp3",
        sample_rate=22050,
    )
    with open("output.mp3", "wb") as output:
        output.write(audio)
```

### 2. 流式接收音频

`stream()` 返回惰性的音频迭代器，首次迭代时才建立连接。内部一次发送全部文本，然后调用 `stop()`；流式的是输出，不是文本输入。

```python
from dashscope import Client

with Client() as client:
    chunks = client.audio.tts.stream(
        "你好，这段音频将按块写入文件。",
        format="mp3",
    )
    try:
        with open("stream-output.mp3", "wb") as output:
            for chunk in chunks:
                output.write(chunk)
    finally:
        chunks.close()
```

业务需要提前 `break` 或处理音频时发生异常，也应保留 `finally` 中的 `chunks.close()`，释放内部 WebSocket 会话。提前退出的文件仅包含已消费的音频，不代表完整合成结果。音频块不保证可独立解码，也不保证与句子一一对应。

### 3. 双向流式：边发送文本，边接收音频与事件

使用 `connect()` 创建会话，在一个线程发送文本、另一个线程读取结果。示例输入为有限元组；接入 LLM 时可替换为逐步产出字符串片段的可迭代对象，无需先收集完整文本。

```python
import json
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

from dashscope import Client


def send_parts(session, parts):
    for text in parts:
        session.send_text(text)
    session.stop()


def receive_results(session):
    with open("duplex-output.mp3", "wb") as output:
        for frame in session.events():
            if isinstance(frame, bytes):
                output.write(frame)
                continue
            print("JSON 事件:", json.dumps(frame, ensure_ascii=False))
            payload = frame.get("payload") or {}
            sentence = (payload.get("output") or {}).get("sentence")
            if isinstance(sentence, dict):
                for word in sentence.get("words") or ():
                    print(
                        "词时间戳:", word.get("text"),
                        word.get("begin_time"), word.get("end_time"),
                    )
            if payload.get("usage") is not None:
                print("本次 usage:", payload["usage"])


parts = ("你好，", "这是双向流式语音合成。", "文本与音频可以并行传输。")
with Client() as client:
    session = client.audio.tts.connect(format="mp3", timeout=30)
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            session.start()
            print("task_id:", session.session.task_id)
            print("启动事件:", session.last_response)
            sender = executor.submit(send_parts, session, parts)
            receiver = executor.submit(receive_results, session)
            done, _ = wait((sender, receiver), return_when=FIRST_EXCEPTION)
            for future in done:
                future.result()
            print("最近 usage:", session.usage)
            print("首包延迟（毫秒）:", session.first_package_delay)
        finally:
            session.close()
```

`wait(..., FIRST_EXCEPTION)` 在任一任务失败或两者全部成功时返回，`Future.result()` 将线程内异常传播到调用方。`finally` 位于线程池上下文内部，先关闭会话以解除 WebSocket 阻塞，再由线程池等待线程退出。实际 LLM 上游若可能阻塞，还需由业务为上游设置超时或取消机制。

发送线程的 `stop()` 只结束输入，接收线程仍持续读取至 `task-finished`。不要在发送结束后立即关闭会话，否则可能丢失尾部音频、时间戳及用量。

## 请求参数

`connect()`、`synthesize()`、`stream()` 共享以下连接与合成参数；除 `text` 外，均以关键字传入。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `text` | `str` | `synthesize(text, ...)`、`stream(text, ...)` 必填的完整文本。`connect()` 不接收此参数，改用会话的 `send_text(text)` 分段输入。 |
| `model` | `str` | 默认 `"cosyvoice-v1"`。模型名称。 |
| `voice` | `str` | 默认 `"longxiaochun"`。音色须与模型匹配。 |
| `format` | `str` | 默认 `"mp3"`。输出音频编码格式；支持范围由模型服务决定。 |
| `sample_rate` | `int` | 默认 `22050`，单位 Hz。可用采样率及其与格式的组合由服务端校验。 |
| `parameters` | `Optional[dict]` | 默认 `None`。额外模型参数，原样合并进任务参数；同名键覆盖显式参数，详见下文。 |
| `ws_url` | `Optional[str]` | 默认 `None`。自定义 WebSocket 地址；未指定时优先使用全局 WebSocket 地址配置，否则按工作空间及地域生成默认地址。不是 HTTP `base_url`。 |
| `connector` | `Optional[Callable]` | 默认 `None`，使用 `websocket-client`。自定义连接函数签名为 `connector(url, headers)`，返回 WebSocket 兼容对象。 |
| `timeout` | `Optional[float]` | 单位秒，须大于 0。默认 `None` 表示沿用 `client.config.timeout`，并非无限等待。默认连接器将其用于建连，并在连接后设置 socket 超时；不是整个合成任务的总时限。 |
| `headers` | `Optional[Mapping[str, str]]` | 默认 `None`。附加 WebSocket 握手请求头。SDK 根据客户端配置填入鉴权和工作空间请求头；配置存在时会覆盖同名 `Authorization`、`X-DashScope-WorkSpace`。 |

合成参数先由 `voice`、`format`、`sample_rate` 和固定默认值 `text_type="PlainText"` 组成，再合并 `parameters`。

> **覆盖提示**：`parameters` 中的 `voice`、`format`、`sample_rate`、`text_type` 会覆盖前面的值。例如 `format="mp3"` 与 `parameters={"format": "pcm"}` 同时传入，实际请求为 PCM。请同步调整文件后缀和播放器配置。模型特有参数、取值范围及组合约束由服务端校验，SDK 不保证所有模型支持同一组参数。

自定义 `connector` 的返回对象需支持 `send()`、`recv()`、`close()`、`settimeout()`；SDK 不向连接函数传递第三个超时参数，因此自定义建连过程的超时由连接函数负责，SDK 在返回后设置 socket 超时。其 `close()` 应能解除阻塞的接收操作。

## 关键接口

### 入口方法

| 接口 | 返回值 | 说明 |
| --- | --- | --- |
| `client.audio.tts.connect(...)` | `SynthesisSession` | 仅创建惰性会话对象，不立即联网。进入上下文或调用 `start()` 才打开连接。 |
| `client.audio.tts.synthesize(text, ...)` | `bytes` | 完整发送文本、结束输入、读取全部音频并拼接；内部自动管理连接。 |
| `client.audio.tts.stream(text, ...)` | `Iterator[bytes]` | 惰性建立会话，一次发送文本并结束输入，逐块输出音频；正常耗尽时自动关闭连接，提前退出需显式关闭迭代器。 |

### 会话方法

除 `send_text(text)` 外，下列会话方法均无参数。

| 接口 | 返回值 | 说明 |
| --- | --- | --- |
| `start()` | `SynthesisSession` | 打开 WebSocket，发送 `run-task` 并等待 `task-started`，返回自身。启动事件记录于 `last_response`，不再由结果迭代器输出。 |
| `send_text(text)` | `None` | 发送一个 `str` 文本片段；非字符串抛出 `TypeError`。只能在启动成功、输入尚未结束时调用。 |
| `flush()` | `None` | 请求合成已缓冲文本，发送 `continue-task`，输入为 `{"flush": true}`。不结束文本输入，之后仍可 `send_text()`。 |
| `stop()` | `None` | 发送 `finish-task` 结束输入，不等待合成完成；之后须继续消费结果。结束输入使用此方法，没有 `finish()` 别名。 |
| `cancel()` | `None` | 发送带 `{"directive": "cancel"}` 的 `finish-task`。终态到来前，即使已 `stop()` 也可取消；取消不是暂停，不能恢复输入。 |
| `events()` | `Iterator[Union[dict, bytes]]` | 按到达顺序输出原始 JSON 事件与音频字节，包括 `task-finished` 后结束；遇 `task-failed` 抛异常而非输出失败事件。 |
| `audio_chunks()` | `Iterator[bytes]` | 仅输出音频，但内部仍读取 JSON 并更新 `last_response`、`usage` 等属性，直至任务终态。 |
| `close()` | `None` | 幂等关闭连接，不等待结果，也不替代正常的 `stop()` 加结果消费。关闭后的对象不可重启。 |

生命周期与并发约束：

- 启动方式二选一：`with client.audio.tts.connect(...) as session:` 自动启动并关闭；或显式 `start()` 配合 `try/finally: close()`。已 `start()` 的对象不要再次进入 `with session:`，也不要重复 `start()`。
- `stop()` 或 `cancel()` 后均不可再调用 `send_text()`、`flush()`。连接未关闭时，重复 `stop()` 或 `cancel()` 不会重复发送相同结束动作；终态后也不再发送。关闭后调用这些操作会报错。
- 同一会话允许一个发送者和一个接收者并行。接收迭代器由锁保护：`events()` 与 `audio_chunks()` 不能同时消费，也不能同时运行两个同类接收迭代器，否则抛出 `RuntimeError`。
- 若手动持有会话的接收迭代器，提前停止时调用该迭代器的 `close()` 释放读取锁；这不等于关闭会话。若不再继续读取，还需关闭会话。需要音频和元数据时，优先统一使用一个 `events()`。
- 每次新任务建立独立 WebSocket，无连接池或任务间复用。任务结束或会话关闭后，使用新的 `connect()` 对象发起下一任务。仅关闭 `Client` 不能替代显式会话的资源清理。

## 返回事件与属性

### 原始事件

JSON 事件保留服务端结构，不转换为专用响应模型；音频帧直接返回 `bytes`。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `header.event` | `str` | 事件名称，例如 `task-started`、`result-generated`、`task-finished`、`task-failed`。 |
| `header.task_id` | `str` | 服务端事件中的任务标识。 |
| `header.request_id` | `str`（若返回） | 服务端请求标识，以原始事件为准。 |
| `payload.output` | `dict`（若返回） | 合成结果元数据，具体字段取决于模型及事件。 |
| `payload.output.sentence.words` | `list[dict]`（若返回） | 词级信息；`begin_time`、`end_time` 为毫秒时间戳。并非所有模型或事件均返回。 |
| `payload.usage` | `dict`（若返回） | 服务端原始用量信息。CosyVoice 通常使用 `characters`；仅在服务端实际返回时才有 `tokens` 等字段。 |
| 音频帧 | `bytes` | 按请求格式编码的音频分片，不是 JSON，也不是 Base64 字符串。 |

以下仅为字段位置示意，不是真实响应，不承诺所有模型均返回这些字段或相同结构：

```json
{
  "header": {"event": "result-generated", "task_id": "示意任务标识"},
  "payload": {
    "output": {
      "sentence": {
        "words": [{"text": "你好", "begin_time": 0, "end_time": 420}]
      }
    },
    "usage": {"characters": 2}
  }
}
```

如需逐句或逐词时间戳，请在消费 `events()` 时处理每个 JSON 事件，不要等任务完成后只查询 `last_response`。

### 会话属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `last_response` | `Optional[dict]` | 最近一次 JSON 事件，初始为 `None`；启动后是 `task-started`，正常完整消费后通常为 `task-finished`，会覆盖之前的结果事件。字节帧不覆盖它。失败事件也会被保留。 |
| `usage` | `Optional[dict]` | 最近一次非 `null` 的 `payload.usage` 原始值，初始为 `None`。后续事件缺失用量或返回 `null` 时保留上次值；不是 SDK 累加值，不保证包含 token 计数。 |
| `first_package_delay` | `Optional[float]` | 单位毫秒，从首次 `send_text()` 调用时的计时点到接收迭代器首次消费音频字节。尚未消费音频时为 `None`；包括调用方延迟读取造成的等待，不是纯服务端耗时。 |
| `session` | `WsSession` | 底层会话，仅在已启动且未关闭时可访问，否则抛出 `RuntimeError`。通常无需直接调用底层收发方法。 |
| `session.session.task_id` | `str` | 在示例的 `session` 变量上获取底层任务 ID。建议在会话关闭前保存用于排查。 |

`SynthesisSession` 没有直接的 `request_id` 便捷属性。请求标识可查看 JSON 原始 `header`；异常也提供 `request_id`。属性只随实际接收而更新，不会由后台线程自动刷新；若只调用 `stop()` 而不消费结果，元数据不会自动完整。

## 异常处理

服务端业务失败与底层连接异常需分别处理；不要假设所有 WebSocket 错误都会被统一转换成 HTTP 风格异常。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `InvalidApiKey` | `AuthenticationError` | 服务端 `task-failed` 的鉴权错误映射，检查 API Key。 |
| `InvalidParameter` | `BadRequestError` | 检查模型、音色、格式、采样率和模型专属参数。 |
| `Throttling` | `RateLimitError` | 检查并发、速率及配额。由业务决定是否新建任务重试。 |
| 其他业务错误码 | `DashScopeError` 或对应子类 | 按 SDK 业务码映射；未知码保留在基础异常中。 |
| 提前 EOF | `APIConnectionError` | 底层接收返回空字符串或 `None`，且尚未收到 `task-finished`，视为连接异常而非成功结束。 |
| socket 超时或断连 | 原生异常 | 如 `websocket.WebSocketTimeoutException`、`websocket.WebSocketConnectionClosedException`、`OSError`，可能原样传播；并不保证变为 `APITimeoutError`。 |
| 非法操作或输入 | `RuntimeError` / `TypeError` / `ValueError` | 包括重复启动、并发读取、结束输入后继续发送、非字符串文本及非正数超时。 |

收到 `task-failed` 时，SDK 根据 `header.error_code`、`header.error_message` 抛出异常，异常对象的 `body` 保留完整失败事件，`code`、`message`、`request_id` 可用于定位问题。会话的 `last_response` 同样保留失败事件。异常的 `status_code` 不能作为 WebSocket 任务成功的判断依据。

启动、收发失败时 SDK 会清理连接；调用方仍应使用上下文管理或 `finally` 释放资源。协议头异常可能抛出 `WsEventError`，非法 JSON 可能抛出原生 JSON 解析异常。

WebSocket 路径不提供自动重连、重放或断点续传；客户端 HTTP 重试配置不会自动重放 TTS 任务。确需重试时，应新建会话并由业务确定文本边界，避免重复播放、重复合成或重复计费。

## 常见问题

**如何衔接 LLM 的流式输出？**

使用 `connect()`，将 LLM 字符串片段逐次传给 `send_text()`，同时在另一个线程持续消费一个结果迭代器。`stream(text)` 仍要求完整字符串，不接受文本迭代器作为双向输入。

**`flush()`、`stop()`、`cancel()` 和 `close()` 有何区别？**

`flush()` 请求处理缓冲文本但保留输入；`stop()` 正常结束输入并保留输出；`cancel()` 请求取消且不可恢复，`stop()` 后终态前仍可调用；`close()` 直接释放连接。`stop()`、`cancel()` 均不等待任务结果。需要确认终态时继续读取；无需保留输出时可显式关闭。

**为什么拿不到时间戳、用量或首包延迟？**

先确认已消费结果。`audio_chunks()` 不输出 JSON，但会更新会话属性；逐词信息应通过 `events()` 当场读取。模型可能不返回词时间戳或用量。`usage` 不等于固定的 token 统计，首包延迟也仅在首次消费音频后产生。

**为什么文件内容与扩展名不匹配？**

检查 `parameters` 是否覆盖 `format`、`sample_rate`，并按实际编码保存或播放。SDK 只返回音频字节，不负责转码，也不会因文件名后缀改变音频格式。

**能否复用会话或使用异步客户端？**

不能用同一会话启动下一任务，也不能重启已关闭对象。每个任务需新建会话和连接；当前没有 `AsyncClient.audio.tts`，异步应用需自行安排同步调用的线程及生命周期。

## HTTP 合成（client.audio.tts.http）

ws 入口之外，`client.audio.tts.http` 提供纯 HTTP POST 的合成方式（不经过 WebSocket）：一次性返回整段音频的临时 URL，或以 SSE 流式逐段返回音频帧。这是 1.x `HttpSpeechSynthesizer`（sambert/cosyvoice）与 qwen-tts `SpeechSynthesizer` 两个模型族的 v2 形态，共用同一入口，SDK 按模型名前缀路由：

- `qwen*`（如 `qwen-tts`、`qwen3-tts-flash`）→ `services/aigc/multimodal-generation/generation`
- 其他（sambert/cosyvoice 等）→ `services/audio/tts/SpeechSynthesizer`

两个端点的请求体同为 `{model, input}`、响应同为 `output.audio`，因此调用代码完全一致，只需替换 `model` 与 `voice`。

### 一次性合成（sambert 官方示例平移）

对应 1.x `HttpSpeechSynthesizer.call(stream=False)`；音频以临时 URL 返回，需自行下载。

```python
import requests
from dashscope.v2 import Client

client = Client()
result = client.audio.tts.http.synthesize(
    "今天天气怎么样？",
    model="sambert-zhichu-v1",
    voice="zhichu",
    format="wav",
    sample_rate=16000,
)
print(result.audio_url, result.expires_at)
with open("output.wav", "wb") as f:
    f.write(requests.get(result.audio_url).content)
```

### 流式合成（qwen-tts 官方示例平移）

对应 1.x qwen `SpeechSynthesizer.call(stream=True)`；SSE 由 SDK 自动开启（`X-DashScope-SSE: enable` 请求头），音频帧已按 base64 解码为字节。

```python
from dashscope.v2 import Client

client = Client()
with open("output.mp3", "wb") as f:
    for chunk in client.audio.tts.http.stream(
        "那我来给大家推荐一款T恤，这款T恤是纯棉的，穿着很舒服。",
        model="qwen3-tts-flash",
        voice="Cherry",
        language_type="Chinese",
    ):
        if chunk.audio:
            f.write(chunk.audio)
        if chunk.is_final:
            print(chunk.audio_url, chunk.usage)
```

cosyvoice 的流式分片额外携带 `chunk.sentence`（`sentence-*` 类型的句级元数据）；qwen 族分片只有 `audio` 与末尾的整段 URL 引用。末片 `is_final`（`finish_reason == "stop"`）不保证携带音频字节，整段音频以 `audio_url` 引用为准。

### HTTP 请求参数

`http.synthesize()` 与 `http.stream()` 参数相同；除 `text` 外均以关键字传入。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `text` | `str` | 必填。待合成文本。 |
| `model` | `str` | 必填。模型名称，同时决定端点路由（见上）。 |
| `voice` | `str` | 必填。音色须与模型匹配。 |
| `format` | `Optional[str]` | 默认 `None`（服务端默认）。输出音频编码格式。 |
| `sample_rate` | `Optional[int]` | 默认 `None`（服务端默认）。采样率，单位 Hz。 |
| `language_type` | `Optional[str]` | 默认 `None`。语种类型，如 `"Auto"`、`"Chinese"`、`"English"`；主要面向 qwen-tts 族，写入 `input.language_type`。 |
| `parameters` | `Optional[dict]` | 默认 `None`。请求体顶层 `parameters`，原样透传。 |
| `extra_input` | `Optional[dict]` | 默认 `None`。额外 `input` 字段（如 cosyvoice 的 `volume`/`rate`/`pitch`），原样合并；显式参数同名时覆盖它。 |
| `timeout` | `Optional[float]` | 默认 `None`，沿用客户端超时配置。 |
| `headers` | `Optional[Mapping[str, str]]` | 默认 `None`。附加请求头；`stream()` 的 SSE 头已由 SDK 设置，同名传入可覆盖。 |

### HTTP 返回类型

`http.synthesize()` 返回 `HttpSynthesis`；`http.stream()` 逐片返回 `HttpSynthesisChunk`。两者都保留原始 `output` 字典、`usage` 与 `request_id`。

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `audio_url` | `Optional[str]` | 整段音频的临时 URL；`HttpSynthesis` 直接携带，`HttpSynthesisChunk` 仅末片携带。 |
| `audio_id` | `Optional[str]` | 音频 ID。 |
| `expires_at` | `Optional[int]` | `audio_url` 的过期时间戳。 |
| `audio` | `Optional[bytes]` | 仅 `HttpSynthesisChunk`：本片音频帧（已 base64 解码），无音频的分片为 `None`。 |
| `sentence` | `Optional[dict]` | 仅 `HttpSynthesisChunk`：cosyvoice `sentence-*` 分片的句级元数据。 |
| `finish_reason` / `is_final` | `Optional[str]` / `bool` | 仅 `HttpSynthesisChunk`：末片 `finish_reason == "stop"`，`is_final` 为其便捷判断。 |

HTTP 路径的错误处理与其他 HTTP 资源一致：错误状态码按错误目录映射（如 `InvalidParameter` → `BadRequestError`）；SSE 流中的 `event:error` 帧在迭代中途抛出同样映射的异常。ws 路径的连接类注意事项（重连、异步）不适用于此入口。

## 连接复用

`connect(..., connection=conn)` 把合成任务跑在共享的
`WsConnection`（`client.ws_connection()` 创建）上，逐句高频合成免去反复
握手。合成输出是二进制音频（线上不带 task_id），所以每条连接同一时刻
只能有一路活跃合成——`start()` 会自动占用下行，第二路并发合成会直接
报错（要并发请再开一条连接）；顺序合成则天然复用。详见
[连接复用指南](dashscope-2.0-connection-reuse-sdk.md)。

## 相关文档

- [CosyVoice 客户端事件](https://docs.bailian.console.aliyun.com/zh/model-studio/cosyvoice-client-events)
- [CosyVoice 服务端事件](https://docs.bailian.console.aliyun.com/zh/model-studio/cosyvoice-server-events)
- [Sambert Python SDK](https://help.aliyun.com/zh/model-studio/sambert-python-sdk)
- [Qwen-TTS API](https://www.alibabacloud.com/help/zh/model-studio/qwen-tts-api)
- [连接复用指南](dashscope-2.0-connection-reuse-sdk.md)
- [DashScope 2.0 ASR SDK](dashscope-2.0-asr-sdk.md)
- [DashScope 2.0 Realtime SDK](dashscope-2.0-realtime-sdk.md)
