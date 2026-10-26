# Realtime 实时交互 Python SDK

`client.audio.realtime` 使用同一个同步 WebSocket 会话接口，支持 Omni 实时多模态交互、Qwen-Audio 实时音频对话和 Qwen-Livetranslate 实时翻译。本文介绍当前仓库的 DashScope 2.0 实现，不适用于旧版 `OmniRealtimeConversation` / `OmniRealtimeCallback`。

SDK 通过 `recv_event()` 或 `events()` 返回原始 JSON 事件，不启动后台回调线程。音频输入可使用原始 `bytes`；音频输出位于 JSON 事件的 Base64 字段中，需要调用方解码。

## 前期准备

1. 在当前仓库根目录安装源码，确保使用的是包含本文接口的版本。本文不声明线上 PyPI 包的最低版本。

   ```bash
   python -m pip install -e .
   ```

2. 配置 `DASHSCOPE_API_KEY`；推荐同时配置 `DASHSCOPE_WORKSPACE`，并按模型的服务地域设置 `DASHSCOPE_REGION`。不应把 API Key 写入代码或提交到仓库。
3. 示例输入 `input.pcm` 为无文件头的 16 kHz、16 bit、单声道 PCM。SDK 不会自动重采样、解码 WAV 文件或采集麦克风；实际音频格式必须与会话配置一致。

```python
from dashscope import Client

with Client(timeout=30.0) as client:
    print(client.config.region)
```

`Client` 可通过 `api_key`、`workspace`、`region`、`timeout` 显式覆盖配置。默认地域为 `cn-beijing`，默认超时为 300 秒；地域可配置不代表该地域提供所有模型。未配置 workspace 时，SDK 回退至共享网关并发出警告。

> Realtime 当前仅支持同步 `Client`，不提供 `AsyncClient.audio.realtime`。需要双向并发时，由应用安排一个发送线程和一个接收线程；同一时刻只能有一个接收者。

## 快速开始

### Omni / Qwen-Audio：手动提交一轮音频

下面的完整示例同时发送和接收。默认使用 Omni；将 `MODEL` 改为 `qwen-audio-3.1-realtime-plus`，即可使用 Qwen-Audio 的配置和结束流程。

```python
import base64
import time
from concurrent.futures import ThreadPoolExecutor

from dashscope import Client

MODEL = "qwen-omni-turbo-realtime"
IS_QWEN_AUDIO = MODEL.startswith("qwen-audio-")


def receive_response(session):
    with open("response.pcm", "wb") as output:
        for event in session.events():
            kind = event.get("type")
            if kind == "response.audio.delta":
                output.write(base64.b64decode(event["delta"]))
            elif kind in (
                "response.text.delta",
                "response.audio_transcript.delta",
            ):
                print(event.get("delta", ""), end="", flush=True)
            elif kind == "response.done":
                print("\nusage:", event.get("response", {}).get("usage"))
                return
    raise RuntimeError("连接已关闭，未收到本轮 response.done")


with Client(timeout=30.0) as client:
    with client.audio.realtime.connect(model=MODEL) as session:
        settings = {
            "input_audio_format": "pcm" if IS_QWEN_AUDIO else "pcm16",
            "output_audio_format": "pcm" if IS_QWEN_AUDIO else "pcm16",
            "enable_turn_detection": False,
        }
        if not IS_QWEN_AUDIO:
            settings["voice"] = "Chelsie"
        session.update_session(["text", "audio"], **settings)
        for event in session.events():
            if event.get("type") == "session.updated":
                break
        else:
            raise RuntimeError("连接已关闭，未收到 session.updated")

        with ThreadPoolExecutor(max_workers=1) as pool:
            receiver = pool.submit(receive_response, session)
            try:
                with open("input.pcm", "rb") as source:
                    while True:
                        chunk = source.read(3200)
                        if not chunk:
                            break
                        session.append_audio(chunk)
                        time.sleep(0.1)
                session.commit()
                session.create_response()
                receiver.result()
                if not IS_QWEN_AUDIO:
                    session.end_session()
                    for event in session.events():
                        print(event)
            finally:
                session.close()
```

3200 字节约等于上述输入格式的 100 ms 音频，示例用等待模拟实时采集；麦克风本身按实时节奏产帧时，不需要再等待。`response.pcm` 没有 WAV 文件头，播放时必须使用服务端实际输出的采样率、位宽和声道配置。

- **Omni / Qwen-Audio 的 Manual 模式**：`append_audio()` → `commit()` → `create_response()` → 读取至 `response.done`。
- **Qwen-Audio 关闭连接**：读完当前响应后调用 `close()` 或退出上下文，不调用 `end_session()`。
- **Omni 结束会话**：调用 `end_session()`，继续接收至 `session.finished`。
- 多轮对话时，不要每轮退出会话上下文；保持连接，重复输入和响应流程。

### Livetranslate：边输入边翻译，并读取最终结果

下面以 `qwen3.5-livetranslate-flash-realtime` 为例，使用服务端默认语音检测。文件发送结束后请求结束会话，接收线程继续读取最终转写、译文、用量和 `session.finished`。

```python
import base64
import time
from concurrent.futures import ThreadPoolExecutor

from dashscope import Client


def receive_translation(session):
    with open("translation.pcm", "wb") as output:
        for event in session.events():
            kind = event.get("type")
            if kind == "response.audio.delta":
                output.write(base64.b64decode(event["delta"]))
            elif kind in (
                "response.text.text",
                "response.audio_transcript.text",
            ):
                print("text:", event.get("text"), "stash:", event.get("stash"))
            elif kind in (
                "response.text.delta",
                "response.audio_transcript.delta",
            ):
                print(event.get("delta", ""), end="", flush=True)
            elif kind == "response.done":
                print("usage:", event.get("response", {}).get("usage"))
            else:
                print(event)


with Client(timeout=30.0) as client:
    with client.audio.realtime.connect(
        model="qwen3.5-livetranslate-flash-realtime",
    ) as session:
        session.update_session(
            output_modalities=["text", "audio"],
            translation_params={"language": "en"},
        )
        for event in session.events():
            if event.get("type") == "session.updated":
                break
        else:
            raise RuntimeError("连接已关闭，未收到 session.updated")
        with ThreadPoolExecutor(max_workers=1) as pool:
            receiver = pool.submit(receive_translation, session)
            try:
                with open("input.pcm", "rb") as source:
                    while True:
                        chunk = source.read(3200)
                        if not chunk:
                            break
                        session.append_audio(chunk)
                        time.sleep(0.1)
                session.end_session()
                receiver.result()
            finally:
                session.close()
```

此示例也可将模型替换为 `qwen3-livetranslate-flash-realtime` 或 `qwen3.8-livetranslate-flash-realtime`；只传输出模态和目标语种时，SDK 会按模型转换字段名，其他参数保留服务端默认值。

> Livetranslate 不使用上面的对话触发流程。Manual 模式下 `commit()` 会让服务端自动生成翻译响应，不再发送 `create_response()`；默认语音检测模式则自动提交。不要在每次 `response.done` 时关闭整个会话。

## 请求参数

### 创建会话：`client.audio.realtime.connect()`

返回 `RealtimeSession`，此时尚未连接。进入 `with` 上下文或调用返回对象的 `connect()` 后才建立连接。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `model` | `str` | 必填，模型名称；没有默认模型。 |
| `ws_url` | `str` 或 `None` | 默认 `None`，根据 workspace 和 region 选择 Realtime 网关。显式设置时覆盖默认地址。 |
| `timeout` | `float` 或 `None` | 秒；默认继承 `Client` 配置。显式值必须大于 0，作用于连接和 socket 操作，不是会话总时长。 |
| `headers` | `Mapping[str, str]` 或 `None` | 额外握手头；按不区分大小写的名称覆盖默认头，包括鉴权头。不要在日志中输出 API Key。 |
| `connector` | `Callable[[str, dict], Any]` 或 `None` | 自定义连接函数，参数为 URL 和 headers；返回支持 `send`、`recv`、`settimeout`、`close` 的对象。默认使用 `websocket-client`。 |

如果 `ws_url` 已包含 `model` 查询参数，该参数决定实际服务端模型和 SDK 配置映射；`session.model` 仍保留构造时的 `model` 参数。建议两处保持一致。

Realtime 使用独立的 `/api-ws/v1/realtime` 路径，不受 `DASHSCOPE_WEBSOCKET_BASE_URL` 的 ASR/TTS inference 地址覆盖影响；需要覆盖时传 `ws_url`。`Client.base_url` 是 HTTP 地址，也不替代 `ws_url`。

自定义 `connector` 不接收第三个超时参数，建立连接阶段的超时由连接函数负责；SDK 在连接返回后设置 socket 超时。其 `close()` 应能解除正在阻塞的接收操作。

### 更新会话：`session.update_session()`

发送 `session.update`，返回 `None`，不等待 `session.updated`。应用应确认配置成功后再发送依赖这些配置的音频。

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `output_modalities` | `list[str]` | Omni 必填；Qwen-Audio/Live 可省略。常用 `['text', 'audio']`，取值组合须由模型支持。Live 3.8 映射为 `output_modalities`，其他模型映射为 `modalities`。 |
| `voice` | `str` 或 `None` | 音色。Omni 未传时发送 `None`；新模型未传时不发送，使用服务端默认值。Live 3.8 写入 `audio.output.voice`。 |
| `input_audio_format` | `str` | Omni 默认 `pcm16`；新模型不补默认值。Qwen-Audio 的 PCM 使用 `pcm`，而非 `pcm16`。Live 3.8 映射为 `audio.input.format.type`。 |
| `output_audio_format` | `str` | Omni 默认 `pcm16`；新模型不补默认值。Live 3/3.5 的输出 PCM 使用 `pcm`；Live 3.8 写入 `audio.output.format.type`。 |
| `enable_input_audio_transcription` | `bool` | Omni 默认 `True`。Live 3/3.5 显式 `True` 选择 `qwen3-asr-flash-realtime`，`False` 发送 `{'model': None}`。Live 3.8 ASR 始终开启，显式关闭会抛出 `ValueError`。Qwen-Audio 示例不主动设置此项。 |
| `input_audio_transcription_model` | `str` 或 `None` | 指定转写模型；Omni 默认 `None`。Live 3/3.5 显式 `None` 关闭 ASR，Live 3.8 禁止这一关闭配置。 |
| `enable_turn_detection` | `bool` | Omni 默认 `True`；新模型未传不更改。显式 `False` 发送语音检测字段为 `None`，用于模型支持的 Manual 模式。 |
| `turn_detection_type` | `str` | Omni 默认 `server_vad`。新模型只有显式设置才发送；显式开启检测且未给类型时，Live 3.8 选择 `speaker_detection`，其他模型选择 `server_vad`。 |
| `prefix_padding_ms` | `int` | 检测起点前保留的音频时长，毫秒；Omni 默认 300，新模型未传不发送，支持情况按模型协议。 |
| `turn_detection_threshold` | `float` | 检测阈值；Omni 默认 0.2。新模型未传不发送。Live 3.8 的 `speaker_detection` 阈值由协议固定为 0.5，不应使用 Omni 默认值替代。 |
| `turn_detection_silence_duration_ms` | `int` | 判定结束的静音时长，毫秒；Omni 默认 800，新模型未传不发送。 |
| `turn_detection_param` | `dict` 或 `None` | 额外语音检测参数，字典同名字段覆盖上述字段。Omni 的 `None` 表示不追加；新模型显式 `None` 表示关闭检测，未传则不更改。 |
| `translation_params` | `dict` 或 `None` | 翻译配置，写入 `translation`，如 `{'language': 'en'}`。Live 3/3.5 可按协议设置 `corpus.phrases` 热词映射。新模型空字典保持为空，不补 `language: None`。 |
| `transcription_params` | `dict` 或 `None` | 输入转写配置；`language` 等字段进入 `input_audio_transcription`，`corpus_text` 转为 `corpus.text`，`input_audio_format`/`sample_rate` 提升为音频配置。详见下方规则。 |
| `input_audio_config` | `dict` 或 `None` | 嵌套输入格式，如 `{'type': 'pcm', 'sample_rate': 16000}`，写入 `audio.input.format`。优先用于支持新结构的模型。 |
| `output_audio_config` | `dict` 或 `None` | 嵌套输出格式，如 `{'type': 'pcm', 'sample_rate': 24000}`，写入 `audio.output.format`。 |
| `input_video_representation_compact` | `str` 或 `None` | 写入 `video.input.representation_compact`；是否支持取决于模型，并非所有模型都接收图像。 |
| `**kwargs` | `Any` | 额外 session 字段，如 `instructions`、`tools`、`sample_rate`、`audio`。字段按协议透传，同名原始字段优先；SDK 不保证所有模型支持所有扩展参数。 |

### 默认值、增量更新和覆盖规则

**Omni** 保留完整默认配置行为：每次调用 `update_session()` 都需要输出模态，并重新构建默认音色、格式、ASR、VAD 等字段，不是只更新显式参数。只修改一个原始字段时可使用 `send_event()`：

```python
session.send_event({
    "type": "session.update",
    "session": {"instructions": "请简洁回答。"},
})
```

**Qwen-Audio、Live 3/3.5/3.8** 只发送显式参数，没有传入的字段不发送，也不会从上一次调用缓存回填：

```python
session.update_session(instructions="请简洁回答。")
```

对应的 session 内容仅为 `{'instructions': '请简洁回答。'}`。此示例适用于支持 `instructions` 的模型；透传能力不代表模型能力。

- 新模型区分“未传”和“显式 `None`”；保留显式的 `False`、`0`、空字符串、空字典和空列表。保留这些值不意味着服务端一定接受，字段的合法性仍以模型协议为准。
- Omni 的 `transcription_params={}` 会补输入 `pcm` / 16000，并过滤该字典的 `None` 值；新模型不补这两个默认值，也不统一过滤 `None`。
- Live 3/3.5 的显式 ASR 关闭开关优先于转写模型和 `transcription_params`。原始 `input_audio_transcription` kwargs 则可整体覆盖便捷参数生成的字段。
- Omni 任一不为 `None` 的嵌套格式配置（包括 `{}`）会抑制旧式顶层音频格式；Qwen-Audio/Live 的显式嵌套格式参数也走嵌套序列化，不能据此推断旧模型支持它。
- Live 3.8 将旧式 `voice`、`input_audio_format`、`output_audio_format`、`sample_rate`、`turn_detection` 转为嵌套字段，不混发对应的旧式顶层字段。嵌套字段覆盖同名便捷字段，未冲突的格式字段会合并；显式 `None` 可整体覆盖子配置。
- `**kwargs` 的原始 `audio` 会先替换由 `input_audio_config` / `output_audio_config` 生成的 `audio`。不建议同时用两种入口配置同一个子对象；Live 3.8 随后会合并仍存在的顶层便捷字段。
- Live 3.8 禁止关闭 ASR，并拒绝 `same_language_skip_options`。这是 `update_session()` 的检查；`send_event()` 不执行该检查，原始发送也不代表服务端会接受非法配置。

Live 3.8 的配置示例（在已打开的会话中执行）：

```python
session.update_session(
    output_modalities=["text", "audio"],
    voice="Tina",
    input_audio_config={"type": "pcm", "sample_rate": 16000},
    output_audio_config={"type": "pcm", "sample_rate": 24000},
    turn_detection_param={"type": "speaker_detection", "threshold": 0.5},
    translation_params={"language": "en"},
)
```

生成的 session 内容如下；`event_id` 由 SDK 自动补充：

```json
{
  "output_modalities": ["text", "audio"],
  "translation": {"language": "en"},
  "audio": {
    "input": {
      "format": {"type": "pcm", "sample_rate": 16000},
      "turn_detection": {"type": "speaker_detection", "threshold": 0.5}
    },
    "output": {
      "format": {"type": "pcm", "sample_rate": 24000},
      "voice": "Tina"
    }
  }
}
```

## 关键接口

### 连接与事件收发

| 方法 | 参数 | 返回值 | 描述 |
| --- | --- | --- | --- |
| `session.connect()` | 无 | 当前 `RealtimeSession` | 建立连接，不自动读取 `session.created`；重复打开会抛出 `RuntimeError`。上下文入口会自动调用，勿先 connect 再进入同一对象的 with。 |
| `send_event(event)` | `event: dict` | `None` | 发送 JSON 事件；缺少 `event_id` 时生成 `event_` 加 UUID。显式 ID 保留，不修改调用方字典。 |
| `recv_event()` | 无 | `dict` 或 `None` | 阻塞接收一条事件。正常关闭后返回 `None`；活动响应或结束确认未完成时断线会抛异常。 |
| `events()` | 无 | `Iterator[dict]` | 连续读取。`response.done` 不自动终止迭代；收到 `session.finished` 后产出该事件一次并结束。 |
| `close()` | 无 | `None` | 幂等关闭本地连接，不发送 `session.finish`、不等待残留结果。重新 `connect()` 会建立新 socket 并清空会话统计。 |

### 音频与图像输入

| 方法 | 参数 | 返回值 | 描述 |
| --- | --- | --- | --- |
| `append_audio(audio_b64)` | `str` 或 `bytes` | `None` | 发送 `input_audio_buffer.append`；字符串视为已编码的 Base64，bytes 自动 Base64 编码。不是发送文本转语音输入。 |
| `append_video(video_b64)` | `str` 或 `bytes` | `None` | 发送单帧 `input_image_buffer.append`；不是视频文件上传，也不从视频自动抽帧。 |
| `commit()` | 无 | `None` | 发送 `input_audio_buffer.commit`，不等待提交确认，也不隐式调用 `create_response()`。 |
| `clear_appended_audio()` | 无 | `None` | 发送 `input_audio_buffer.clear`，清除服务端未提交的音频缓冲；不是取消已在生成的响应。 |

适用模型的图像输入可以在已追加音频后执行：

```python
from pathlib import Path

session.append_video(Path("frame.jpg").read_bytes())
```

Livetranslate 的图像使用 JPG/JPEG，编码前不超过 500 KB、帧率不超过 2 fps，且必须先至少追加一次音频；模型版本是否支持以协议为准。Qwen-Audio 的所列客户端协议没有图像事件，不要默认套用。SDK 不进行图像转换、大小检查或自动限帧。

### 对话和响应控制（Omni / Qwen-Audio）

| 方法 | 参数 | 返回值 | 描述 |
| --- | --- | --- | --- |
| `create_item(item, *, previous_item_id=None)` | `item: dict`；前置条目 ID 可选 | `None` | 发送 `conversation.item.create`；可创建文本消息或函数调用结果。指定 `previous_item_id` 时传给服务端决定插入位置。 |
| `retrieve_item(item_id)` | `item_id: str` | `None` | 发送 `conversation.item.retrieve`，从后续事件读取结果。 |
| `delete_item(item_id)` | `item_id: str` | `None` | 发送 `conversation.item.delete`。 |
| `create_response(modalities, instructions, **kwargs)` | 两个参数均可省略；类型分别为 `list[str]`、`str`，均可显式 `None` | `None` | 发送 `response.create`。Omni 无参时发送 `{'instructions': None}`，Qwen-Audio 无参时发送空 response 对象；其他字段按 kwargs 透传。 |
| `cancel_response()` | 无 | `None` | 发送 `response.cancel`；继续接收服务端取消结果和 `response.done`，不关闭会话。 |

在已打开的 Omni 会话中，文本输入示例为：

```python
session.create_item({
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "请简单介绍一下你自己。"}],
})
session.create_response(modalities=["text", "audio"])
```

工具结果通过 `create_item({'type': 'function_call_output', 'call_id': ..., 'output': ...})` 回填，再按模型协议创建响应。参数中的 ID 和 output 需使用真实函数调用结果。Livetranslate 的翻译流程不使用上述客户端对话和响应控制事件。

### 结束会话：`end_session()`

| 模型 | 行为 |
| --- | --- |
| Omni | 发送 `session.finish`，立即返回；继续读取至 `session.finished`。 |
| Live 3/3.5/3.8 | 发送 `session.finish`；继续读取剩余输入转写、翻译音频、用量和最终确认。收到 `session.finished` 时 SDK 自动关闭连接。 |
| Qwen-Audio | 抛出 `NotImplementedError`，不发送事件、不关闭连接、不改变会话状态。应读完或取消当前响应，再 `close()`。 |

请求结束后不能再发送音频、修改配置或创建响应；连接仍打开时重复 `end_session()` 不重复发送。若只收到 `response.done` 就断线，但尚未收到请求的 `session.finished`，SDK 仍将其视为未完整结束。

## 模型差异与调用链路

| 场景 | Omni | Qwen-Audio | Live 3/3.5 | Live 3.8 |
| --- | --- | --- | --- | --- |
| 配置方式 | helper 每次补 Omni 默认值 | 仅发送显式字段 | 仅发送显式字段 | 仅发送显式字段并映射嵌套音频配置 |
| 输出模态协议字段 | `modalities` | `modalities` | `modalities` | `output_modalities` |
| 输入转写 | 按会话配置 | 不注入 Omni 转写配置 | 默认开启，可显式关闭 | 始终开启，不可关闭 |
| Manual 提交后的响应 | `commit` 后 `create_response` | `commit` 后 `create_response` | `commit` 后自动生成 | 按模型支持的检测模式运行，不照搬 Omni 响应控制 |
| 主要输出文本事件 | `.delta` | `.delta` | `.text`，含 `text` / `stash` | `.delta` |
| 结束整个会话 | finish + finished | 读完当前响应后 close | finish + finished | finish + finished |

```text
Omni / Qwen-Audio Manual
connect → update_session → session.updated
        → append_audio（发送与接收并发）→ commit → create_response
        ← response.created / 文本与音频 / response.done
        → 下一轮输入，或按模型结束连接

Livetranslate
connect → update_session → session.updated
        → append_audio（语音检测自动提交，或支持的 Manual 模式 commit）
        ← 输入转写 / response.created / 译文与音频 / response.done
        → end_session
        ← 剩余结果 / session.finished → 自动关闭
```

Qwen-Audio 支持 `server_vad`、`smart_turn` 和 Manual 等模式，不能套用 Omni 的所有检测参数；其音色仅首次更新有效，输入格式和检测设置有首次音频输入之前的限制。SDK 不维护服务端完整的 IDLE/生成/打断状态机，应按模型协议安排配置时机。

## 返回事件与会话属性

### 事件内容

| 事件 | 主要内容 | 处理方式 |
| --- | --- | --- |
| `session.created` / `session.updated` | `session` | 确认会话和配置，SDK 更新 `session_id`。 |
| `input_audio_buffer.speech_started` / `.speech_stopped` | 音频起止毫秒、条目 ID；部分版本提供 `speaker_id` | 原样保留，按 `item_id` 关联，不换算成 SDK 首包延迟。 |
| `conversation.item.input_audio_transcription.*` | 输入转写、语言等模型字段 | 原样读取；不计入译文/回复首文本延迟。 |
| `response.created` | `response.id` | 开始一轮响应，重置该轮首文本/首音频延迟。 |
| `response.text.delta` / `response.audio_transcript.delta` | `delta` | 按响应和条目分组，依次累加文本。 |
| `response.text.text` / `response.audio_transcript.text` | `text`、`stash` | Live 3/3.5 的文本片段及临时文本。`stash` 可能修正，不应作为确定文本永久累加；以最终 done 事件校准。 |
| `response.audio.delta` | Base64 `delta` | 解码后写文件或播放；没有单独的 bytes 迭代器。 |
| `response.text.done` / `response.audio_transcript.done` | 最终 `text` / `transcript` | 用于确认最终文本，具体结构依模型协议。 |
| `response.done` | `response`，可能含 `usage`、状态及输出条目 | 保存每轮用量和完成状态，不能仅凭事件类型认定生成成功。 |
| `session.finished` | 会话最终确认 | 结束迭代并自动关闭。 |
| 其他事件 | 函数参数、环境音转写、自定义扩展等 | 不进行字段白名单过滤，原样返回。 |

Realtime 不提供独立 `usage` 属性或完整事件历史缓存。每轮用量从 `event['response'].get('usage')` 获取；原始字段可以包含 `input_tokens`、`output_tokens`、`total_tokens` 及细分类别，以该模型实际返回为准。

### 属性

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `model` | `str` | 构造时的模型参数。 |
| `url` | `str` | 实际连接 URL，包括模型查询参数。 |
| `session_id` | `str` 或 `None` | 从已读取的 session.created / updated 中更新。 |
| `last_message` | `dict` 或 `None` | 最近读取的 JSON 事件，包括服务端 error；后续事件会覆盖它，不是历史。 |
| `last_response_id` | `str` 或 `None` | 最近一次 response.created 中的响应 ID。 |
| `last_response_create_time` | `float` 或 `None` | 收到 response.created 时的本地单调时钟毫秒值，不是 Unix 时间戳。 |
| `last_first_text_delay` | `float` 或 `None` | 从读取 response.created 到读取该轮首条输出文本事件的毫秒数；支持 `.text` 和 `.delta`。 |
| `last_first_audio_delay` | `float` 或 `None` | 从读取 response.created 到读取该轮首条 response.audio.delta 的毫秒数。 |

统计在应用消费事件时更新，不是服务端耗时，也不是从发送音频起算。相同响应的重复首包、已结束响应的迟到包和显式不同 response_id 的包不会覆盖该轮首包延迟。只发送不读取时，属性不会由后台自动更新。

## 异常处理

| 场景 | 表现与处理 |
| --- | --- |
| 服务端 `error` 事件 | 转为 `dashscope.v2.errors` 中的业务异常，如 `BadRequestError`、`RateLimitError`；原始事件在异常 `body` 和 `last_message` 中，不作为普通事件 yield。 |
| `invalid_request_error` 且未请求结束 | SDK 保持连接；修正参数后可继续发送。若通过 `events()` 遇到异常，该迭代器已经终止，需要创建新迭代器或调用 `recv_event()`。 |
| 其他服务端错误，或结束期间的错误 | 关闭连接并抛异常。不要继续向旧会话发送。 |
| 活动响应缺少 `response.done`，或已请求结束但缺少 `session.finished` 时断线 | 抛出 `APIConnectionError`，不把截断输出当作完整结果。 |
| socket 超时、网络重置、发送失败 | 关闭连接并传播底层异常，包括 `websocket.WebSocketTimeoutException`、`OSError` 等；不保证全部转换为 SDK 连接异常。 |
| 非法媒体类型或非法 JSON | 类型/解析异常；媒体只支持字符串或 bytes，不自动读取路径。 |
| 未连接、重复连接、结束后发送 | `RuntimeError`。 |
| Live 3.8 的禁止配置 | `ValueError`，在发送前拒绝，不破坏当前连接。 |
| Qwen-Audio 调用 `end_session()` | `NotImplementedError`；改用读取完成后关闭连接。 |

WebSocket 不使用 HTTP 自动重试来重放音频或响应请求，`Client.max_retries` 不提供断线续传。应用需要自行决定是否重建会话，以及如何处理已发送但未确认的输入。

## 常见问题

### 为什么 `response.done` 后 `events()` 还在等待？

它只代表当前响应结束，会话可以继续多轮交互。单轮读取可以在该事件后退出循环；结束整个会话需按对应模型执行 finish 或 close 流程。`break` 不会自动关闭会话。

### 能否同时在两个线程调用 `events()`？

不应这样做，事件会被两个消费者分流。只保留一个读取者；它可以将事件分发给播放、字幕和统计逻辑。发送与接收可以并行。

### 是否有 `start()` / `stop()`？

Realtime 的生命周期接口为 `connect()` / `end_session()` / `close()`，不是 ASR/TTS 的 `start()` / `stop()`。`cancel_response()` 取消单轮响应，与关闭整个会话不同。

### 同一连接可以复用吗？

Omni/Qwen-Audio 的同一会话可完成多轮对话，Live 也可在会话中连续翻译多段音频。`close()` 后重新 `connect()` 是新连接、新会话，不是恢复旧上下文。这是协议设计而非 SDK 限制：realtime 协议没有 task_id，一条连接就是一个会话，会话结束服务端即关闭连接，因此 `WsConnection` 多任务复用对它不适用（ASR/TTS/翻译/TingWu 等 run-task 族可以，见[连接复用指南](dashscope-2.0-connection-reuse-sdk.md)）。要压首开延迟，建议在用户开始说话前提前 `connect()` 预热。仅关闭 `Client` 不替代实时会话的关闭；重连前应等待旧的收发线程退出。

## 相关文档

- [ASR Python SDK](dashscope-2.0-asr-sdk.md)
- [TTS Python SDK](dashscope-2.0-tts-sdk.md)
- [连接复用指南](dashscope-2.0-connection-reuse-sdk.md)
- [2.0 接口示例总览](dashscope-2.0-interfaces-examples.md)
- [Omni Python SDK（本文结构参考，接口为旧版）](https://help.aliyun.com/zh/model-studio/omni-realtime-python-sdk)
- [Qwen-Audio 客户端事件](https://help.aliyun.com/zh/model-studio/fun-audiochat-client-events)
- [Qwen-Audio 服务端事件](https://help.aliyun.com/zh/model-studio/qwen-audio-realtime-server-events)
- [Livetranslate 客户端事件](https://help.aliyun.com/zh/model-studio/live-translator-client-events)
- [Livetranslate 服务端事件](https://help.aliyun.com/zh/model-studio/live-translator-server-events)
