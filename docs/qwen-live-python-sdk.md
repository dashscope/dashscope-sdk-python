# Qwen Live Python SDK

Qwen Live 是面向实时语音交互的 WebSocket 模型：客户端持续上行音频，服务端返回音频、转写与用量等事件。接口位于 `client.audio.live`，提供连接管理、会话配置、音频上行、委派回注与事件迭代。

本文介绍如何用 Python SDK 接入 Qwen Live，完成「发送音频 → 接收并保存回复」的最小闭环，以及文本输入与委派模式。Qwen Live 使用 `session.*` 事件族，与 Omni／Qwen-Audio 的 `input_audio_buffer.*`、`response.*` 不是同一套协议。

---

## 前提条件

- **安装 SDK**：需要 Python 3.9 及以上，并安装包含 `client.audio.live` 的 DashScope 2.1 SDK。本文使用 `dashscope`、`dashscope.resources.audio` 导入路径，不使用 `dashscope.v2`。

  ```bash
  python -m pip install -U "dashscope>=2.1.0"
  python -c "from dashscope import Client; from dashscope.resources.audio import LiveSession"
  ```

  若升级后仍无法导入 `LiveSession`，请使用服务提供方指定的支持 Live 的 SDK 版本。

- **配置 API Key**：推荐通过环境变量 `DASHSCOPE_API_KEY` 提供，不要把密钥写入代码。Client 按 `显式参数 → 进程级 settings → 环境变量 → config.toml → 密钥文件` 的顺序解析配置；密钥须有目标接入环境的访问权限。

  ```bash
  export DASHSCOPE_API_KEY="<your-api-key>"
  export DASHSCOPE_WORKSPACE="<your-workspace-id>"
  ```

- **确认接入地址与模型**：Qwen Live 走业务空间专属网关，默认地址模板为

  ```
  wss://{workspace}.{region}.maas.aliyuncs.com/api-ws/v1/realtime
  ```

  `region` 默认 `cn-beijing`，另支持 `ap-southeast-1`、`cn-hongkong`、`us-east-1`、`eu-central-1`、`ap-northeast-1`。未配置业务空间时回退到该地域的共享网关（北京为 `wss://dashscope.aliyuncs.com`）并打印 `UserWarning`。

  > **重要**：SDK 会自动在地址上追加 `?model=<路由模型>`；若 `ws_url` 已带 `model` 查询参数，以已有值为准。接入私有化或预发部署时用 `ws_url` 显式指定完整地址。路由模型名（`connect`）与会话模型名（`update_session`）以你的环境实际提供为准。

---

## 请求参数

参数分三层：**连接参数**决定连到哪、如何鉴权；**会话配置**作为 `session.update` 报文下发，决定会话行为；**委派配置**决定由谁产生回复。路由模型、前端模型、后端模型分属这三层，不要混用同一个名称。

### 一、连接参数 `client.audio.live.connect(...)`

返回尚未建连的 `LiveSession`；调用 `connect()` 或进入 `with` 上下文才真正握手。

| 参数 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| `model` | `str` | 是 | 网关路由模型名，写入地址的 `?model=`。 |
| `ws_url` | `str` | 否 | 显式完整地址，优先级高于模板。 |
| `timeout` | `float` | 否 | 握手及单次 socket 收发超时（秒），须为正数；省略时取 Client 配置，不等于整场会话时长上限。 |
| `headers` | `Mapping[str, str]` | 否 | 自定义握手请求头，同名头以传入值为准。 |
| `api_key` | `str` | 否 | 本连接使用的密钥，写入 `Authorization: Bearer`；省略或空字符串时使用 Client 已解析的密钥。 |
| `workspace` | `str` | 否 | 业务空间 ID，省略时使用 Client 配置；影响默认地址及 `X-DashScope-WorkSpace` 请求头，显式 `ws_url` 不会被改写。 |
| `connector` | `Callable` | 否 | 自定义建连函数，仅用于测试或特殊传输。 |

```python
import os
from dashscope import Client

with Client(region="cn-beijing") as client:
    session = client.audio.live.connect(
        model=os.getenv("LIVE_ROUTE_MODEL", "qwen-live"),
        ws_url=os.getenv("LIVE_WS_URL"),
        timeout=30,
    )  # 此时尚未握手；完整用法见下方示例
```

### 二、会话配置 `update_session(...)`

序列化为 `session.update` 的 `session` 节点，只发送显式传入的字段。服务端校验、准备并提交成功后返回 `session.updated`（首次成功 `revision=1`）；失败时不提交新配置，只返回 `error`。

| 参数 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| `model` | `str` | 是 | 会话（前端）模型名，如 `audio-live`、`qwen-audio-realtime`；可用名称由部署决定，SDK 不设模型白名单。 |
| `delegation` | `Dict` | 是 | 委派配置，见下。 |
| `instructions` | `str` | 否 | 前端模型系统提示词。 |
| `voice` | `str` | 否 | 音色，枚举 `台湾女声`、`星悦`、`古风女声`、`古风男声`；省略时用部署默认值。 |
| `tts` | `Dict` | 否 | Audio Live 的语音配置，仅支持 `system_prompt`。 |
| `input_audio_format` | `str` | 否 | 上行音频格式，仅接受 `pcm`。 |
| `output_audio_format` | `str` | 否 | 下行音频格式，仅接受 `pcm`。 |
| `modalities` | `List[str]` | 否 | 输出模态，默认 `["text", "audio"]`。 |
| `tools` | `List[Dict]` | 否 | 前台 function 工具，默认空数组。 |
| `context` | `Dict` | 否 | 上下文节点；服务端实际必填，见下方说明。 |
| `turn_detection` | `Dict` 或 `None` | 否 | 轮次检测配置；省略与显式 `None` 不同，后者发送 JSON `null`，可用配置由前端模型决定。 |
| `input_audio_transcription` | `Dict` 或 `None` | 否 | 输入转写配置；显式 `None` 原样发送为 JSON `null`。 |
| `transport` | `Dict` | 否 | 省略或空对象为默认 WebSocket；显式传输仅实现 `aoq`。 |
| `parameters` | `Dict` | 否 | 服务端新增字段的透传容器，按键合并进 `session` 节点；不能覆盖上述具名字段。 |
| `event_id` | `str` | 否 | 本次请求的客户端事件 ID，长度为 1～128 字符；省略时由 SDK 生成。其他发送方法同样支持。 |

> **说明**：服务端要求配置 `context.compaction`，否则返回 `invalid_session_configuration`、消息 `session.update: context_compaction_not_configured`。即使不启用压缩也需显式下发：`{"compaction": {"enabled": False, "mode": "local"}}`。显式设置 `audio-live` 的 `tts.system_prompt` 时，需以 `You are a helpful assistant and asked to generate both text and speech tokens at the same time.` 原句开头；不自定义时省略 `tts`，SDK 不替你补前缀。

委派类型非法、后端模型为空、音频格式不是 `pcm` 等情况在 SDK 本地抛 `ValueError`，不发出网络请求。

```python
session.update_session(
    model="audio-live",
    instructions="用简洁中文回答用户的语音问题",
    input_audio_format="pcm",
    output_audio_format="pcm",
    context={"compaction": {"enabled": False, "mode": "local"}},
    delegation={"type": "none"},
)
```

### 三、委派配置 `delegation`

决定「谁来产生回复」。

| 参数 | 类型 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| `type` | `str` | 是 | `none`（不委派）、`client`（外部 Agent）、`responses`（后台模型）。 |
| `responses` | `Dict` | `responses` 模式必填 | 后台模型配置；其他模式不得携带。 |
| `responses.model` | `str` | 是 | 后台模型名，须匹配当前部署，以环境实际可用为准。 |
| `responses.instructions` | `str` | 否 | 后台模型提示词。 |
| `responses.tools` | `List[Dict]` | 否 | 后台模型可用工具。 |

- **`none`**：由前端模型自行产生回复，客户端只需上行音频。
- **`client`**：服务端不产生回复，而是下发 `session.delegation.created`，由你的外部 Agent 通过 `append_instruction` / `append_thinking` / `append_commentary` 回注内容，再由前端模型组织为语音。
- **`responses`**：由后台模型直接产生回复，客户端只需上行音频（可选用 `create_response_item` / `create_response` 补充输入或主动请求回复）。

配置方式见示例二；SDK 只发送与接收事件，不会自动执行外部 Agent 或工具。

---

## 关键接口 `LiveSession`

同一实例同一时刻只能打开一条连接；关闭完成后可显式重新 `connect()`，SDK 会清空本地会话状态，但不会恢复旧会话或重放音频。所有发送方法返回**本次请求的 `event_id`**，不是服务端确认。

| 方法／属性 | 触发的服务端事件 | 说明 |
| --- | --- | --- |
| `connect()` | `session.created` | 建立 WebSocket，也可用 `with` 自动打开与释放。仅表示连接已打开，**不代表会话已配置**。 |
| `update_session(...)` | `session.updated` | 下发会话配置。 |
| `append_audio(audio, *, event_id=None)` | — | 追加音频帧，接受 `bytes` 或 Base64 `str`；空数据或奇数字节抛 `ValueError`。 |
| `mute_input_audio()` / `unmute_input_audio()` | `session.input_audio.muted` / `.unmuted` | 上行静音／解除静音；解除后不回放静音期间的音频。 |
| `append_commentary(content, *, delegation_id=None, ...)` | `session.commentary.appended` | 不带 `delegation_id` 时发送用户文本；带 `delegation_id` 时回注 `client` 委派的任务结果。 |
| `append_instruction(content, *, delegation_id, ...)` | `session.instructions.appended` | 向指定委派回注提示词（`client` 模式）。 |
| `append_thinking(content, *, delegation_id, ...)` | `session.thinking.appended` | 回注静默推理上下文，不保证产生语音。 |
| `create_session_item(item=省略)` | — | 发送 `session.item.create`；省略 `item` 时不发送该字段，不等同于显式传入 `None`。 |
| `submit_function_output(call_id, output)` | — | 回应 `session.event` 中的前台 function call，不用于后端工具。 |
| `create_response_item(item)` | — | 向 Responses 后端追加输入项，不触发回复。 |
| `create_response(response=None)` | 取决于模式 | 在 `responses` 模式启动后台任务；在支持主动回复的前端请求作答。纯文本用法见示例二。 |
| `end_session()` | `session.closed` | 发送业务事件 `session.close`；发送后禁止其他输入，需继续接收直到 `session.closed`。关闭请求待确认期间重复调用返回同一请求 ID。 |
| `close()` | — | 最多等待在途发送 1 秒，再发送标准 WebSocket Close（1000）并独立等待握手最多 1 秒；任一阶段超时则强制释放，不保证此时对端收到 Close 帧。不等待业务事件 `session.closed`，重复调用安全。 |
| `cancel()` | — | 立即中断并释放 socket，不进行关闭握手。对应 Java SDK 的 `cancel()`。 |
| `send_event(event)` | 取决于报文 | 发送任意协议事件的底层入口；`type` 必须是非空白字符串，`event_id` 省略时自动生成。 |
| `model` / `url` | — | 路由模型名／实际握手地址（含 `?model=`）。 |
| `session_id` / `close_reason` | — | `session.created` 的 `session.id`／`session.closed` 的 `reason`，收到前为 `None`。 |
| `last_message` | — | 最近一条服务端事件。 |

> **音频要求**：上行为 16 kHz、单声道、16-bit 有符号小端裸 PCM（PCM16LE）；下行为 24 kHz、单声道、PCM16LE，Base64 编码，无 WAV／容器头。SDK 不重采样、不识别 WAV 容器、不改变字节顺序，WAV 文件请先剥离文件头。必须等 `session.updated` 后再上行：每帧 3200 字节（100 ms），每次 `append_audio()` 返回后等待 100 ms，延迟后**不追赶补发、不丢弃输入帧**，实际发送速率略低于 1x。节奏由应用负责，SDK 不限速。文件结束后继续发送静音帧以维持 `audio-live` 的双工音频时钟，直至应用决定结束会话。

---

## 事件迭代与错误契约

使用一个接收线程消费事件，音频在另一个线程上行；也可以像完整示例一样由主线程接收。SDK 对收发两个方向分别加锁，不自动创建后台线程。不要同时用 `events()` 与 `recv_event()` 消费同一连接。

| 方法 | 说明 |
| --- | --- |
| `events()` | 迭代器，持续产出事件直到正常关闭。可恢复的 `error` 事件产出不抛出；致命错误抛出异常。 |
| `recv_event()` | 读取一条事件；服务端错误均抛异常，仅可恢复错误保留连接。正常关闭后返回 `None`。 |

> **错误契约**：`error.type == "invalid_request_error"` 为可恢复业务错误，其余业务错误关闭连接。意外断连、超时、非法事件分别抛 `APIConnectionError`、`APITimeoutError`、`WsEventError`；握手 HTTP 错误映射为对应异常（401 → `AuthenticationError`，429 → `RateLimitError`）。这些异常均继承 `DashScopeError`；WebSocket 业务错误的 `status_code` 为 `None`。致命失败会被记住，后续接收仍抛出该异常，`close()` 不清除失败；没有已记录失败的本地关闭／取消后接收返回 `None`。`KeyboardInterrupt` 与 `SystemExit` 原样传播。

```python
from dashscope import DashScopeError

try:
    for event in session.events():
        if event["type"] == "error":
            print(event["error"])  # 可恢复；根据业务决定是否继续
except DashScopeError as exc:
    print(f"[错误] {exc}")
```

### 服务端事件类型

在事件循环中通过 `type` 字段区分。事件名常量从 `dashscope.resources.audio` 或 `dashscope.resources.audio.live` 导入，如 `EVENT_SESSION_UPDATED`、`EVENT_SESSION_OUTPUT_AUDIO_DELTA`，与下表字面值一一对应：

| 事件 `type` | 含义 |
| --- | --- |
| `session.created` | 会话已创建，含 `session.id` 与服务端默认配置；一次连接可能收到多条（路由层与前端各一条）。 |
| `session.updated` | 会话配置已生效（**上行音频前应等待此事件**），含完整配置与 `revision`。 |
| `session.input_transcript.delta` | 用户上行语音转写分片；时间戳以服务端实际返回为准。 |
| `session.output_transcript.delta` | 助手回复文本分片。 |
| `session.output_audio.delta` | 下行音频分片，`delta` 为 Base64 裸 PCM16LE。 |
| `session.output_audio.interrupted` | 当前下行音频被打断，**不是**回复结束信号。 |
| `session.commentary.appended` / `session.instructions.appended` / `session.thinking.appended` | 委派回注内容确认。 |
| `session.input_audio.muted` / `session.input_audio.unmuted` | 静音状态确认。 |
| `session.delegation.created` | 前端派发新任务或补充，含 `delegation_id`、`offset_ms`、`summary`、`context`；`client` 与 `responses` 模式都会发。 |
| `session.event` | 前台模型的嵌套事件；其中的 function call 可用 `submit_function_output` 回应。 |
| `response.event` | 后端 Responses 模型的原生事件，后台 token 用量在此观察。 |
| `session.usage.updated` | 用量更新，如累计音频时长 `usage.seconds`；上报节奏与字段以服务端返回为准。 |
| `session.closed` | 会话已关闭，含 `reason`（`close_requested`／`expired`／`content`／`remote_hangup`／`connection_lost`）与最终 `usage`。 |
| `error` | 服务端错误，含 `error.type`（`invalid_request_error`／`server_error`）、`error.code`、`error.message`。 |

协议可能新增事件类型，遇到未知 `type` 应忽略而不是报错。

---

## 完整示例

### 示例一：最小闭环（语音文件输入）

保存为 `qwen_live_quickstart.py`。上行一个 PCM／WAV 文件，打印双向转写并保存回复。发送线程在文件结束后继续上行静音，并独立检查回复是否安静；主线程持续接收，直到 `session.closed`。本例不采集麦克风、不自动播放音频。

```python
import base64
import os
import sys
import time
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

from dashscope import Client

FRONTEND_MODEL = os.getenv("LIVE_FRONTEND_MODEL", "audio-live")
FRAME_BYTES = 3200
IDLE_SECONDS = 5.0
REPLY_TIMEOUT = 90.0
CLOSE_TIMEOUT = 5.0


def read_pcm(path):
    path = Path(path)
    if path.suffix.lower() == ".pcm":
        pcm = path.read_bytes()
    elif path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(),
                    audio.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("WAV 必须是 16 kHz、单声道、16-bit PCM")
            pcm = audio.readframes(audio.getnframes())
            if len(pcm) != audio.getnframes() * 2:
                raise ValueError("WAV 数据被截断")
    else:
        raise ValueError("仅支持 .pcm 或 .wav 文件")
    if not pcm or len(pcm) % 2:
        raise ValueError("音频必须是非空、偶数字节的 PCM16LE")
    return pcm


def send_audio(session, pcm, stopped, progress, lock):
    try:
        for offset in range(0, len(pcm), FRAME_BYTES):
            if stopped.is_set():
                return
            session.append_audio(pcm[offset:offset + FRAME_BYTES].ljust(FRAME_BYTES, b"\0"))
            if stopped.wait(0.1):
                return
        input_finished = time.monotonic()
        while not stopped.is_set():
            now = time.monotonic()
            with lock:
                text_at, audio_at = progress["text"], progress["audio"]
            if text_at is not None and audio_at is not None:
                if now - max(input_finished, text_at, audio_at) >= IDLE_SECONDS:
                    with lock:
                        progress["closing"] = True
                    session.end_session()  # 不再发送音频；主线程继续等关闭确认
                    if not stopped.wait(CLOSE_TIMEOUT):
                        raise TimeoutError("未收到 session.closed")
                    return
            if now - input_finished >= REPLY_TIMEOUT:
                raise TimeoutError("回复未在等待窗口内结束")
            session.append_audio(bytes(FRAME_BYTES))
            stopped.wait(0.1)
    except Exception:
        session.cancel()  # 发送失败时唤醒接收方；异常由 sender.result() 上抛
        raise


def main(input_path, output_path):
    pcm = read_pcm(input_path)
    stopped, lock = Event(), Lock()
    progress = {"text": None, "audio": None, "closing": False}
    with Client() as client:
        with client.audio.live.connect(
            model=os.getenv("LIVE_ROUTE_MODEL", "qwen-live"),
            ws_url=os.getenv("LIVE_WS_URL"), timeout=30,
        ) as session:
            with ThreadPoolExecutor(max_workers=1) as pool, open(output_path, "xb") as out:
                sender = None
                try:
                    session.update_session(
                        model=FRONTEND_MODEL, delegation={"type": "none"},
                        instructions="用简洁中文回答用户的语音问题",
                        input_audio_format="pcm", output_audio_format="pcm",
                        context={"compaction": {"enabled": False, "mode": "local"}},
                    )
                    for event in session.events():
                        kind = event["type"]
                        if kind == "session.updated" and sender is None:
                            sender = pool.submit(send_audio, session, pcm, stopped, progress, lock)
                        elif kind == "session.input_transcript.delta":
                            print(f"\n[用户] {event['delta']}", flush=True)
                        elif kind == "session.output_transcript.delta":
                            if event["delta"]:
                                with lock:
                                    progress["text"] = time.monotonic()
                            print(event["delta"], end="", flush=True)
                        elif kind == "session.output_audio.delta":
                            data = base64.b64decode(event["delta"], validate=True)
                            if not data or len(data) % 2:
                                raise ValueError("下行音频不是有效 PCM16LE")
                            out.write(data)
                            samples = array("h", data)
                            if sys.byteorder != "little":
                                samples.byteswap()
                            if sum(x * x for x in samples) >= len(samples) * 150 ** 2:
                                with lock:
                                    progress["audio"] = time.monotonic()
                        elif kind == "error":
                            raise RuntimeError(event["error"])  # 本示例选择失败即退出
                        elif kind == "session.closed":
                            with lock:
                                requested = progress["closing"]
                            if not requested or event.get("reason") != "close_requested":
                                raise RuntimeError(f"会话提前结束：{event.get('reason')}")
                            print(f"\n关闭原因：{event.get('reason')}")
                            break
                finally:
                    stopped.set()
                    session.close()  # 先有界关闭 WebSocket，再等待发送线程退出
                if sender is not None:
                    sender.result()
    print()


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except KeyboardInterrupt:
        print()
```

配置业务空间，或通过 `LIVE_WS_URL` 显式指定接入地址；`LIVE_ROUTE_MODEL` 设置路由模型，`LIVE_FRONTEND_MODEL` 设置前端模型。这三个 `LIVE_*` 变量由示例读取，不是 SDK 全局配置项。运行时 `output.pcm` 必须不存在：

```bash
python qwen_live_quickstart.py input.pcm output.pcm
```

`output.pcm` 为 24 kHz、单声道、16-bit 裸 PCM，含收到的静音帧。示例以 RMS ≥ 150 判断有声输出，收到文本和有声音频后，待输入结束且二者连续安静 5 秒再请求关闭；这不是服务端完成确认，长停顿或背景音可能影响判断。输入结束后最多等待回复 90 秒，发出关闭请求后最多再等 5 秒；超时会取消连接并抛异常。未满足完成条件便收到关闭事件也会报错。

### 示例二：文本输入与委派模式

**纯文本输入**：在支持主动文本回复的前端使用 `append_commentary()` 后还需 `create_response()`；前者的回执不代表已开始作答。将示例一 `main()` 中的 `pcm = read_pcm(input_path)` 改为 `pcm = b""`，并用下面代码替换 `session.updated` 分支中的 `pool.submit(...)` 一行，其余不变；空输入只发送静音以维持音频时钟：

```python
session.append_commentary("你好，请用一句话介绍自己。")
session.create_response()
sender = pool.submit(send_audio, session, pcm, stopped, progress, lock)
```

**后台模型委派**：保留示例一的语音输入，在配置前定义下列变量，并将 `update_session()` 的 `delegation={"type": "none"}` 替换为 `delegation=delegation`。`create_response_item()` 仅用于此模式的后台输入，不能代替 `none` 模式下的 `append_commentary()`：

```python
delegation = {
    "type": "responses",
    "responses": {
        "model": "<backend-model>",
        "instructions": "直接回答用户问题，不使用外部工具",
    },
}
```

**外部 Agent 委派**：将示例一配置改为 `delegation={"type": "client"}`，并在事件循环中增加下面分支。这里用固定文本演示回注，实际结果由你的 Agent 产生：

```python
elif kind == "session.delegation.created":
    session.append_commentary("杭州明天多云转晴，18 到 26 度。",
                              delegation_id=event["delegation_id"])
```

当前接入服务对同一 `delegation_id` 只接受一次回注，重复回注可能返回 `CONFLICT`；不要连续调用三种回注方法。`append_instruction` 用于让出等待用户输入，`append_thinking` 用于静默上下文，`append_commentary` 用于任务结果；仅收到回执不能视为语音播放完成。耗时的 Agent 处理应移出接收线程。

---

## 重要说明与限制

- **多轮对话复用同一会话**：首轮后保持收发循环，在同一连接继续发送下一轮音频；不要在每轮结束时调用 `end_session()`。完整示例在一轮后关闭，仅用于最小接入。
- **没有下行音频完成事件**：`session.usage.updated`、`session.output_audio.interrupted` 都不能作为回复结束信号；业务结束确认应看 `session.closed` 的事件类型，而不是 `session.status`。
- **配置更新不等于重建连接**：音色或传输等连接级选项可能不支持热更新；服务端返回 `Changing connection options requires a new session` 时，应重新建连并初始化。
- **重连需重新初始化**：SDK 不自动重连或重发，已打开或正在关闭时再次 `connect()` 会抛 `RuntimeError`。重连后须发送配置并创建新的 `events()` 迭代器，不能继续使用绑定旧连接的迭代器。
- **每个会话必须释放连接**：用 `with` 或 `finally: session.close()` 兜底；多线程时先停止发送并关闭连接，再等待工作线程退出。需要业务关闭确认时，先停止音频上行，再发 `end_session()` 并保持接收。
- **握手鉴权失败**：401 表示连接未建立，先核对 API Key、目标地址、地域及业务空间的访问权限；不能通过更换音频格式或调节发送间隔解决。向服务提供方排查时提供 `exc.request_id`，不要提供 API Key。
- **按错误阶段排查**：握手成功不代表前端模型可用；`session.update` 的参数错误与后续推理错误需分别按服务端 `error.code` / `message` 处理，不要把单次 `UNSUPPORTED` 视为整个模型不可用。
