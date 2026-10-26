# DashScope 2.0 连接复用（语音实时面 + 全模型 HTTP）

对应《dashscope-2.0-vs-1.x-alignment-review.md》§7.4 的 **M6**。

## 背景与动机

1.x 有大量用户以**连接复用**模式接入（主要是 Java 侧）。语音业务、尤其
实时交互对延时极敏感：每省一次握手（TCP + TLS + websocket 升级，数百
毫秒级）就直接降低首包延迟。因此连接复用是硬需求而非优化项。

- **ws 复用**：1.x 的 `WsSession` 形态是"一连接一任务"，短任务高频调用
  （如逐句 TTS）下握手开销占比大；多路并发（如多路 ASR）下每路一条连接
  也浪费 fd。2.0 引入 `WsConnection`：一条物理连接上按 `header.task_id`
  解复用多个任务，支持**并发多任务**与**顺序复用**（免再次握手）。
- **HTTP 复用（所有模型）**：不止语音——2.0 所有资源的 HTTP 请求都
  走同一个 keep-alive 连接池（每个 `Client` 一个 httpx client，1.x 是
  每次调用新建 `requests.Session`）。M6 把池参数开放为 settings 配置，
  并给出同一 TCP 连接被复用的可测证明。

## 1. ws 复用：`WsConnection` + `connection=`

### 1.1 基本用法

```python
import dashscope

client = dashscope.Client(api_key="sk-xxx")

# 工厂方法继承 client 的 api_key / workspace / region / timeout；
# 连接由调用方持有并负责关闭。
conn = client.ws_connection()

# 顺序复用：两个识别任务共用一条连接，第二个任务免握手
for audio in (pcm_1, pcm_2):
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connection=conn,
    ) as rec:
        rec.send_audio_frame(audio)
        rec.stop()
        for result in rec.results():
            ...

# 并发：ASR 上行 + TTS 下行跑在同一条连接上
rec = client.audio.asr.recognize(model="paraformer-realtime-v2", connection=conn).start()
tts = client.audio.tts.connect(model="cosyvoice-v1", connection=conn).start()
rec.send_audio_frame(chunk)      # 二进制上行
for piece in tts.audio_chunks(): # 二进制下行
    play(piece)

conn.close()
```

支持 `connection=` 的入口（run-task 帧协议族）：

| 入口 | 方向 |
| --- | --- |
| `client.audio.asr.recognize(...)` | JSON 下行 + 二进制上行 |
| `client.audio.tts.connect(...)` | JSON + 二进制下行 |
| `client.audio.translation.recognize(...)` | JSON 下行 + 二进制上行 |
| `client.tingwu.realtime(...)` | JSON 下行 + 二进制上行 |

传入 `connection=` 后，`ws_url` / `connector` / `api_key` / `workspace` /
`headers` 等会话级参数被忽略（它们属于连接）。会话的 `close()` 只从连接
上摘离本任务，**不会关闭共享 socket**；`WsConnection.close()` 才关闭物理
连接并使所有未结束任务报错。

### 1.2 并发约束（协议决定）

websocket 二进制帧**不携带 task_id**，无法解复用，因此每条连接：

- **二进制上行单 owner**：第一个 `send_audio_frame` 的任务占用上行；其他
  任务再发二进制会 `RuntimeError`。任务结束/关闭后释放。
- **二进制下行单 owner**：TTS 会话在 `start()` 时自动
  `claim_binary_downlink()`；连接上同时只能有一个活跃合成（第二个
  `start()` 直接报错，要并发 TTS 请用多条连接）。仅有一个活跃任务时下行
  会隐式归属该任务。
- **纯 JSON 任务完全并发**：事件按 `header.task_id` 路由到各自队列，无
  数量限制。

典型可行组合：1 路 ASR（二进制上行）+ 1 路 TTS（二进制下行）+ 任意多
JSON 任务。典型不可行组合：2 路并发 TTS（下行无法区分）——开两条
`WsConnection` 即可。

底层对象也可单独使用（`dashscope.WsConnection` /
`dashscope.v2.transport.ws_connection.WsTask`）：`conn.open_task()` 得到与
`WsSession` 同接口的任务视图（`run_task` / `continue_task` /
`finish_task` / `send_binary` / `recv` / `recv_event` / `close`）。

### 1.3 读线程与超时

`WsConnection` 内部有一个后台读线程，socket 读永久阻塞；每个任务的接收
超时在解复用队列上生效（`open_task(timeout=...)`，默认取连接构造时的
`timeout`）。socket 断开或协议错误会广播给所有任务。

### 1.4 realtime 族：协议不可复用（验证结论）

realtime 协议（Omni / Qwen-Audio / Livetranslate，与 OpenAI Realtime
同构）**一条连接就是一个会话**：事件按会话级类型流转、没有 task_id，且
会话结束服务端即关闭连接——既无法并发解复用，也无法顺序复用。这不是
SDK 限制，是协议设计。`client.audio.realtime.connect` 每次新建连接即协议
语义的直接体现（loopback 测试已固化该行为）。

要压首开延迟，方向是**预热连接**（在用户开始说话前提前 `connect()`），
而不是复用；realtime 会话通常持续数十秒到数分钟，握手开销摊销后占比很
小。受益最大的是短任务高频的 TTS 与多路并发的 ASR——即 1.1 的形态。

## 2. HTTP keep-alive（所有模型生效）

### 2.1 默认行为

每个 `Client` 持有一个 httpx 连接池，`client.chat` / `images` / `audio` /
`files` … 所有资源请求共享：默认 `max_connections=100`、
`max_keepalive_connections=20`、`keepalive_expiry=5s`（与 httpx 默认一致；
1.x 每次调用新建 `requests.Session`，无跨调用复用）。
`client.with_options(...)` 在未改连接级选项时**共享**同一个池。

注意：池是 per-Client 的——频繁新建 `Client` 实例会击穿复用；长驻一个
client（或 with_options 派生）才是正确姿势。

### 2.2 settings 配置

```python
import dashscope

dashscope.settings.http_max_connections = 50            # 默认 100
dashscope.settings.http_max_keepalive_connections = 10  # 默认 20；0 = 关闭 keep-alive
dashscope.settings.http_keepalive_expiry = 30.0         # 默认 5.0 秒
```

环境变量 / `~/.dashscope/config.toml` 同名键：

| settings | 环境变量 | 默认 |
| --- | --- | --- |
| `http_max_connections` | `DASHSCOPE_HTTP_MAX_CONNECTIONS` | 100 |
| `http_max_keepalive_connections` | `DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS` | 20 |
| `http_keepalive_expiry` | `DASHSCOPE_HTTP_KEEPALIVE_EXPIRY` | 5.0 |

优先级与其他配置一致：settings 显式赋值 > 环境变量 > config.toml > 默认。
校验：`max_connections >= 1`、`max_keepalive_connections >= 0`、
`keepalive_expiry > 0`。

### 2.3 复用证明

`tests/v2/test_http_pool.py` 起本地 HTTP/1.1 服务器记录对端端口：

- 同一 `Client` 顺序 3 次请求 → 全部命中**同一 TCP 连接**（端口一致）；
- `http_max_keepalive_connections = 0` → 每次请求新连接，证明 settings →
  `httpx.Limits` → 传输行为端到端生效。

## 3. 1.x 对照

| 1.x | 2.0 |
| --- | --- |
| Java 侧连接池/复用（Python 1.x 无对应物，`WsSession` 一连接一任务） | `client.ws_connection()` + 各实时入口 `connection=`，并发 + 顺序复用 |
| 每次调用新建 `requests.Session` | 每 `Client` 一个 httpx 池，默认 keep-alive；池参数开放 settings |
| realtime（omni）一会话一连接 | 同（协议决定，不可复用；建议预热） |

## 相关文档

- dashscope-2.0-asr-sdk.md / dashscope-2.0-tts-sdk.md /
  dashscope-2.0-translation-sdk.md / dashscope-2.0-realtime-sdk.md
- dashscope-2.0-vs-1.x-alignment-review.md（§7.2-3、§7.4 M6）
