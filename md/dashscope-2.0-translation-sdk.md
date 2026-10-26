# 实时语音翻译（Gummy）Python SDK

通过 `client.audio.translation`，您可以在一条 WebSocket 双工会话上边发送音频边获取识别与翻译结果（对应 1.x `TranslationRecognizerRealtime` / `TranslationRecognizerChat`，模型如 `gummy-realtime-v1`、`gummy-chat-v1`）。本文对应当前仓库的 SDK 实现；模型能力、音频限制及服务端字段以相应模型协议为准。

更新的 qwen3-livetranslate 实时翻译模型（如 `qwen3.5-livetranslate-flash-realtime`）走 Realtime 协议，请使用 [`audio.realtime.connect`](dashscope-2.0-realtime-sdk.md) 并配置 `update_session(translation_params=...)`，不在本资源面。

| 场景 | 同步客户端 `Client` | 异步客户端 `AsyncClient` |
| --- | --- | --- |
| 实时语音翻译 | `audio.translation.recognize()` | 不支持 |
| 本地文件翻译 | `audio.translation.recognize_file()` | 不支持 |

## 前期准备

1. 准备 API Key，并通过环境变量 `DASHSCOPE_API_KEY` 配置，避免将密钥写入代码。
2. 建议通过 `DASHSCOPE_WORKSPACE` 配置工作空间，并确保 API Key、工作空间和模型所在地域匹配。地域默认是 `cn-beijing`，可通过 `DASHSCOPE_REGION` 或 `Client(region=...)` 设置。
3. 在本仓库根目录执行源码安装：`python -m pip install -e .`。
4. 实时示例需要当前目录下的 `input.pcm`：有限长度、无文件头、16 kHz、16 bit、单声道的原始 PCM 音频。

## 快速开始

### 实时语音翻译

以下代码可保存为 `translation_example.py`，在音频文件所在目录运行 `python translation_example.py`。发送和接收分别运行在两个线程中，逐块读取文件、逐条处理双语结果。

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


def receive_results(session):
    for result in session.results():
        transcription = result.get_transcription()
        if transcription and transcription.get("text"):
            label = "最终" if result.is_sentence_end() else "中间"
            print(label, "识别:", transcription["text"], flush=True)
        for translation in result.get_translations():
            label = "最终" if translation.get("sentence_end") else "中间"
            print(label, f"翻译({translation['lang']}):", translation["text"], flush=True)


def main(audio_path):
    with Client() as client:
        session = client.audio.translation.recognize(
            model="gummy-realtime-v1",
            format="pcm",
            sample_rate=16000,
            source_language="zh",
            translation_enabled=True,
            translation_target_languages=["en"],
            timeout=30.0,
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                session.start()
                futures = [
                    executor.submit(receive_results, session),
                    executor.submit(send_audio, session, audio_path),
                ]
                for future in as_completed(futures):
                    future.result()
            finally:
                session.close()
        print("request_id:", session.request_id)
        print("usage:", session.usage)


if __name__ == "__main__":
    main("input.pcm")
```

对应 1.x `TranslationRecognizerRealtime.start()` + `send_audio_frame()` + `stop()` 的回调流程：v2 用主动迭代 `session.results()` 替代 `TranslationRecognizerCallback.on_event`，每一项就是 1.x 回调收到的 `(transcription_result, translation_result, usage)` 组合。

### 本地文件翻译

对应 1.x `TranslationRecognizerRealtime.call(file)`：文件自动按 12800 字节分块、单会话跑到底，逐项产出类型化结果（1.x `TranslationRecognizerResultPack` 的逐项形态）。

```python
from dashscope import Client

with Client() as client:
    for result in client.audio.translation.recognize_file(
        "input.pcm",
        model="gummy-realtime-v1",
        format="pcm",
        sample_rate=16000,
        source_language="zh",
        translation_enabled=True,
        translation_target_languages=["en"],
    ):
        transcription = result.get_transcription()
        if transcription and result.is_sentence_end():
            print("识别:", transcription["text"])
        for translation in result.get_translations():
            if translation.get("sentence_end"):
                print("翻译:", translation["text"])
```

### gummy-chat 单句模型

`gummy-chat-v1` 与实时模型共用同一协议与入口，直接 `recognize(model="gummy-chat-v1", ...)` 即可。1.x `TranslationRecognizerChat` 与实时版的差别仅在客户端行为（收到首个句结束后停止发送音频）；v2 不单独建模，需要该行为时在 `results()` 里判断 `is_sentence_end()` 后自行停止发送。

## 接口说明

`Translation` 的推荐调用顺序与 `Recognition` 相同：创建对象 → `start()` → 并发发送音频和读取结果 → `stop()` → 读到 `task-finished` → `close()`。`with client.audio.translation.recognize(...) as session:` 自动执行 `start()` / `close()`。

| 接口 | 说明 |
| --- | --- |
| `translation.recognize(...)` | 创建 `TranslationSession`。除下表的翻译参数外，与 `asr.recognize` 相同（`format`/`sample_rate`/`parameters`/`input`/`phrase_id`/`ws_url`/`connector`/`timeout`/`headers`/`api_key`/`workspace`）。 |
| `translation.recognize_file(path, ...)` | 本地文件一次性翻译，产出 `Iterator[TranslationResult]`；路径校验在调用时即生效。 |
| `session.send_audio_frame(data)` | 二进制音频帧，通常 `bytes`。 |
| `session.stop()` | 结束输入，不等待最终结果、不关闭连接。 |
| `session.events()` | 原始事件字典流（含 `task-finished`）。 |
| `session.results()` | 类型化双语结果流，过滤心跳与 `task-finished`。 |
| `session.close()` | 关闭底层连接，可重复调用。 |

翻译参数（缺省时不发送，由服务端默认值接管：识别开、翻译关、源语言自动）：

| 参数 | 说明 |
| --- | --- |
| `source_language` | 源语言代码（如 `"zh"`、`"en"`），缺省为自动检测。 |
| `transcription_enabled` | 是否输出源语言识别结果。 |
| `translation_enabled` | 是否启用翻译；启用时需给 `translation_target_languages`。 |
| `translation_target_languages` | 目标语言列表（当前仅支持一个目标语言），如 `["en"]`。 |
| `max_end_silence` | VAD 断句静音阈值（毫秒，200–6000）。 |

`TranslationResult` 是单个结果事件的类型化视图：

| 接口 | 说明 |
| --- | --- |
| `get_transcription()` | 源语言句子字典（`text`/`begin_time`/`end_time`/`words`/...），该帧无识别结果时为 `None`。 |
| `get_translations()` | 各目标语言句子字典列表（`lang`/`text`/`sentence_end`/...），无翻译结果时为空列表。 |
| `is_sentence_end()` | 识别或任一翻译到达句尾时为 `True`。 |
| `get_usage()` | 该帧携带的 `usage`，无则 `None`。 |
| `request_id` / `usage` / `output` | 原始字段属性；`output` 保留完整服务端输出。 |

## 连接复用

`recognize(..., connection=conn)` 把翻译任务跑在共享的
`WsConnection`（`client.ws_connection()` 创建）上：顺序任务免再次握手，
纯 JSON 事件并发无限制；二进制上行每条连接单 owner（一路活跃翻译/识别）。
详见 [连接复用指南](dashscope-2.0-connection-reuse-sdk.md)。

## 相关文档

- [连接复用指南](dashscope-2.0-connection-reuse-sdk.md)
- [ASR Python SDK](dashscope-2.0-asr-sdk.md)
- [Realtime Python SDK](dashscope-2.0-realtime-sdk.md)（qwen3-livetranslate 协议族）
