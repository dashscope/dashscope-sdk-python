# 语音接口调用示例

所有命令在仓库根目录执行，使用 V2 SDK：`from dashscope import Client`。

## 准备

```bash
python -m pip install -e .
# 使用麦克风时安装，并允许终端或 IDE 访问系统麦克风
python -m pip install sounddevice

OUT="$(mktemp -d)"
```

- 配置环境变量 `DASHSCOPE_API_KEY`，按需设置 `DASHSCOPE_REGION`、`DASHSCOPE_WORKSPACE`；示例不自动加载 `.env`。真实服务调用可能计费。
- 输出不覆盖已有文件；`--output-dir` 必须是尚不存在的目录，父目录需已存在。
- 本地 ASR WAV 使用 **16 kHz、16-bit PCM、单声道**，不自动重采样。
- TTS 默认模型为 `qwen-audio-3.1-tts-flash`，音色为 `longanhuan_v3.1`；默认保存 WAV 后播放，添加 `--no-play` 只保存。
- 各命令可加 `--help` 查看完整参数；均支持 `--region`、`--workspace`、`--timeout`。

## 直接调用版（不使用 `with`）

`direct_call/` 提供全部 12 个调用示例的同名副本，参数和业务流程不变。将原版命令中的 `examples.01_atomic_capabilities.audio.` 替换为 `examples.01_atomic_capabilities.audio.direct_call.` 即可；原版仍保留。

```bash
python -m examples.01_atomic_capabilities.audio.direct_call.asr_websocket tests/data/asr_example.wav \
  --transcript "$OUT/asr-direct.txt"
python -m examples.01_atomic_capabilities.audio.direct_call.realtime_omni \
  --wav tests/data/asr_example.wav --output "$OUT/omni-direct.wav" --no-play
```

副本显式调用 ASR/TTS 会话的 `start()`、Realtime 会话的 `connect()`，用 `try/finally` 关闭文件、客户端和会话；先关闭会话，再 `shutdown()` 等待接收线程退出。HTTP 结果下载仍使用不携带 SDK 凭据的独立客户端。

## asr_file.py — HTTP 录音文件识别

接口：`client.audio.asr.jobs.create()` / `jobs.wait()`。默认模型：`qwen-audio-3.1-asr-flash-filetrans`。

将服务可下载的、你拥有或获授权使用的 HTTPS 音频地址设置为 `AUDIO_URL`，每个任务传一个 URL；不接受本地路径，也不上传文件。输出任务记录 `task.json` 和逐文件转写 JSON/TXT。

```bash
python -m examples.01_atomic_capabilities.audio.asr_file \
  --url "$AUDIO_URL" --output-dir "$OUT/batch"
```

超时或中断后，从 `task.json` 取出 `task_id` 设置为 `TASK_ID`，在同一地域和工作空间恢复，不重复提交：

```bash
python -m examples.01_atomic_capabilities.audio.asr_file \
  --task-id "$TASK_ID" --output-dir "$OUT/batch-resumed"
```

## asr_http_stream.py — RESTful 流式返回识别

接口：`client.transport.open_stream()`，POST `/services/aigc/multimodal-generation/generation`，通过 SSE 接收文本。默认模型：`qwen-audio-3.1-asr-flash`。

一次提交完整音频，不是双工上传；返回累计文本，TXT 始终保存最新结果。短音频可能只返回一次文本。

```bash
python -m examples.01_atomic_capabilities.audio.asr_http_stream \
  --file tests/data/asr_example.wav --transcript "$OUT/asr-rest.txt"
```

可将 `--file` 替换为 `--url "$AUDIO_URL"`；加 `--events "$OUT/asr-rest.jsonl"` 保存原始事件。

## asr_websocket.py — WebSocket 双工识别

接口：`client.audio.asr.recognize()`。默认模型：`qwen-audio-3.1-asr-flash-streaming`。

读取本地 WAV，分片发送 PCM，同时接收识别结果；最终文本保存到 TXT。

```bash
python -m examples.01_atomic_capabilities.audio.asr_websocket tests/data/asr_example.wav \
  --transcript "$OUT/asr-wav.txt" --partials
```

`--partials` 显示临时结果，不写入最终 TXT。

## asr_microphone.py — 麦克风实时识别

接口：`client.audio.asr.recognize()`，模型同上。通过 `sounddevice` 采集系统麦克风，实时识别并保存录音与转写。

```bash
python -m examples.01_atomic_capabilities.audio.asr_microphone --list-devices
python -m examples.01_atomic_capabilities.audio.asr_microphone --seconds 5 \
  --recording "$OUT/microphone.wav" --transcript "$OUT/microphone.txt"
```

`--device` 指定设备；用 `--no-recording` 替换 `--recording` 可不保存录音。采集中 Ctrl+C 停止输入并等待最终结果。

## tts_http.py — HTTP 一次性合成

接口：`client.audio.tts.http.synthesize()`。一次提交文本，下载返回的音频 URL，保存 WAV 后播放。

```bash
python -m examples.01_atomic_capabilities.audio.tts_http \
  --text "你好，这是 HTTP 一次性合成。" --output "$OUT/tts-http.wav"
```

## tts_http_stream.py — HTTP/SSE 流式合成

接口：`client.audio.tts.http.stream()`。一次提交完整文本，流式接收音频分片，保存为一个 WAV 后播放。

```bash
python -m examples.01_atomic_capabilities.audio.tts_http_stream \
  --text "你好，这是 HTTP 流式合成。" --output "$OUT/tts-http-stream.wav"
```

两个 HTTP TTS 示例均支持用 `--text-file article.txt` 替换 `--text`。

## tts_websocket.py — WebSocket 双工流式合成

接口：`client.audio.tts.connect()`，通过 `send_text()` 追加文本，`stop()` 结束输入；同时接收音频并保存 WAV，完成后播放。

```bash
python -m examples.01_atomic_capabilities.audio.tts_websocket \
  --text "你好，这是第一段。" --text "这是第二段。" \
  --output "$OUT/tts-websocket.wav"
```

`--text-file article.txt` 可按行发送；`--events` 可保存 JSONL 事件。也可朗读 LLM 流式回答，此时额外调用 `client.chat.completions.create(stream=True)`：

```bash
python -m examples.01_atomic_capabilities.audio.tts_websocket \
  --prompt "请用三句话介绍春天。" --llm-model qwen-plus \
  --output "$OUT/tts-llm.wav"
```

`--text`、`--text-file`、`--prompt` 三选一。

## voice_clone.py — 声音复刻管理

接口：`client.audio.voices.create()` / `get()` / `list()` / `delete()`。默认目标模型：`qwen-audio-3.0-tts-flash`。

将自己或获授权的参考语音 HTTPS 地址设置为 `CLONE_AUDIO_URL`；推荐 10～20 秒清晰语音。本脚本不上传音频、不播放、不自动删除音色。

```bash
python -m examples.01_atomic_capabilities.audio.voice_clone create \
  --url "$CLONE_AUDIO_URL" --prefix demo --output "$OUT/voice.json"
```

将输出的 `voice_id` 设置为 `VOICE_ID`，查询状态或列出音色：

```bash
python -m examples.01_atomic_capabilities.audio.voice_clone get --voice-id "$VOICE_ID"
python -m examples.01_atomic_capabilities.audio.voice_clone list --prefix demo
```

状态为 `OK` 后，用创建时相同的模型合成：

```bash
python -m examples.01_atomic_capabilities.audio.tts_websocket \
  --model qwen-audio-3.0-tts-flash --voice "$VOICE_ID" \
  --text "你好，这是复刻音色。" --output "$OUT/cloned.wav"
```

仅在确定不再使用时手动删除：

```bash
python -m examples.01_atomic_capabilities.audio.voice_clone delete --voice-id "$VOICE_ID" --yes
```

## realtime_omni.py — Omni 单轮语音对话

接口：`client.audio.realtime.connect()`。固定模型：`qwen3.8-omni-flash-realtime`，默认音色：`longanlingxin`。

输入语音，通过 `commit()` / `create_response()` 请求回答，接收至 `response.done`，保存 WAV 后播放。

```bash
python -m examples.01_atomic_capabilities.audio.realtime_omni \
  --wav tests/data/asr_example.wav --output "$OUT/omni-reply.wav"
```

## realtime_qwen_audio.py — Qwen-Audio 单轮语音对话

接口：`client.audio.realtime.connect()`。固定模型：`qwen-audio-3.0-realtime-plus`，默认音色：`longanqian`。

按 Qwen-Audio 的音频格式配置会话，输入结束后 `commit()` / `create_response()`，接收至 `response.done`，保存 WAV 后播放。

```bash
python -m examples.01_atomic_capabilities.audio.realtime_qwen_audio \
  --wav tests/data/asr_example.wav --output "$OUT/qwen-audio-reply.wav"
```

## realtime_translation.py — 实时语音翻译

接口：`client.audio.realtime.connect()`。固定模型：`qwen3.8-livetranslate-flash-realtime`；`--language` 指定目标语种，默认 `en`（英语）。

边输入边翻译，打印原文与译文，保存翻译音频后播放。使用服务端默认说话人检测，不调用 `commit()` / `create_response()`；输入结束调用 `end_session()`，继续接收至 `session.finished`，不在第一条 `response.done` 提前退出。

```bash
python -m examples.01_atomic_capabilities.audio.realtime_translation \
  --wav tests/data/asr_example.wav --language en --output "$OUT/translation.wav"
```

以上三个示例均可将 `--wav` 替换为 `--microphone --seconds 5`，加 `--no-play` 只保存。输入为 0.1～20 秒、16 kHz、16-bit PCM、单声道 WAV，输出为 24 kHz WAV；麦克风输入不会另存录音。各文件固定对应模型，不再使用原 `realtime.py` 的 `--model` 切换；新拆分示例尚未做云端验证。

## connection_reuse.py — 同一连接运行五个任务

接口：`client.ws_connection()`，通过 `connection=` 传给 `client.audio.asr.recognize()` 或 `client.audio.tts.connect()`。

ASR 使用 `qwen-audio-3.1-asr-flash-streaming`，TTS 使用 `qwen-audio-3.1-tts-flash` / `longanhuan_v3.1`。每次运行创建一条 WebSocket，顺序完成五个独立任务，输出逐轮文件与 `summary.json`；不录音、不播放，失败后不自动重连。

```bash
python -m examples.01_atomic_capabilities.audio.connection_reuse asr tests/data/asr_example.wav \
  --output-dir "$OUT/reuse-asr"
python -m examples.01_atomic_capabilities.audio.connection_reuse tts \
  --text "你好，这是连接复用。" --output-dir "$OUT/reuse-tts"
```

## tingwu.py — 听悟任务

接口：`client.tingwu`。听悟离线任务与实时会话示例。

## _common.py — 本地音频工具

提供 WAV 读写、麦克风采集、系统播放和参数校验，供上述示例使用；不调用云端接口，无独立命令行入口。

## 生产化模式

以上示例保持最小可运行；生产化请参考 `02_advanced_engineering/local_mock_debugging.py`（离线 mock 调试）、`02_advanced_engineering/circuit_breaker_retry.py`（退避与熔断容灾）、`04_real_world_scenarios/realtime_audio_meeting_harness.py`（实时音频会议 harness）。
