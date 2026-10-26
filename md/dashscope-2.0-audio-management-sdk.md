# DashScope 2.0 Python SDK：语音管理面（词表 / 热词 / 声音复刻 / 声音设计）

本文覆盖 `client.audio` 的四个管理面子资源，对应 1.x 的三个管理类与官方声音设计 REST API：

| v2 入口 | 1.x / 官方对应 | 说明 |
| --- | --- | --- |
| `client.audio.vocabularies` | `VocabularyService`（`audio/asr/vocabulary.py`） | ASR 定制词表（热词表）CRUD |
| `client.audio.phrases` | `AsrPhraseManager`（`audio/asr/asr_phrase_manager.py`） | paraformer 热词编译 CRUD |
| `client.audio.voices` | `VoiceEnrollmentService`（`audio/tts_v2/enrollment.py`） | 声音复刻（克隆音色）CRUD |
| `client.audio.voice_design` | 官方[声音设计 API](https://help.aliyun.com/zh/model-studio/voice-design-api-references)（1.x 无 SDK 类） | qwen 声音设计 CRUD |

词表 / 声音复刻 / 声音设计三者共用同一信封：`POST /services/audio/{asr|tts}/customization`，请求体 `{model, input: {action, ...}, parameters?}`，仅 action 词与 ID 字段随族不同；v2 把 action 收敛为统一的 `create/list/get/update/delete` 方法名（1.x 的 `query_*` 统一为 `get`）。热词（phrases）实为一条编译任务：`create/update` POST `/fine-tunes`（`mode=compile_asr_phrase`、空文件集），`get/list/delete` 走 `/fine-tunes/outputs`。

所有方法共享关键字参数 `timeout` / `headers` / `api_key` / `workspace`；`model=` 可覆盖各资源默认服务模型（下表）；模型专属的 `input` 字段经 `extra_input=` 透传（具名参数同名优先），顶层 `parameters=` 原样放入请求体。错误按 SDK 错误目录映射（如 `InvalidParameter` → `BadRequestError`）；`raise_on_error=False` 时错误体现在返回对象的 `code` / `message` / `is_error` 上。返回值均携带 `request_id`；`update` / `delete` 返回 `JobStatus`。

| 资源 | 默认 `model` | 端点 |
| --- | --- | --- |
| `vocabularies` | `speech-biasing` | `/services/audio/asr/customization` |
| `voices` | `voice-enrollment` | `/services/audio/tts/customization` |
| `voice_design` | `qwen-voice-design` | `/services/audio/tts/customization` |
| `phrases` | 调用方 ASR 模型 | `/fine-tunes`、`/fine-tunes/outputs` |

> 1.x 这两个服务内部对超时/断连做 3 次重试；v2 交由 transport 的重试循环统一处理（默认 2 次），无需在资源层重复。1.x `list_phrases(page=...)` 的 `page` 实参在 `**kwargs` 中被丢弃、从未到达线缆；v2 发送文档化的 `page_no`，不复刻该行为。

## 定制词表（vocabularies）

对应官方[定制热词](https://help.aliyun.com/zh/model-studio/custom-hot-words/)示例：为 paraformer 系列识别模型创建词表，返回 `vocabulary_id`。

```python
from dashscope.v2 import Client

client = Client()
created = client.audio.vocabularies.create(
    target_model="paraformer-realtime-v2",
    prefix="med",                      # 仅小写字母/数字，10 字符以内
    vocabulary=[
        {"text": "心肌梗死", "weight": 4},
        {"text": "脑梗死", "weight": 4},
    ],
)
print(created.vocabulary_id)

# 列表 / 查询内容 / 整体替换 / 删除
tables = client.audio.vocabularies.list(prefix="med", page_index=0, page_size=10)
for item in tables.vocabularies:
    print(item["vocabulary_id"], item.get("prefix"))

table = client.audio.vocabularies.get(created.vocabulary_id)
print(table.vocabulary)                # 词表内容

client.audio.vocabularies.update(
    created.vocabulary_id,
    vocabulary=[{"text": "冠心病", "weight": 4}],
)
client.audio.vocabularies.delete(created.vocabulary_id)
```

## 热词（phrases，paraformer）

对应官方 [paraformer 热词](https://help.aliyun.com/zh/model-studio/paraformer-asr-phrase-manager)示例：`create` 返回编译任务，`phrase_id`（即任务 `job_id`）编译完成后即可传给识别：

```python
from dashscope.v2 import Client

client = Client()
phrase = client.audio.phrases.create(
    model="paraformer-v2",
    phrases={"下一首": 90, "上一首": 90},   # 词 -> 权重
)
print(phrase.phrase_id, phrase.status)

# 编译完成后在实时识别中引用（run-task 顶层 resources，见 ASR 指南）
# client.audio.asr.recognize(..., phrase_id=phrase.phrase_id)

client.audio.phrases.update(
    phrase.phrase_id,
    model="paraformer-v2",
    phrases={"暂停": 90},
)
fetched = client.audio.phrases.get(phrase.phrase_id)
page = client.audio.phrases.list(page_no=1, page_size=10)
client.audio.phrases.delete(phrase.phrase_id)
```

`create` / `update` 的 `training_type` 默认 `"compile_asr_phrase"`；空 `phrases` 与空 `phrase_id` 在本地直接 `ValueError`。

## 声音复刻（voices）

对应官方[声音复刻](https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide)示例：以一段音频 URL 克隆音色，返回 `voice_id`，随后即可作为 TTS 的 `voice` 使用。

```python
from dashscope.v2 import Client

client = Client()
voice = client.audio.voices.create(
    target_model="cosyvoice-v3-plus",
    prefix="demo",
    url="https://your-audio-url/sample.wav",
    language_hints=["zh"],
)
print(voice.voice_id)

# audio = client.audio.tts.synthesize("你好", model="cosyvoice-v3-plus", voice=voice.voice_id)

voices = client.audio.voices.list(prefix="demo")
detail = client.audio.voices.get(voice.voice_id)   # status: OK / DEPLOYING / UNDEPLOYED
client.audio.voices.update(voice.voice_id, url="https://your-audio-url/new-sample.wav")
client.audio.voices.delete(voice.voice_id)
```

## 声音设计（voice_design）

对应官方[声音设计](https://help.aliyun.com/zh/model-studio/voice-design-user-guide)示例：用文字描述生成音色，ID 字段为 `voice`，并随附 base64 预览音频。

```python
import base64
from dashscope.v2 import Client

client = Client()
designed = client.audio.voice_design.create(
    target_model="qwen3-tts-vd-realtime-2026-01-15",
    voice_prompt="沉稳的中年男性，音色低沉浑厚",
    preview_text="各位听众朋友，大家好",
    preferred_name="announcer",
    language="zh",
    parameters={"sample_rate": 24000, "response_format": "wav"},
)
print(designed.voice)
preview = designed.preview_audio          # {data(base64), sample_rate, response_format}
with open("preview.wav", "wb") as f:
    f.write(base64.b64decode(preview["data"]))

voices = client.audio.voice_design.list(page_index=0, page_size=10)
detail = client.audio.voice_design.get("announcer")
client.audio.voice_design.delete("announcer")
```

## 返回类型速查

| 类型 | 关键字段 | 说明 |
| --- | --- | --- |
| `Vocabulary` | `vocabulary_id`、`vocabulary` | `create` 只带 ID；`get` 带词表内容；其余字段随 `output` 保留 |
| `VocabularyList` | `vocabularies` | 词表条目 dict 列表 |
| `Voice` | `voice_id`、`status`、`preview_audio` | 克隆音色 |
| `VoiceList` | `voices` | 音色条目 dict 列表 |
| `DesignedVoice` | `voice`、`preview_audio`、`language`、`target_model` | 设计音色 |
| `DesignedVoiceList` | `voices`（另含 `total_count`） | 设计音色条目 |
| `Phrase` | `phrase_id`、`status` | 编译任务即热词集 |
| `PhraseList` | `phrases`、`page_no`、`page_size`、`total_count` | 热词集分页 |
| `JobStatus` | `status` | `update` / `delete` 的应答 |

## 相关文档

- [DashScope 2.0 ASR SDK](dashscope-2.0-asr-sdk.md)（`phrase_id` 在识别中的使用）
- [DashScope 2.0 TTS SDK](dashscope-2.0-tts-sdk.md)（克隆/设计音色在合成中的使用）
- [定制热词](https://help.aliyun.com/zh/model-studio/custom-hot-words/) / [paraformer 热词](https://help.aliyun.com/zh/model-studio/paraformer-asr-phrase-manager)
- [声音复刻](https://help.aliyun.com/zh/model-studio/voice-cloning-user-guide) / [声音设计](https://help.aliyun.com/zh/model-studio/voice-design-user-guide)
