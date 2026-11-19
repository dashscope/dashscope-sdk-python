---
description: "dashscope 各能力的即抄即用示例：多模态、图像、视频、向量、重排序、语音、应用、文件与模型。"
---
# 能力速查示例

每个能力一段可直接运行的短示例。均假设已导出 `DASHSCOPE_API_KEY`
（见[快速开始](../quickstart.md)），并使用同步客户端。`AsyncClient`
以 `await` 镜像 HTTP 调用；下方基于 WebSocket 的语音合成与文件识别
示例仅支持同步客户端。响应字段均为类型化对象，
[API 参考](../reference/client.md)逐一说明。

## 多模态理解

```python
from dashscope import Client

with Client() as client:
    resp = client.multimodal.chat.create(
        model="qwen-vl-plus",
        messages=[{
            "role": "user",
            "content": [
                {"image": "https://example.com/dog_and_girl.jpeg"},
                {"text": "这张图里有什么？"},
            ],
        }],
    )
    print(resp.output_text)
```

## 图像生成

`generate()` 提交任务并等待完成；完成后的 job 携带结果 URL。

```python
job = client.images.generate(
    model="wan2.2-t2i-flash",
    prompt="一只漂浮在太空中的猫宇航员",
)
print(job.image_urls)
```

## 视频生成

```python
job = client.videos.generate(
    model="wan2.2-t2v-plus",
    prompt="一只漂浮在太空中的猫宇航员",
)
print(job.video_url)
```

## 文本向量

```python
resp = client.embeddings.create(
    model="text-embedding-v4",
    input=["hello world", "你好世界"],
)
print(len(resp.embeddings[0].embedding))
```

## 重排序

```python
resp = client.rerank.create(
    model="gte-rerank-v2",
    query="什么是 DashScope？",
    documents=["DashScope 是模型服务。", "今天天气不错。"],
    top_n=2,
)
print(resp.results)
```

## 语音合成

!!! note "仅支持同步的 WebSocket API"
    此示例需使用 `Client`。`AsyncClient.audio.tts` 不提供
    `synthesize()`，不能仅通过添加 `await` 改为异步调用。

`synthesize()` 返回拼接后的音频字节。

```python
audio = client.audio.tts.synthesize(
    "你好，DashScope",
    model="cosyvoice-v2",
    voice="longxiaochun",
)
with open("out.mp3", "wb") as f:
    f.write(audio)
```

## 文件识别

!!! note "仅支持同步的 WebSocket API"
    此示例需使用 `Client`。`AsyncClient.audio.asr` 不提供
    `recognize_file()`，不能仅通过添加 `await` 改为异步调用。

```python
for event in client.audio.asr.recognize_file(
    "audio.pcm",
    model="paraformer-realtime-v2",
):
    print(event)
```

实时会话见[语音](../reference/audio.md)与
[Qwen Live 指南](qwen-live-python-sdk.md)。

## 应用调用

```python
resp = client.apps.create(app_id="your-app-id", prompt="介绍一下你自己")
print(resp.text)
```

## 文件上传

```python
file = client.files.upload(file_path="train.jsonl", purpose="fine-tune")
print(file.id)
```

## 模型列表

```python
page = client.models.list()
print(len(page.items), "个可用模型")
```

完整资源接口（含批量任务与微调）见
[客户端资源](../reference/resources.md)。
