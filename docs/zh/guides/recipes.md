---
description: "dashscope 各能力的即抄即用示例：多模态、图像、视频、向量、重排序、决策、Token 计数、语音、听悟、应用、文件与模型。"
---
# 能力速查示例

每个能力一段可直接运行的短示例，均自包含客户端的创建与释放，
可单独复制运行。均假设已导出 `DASHSCOPE_API_KEY`
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
                {"image": "https://help-static-aliyun-doc.aliyuncs.com/"
                          "file-manage-files/zh-CN/20241022/emyrja/"
                          "dog_and_girl.jpeg"},
                {"text": "这张图里有什么？"},
            ],
        }],
    )
    print(resp.output_text)
```

## 图像生成

`generate()` 提交任务并等待完成；完成后的 job 携带结果 URL。

```python
from dashscope import Client

with Client() as client:
    job = client.images.generate(
        model="wan2.2-t2i-flash",
        prompt="一只漂浮在太空中的猫宇航员",
    )
    print(job.image_urls)
```

## 视频生成

```python
from dashscope import Client

with Client() as client:
    job = client.videos.generate(
        model="wan2.2-t2v-plus",
        prompt="一只漂浮在太空中的猫宇航员",
    )
    print(job.video_url)
```

## 文本向量

```python
from dashscope import Client

with Client() as client:
    resp = client.embeddings.create(
        model="text-embedding-v4",
        input=["hello world", "你好世界"],
    )
    print(len(resp.embeddings[0].embedding))
```

## 重排序

```python
from dashscope import Client

with Client() as client:
    resp = client.rerank.create(
        model="gte-rerank-v2",
        query="什么是 DashScope？",
        documents=["DashScope 是模型服务。", "今天天气不错。"],
        top_n=2,
    )
    print(resp.results)
```

## 决策模型

`client.decision` 调用 System One 决策模型：对业务 `state` 一次性回答
多选（choice）、是否（noul）、等级打分（score）问题，返回概率分布，
不生成文本。详见[客户端资源](../reference/resources.md)。

```python
from dashscope import Client
from dashscope.resources.decision import ChoiceQuestion, NoulQuestion

with Client() as client:
    resp = client.decision.create(
        model="decision-model-preview",
        state={"ticket_id": "T-1001", "content": "我的账单被重复扣款了"},
        questions={
            "department": ChoiceQuestion(
                instructions="这个工单应该由哪个团队处理？",
                criteria={
                    "billing": "账单与退款问题",
                    "technical": "产品故障与使用问题",
                },
            ),
            "escalate": NoulQuestion(
                instructions="是否需要升级给值班工程师？",
            ),
        },
    )
    print(resp.answers["department"].choice)
    print(resp.answers["escalate"].noul)  # P(是)，0=否 1=是
```

## Token 计数

也可传 `messages=[...]` 统计对话消息列表的 token。

```python
from dashscope import Client

with Client() as client:
    resp = client.tokenization.count(
        model="qwen3.7-plus",
        prompt="请用一句话介绍杭州",
    )
    print(resp.input_tokens)
```

## 语音合成

!!! note "仅支持同步的 WebSocket API"
    此示例需使用 `Client`。`AsyncClient.audio.tts` 不提供
    `synthesize()`，不能仅通过添加 `await` 改为异步调用。

`synthesize()` 返回拼接后的音频字节。

```python
from dashscope import Client

with Client() as client:
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
from dashscope import Client

with Client() as client:
    for event in client.audio.asr.recognize_file(
        "audio.pcm",
        model="paraformer-realtime-v2",
    ):
        print(event)
```

实时会话见[语音](../reference/audio.md)与
[Qwen Live 指南](qwen-live-python-sdk.md)。

## 听悟（离线分析）

`call()` 一次 POST 返回解析后的结果。下方 `fileUrl` 与 `appid` 为
占位符，需替换为你自己的听悟应用 ID 和可公网访问的音视频地址。

```python
from dashscope import Client

with Client() as client:
    result = client.tingwu.call(
        model="tingwu-automotive-service-inspection",
        input={
            "fileUrl": "https://your-bucket.oss-cn-beijing.aliyuncs.com/meeting.mp3",
            "appid": "your-tingwu-app-id",
        },
    )
    print(result.output)
```

## 应用调用

```python
from dashscope import Client

with Client() as client:
    resp = client.apps.create(app_id="your-app-id", prompt="介绍一下你自己")
    print(resp.text)
```

## 文件上传

```python
from dashscope import Client

with Client() as client:
    file = client.files.upload(file_path="train.jsonl", purpose="fine-tune")
    print(file.file_id)
```

## 模型列表

```python
from dashscope import Client

with Client() as client:
    page = client.models.list()
    print(len(page.items), "个可用模型")
```

其余能力的完整接口——批量任务、微调、模型部署、智能体强化学习、
多模态向量——见[客户端资源](../reference/resources.md)；实时语音见
[Qwen Live 指南](qwen-live-python-sdk.md)，声音与热词定制见
[语音](../reference/audio.md)，托管智能体见
[AgentStudio](../reference/agentstudio.md)。对话补全示例见
[快速开始](../quickstart.md)。