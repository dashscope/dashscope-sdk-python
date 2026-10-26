# 决策模型（Decision / System One）Python SDK

通过 `client.decision`，您可以针对一段业务状态（`state`），在一次前向推理中同时回答一组**类型化问题**：多选一（`choice`）、是非判断（`noul`）和有序评分（`score`）。每个答案都带有概率分布，`choice` / `score` 还附带置信度（`confidence`）。

决策模型（当前为 `decision-model-preview`）**不生成文本**，因此延迟和成本不随输出长度增长，适合替代"用大模型生成 JSON 再解析"的判断类环节。典型场景：工单路由、内容审核、Agent 路由、生成结果校验。

本文对应当前仓库的 SDK 实现；问题类型的语义、字段取值及服务端限制以决策模型 API 文档为准。

| 场景 | 同步客户端 `Client` | 异步客户端 `AsyncClient` |
| --- | --- | --- |
| 单次决策 | `client.decision.create()` | `await client.decision.create()` |

> 该接口为一次性请求-响应：不支持流式输出、函数调用（function calling）和联网搜索。

## 前期准备

1. 准备 API Key，并通过环境变量 `DASHSCOPE_API_KEY` 配置，避免将密钥写入代码。
2. 建议通过 `DASHSCOPE_WORKSPACE` 配置工作空间，并确保 API Key、工作空间和模型所在地域匹配。地域默认是 `cn-beijing`，可通过 `DASHSCOPE_REGION` 或 `Client(region=...)` 设置，目前支持 `cn-beijing`、`ap-southeast-1` 等地域。不配置工作空间且未指定端点时，当前 SDK 会回退到共享网关并发出警告。
3. 在本仓库根目录执行源码安装：

   ```bash
   python -m pip install -e .
   ```

客户端使用公开入口 `from dashscope import Client, AsyncClient`。也可以在创建客户端时显式传入 `api_key`、`workspace`、`region`；本文示例从环境配置读取凭据。

## 快速开始

### 同步调用：工单路由

以下示例对一张工单同时提出三个问题：应由哪个团队处理（多选一）、是否需要通知值班人员（是非）、严重程度（评分）。可保存为 `decision_example.py` 直接运行。

```python
from dashscope import Client
from dashscope.v2.resources.decision import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)

state = {
    "ticket_id": "T-1001",
    "content": "订单支付后超过 24 小时仍未到账，用户无法继续使用核心服务。",
}

questions = {
    "department": ChoiceQuestion(
        instructions="应该由哪个团队处理？",
        criteria={
            "billing": "支付、退款和账单问题",
            "technical": "产品故障和集成问题",
        },
    ),
    "escalate": NoulQuestion(instructions="是否需要立即通知值班人员？"),
    "severity": ScoreQuestion(
        instructions="这个问题有多严重？",
        criteria=[
            "轻微问题，不影响功能",
            "部分功能受影响，但存在替代方案",
            "核心功能不可用，没有替代方案",
            "造成严重业务或安全影响",
        ],
    ),
}

with Client() as client:
    resp = client.decision.create(
        model="decision-model-preview",
        state=state,
        questions=questions,
    )

print("request_id:", resp.request_id)

department = resp.answers["department"]
print("department:", department.choice)            # billing
print("confidence:", department.confidence)        # 0.88
print("probabilities:", department.probabilities)  # {"billing": 0.94, "technical": 0.06}

escalate = resp.answers["escalate"]
print("escalate P(yes):", escalate.noul)           # 0.99

severity = resp.answers["severity"]
print("severity:", severity.score)                 # 2.25（概率加权等级索引）
print("legend:", severity.legend["2"])             # 核心功能不可用，没有替代方案

print("input_tokens:", resp.usage.input_tokens)    # 125
print("latency_ms:", resp.latency_ms)              # 52.9
```

- `questions` 的键是您自定义的问题 ID，`answers` 按相同的键返回，一一对应。
- 一次调用可提多个问题，只产生一次模型前向推理，按输入 token 计费（`usage` 中只有 `input_tokens`）。
- `score` 答案是**概率加权等级索引**，可能落在两个等级之间（如上例 `2.25` 介于等级 2 和 3 之间）；`legend` 把等级索引映射回您给的等级描述，索引以字符串为键。

### 三种问题类型

| 类型 | 模型类 | `criteria` | 答案字段 | 说明 |
| --- | --- | --- | --- | --- |
| 多选一 | `ChoiceQuestion` | `Dict[str, str]`，必填，1–255 个选项 | `choice`（选中的选项名）、`confidence`、`probabilities` | `criteria` 映射"选项名 → 选项描述"；请给全选项集合，建议增加 `other` 兜底选项。 |
| 是非判断 | `NoulQuestion` | `Dict[str, str]`，可选 | `noul`（P(yes)，0=否、1=是）、`probabilities` | `criteria` 可描述两个结果，如 `{"true": "...", "false": "..."}`；判定阈值由应用自定（如 0.5）。 |
| 有序评分 | `ScoreQuestion` | `List[str]`，必填，2–255 级 | `score`（概率加权等级索引）、`confidence`、`probabilities`、`legend` | `criteria` 按从低到高描述各等级；3–7 个清晰可区分的等级效果最佳。 |

三个模型类都有可选字段 `instructions`（对该问题的补充说明）。未设置的可选字段不会出现在请求报文中（不会发送显式 `null`）。

### 原始字典形式（前向兼容）

`questions` 的值也可以直接传原始 wire 字典，SDK 原样透传（包括 SDK 模型尚未收录的未来字段）：

```python
resp = client.decision.create(
    model="decision-model-preview",
    state="这段评论是否在索要退款？",  # state 也可以是纯文本
    questions={
        "refund": {
            "type": "noul",
            "instructions": "是否在索要退款？",
            "future_field": 1,  # 未知字段透传，便于前向兼容
        },
    },
)
print(resp.answers["refund"].noul)
```

类型化模型（`ChoiceQuestion` 等）会在本地做 pydantic 校验，能在发送前发现错误；原始字典不做本地校验，适合使用 SDK 尚未支持的新字段。

### 异步调用

`AsyncClient` 提供相同的接口，需 `await`：

```python
import asyncio

from dashscope import AsyncClient


async def main():
    async with AsyncClient() as client:
        resp = await client.decision.create(
            model="decision-model-preview",
            state={"ticket_id": "T-1001", "content": "..."},
            questions={"q": {"type": "noul", "instructions": "是否需要人工介入？"}},
        )
        print(resp.answers["q"].noul)


asyncio.run(main())
```

## 请求参数

### `client.decision.create()`

所有参数均为关键字参数。

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `model` | `str` | 是 | 模型名称，当前为 `decision-model-preview`。 |
| `state` | `str \| dict \| list` | 是 | 待决策的业务状态：自由文本、一段对话或结构化对象；结构化对象会被序列化后交给模型。 |
| `questions` | `Dict[str, QuestionInput]` | 是 | 问题集合，键为问题 ID，值为 `ChoiceQuestion` / `NoulQuestion` / `ScoreQuestion` 实例或原始字典。不能为空，否则抛出 `ValueError`。 |
| `timeout` | `float` | 否 | 本次请求超时时间（秒），覆盖客户端级 `timeout`。 |
| `api_key` | `str` | 否 | 本次请求使用的 API Key，覆盖客户端级配置。 |
| `workspace` | `str` | 否 | 本次请求使用的工作空间，覆盖客户端级配置。 |

### 问题模型字段

`ChoiceQuestion`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `Literal["choice"]` | 固定值 | 默认 `"choice"`，无需设置。 |
| `instructions` | `str` | 否 | 问题说明。 |
| `criteria` | `Dict[str, str]` | 是 | 选项名 → 选项描述，1–255 个；为空或缺失会抛出 `ValidationError`。 |

`NoulQuestion`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `Literal["noul"]` | 固定值 | 默认 `"noul"`，无需设置。 |
| `instructions` | `str` | 否 | 问题说明。 |
| `criteria` | `Dict[str, str]` | 否 | 描述两个结果，如 `{"true": "...", "false": "..."}`；裸 `NoulQuestion()` 合法。 |

`ScoreQuestion`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `Literal["score"]` | 固定值 | 默认 `"score"`，无需设置。 |
| `instructions` | `str` | 否 | 问题说明。 |
| `criteria` | `List[str]` | 是 | 从低到高的等级描述，2–255 级；少于 2 级会抛出 `ValidationError`。 |

三个模型类均允许额外字段（`extra="allow"`），额外字段随报文透传。

## 返回字段

### `DecisionResponse`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `answers` | `Dict[str, DecisionAnswer]` | 各问题的答案，键与请求的 `questions` 键一致。 |
| `model` | `str` | 实际应答的模型名。 |
| `usage` | `Usage` 或 `None` | 用量对象；决策模型只产生 `input_tokens`（如 `125`），不生成输出 token。 |
| `latency_ms` | `float` 或 `None` | 服务端推理耗时（毫秒）。 |
| `request_id` | `str` | 请求 ID，排查问题时请提供。 |
| `headers` | `Dict[str, str]` | 响应头的小写字典副本，如 `resp.headers["x-request-id"]`。 |
| `status_code` / `code` / `message` / `is_error` | - | 错误信封字段，正常响应时 `is_error` 为 `False`；见"异常处理"。 |

### `DecisionAnswer`

按 `type` 填充对应字段，未涉及的字段为 `None`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | `str` | 答案类型：`"choice"` / `"noul"` / `"score"`。 |
| `choice` | `str` | 多选一答案：选中的选项名。 |
| `noul` | `float` | 是非答案：P(yes)，0 表示否、1 表示是。 |
| `score` | `float` | 评分答案：概率加权等级索引，可能落在两个等级之间（如 `1.43`）。 |
| `probabilities` | `Dict[str, float]` | 每个选项 / 等级的概率，总和为 1。 |
| `confidence` | `float` | 置信度；仅 `choice` / `score` 返回答案时携带。 |
| `legend` | `Dict[str, str]` | 仅 `score`：等级索引（字符串键）→ 等级描述。 |

响应为平坦结构（无 `output` 信封），额外字段同样透传保留。

## 端点与路由

决策模型的服务端点为 `POST {网关}/compatible-mode/v1/systemone`。它挂在每个网关的 compatible-mode 路径下，**与客户端的 `protocol` 设置无关**——无论 chat 协议配置为 `native`、`openai` 还是 `anthropic`，决策请求都发往同一路径：

- 默认（配置工作空间）：`https://{workspace}.{region}.maas.aliyuncs.com/compatible-mode/v1/systemone`。
- 显式 `base_url`：以您配置的 `base_url` 为服务根，请求发往 `{base_url}/systemone`（与 chat 的 base_url 规则一致）。
- 未配置工作空间且未指定端点：回退到共享网关（如 `https://dashscope.aliyuncs.com/compatible-mode/v1/systemone`）并发出 `UserWarning`；生产环境请配置工作空间。

## 异常处理

默认 `raise_on_error=True` 时，以下失败都会抛出异常：

| 情况 | 行为与处理建议 |
| --- | --- |
| HTTP 错误（4xx/5xx） | 按服务端错误码映射为 `DashScopeError` 子类抛出，如参数错误抛 `BadRequestError`、限流抛 `RateLimitError`、鉴权失败抛 `AuthenticationError`；检查异常的 `code`、`message`、`status_code`、`request_id`。 |
| 业务错误层（HTTP 200 + 错误体） | 服务端以 200 返回 `{"code", "message"}` 时同样按错误码抛出（如 `BadRequestError`），此时 `status_code` 为 200；不要仅凭 HTTP 状态判断成功。 |
| 本地参数错误 | `questions` 为空字典抛出 `ValueError`；问题模型字段不满足约束（如 `ChoiceQuestion` 无选项、`ScoreQuestion` 少于 2 级）抛出 pydantic `ValidationError`，请求不会发出。 |

`Client(raise_on_error=False)` 时不抛异常，改为填充返回对象的错误信封：

```python
client = Client(raise_on_error=False)
resp = client.decision.create(
    model="decision-model-preview",
    state="...",
    questions={"q": {"type": "noul"}},
)
if resp.is_error:
    print(resp.code, resp.message, resp.request_id)
    # answers 为空字典，不要按正常答案读取
```

- 网络层失败（连接失败、超时）抛出 `APIConnectionError` / `APITimeoutError`。
- 排查时请记录异常类型与 `request_id`；不要记录 API Key 或未经脱敏的业务状态内容。

## 常见问题

**为什么 `score` 答案是小数？**

`score` 是各等级概率加权后的索引，不是离散等级。例如 `2.25` 表示答案主要分布在等级 2、部分落在等级 3（可结合 `probabilities` 查看分布）。需要离散等级时可四舍五入或按阈值判定。

**`noul` 返回 0.99 可以直接当作"是"吗？**

`noul` 是 P(yes) 概率，不是布尔值。判定阈值由应用决定（常用 0.5）；对关键操作可结合 `probabilities` 或设置更高阈值。

**为什么有的答案没有 `confidence`？**

`confidence` 仅随 `choice` / `score` 答案返回；`noul` 答案可通过 `probabilities` 自行评估确定程度。

**配置了 `protocol="anthropic"`（或 `openai`），决策请求的端点会跟着变吗？**

不会。`protocol` 只影响 chat 族接口的路由；决策模型固定走 compatible-mode 路径，三种协议配置下URL 完全一致。

**可以一次传多少个问题？**

`questions` 至少一个（空字典会抛出 `ValueError`），多个问题共享一次前向推理。问题数量、单字段长度等上限以服务端限制为准；建议问题 ID 使用简短、可读、稳定的标识，答案按键原样返回。

**支持流式输出或 function calling 吗？**

不支持。决策模型不生成文本，一次性返回全部答案；需要生成能力请使用 chat 族接口，再把生成结果交给 `decision.create()` 做校验是常见的组合用法。

## 相关文档

- [DashScope 2.0 设计](dashscope-2.0-design.md)
- [DashScope 2.0 迁移指南](dashscope-2.0-migration-guide.md)
- [语音识别 Python SDK](dashscope-2.0-asr-sdk.md)
- [语音合成 Python SDK](dashscope-2.0-tts-sdk.md)
