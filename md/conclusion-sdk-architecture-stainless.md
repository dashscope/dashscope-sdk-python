# 结论：SDK 架构（Stainless 孪生架构实证）

> 来源：钉钉文档（alidocs YMyQA2dXW7gYo6MzcZR0MPdZWzlwrZgb）

---

## 实证：两个仓库的目录几乎完全同构

`comm -3` 对比两个 src/ 目录，只有 14 项不同，其余 17 项（_base_client.py、_client.py、_exceptions.py、_streaming.py、_models.py、_response.py、_resource.py、_qs.py、pagination.py、resources/、types/、lib/……）文件名逐一对应。且文件头直接写着：

```
# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.
```

连 _streaming.py 里的注释都一样（"initially copied from httpx-sse"）——因为是同一份模板复制出去的。

## Stainless 是什么

Stainless（stainlessapi.com）是一家专门做"SDK 生成器"的公司。工作方式：

```
厂商的 OpenAPI spec（机器可读的 API 定义）
        │
        ▼
Stainless 生成器（模板 + 各语言运行时）
        │
        ▼
完整的类型安全 SDK：Python / TypeScript / Go / Java / Kotlin / Ruby
```

客户包括 OpenAI、Anthropic、Cloudflare、Cohere、Lithic 等。厂商不手写 SDK，而是维护 OpenAPI spec，Stainless 的 CI 在 spec 变更后自动重新生成 SDK 并发 PR。

## "孪生架构"具体指什么

同一个模板生成，所以两个 SDK 的"骨架"完全一样：

| 层 | OpenAI | Anthropic | 来源 |
| --- | --- | --- | --- |
| HTTP 基座 | SyncAPIClient/AsyncAPIClient（httpx） | 同名同实现 | 模板 |
| 重试逻辑 | _should_retry + Retry-After + 指数退避 | 逐行同构 | 模板 |
| 异常层次 | APIStatusError → 400/401/403/404/409/422/429/5xx | 同一棵树 | 模板 |
| 流式机制 | Stream[T] / AsyncStream[T] SSE 迭代器 | 同名同构 | 模板 |
| 未传参哨兵 | NOT_GIVEN | NOT_GIVEN | 模板 |
| 响应模型 | Pydantic + model_construct | 同 | 模板 |

差异只在两处：

1. **API 表面**——resources/ 和 types/ 目录由各自的 OpenAPI spec 生成，内容完全不同（chat.completions vs messages）；
2. **手写扩展**——就是 comm 输出里那 14 个不同项：OpenAI 的 providers/（多云）、_provider.py、realtime/websocket 相关；Anthropic 的 lib/credentials/（OAuth 凭证链）、_middleware.py、AnthropicAWS/AnthropicFoundry 子类。

## 这对我们的对比意味着什么

三点推论：

1. 对比 OpenAI SDK vs Anthropic SDK 时，"框架能力"那部分结论对两者同时成立。重试、异常驱动、类型系统、四象限覆盖（同步/异步 × 流式/非流式）这些优点不是某一家的设计选择，而是 Stainless 模板的标配。真正值得单独评的是两家的手写扩展（Anthropic 的 text_stream、OpenAI 的 parse() 等）。
2. DashScope 的差距要重新定性：不是"DashScope 工程师写得不如人"，而是**手写实现 vs 工业化流水线的差距**。Stainless 模式下，API 加了个新参数 → spec 更新 → SDK 自动生成带完整类型和测试的版本，天级送达；手写模式下要人肉改签名、补文档、补测试，还容易漂移（DashScope Generation.call 里那 30 个 Any 参数和手工 `if x is not None` 搬运就是这种漂移的产物）。
3. 这也解释了之前建议里 P2 那条"代码生成"：如果 DashScope 要追平，方向不是继续手写补丁，而是把百炼 API 沉淀成 OpenAPI spec，接入生成器（Stainless 是商业服务，也可用开源方案如 oapi-codegen/自研模板），让 SDK 从 spec 自动长出类型和资源类。
