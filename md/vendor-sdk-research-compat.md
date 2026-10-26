# 国内/第三方大模型厂商 SDK 调研：接入百炼时 dashscope SDK/CLI 的兼容考量

> 调研对象：MiniMax、DeepSeek、Baseten、Kimi（Moonshot）
> 调研时间：2026-08-24
> 问题：这些厂家的模型接入百炼（DashScope 平台）后，dashscope SDK/CLI 需要兼容什么，还是不需要？

---

## 1. 四家 API/SDK 形态速览

| 厂商 | 协议形态 | 认证 | 自有 SDK | 特有扩展 |
| --- | --- | --- | --- | --- |
| **DeepSeek** | OpenAI/Anthropic 双兼容（官方原话 "API format compatible with OpenAI/Anthropic"），base_url `https://api.deepseek.com` + `/chat/completions` | Bearer token | 无自有 SDK，直接用 OpenAI SDK 改 base_url | `thinking: {type: enabled}` 思考模式、`reasoning_effort` 推理深度 |
| **Kimi（Moonshot）** | 完全 OpenAI 兼容（chat completions、models、token 估算、余额查询、files），改 base_url 即可 | Bearer token | 官方推荐用 openai 包 + 自定义 base_url，有最低版本要求 | `thinking` 参数走 `extra_body`；partial 标记挂在消息上而非请求级 |
| **MiniMax** | 文本模型支持 **Anthropic SDK 和 OpenAI SDK 双接入**；非文本能力（视频、语音合成/声音克隆、图片、音乐、文档处理）走自有 REST API | API Key（按量）或 Subscription Key（预付）双形态 | 无统一自有 SDK，文本靠第三方 SDK + MCP | 语音合成走 socket 流式；异步媒体任务返回错误码 |
| **Baseten**（推理平台，非国内厂商） | OpenAI 兼容端点 + beta 的 Anthropic Messages 协议 | Bearer token（需覆盖 SDK 默认 key 传法） | 无自有，用 OpenAI/Anthropic 官方 SDK + model slug | 标准 HTTP 错误码 + JSON 错误体 |

**共同事实：四家在"对话/推理"这条主线上全部收敛到 OpenAI 兼容协议**（MiniMax/Baseten 额外提供 Anthropic 协议）。没有一家要求接入方实现自己的私有 chat 协议。

---

## 2. 百炼侧现状

百炼托管第三方模型（DeepSeek、Kimi 等已在列）提供两条通路：

1. **OpenAI 兼容模式**：`base_compatible_api_url`（`/compatible-mode/v1`），第三方模型的主推通路；
2. **DashScope native 协议**：部分第三方模型同时以 native 形态暴露（走 Generation 类接口）。

即：**厂商协议的差异在百炼网关侧已被翻译成上述两种形态之一**，dashscope SDK 面对的从来不是厂商原始端点，而是百炼网关。

---

## 3. 结论：需要做专门兼容吗？——基本不需要，但有 6 个具体落点

**总判断**：不需要为任何一家厂商做专属 adapter/资源类。厂商协议已行业收敛（OpenAI 为主、Anthropic 为辅），2.0 方案 §3 的 `protocol=openai|anthropic|native` 三适配器就是覆盖这四家（及未来所有厂商）的通用机制。真正要做的是把"厂商特有语义"在通用协议里表达出来：

| # | 落点 | 说明 | 对应 2.0 方案 |
| --- | --- | --- | --- |
| 1 | **推理/思考参数的透传与一等化** | DeepSeek `thinking`/`reasoning_effort`、Kimi `thinking`（走 extra_body）在百炼兼容模式下同样存在。SDK 需要：`extra_body`/未知参数透传逃生舱（默认）；高频参数如 `enable_thinking`/`reasoning_effort` 升级为显式签名参数，思考内容在响应模型中有字段（`reasoning_content`），流式中单独事件化 | §2.2 类型化 + 逃逸舱 |
| 2 | **模型目录可发现性** | 第三方模型在百炼的 model id 与厂商原始 id 不同（带平台前缀/版本）。`client.models.list()` 需支持按 provider/来源过滤，文档给出"厂商名 → 百炼 model id"对照表；CLI `dashscope models list --provider deepseek` | §2.2 models 资源 |
| 3 | **错误语义映射** | 厂商错误码（DeepSeek/Kimi 的结构化 error 对象）经百炼网关后会变成 DashScope 字符串 code。异常层需保证兼容模式下也能拿到可读错误；`ErrorCode` 枚举覆盖第三方模型常见失败（内容审核、配额、模型不存在） | §2.2 错误模型 |
| 4 | **多 API Key 形态**（仅 MiniMax 相关） | MiniMax 有 API Key / Subscription Key 双形态；百炼侧统一为一个 key，**无需兼容**。但若用户用 `protocol=openai` 直连厂商端点（SDK 支持任意兼容端点），key 形态对 SDK 透明（都是 Bearer），也无需处理 | 无需改动 |
| 5 | **非 chat 厂商能力不做兼容** | MiniMax 的视频生成/声音克隆/音乐、Baseten 的自定义模型部署，是厂商私有能力。百炼若要提供，会以 dashscope native 异步任务形态落地（走现有 images/videos/audio 资源树 + jobs 模式），**不是协议兼容问题** | §2.2 长任务统一模式 |
| 6 | **CLI 文档与示例** | `dashscope chat create -m deepseek-...` / `-m kimi-...` 直接可用是卖点，但需示例展示厂商特有参数（`--extra-body '{"thinking":{"type":"enabled"}}'`）和 compatible-mode 与 native 的选择指引 | §4 CLI |

## 4. 与 2.0 方案的校验

- §3 协议层设计（openai/anthropic/native 三 adapter + `strict` 参数翻译策略）**足以覆盖这四家**，无需新增"vendor adapter"维度；
- 唯一需要在 §3 补充的设计细节：**"未知参数透传"必须是默认行为**（strict=False 静默透传而非丢弃）——厂商新参数（如 reasoning 类）上线快于 SDK 发版，透传是兼容性保险；
- acli/expert 侧：providers 已按协议接入，第三方模型在 expert 里天然可用，无额外工作。

## 5. 一句话总结

**不需要为 MiniMax/DeepSeek/Baseten/Kimi 做专门兼容**——它们的对话协议已全部收敛到 OpenAI/Anthropic 兼容，百炼网关又做了一层收敛。dashscope 2.0 要做的是三件通用事：协议适配层（已规划）、厂商特有参数的透传 + 高频参数一等化（reasoning/thinking）、第三方模型在 models 目录中的可发现性。

---

## 参考

- MiniMax API Overview：https://platform.minimax.io/docs/api-reference/api-overview
- DeepSeek API Docs：https://api-docs.deepseek.com/
- Baseten Inference APIs：https://docs.baseten.co/inference/model-apis/overview
- Kimi API Overview：https://platform.kimi.ai/docs/api/overview
