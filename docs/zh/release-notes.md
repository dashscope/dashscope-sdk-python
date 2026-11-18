---
description: "dashscope Python SDK 发行说明与变更日志指引，含 2.0 亮点。"
---
# 发行说明

逐版本的权威变更日志发布在
[GitHub Releases](https://github.com/dashscope/dashscope-sdk-python/releases)。
本页概述主要版本线并链接近期 tag。
当前 SDK 版本：**{{ sdk_version }}**。

## 近期发布

| Tag | 说明 |
| --- | --- |
| v2.1.0 | [请参阅发布页面](https://github.com/dashscope/dashscope-sdk-python/releases/tag/v2.1.0) |
| v2.0.4 | [请参阅发布页面](https://github.com/dashscope/dashscope-sdk-python/releases/tag/v2.0.4) |
| v2.0.3 | [请参阅发布页面](https://github.com/dashscope/dashscope-sdk-python/releases/tag/v2.0.3) |
| v2.0.2 | [请参阅发布页面](https://github.com/dashscope/dashscope-sdk-python/releases/tag/v2.0.2) |
| v2.0.1 | [请参阅发布页面](https://github.com/dashscope/dashscope-sdk-python/releases/tag/v2.0.1) |

## 2.0 亮点

2.0 引入了沿用 OpenAI SDK 使用体验的资源式客户端：

- **入口** —— `client.chat.completions.create(...)` 取代模块级静态类；
  一行式 `dashscope.chat("qwen-plus", "...")`。
- **响应** —— 类型化对象（`resp.output_text`、`resp.usage.total_tokens`），
  取代原始 dict 式结果。
- **错误** —— 默认抛出类型化 `DashScopeError` 异常。
- **流式** —— `ChatStream` 提供 `.text_stream`、`.reasoning_stream` 与
  `.get_final_message()`。
- **协议** —— 一个客户端通过 `protocol=` 使用 `native`、`openai` 或
  `anthropic`。
- **异步** —— `dashscope.AsyncClient()` 镜像同步接口。
- **连接** —— 每客户端一个 keep-alive 连接池，跨资源共享。
- **配置** —— 分层：参数 → `dashscope.settings` → 环境变量 →
  `~/.dashscope/config.toml`。

## 1.x 版本线

1.x API 为向后兼容保留在 `dashscope.legacy_1x` 下，仍可从 `dashscope`
根模块导入。该层已冻结：新能力只进入 2.x 接口。
见[旧版 1.x](reference/legacy.md)。