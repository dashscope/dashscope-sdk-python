# 调研：OpenAI/Athropic SDK/CLI

> 来源：钉钉文档（alidocs YMyQA2dXW7gYo6MzcZR0MPdZWzlwrZgb）

---

## 参考

### OpenAI

- OpenAI SDK：https://developers.openai.com/api/docs/libraries
- OpenAI CLI：https://developers.openai.com/api/docs/libraries/openai-cli
- OpenAI python Git：https://github.com/openai/openai-python
- OpenAI Agents python：https://github.com/openai/openai-agents-python
  - https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md
- OpenAI Skills：https://github.com/openai/

### Anthropic

- Anthropic SDK：https://github.com/anthropics/anthropic-sdk-python
- Docs：https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python
- CLI：https://platform.claude.com/docs/en/cli-sdks-libraries/cli/using
- Anthropic Claude API：https://platform.claude.com/docs/en/api/overview
- Anthropic Claude-code：https://github.com/anthropics/claude-code
- Docs：https://code.claude.com/docs/zh-CN/overview
- Agent SDK：https://code.claude.com/docs/zh-CN/agent-sdk/overview
- CLI：https://code.claude.com/docs/zh-CN/cli-reference
- Cookbooks：https://github.com/anthropics/claude-cookbooks
- Anthropic Skills：https://github.com/anthropics/skills/tree/main
- Anthropic Prompts：https://github.com/anthropics/prompt-eng-interactive-tutorial

### Stainless

- https://www.stainless.com/docs/

### oapi-codegen

- https://github.com/oapi-codegen/oapi-codegen

---

## 现状

三者深度对比：

- https://alidocs.dingtalk.com/i/nodes/YQBnd5ExVEjea40qCgXO1470JyeZqMmz
- https://alidocs.dingtalk.com/i/nodes/N7dx2rn0JbxOaqnACZDa3wRpWMGjLRb3

结论：

- https://alidocs.dingtalk.com/i/nodes/m9bN7RYPWdyrPBREcKpOnzwgVZd1wyK0
- dashscope vs openai
  - https://alidocs.dingtalk.com/i/nodes/NZQYprEoWoxKPoqwCB7dobYDV1waOeDk
- dashscope vs athropic
  - https://alidocs.dingtalk.com/i/nodes/NZQYprEoWoxKPoqwCB7dobYDV1waOeDk

---

## 三个对比分析

### 三者对比总结（OpenAI / Anthropic / DashScope）

| 维度 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 上手体验 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| API 简洁度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 流式输出 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Tool Use | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 错误处理 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 多模态支持 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 模型多样性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 工程化程度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**一句话总结**：

- **OpenAI**：上手最快，API 最简洁（output_text 一行直达）
- **Anthropic**：工程化程度最高（异常处理、自动重试、Tool 装饰器、流式助手）
- **DashScope**：模型多样性最强（文本/图像/视频/语音全覆盖），但需要在异常处理、Client 实例化、Tool Use 简化 三个方向补齐短板
