---
description: "dev mock 模式与本地自诊断 API 参考。"
---
# 开发工具（Devkit：Mock 与诊断）

`dashscope.devkit` 提供两个可选的开发辅助能力，关闭时几乎零开销，
生产成功路径上不会导入本包。

- **Dev mock** —— `static` / `replay` / `record` 三种模式在传输层
  拦截 HTTP 请求，仅覆盖 HTTP，不覆盖 WebSocket 传输。
  代码中用 `dashscope.enable_dev_mock(...)` 启用，
  或设置 `DASHSCOPE_DEV_MOCK=<mode>`。各模式能力范围见下表。
- **自诊断** —— 本地双语规则表，把失败调用转化为结构化的修复建议，
  通过 `DashScopeError.diagnosis` 呈现。

| 模式 | 网络 | 能力范围 |
| --- | --- | --- |
| `static` | 离线 | 仅 chat / multimodal 的解析冒烟：本地合成响应，走通协议适配、SSE 解析与类型化模型。不模拟工具调用、结构化输出或任务状态机；其他资源（向量、rerank 等）会抛出 `DevMockMiss` |
| `replay` | 离线 | 回放录制的真实响应；未命中抛出 `DevMockMiss`，不会回落到网络 |
| `record` | 真实请求 | 透传真实服务并录制响应，可能产生费用，供后续离线回放 |
| WebSocket | — | 三种模式均不覆盖 |

根模块转发了 mock 开关：`dashscope.enable_dev_mock`、
`dashscope.disable_dev_mock`、`dashscope.dev_mock_active`。

!!! warning "共享录制文件"
    `record` 模式会把完整真实响应写盘，可能包含生成内容、
    工具调用参数或账号数据。共享 fixture 前先用 `set_redactor`
    注册脱敏器，但注意：脱敏器只处理能解析为单个 JSON 文档的
    响应体，**SSE 流与二进制响应原样落盘、不会被脱敏**——
    注册脱敏器不等于所有 fixture 都已脱敏。

## Dev mock 开关

::: dashscope.devkit.enable_dev_mock

::: dashscope.devkit.disable_dev_mock

::: dashscope.devkit.is_active

::: dashscope.devkit.mock.DevMockMiss

::: dashscope.devkit.mock.set_redactor

## 自诊断

::: dashscope.devkit.diagnosis.Diagnosis

::: dashscope.devkit.diagnosis.diagnose

::: dashscope.devkit.diagnosis.render

::: dashscope.devkit.diagnosis.install_excepthook

::: dashscope.devkit.diagnosis.Matcher

::: dashscope.devkit.diagnosis.Rule
