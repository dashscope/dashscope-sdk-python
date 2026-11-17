---
description: "dev mock 模式与本地自诊断 API 参考。"
---
# 开发工具（Devkit：Mock 与诊断）

`dashscope.devkit` 提供两个可选的开发辅助能力，关闭时几乎零开销，
生产成功路径上不会导入本包。

- **Dev mock** —— `static` / `replay` / `record` 三种模式走通
  2.0 HTTP 请求路径。`static` 与 `replay` 离线运行，不消耗云端
  token；`record` 会向服务发送真实请求，可能产生费用，并保存响应
  供后续回放。仅覆盖 HTTP，不覆盖
  WebSocket 传输。代码中用 `dashscope.enable_dev_mock(...)` 启用，
  或设置 `DASHSCOPE_DEV_MOCK=<mode>`。
- **自诊断** —— 本地双语规则表，把失败调用转化为结构化的修复建议，
  通过 `DashScopeError.diagnosis` 呈现。

根模块转发了 mock 开关：`dashscope.enable_dev_mock`、
`dashscope.disable_dev_mock`、`dashscope.dev_mock_active`。

!!! tip "共享录制文件"
    `record` 模式会把真实响应写盘；共享 fixture 前先用
    `set_redactor` 注册脱敏器。

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
