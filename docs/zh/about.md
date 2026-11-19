---
description: "关于 DashScope Python SDK：项目链接、Apache-2.0 许可证、兼容性与版本策略。"
---
# 关于

## 项目

DashScope Python SDK（PyPI 包名 `dashscope`）是
[阿里云百炼（Model Studio）](https://help.aliyun.com/zh/model-studio/)
的官方 Python 客户端，覆盖文本生成、多模态理解、向量嵌入、重排序、
图像/视频生成、语音合成与识别、微调、智能体强化学习、托管智能体与应用调用。

- 源码：[github.com/dashscope/dashscope-sdk-python](https://github.com/dashscope/dashscope-sdk-python)
- 包：[pypi.org/project/dashscope](https://pypi.org/project/dashscope/)
- 服务文档：[百炼帮助中心](https://help.aliyun.com/zh/model-studio/)

## 许可证

SDK 采用
[Apache License 2.0](https://github.com/dashscope/dashscope-sdk-python/blob/main/LICENSE)
授权。第三方声明（如内置的 certifi CA 证书包）见
[NOTICE](https://github.com/dashscope/dashscope-sdk-python/blob/main/NOTICE)。

## 兼容性与版本策略

- 2.x 是积极开发的主线：资源式 `Client` / `AsyncClient`、
  类型化响应与异常。
- 完整 1.x 公共接口仍可从历史路径导入，由 `dashscope.legacy_1x` 下
  的冻结门面实现；不再新增能力。
- AgentStudio 与 Agentic RL 是应用子 SDK，拥有独立的客户端与返回类型
  （AgentStudio 还有专用的资源与分页接口）；错误则统一构建在
  `dashscope.errors` 体系上：AgentStudio 直接抛出主 SDK 异常，
  `AgentStudioError` 等历史名称只是别名；Agentic RL 保留专用的
  `AgenticRLError` 子类，但同属该体系。
- 需要 Python 3.9+；可选交互式助手（`dashscope[expert]`）需要 Python 3.10+。
- 当前 SDK 版本：**{{ sdk_version }}** —— 见[发行说明](release-notes.md)。

## 获取帮助

- 在 [GitHub Issues](https://github.com/dashscope/dashscope-sdk-python/issues)
  反馈缺陷与需求。
- 浏览 [API 参考](reference/client.md)或[快速开始](quickstart.md)。
- 失败的调用可以自我解释：启用 devkit 诊断后，
  `DashScopeError.diagnosis` 会给出结构化修复建议
  （见[开发工具](reference/devkit.md)）。