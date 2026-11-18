---
description: "DashScope SDK Expert：终端里的 AgenticX AI 助手 —— 安装、启动方式、一次性提问与首次运行的 Expert 配置。"
---
# SDK 专家（AgenticX）

SDK 内置了交互式 AI 助手 **DashScope SDK Expert**，由可选依赖
`dashscope-agenticx`（AgenticX）提供。它能直接在终端中解答 SDK/API
问题、生成可运行示例、展示 CLI 用法、诊断错误。

助手需要 **Python 3.10+**（SDK 本身支持 3.9+），默认不安装。

## 安装

```bash
pip install -U "dashscope[acli]"
```

在 `acli` 之上，可选 extra 为助手追加能力：

| Extra | 提供 |
| --- | --- |
| `acli` | 交互式 AI 助手（DashScope SDK Expert）。 |
| `acli-openai` | 让助手使用 OpenAI 兼容的模型服务方。 |
| `acli-anthropic` | 让助手使用 Anthropic 模型服务方。 |
| `acli-voice` | 助手的语音输入与语音播报。 |
| `acli-camera` | 助手的摄像头采集图片输入。 |
| `acli-all` | 以上全部（含 `acli`）。 |

## 启动

```bash
dashscope                     # 交互式助手（默认 TUI）
dashscope expert              # 同上，显式写法
dashscope expert --cli        # 交互式，用纯 REPL 代替 TUI
dashscope expert --tui        # 交互式，强制 TUI
dashscope expert "问题"        # 一次性提问后退出
dashscope expert --help       # 查看全部形式
```

不带参数且 stdin 为管道（非终端）时，输入会被当作一次性提问：

```bash
echo "如何流式输出对话结果？" | dashscope
```

无法识别的命令会报错，而不是被当作提问 —— 请用
`dashscope expert "问题"` 提问。在助手内输入 `/help` 查看可用命令。

助手使用与 SDK 相同的 API Key：导出 `DASHSCOPE_API_KEY` 环境变量，
或在命令行传入：

```bash
dashscope -k sk-xxx expert "如何上传文件？"
```

## 首次运行：Expert 配置

在没有 `./.agenticx/` 目录的环境中首次交互式运行时，助手会提示将
`dashscope-sdk-expert` 配置下载到 `./.agenticx/`：包含 SDK 问答专家
人设、技能模板与 SDK 知识索引。它是纯配置 —— 随时可编辑。

- 下载需要 `git` 并能访问 `github.com`。下载失败时助手仍会启动；
  之后可重试：

  ```bash
  agenticx expert download dashscope-sdk-expert
  ```

- 拒绝下载会在 `./.agenticx/` 留下标记，不再重复提示；之后用同一条
  命令即可安装该配置。

## 故障排查

- **运行 `dashscope` 只打印安装提示** —— 缺少 `acli` extra（或
  Python 低于 3.10）。在 Python 3.10+ 下执行
  `pip install -U "dashscope[acli]"`。
- **提示 `dashscope-agenticx` 已安装但不兼容** —— 运行时未提供本
  SDK 期望的接口。执行 `pip install -U "dashscope[acli]"` 升级；
  仍无法解决请反馈该不匹配问题。

## 完整使用指南

阿里云百炼的
[DashScope SDK Expert 文档](https://help.aliyun.com/zh/model-studio/dashscope-sdk-expert)
提供助手的完整使用指南。`dashscope <命令组> ...` 命令树请见
[命令行工具](../cli.md)。