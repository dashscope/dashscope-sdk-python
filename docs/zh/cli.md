---
description: "dashscope 命令行：全局选项、退出码与全部命令组。"
---
# 命令行工具（CLI）

安装 SDK 时会同时安装 `dashscope` 命令（以及归档 1.x 命令树的
`dashscope1`）。CLI 覆盖主要 SDK 资源，多数能力可以直接
用 shell 脚本完成。

```bash
dashscope version                 # 打印 SDK 版本（是子命令，不是 --version）
dashscope auth login              # 保存 API Key（交互式输入）
dashscope chat create --model qwen-plus --prompt "你好！"
dashscope -k sk-xxx --output json ft list
```

## 全局选项

全局选项放在子命令**之前**：

| 选项 | 含义 |
| --- | --- |
| `-k`, `--api-key` | DashScope API Key；覆盖环境变量与配置文件。 |
| `--protocol` | 传输协议：`native`、`openai` 或 `anthropic`。 |
| `--base-url` | 覆盖服务基础地址。 |
| `--workspace` | 业务空间 ID。 |
| `--region` | 服务地域，如 `cn-beijing`、`ap-southeast-1`。 |
| `--timeout` | 单次请求超时（秒）。 |
| `-o`, `--output` | 输出格式；机器格式保持 stdout 可解析（如 `json`）。 |
| `-c KEY=VALUE` | 本次调用覆盖一个配置键；可重复。SDK 键（`api_key`、`protocol`、`timeout` 等）传给客户端；CLI 键（`output`、`default_model`）设置默认值。 |
| `-v`, `--verbose` | 显示请求元数据，失败时打印完整 traceback。 |
| `--no-color` | 禁用彩色输出；也遵循 `$NO_COLOR`。 |

错误以 `Error [category]: message` 输出到 stderr。退出码：`0` 成功，
`1` 服务端/限流/网络，`2` 命令行错误，`3` 鉴权，`4` 未找到。

## 命令组

| 命令组 | 用途 |
| --- | --- |
| `auth` | `login` / `whoami` / `logout` —— 管理已保存的 API Key。 |
| `config` | `set` / `get` / `list` —— CLI 持久化配置。 |
| `chat` | 对话补全（同步与流式）。 |
| `multimodal` | 多模态理解/生成调用。 |
| `embeddings` | 文本向量调用与批量任务。 |
| `rerank` | 文档重排序。 |
| `tokenization` | Token 计数。 |
| `images` | 图像生成任务。 |
| `videos` | 视频生成任务。 |
| `audio` | TTS / ASR / 实时语音命令。 |
| `files` | 文件上传、列表、查询与删除。 |
| `models` | 浏览模型目录。 |
| `ft` | 微调任务（`fine-tunes` 为隐藏别名）。 |
| `deployments` | 模型部署管理。 |
| `rl` | Agentic RL 数据集、函数与任务（别名 `agentic-rl`）。 |
| `apps` | 百炼应用补全。 |
| `tingwu` | 会议转写与分析。 |
| `decision` | 结构化决策/评判调用。 |
| `agentstudio` | AgentStudio 托管智能体资源。 |
| `oss` | 大体积媒体输入的 OSS 上传辅助。 |

每个命令的具体参数请运行 `dashscope <命令组> --help`
（或 `dashscope <命令组> <命令> --help`）查看。

## 智能体入口

**不带任何参数**运行 `dashscope`（或 `dashscope expert`）会启动交互式
AgenticX 智能体，而不是命令树。需要可选 extra 与 Python 3.10+：

```bash
pip install -U "dashscope[acli]"
dashscope                         # 启动智能体（默认 TUI）
dashscope expert --cli            # 用纯 REPL 代替 TUI
dashscope expert "问题"            # 一次性提问后退出
```

启动方式、首次运行的 Expert 配置与故障排查见
[SDK 专家（AgenticX）](guides/expert.md)。

## 1.x CLI

**旧版命令树（`generation create`、`application create`、`files.upload`、
`login` 等）单独安装为 `dashscope1`，已冻结。**在 `dashscope` 下运行
1.x 命令名只会打印迁移提示，指向 `dashscope1` 或对应的命令。