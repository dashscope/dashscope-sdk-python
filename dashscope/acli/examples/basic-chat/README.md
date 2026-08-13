# Basic Chat — acli 最小可用示例

演示如何用最少的配置启动一个具备联网搜索能力的通用聊天 Agent。**全部智能都在 `.acli/` 配置中，没有任何 Python 启动代码**——下载后直接运行 `acli` 即可。

## 目录结构

```
basic-chat/
└── .acli/
    ├── config.toml                   # 默认 provider/model/user_name
    ├── custom-extensions.toml        # 声明 tongyi provider + capability/skill/shell_tool 注释模板
    ├── hooks.toml                    # 事件钩子（before/after_tool_call、on_error 等）注释模板
    ├── system-prompt.md              # Agent 人设与行为规则
    └── skills/
        ├── research-topic.md         # 联网搜索某主题并出简报（调用 web_search）
        ├── explain-code.md           # 解释代码逻辑
        ├── translate.md              # 中英互译
        └── write-poem.md             # 写七言绝句（演示纯 prompt 模板）
```

## 快速开始

```bash
pip install acli
export DASHSCOPE_API_KEY="sk-xxx"

# 把示例合并到 ./.acli/（同名文件自动备份到 .acli/backup/，可用 example restore 撤销）
acli example download basic-chat

# 修改 .acli/custom-extensions.toml 添加你需要的 provider
# 修改 .acli/system-prompt.md 定义 Agent 人设
# 在 .acli/skills/ 下添加你自己的技能模板

# 启动（无需 cd，配置已在当前目录）
acli
acli --tui
acli -c "你好"
```

> 想放在一个全新目录里？`mkdir my-agent && cd my-agent && acli example download basic-chat`，
> 或用 `acli example download basic-chat --target my-agent`。

## 配置即程序

### custom-extensions.toml — Provider 声明

声明 acli 能用哪些 LLM provider。最小配置只需要一个 `[[providers]]` 块：

```toml
[[providers]]
name = "tongyi"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY"      # ← 只存环境变量名，shell 提供 sk-xxx
default_model = "qwen3.7-max"
models = ["qwen3.7-max", "qwen3.7-plus", "qwen-turbo", "qwen-vl-max"]
vision_models = ["qwen-vl-max"]        # ← 让 acli 知道这些模型支持图片输入
protocol = "openai"                     # ← openai / anthropic / dashscope
```

想用 Claude / GPT / GLM？取消注释 toml 里对应的 `[[providers]]` 块即可。

**API Key 三种写法**（推荐度递减）：

1. `api_key_env = "FOO_API_KEY"` — shell 里 `export FOO_API_KEY=sk-xxx`，toml 可安全提交 git
2. `/provider` 向导交互式录入 — 写入 `api_key = "ENC:..."`（机器绑定加密）
3. 明文 `api_key = "sk-xxx"` — 加载器拒绝，仅占位说明

### system-prompt.md — Agent 人设

定义 Agent "是谁"。acli 启动时自动加载 `.acli/system-prompt.md`（workspace 优先于 `~/.acli/system-prompt.md`）。

### skills/*.md — Prompt 模板

每个 `.md` 文件是一个可复用提示词，带 YAML frontmatter：

```yaml
---
name: research-topic
description: 联网搜索某个主题并给出带来源 URL 的简报
arguments: [topic]
---

使用 web_search 工具研究「{topic}」：
...
```

调用方式：
- `/skill research-topic 量子计算` — 显式调用
- 自然语言："帮我研究一下量子计算的最新进展" — LLM 自动选择是否使用

`research-topic` 演示了如何用 prompt 引导 LLM 调用内置的 `web_search` 工具完成联网信息获取。

### config.toml — 默认值

```toml
user_name = "dashscope"
provider = "tongyi"
model = "qwen3.7-max"
memory_user_id = "acli-basic"
```

## 从这里出发

- **加更多 provider**：在 `custom-extensions.toml` 添加 `[[providers]]` 块
- **加 HTTP 工具**：添加 `[[capabilities]]` + `[[capabilities.tools]]` 块（如调用 Coze、智谱画图等）
- **加视觉能力**：添加 `type = "vision"` 的 capability tool，让 text agent 按需调用 vision LLM
- **加 Shell 工具**：添加 `[[shell_tools]]` 块，封装常用本地命令
- **加 Hooks**：在 `.acli/hooks.toml` 配置工具调用前后的钩子（如写完 `.py` 自动 `py_compile`、`pip install` 前确认、删除文件前阻止）。模板见 `.acli/hooks.toml`，覆盖 `before_tool_call` / `after_tool_call` / `on_error` / `on_message` / `on_response` 全部 5 个事件 × 6 种动作（run/block/confirm/warn/alert/log）。
- **加常驻知识**：把需要**始终**出现在 system prompt 的文档（如 API 索引）放到 `.acli/references/*.md`
- **改人设**：编辑 `system-prompt.md`，例如变成"代码审查员"、"数据分析师"、"客服"

完整功能文档见项目根目录 `README.md`。
