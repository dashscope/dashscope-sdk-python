# DashScope SDK Expert — acli 配置化示例

本示例演示如何用 **AgenticCLI (acli)** 原生配置机制构建一个场景化的 AI 专家 Agent。

**核心理念：配置驱动，零 Python 胶水。** Agent 的身份、能力、技能、知识索引全部由 `.acli/` 目录中的文件定义；下载示例后直接运行 `acli` 即可启动。

## 目录结构

```
dashscope-sdk-expert/
└── .acli/                      # Agent 配置目录
    ├── config.toml              # 模型与用户配置
    ├── custom-extensions.toml   # Provider 声明（tongyi）
    ├── hooks.toml               # 事件钩子
    ├── system-prompt.md         # 系统提示词（Agent 人设与行为规则）
    └── skills/                  # 技能模板（模型按需 use_skill 加载）
        ├── text-generation.md   # 文本生成（Generation / OpenAI 兼容，Python+Java）
        ├── multimodal.md        # 多模态（MultiModalConversation/ImageSynthesis/VideoSynthesis）
        ├── speech.md            # 语音（SpeechSynthesizer/Transcription）
        ├── retrieval.md         # 检索（Embedding/TextReRank/RAG）
        ├── fine-tuning.md       # 微调与部署（SFT/CPT/DPO/Deployments）
        ├── agent.md             # Agent（Application/Assistants/插件与 MCP）
        ├── cli.md               # dashscope CLI 命令参考
        ├── sdk-example.md       # 生成 SDK 代码示例
        ├── api-doc.md           # 查看 API 参数文档
        ├── diagnose.md          # 诊断 SDK 调用错误
        ├── error-code.md        # 解释错误码
        ├── explain-code.md      # 解释代码逻辑
        └── translate.md         # 中英互译
```

## 快速开始

```bash
pip install acli
export DASHSCOPE_API_KEY="sk-xxx"

# 把示例合并到 ./.acli/（同名文件自动备份到 .acli/backup/，可用 example restore 撤销）
acli example download dashscope-sdk-expert

# 启动 —— 无需 cd、无需任何 Python 启动脚本
acli
acli --tui
acli -c "Generation.call 怎么用？"
```

## 配置化理念

### 1. system-prompt.md — Agent 人设

定义 Agent 的身份、知识范围、行为规则。这是 Agent "是谁"的核心：

```markdown
You are DashScope SDK Expert, an intelligent assistant for the DashScope Python SDK...

## Grounded Knowledge First
Before answering, ALWAYS verify against the actual installed SDK...
```

### 2. skills/ — 领域知识库（按需加载）

SDK/CLI 的对外接口知识直接维护在 domain skills 里：每个领域一个文件，包含模型清单、Python 与 Java SDK 签名、输入输出结构和错误码。模型回答 API 问题时通过 `use_skill` 按需加载对应技能，**不常驻 system prompt**——首轮输入 token 因此减少约 16k 字符；skill 未覆盖的细节再回退到 `inspect.signature` / `help()` 验证已安装包。

### 3. skills/ — 任务模板

每个 `.md` 文件是一个可复用的 prompt 模板，带有 frontmatter 元数据：

```yaml
---
name: sdk-example
description: 生成 DashScope SDK 可运行代码示例
arguments: [api_name]
---

生成代码前，请先验证用户安装的 SDK 版本和 API 签名：
1. `run_command("python -c 'import dashscope; ...'")`
...
```

- **name**: 技能标识符，用于 `/skill` 命令调用
- **description**: 简短描述，Agent 据此决定何时使用
- **arguments**: 模板变量，调用时由实际值替换

### 4. config.toml — 运行时配置

```toml
user_name = "dashscope"
provider = "tongyi"
model = "qwen3.7-plus"
memory_user_id = "acli-dashscope"
```

## 复用此模式

要为你自己的场景创建 AI 专家：

1. `acli example download dashscope-sdk-expert`（合并到你的项目 `./.acli/`）
2. 修改 `.acli/system-prompt.md` — 定义你的 Agent 人设
3. 修改 `.acli/skills/` — 添加你的领域知识与技能模板（模型按需加载）
4. 修改 `.acli/config.toml` — 选择合适的模型
5. 运行 `acli`

## 设计启示

| 传统方式 | acli 配置化方式 |
|---------|---------------|
| 代码中硬编码 prompt | `system-prompt.md` 文件 |
| if-else 分支处理不同场景 | `skills/*.md` 模板库 |
| 大文档全量塞进 prompt | `skills/` 领域知识按需加载 |
| 修改代码才能调整行为 | 编辑 Markdown 即可 |
| 无法共享和复用 | 整个 `.acli/` 目录可迁移 |
