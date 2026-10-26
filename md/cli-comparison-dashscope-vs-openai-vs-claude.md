# CLI 深度对比：DashScope CLI vs OpenAI CLI vs Claude CLI

> 基于源码级分析（DashScope）+ 官方文档（OpenAI/Claude）
> 生成时间：2026-07-31
> 来源：钉钉文档（alidocs YMyQA2dXW7gYo6MzcZR0MPdZWzlwrZgb）

---

## 1. 定位差异 — 根本性不同

| 维度 | DashScope CLI | OpenAI CLI | Claude CLI |
| --- | --- | --- | --- |
| 本质 | API 管理工具 | API 测试工具 | AI 编程 Agent |
| 类比 | 类似 aws cli / gcloud | 类似 curl 的高级封装 | 类似 Cursor / Copilot 的终端版 |
| 用户 | 开发者管理模型/微调/部署 | 开发者快速测试 API | 开发者用 AI 写代码 |
| 交互模式 | ❌ 无 | ❌ 无 | ✅ 完整 REPL |
| Agent 能力 | ❌ 无 | ❌ 无 | ✅ 读写文件、执行命令、搜索代码 |
| 安装 | `pip install dashscope` | `pip install openai[cli]` | `npm install -g @anthropic-ai/claude-code` |
| 入口命令 | `dashscope` | `openai` | `claude` |

**核心洞察**：三者不在同一赛道。DashScope 和 OpenAI CLI 是"API 管理工具"，Claude CLI 是"AI Agent 终端"。

---

## 2. 命令体系对比

### 2.1 命令结构

**DashScope CLI**

```
dashscope <resource> <action> [options]
```

```bash
# 文本生成
dashscope generation create -m qwen-plus -p "Hello"

# 文件管理
dashscope files upload -f data.jsonl -p fine_tune
dashscope files list
dashscope files get <file_id>
dashscope files delete <file_id>

# 模型管理
dashscope models list
dashscope models get <model_name>

# 微调
dashscope ft create --training-file-ids file_xxx
dashscope ft list
dashscope ft get <job_id>
dashscope ft stream <job_id>

# 部署
dashscope deployments create --deployed-model xxx
dashscope deployments list

# Embeddings / Rerank / Tokenization
dashscope embeddings create -m text-embedding-v2 -i "Hello"
dashscope rerank create -m gte-rerank -q "query" --documents '["doc1","doc2"]'
dashscope tokenization create -m qwen-plus -p "Hello"

# 多模态
dashscope image-synthesis create -m wanx-v1 -p "A cat"
dashscope video-synthesis create -m video-generation-model -p "A cat walking"
dashscope multimodal-conversation create -m qwen-vl-plus -p "Describe this" --image-url https://...
dashscope speech-synthesis create -m cosyvoice-v1 -p "Hello"
dashscope transcription create -m paraformer-v2 -f audio.wav

# 强化学习
dashscope rl create --config rl_config.yaml
dashscope rl list
dashscope rl get <job_id>
dashscope rl logs <job_id>
```

**OpenAI CLI**

```
openai api <resource>.<method> [options]
```

```bash
# Chat Completions
openai api chat.completions.create -m gpt-4o -g user "Hello"

# 流式
openai api chat.completions.create -m gpt-4o -g user "Hello" --stream

# Files
openai api files.create -f data.jsonl -p fine-tune
openai api files.list
openai api files.retrieve <file_id>
openai api files.delete <file_id>

# Fine-tuning
openai api fine_tuning.jobs.create -m gpt-4o-mini --training_file file-xxx
openai api fine_tuning.jobs.list
openai api fine_tuning.jobs.retrieve <job_id>
openai api fine_tuning.jobs.cancel <job_id>

# Models
openai api models.list
openai api models.retrieve gpt-4o

# Images
openai api images.generate -p "A cat" --size 1024x1024

# Embeddings
openai api embeddings.create -m text-embedding-3-small -i "Hello"

# Audio
openai api audio.transcriptions.create -m whisper-1 -f audio.mp3
openai api audio.speech.create -m tts-1 -i "Hello" --voice alloy

# Assistants (deprecated)
openai api assistants.create --instructions "You are helpful" -m gpt-4o
```

**Claude CLI**

```bash
# 交互模式（REPL）
claude

# 单次提问（脚本模式）
claude -p "Explain this codebase"

# 管道模式
cat error.log | claude -p "What's wrong?"
git diff | claude -p "Review this change"

# 继续上次会话
claude --continue
claude --resume

# 指定模型
claude --model claude-sonnet-4-20250514

# 输出格式
claude -p "Hello" --output-format json
claude -p "Hello" --output-format stream-json

# 权限模式
claude --permission-mode acceptEdits
claude --permission-mode plan

# MCP 服务器
claude --mcp-server "npx @anthropic-ai/mcp-server-github"

# 非交互 + 自动批准所有操作
claude -p "Fix all bugs" --dangerously-skip-permissions
```

### 2.2 命令覆盖范围

| 功能域 | DashScope CLI | OpenAI CLI | Claude CLI |
| --- | --- | --- | --- |
| 文本生成 | ✅ generation create | ✅ chat.completions.create | ✅ 内置（Agent 模式） |
| 流式输出 | ✅ --stream | ✅ --stream | ✅ --output-format stream-json |
| 文件管理 | ✅ upload/get/list/delete | ✅ create/retrieve/list/delete | ❌ 不需要（直接读本地文件） |
| 模型列表 | ✅ models list/get | ✅ models.list/retrieve | ❌ 不需要 |
| 微调 | ✅ ft create/list/get/stream/cancel | ✅ fine_tuning.jobs.* | ❌ 不支持 |
| 部署 | ✅ deployments create/list/get/scale/delete | ❌ 不支持 | ❌ 不支持 |
| Embeddings | ✅ embeddings create | ✅ embeddings.create | ❌ 不支持 |
| Rerank | ✅ rerank create | ❌ 不支持 | ❌ 不支持 |
| Tokenization | ✅ tokenization create | ❌ 不支持 | ❌ 不支持 |
| 图像生成 | ✅ image-synthesis create | ✅ images.generate | ❌ 不支持 |
| 视频生成 | ✅ video-synthesis create | ✅ videos.create | ❌ 不支持 |
| 多模态对话 | ✅ multimodal-conversation create | ✅（通过 messages 传图） | ❌ 不支持 |
| 语音合成 | ✅ speech-synthesis create | ✅ audio.speech.create | ❌ 不支持 |
| 语音识别 | ✅ transcription create | ✅ audio.transcriptions.create | ❌ 不支持 |
| 强化学习 | ✅ rl create/list/get/logs | ❌ 不支持 | ❌ 不支持 |
| OSS 上传 | ✅ oss upload | ❌ 不支持 | ❌ 不支持 |
| 应用管理 | ✅ application | ❌ 不支持 | ❌ 不支持 |
| 代码编辑 | ❌ | ❌ | ✅ 核心能力 |
| 命令执行 | ❌ | ❌ | ✅ 核心能力 |
| 代码搜索 | ❌ | ❌ | ✅ 核心能力 |
| 会话管理 | ❌ | ❌ | ✅ --continue / --resume |
| MCP 集成 | ❌ | ❌ | ✅ --mcp-server |

**结论**：

- DashScope CLI 覆盖面最广（18 个资源域），但全是 API 管理
- OpenAI CLI 覆盖 8 个资源域，纯 API 测试
- Claude CLI 覆盖 0 个 API 资源域，但提供了完全不同的 Agent 能力

---

## 3. 用户体验对比

### 3.1 安装与配置

| 步骤 | DashScope CLI | OpenAI CLI | Claude CLI |
| --- | --- | --- | --- |
| 安装 | `pip install dashscope` | `pip install openai[cli]` | `npm install -g @anthropic-ai/claude-code` |
| 认证 | `export DASHSCOPE_API_KEY=sk-xxx` 或 `-k` 参数 | `export OPENAI_API_KEY=sk-xxx` 或 `openai auth login` | claude 首次运行引导 OAuth 登录 |
| 验证 | 无 auth 命令 | `openai auth login` / `openai auth logout` | OAuth 自动管理 |
| 版本 | `pip show dashscope` | `openai --version` | `claude --version` |

**差距**：

- OpenAI 有 `openai auth login` 交互式认证
- Claude 有 OAuth 引导流程，首次使用零配置
- DashScope 只有环境变量和 `-k` 参数，无 auth 命令

### 3.2 帮助系统

| 特性 | DashScope CLI | OpenAI CLI | Claude CLI |
| --- | --- | --- | --- |
| 全局帮助 | `dashscope --help` | `openai --help` | `claude --help` |
| 子命令帮助 | `dashscope generation --help` | `openai api chat.completions.create --help` | `claude --help` |
| 参数说明 | ✅ Typer 自动生成 | ✅ Click 自动生成 | ✅ 内置 |
| 示例 | ❌ 无 | ❌ 无 | ✅ 帮助中有示例 |
| Shell 补全 | ❌ add_completion=False | ✅ 支持 | ✅ 支持 |
| Rich 格式化 | ✅ 彩色输出 | ❌ 纯文本 | ✅ 彩色 + Markdown 渲染 |

### 3.3 输出格式

| 特性 | DashScope CLI | OpenAI CLI | Claude CLI |
| --- | --- | --- | --- |
| 默认输出 | JSON（Rich 格式化） | JSON（原始） | 纯文本 / Markdown |
| --output-format | ❌ 无 | ❌ 无 | ✅ text / json / stream-json |
| 表格输出 | ✅（rl 命令有 Table） | ❌ | ❌ |
| JSON 美化 | ✅ console.print_json() | ✅ 自动格式化 | ✅ --output-format json |
| 进度条 | ✅（rl 命令有 Progress） | ❌ | ❌ |
| 错误着色 | ✅ `[red]Error:[/red]` | ❌ 纯文本 | ✅ 彩色 |
| 成功标记 | ✅ `[green]✓[/green]` | ❌ | ✅ |

### 3.4 错误处理

**DashScope CLI**

```python
# common.py — 统一错误处理
def print_failed_message(rsp):
    err_console.print(
        f"[red]Failed[/red] request_id: {rsp.request_id}, "
        f"status_code: {rsp.status_code}, "
        f"code: {rsp.code}, message: {rsp.message}",
    )

def ensure_ok(rsp, check_business_error=True):
    if rsp.status_code != HTTPStatus.OK:
        print_failed_message(rsp)
        raise typer.Exit(1)
    # 还检查 HTTP 200 但业务错误的情况
    if check_business_error:
        error_code = output.get("code")
        if error_code:
            err_console.print(f"[red]Business Error[/red] ...")
            raise typer.Exit(1)
    return output
```

**亮点**：

- `ensure_ok()` 统一处理 — 消除重复的 if/else
- `handle_sdk_error` 装饰器 — 捕获 SDK 异常转为友好消息
- 区分 HTTP 错误和业务错误（HTTP 200 但 code 非空）
- Rich 彩色输出

实际错误输出：

```
Failed request_id: req-xxx, status_code: 429, code: Throttling, message: Requests rate limit
```

**OpenAI CLI**

```json
// 错误直接输出 JSON
{
  "error": {
    "message": "You exceeded your current quota...",
    "type": "insufficient_quota",
    "code": "insufficient_quota"
  }
}
```

- 无彩色输出
- 无 request_id 高亮
- 无建议操作

**Claude CLI**

```
Error: Rate limit exceeded. Please wait a moment and try again.
```

- 简洁友好的错误消息
- 权限拒绝时有明确提示
- 工具执行失败时显示具体原因

---

## 4. 架构设计对比

### 4.1 技术栈

| 组件 | DashScope CLI | OpenAI CLI | Claude CLI |
| --- | --- | --- | --- |
| 语言 | Python | Python | TypeScript (Node.js) |
| CLI 框架 | Typer (基于 Click) | Click | Ink (React for CLI) |
| 输出美化 | Rich | 无 | Ink + Markdown 渲染 |
| 异步 | asyncio（仅 rl 命令） | 无 | 完整异步 |
| 包管理 | pip | pip | npm |
| 入口 | dashscope.cli.main() | openai.cli.main() | claude (Node binary) |

### 4.2 代码组织

**DashScope CLI**

```
dashscope/cli/
├── __init__.py          # Typer app + 命令注册 + 兼容层
├── __main__.py          # python -m dashscope.cli
├── common.py            # 共享工具（ensure_ok, error, success）
├── generation.py        # 文本生成
├── files.py             # 文件管理
├── models.py            # 模型管理
├── fine_tunes.py        # 微调
├── deployments.py       # 部署
├── embeddings.py        # 向量化
├── rerank.py            # 重排序
├── tokenization.py      # 分词
├── image_synthesis.py   # 图像合成
├── video_synthesis.py   # 视频合成
├── image_generation.py  # 图像生成
├── multimodal_conversation.py  # 多模态对话
├── multimodal_embedding.py     # 多模态向量化
├── transcription.py     # 语音识别
├── speech_synthesis.py  # 语音合成
├── oss.py               # OSS 上传
├── application.py       # 应用管理
├── agentic_rl.py        # 强化学习（最复杂，21.8KB）
└── code_generation.py   # 代码生成
```

模式：每个资源域一个文件，每个文件定义一个 `typer.Typer()` app。

**OpenAI CLI**

```
openai/cli/
├── main.py              # 入口
├── _cli.py              # Click group 定义
├── _errors.py           # 错误处理
└── (resources 通过 SDK 自动生成)
```

模式：极简 — 只是 SDK 方法的薄封装，大部分代码自动生成。

**Claude CLI**

```
@anthropic-ai/claude-code/
├── src/
│   ├── cli.ts           # 入口 + 参数解析
│   ├── repl/            # 交互式 REPL
│   ├── agent/           # Agent 循环
│   ├── tools/           # 内置工具（Read/Edit/Bash/Glob）
│   ├── permissions/     # 权限系统
│   ├── mcp/             # MCP 服务器集成
│   ├── sessions/        # 会话管理
│   └── hooks/           # 生命周期钩子
└── ...
```

模式：完整的 Agent 框架，CLI 只是入口。

### 4.3 命令模式对比

| 模式 | DashScope | OpenAI | Claude |
| --- | --- | --- | --- |
| 资源-动作 | `dashscope files upload` | `openai api files.create` | N/A |
| 命名风格 | kebab-case (fine-tunes) | dot-notation (fine_tuning.jobs) | 无子命令（flag 驱动） |
| 参数风格 | `--training-file-ids` | `--training_file` | `--model` |
| 输入方式 | 命令行参数 | 命令行参数 + stdin | stdin + 交互式 |
| JSON 输入 | `--messages '[{"role":"user",...}]'` | `-g user "Hello"` | 直接输入 |

---

## 5. 功能深度对比

### 5.1 流式输出

**DashScope CLI**

```bash
dashscope generation create -m qwen-plus -p "Hello" --stream
```

```python
# 实现：逐个 chunk 输出
if stream:
    for rsp in response:
        if rsp.status_code == HTTPStatus.OK:
            typer.echo(rsp.output)  # 原始 output dict
```

问题：

- 输出的是 `rsp.output`（原始 dict），不是纯文本
- 无增量拼接
- 无进度指示

**OpenAI CLI**

```bash
openai api chat.completions.create -m gpt-4o -g user "Hello" --stream
```

- 每个 chunk 输出 JSON
- 支持 `--stream` flag
- 输出格式与非流式一致（JSON）

**Claude CLI**

```bash
claude -p "Hello" --output-format stream-json
```

- 流式 JSON 事件（assistant / tool_use / result）
- 交互模式下实时渲染 Markdown
- 支持 `--output-format text` 纯文本流

### 5.2 多模态输入

**DashScope CLI**

```bash
# 图像理解
dashscope multimodal-conversation create \
  -m qwen-vl-plus \
  -p "Describe this image" \
  --image-url "https://example.com/cat.jpg"

# 语音识别
dashscope transcription create \
  -m paraformer-v2 \
  -f audio.wav
```

**OpenAI CLI**

```bash
# 图像（通过 messages JSON）
openai api chat.completions.create \
  -m gpt-4o \
  -g user '[{"type":"text","text":"Describe"},{"type":"image_url","image_url":{"url":"https://..."}}]'

# 语音
openai api audio.transcriptions.create -m whisper-1 -f audio.mp3
```

**Claude CLI**

```bash
# 不直接支持多模态 API 调用
# 但可以读取本地文件（包括图片）作为上下文
claude -p "Describe this image" < image.png  # 管道输入
```

### 5.3 异步任务管理

**DashScope CLI — 独有优势**

```bash
# 创建异步任务
dashscope image-synthesis create -m wanx-v1 -p "A cat" --async

# 查看任务状态
dashscope image-synthesis get <task_id>

# 等待任务完成（内置轮询）
dashscope video-synthesis create -m xxx -p "A cat" --wait

# 微调任务流式日志
dashscope ft stream <job_id>

# RL 任务日志
dashscope rl logs <job_id> --follow
```

亮点：

- 内置任务轮询（POLL_INTERVAL = 30s）
- Rich Progress 进度条（rl 命令）
- 任务状态跟踪（PENDING → RUNNING → SUCCEEDED/FAILED）

**OpenAI CLI**

```bash
# 微调任务
openai api fine_tuning.jobs.create -m gpt-4o-mini --training_file file-xxx
openai api fine_tuning.jobs.retrieve <job_id>
openai api fine_tuning.jobs.cancel <job_id>
# 无内置轮询，需手动重复 retrieve
```

**Claude CLI**

```
# 不适用 — Claude CLI 不管理异步任务
```

### 5.4 管道与脚本集成

**DashScope CLI**

```bash
# 基本管道
echo '{"prompt": "Hello"}' | dashscope generation create -m qwen-plus -p "$(cat)"

# 批量处理（需要外部脚本）
for file in *.txt; do
  dashscope embeddings create -m text-embedding-v2 -i "$(cat $file)" > "${file}.json"
done
```

问题：

- 无原生 stdin JSON 输入
- 无 `--json` 输入模式
- 批量操作需要外部 shell 脚本

**OpenAI CLI**

```bash
# stdin JSON 输入
echo '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}' | \
  openai api chat.completions.create

# 文件输入
openai api chat.completions.create < request.json
```

亮点：支持从 stdin 读取完整 JSON 请求体。

**Claude CLI**

```bash
# 管道模式 — 最强大
cat error.log | claude -p "What's wrong with this?"
git diff HEAD~1 | claude -p "Write a commit message"
find . -name "*.py" | xargs claude -p "Find security issues" --output-format json

# 脚本集成
result=$(claude -p "What is 2+2?" --output-format json)
echo "$result" | jq '.result'
```

亮点：

- 管道输入是核心设计
- `--output-format json` 便于脚本解析
- 可以处理任意文件内容

---

## 6. Claude CLI 独有能力（DashScope/OpenAI 完全缺失）

### 6.1 交互式 REPL

```
$ claude
╭────────────────────────────────────────────╮
│ Claude Code                                │
│                                            │
│ Type /help for available commands          │
╰────────────────────────────────────────────╯

> Fix the bug in auth.py where login fails for expired tokens

I'll look at the auth.py file to understand the issue.

[Read auth.py]
[Edit auth.py - line 42: Added token expiry check]
[Bash: python -m pytest tests/test_auth.py -v]

All tests pass. The fix adds an expiry check before validating the token.
```

### 6.2 Agent 工具调用

Claude CLI 内置工具：

- Read — 读取文件
- Edit — 编辑文件（精确行级修改）
- Write — 写入新文件
- Bash — 执行 shell 命令
- Glob — 文件模式搜索
- Grep — 内容搜索
- WebSearch — 网络搜索

### 6.3 权限系统

```bash
# 6 种权限模式
claude --permission-mode default        # 每次操作都询问
claude --permission-mode acceptEdits    # 自动批准文件编辑
claude --permission-mode plan           # 只规划不执行
claude --permission-mode auto           # 自动批准安全操作
claude --permission-mode dontAsk        # 从不询问（危险）
claude --permission-mode bypassPermissions  # 完全绕过（极危险）
```

### 6.4 会话管理

```bash
claude                    # 新会话
claude --continue         # 继续上次会话
claude --resume           # 选择历史会话恢复
claude --session-id xxx   # 恢复指定会话
```

### 6.5 MCP 服务器集成

```bash
# 连接 GitHub MCP
claude --mcp-server "npx @anthropic-ai/mcp-server-github"

# 连接数据库 MCP
claude --mcp-server "npx @anthropic-ai/mcp-server-postgres postgresql://..."

# 在会话中使用 MCP 工具
> List all open PRs on the dashscope/dashscope-sdk-python repo
```

---

## 7. 综合评分

| 维度 | DashScope CLI | OpenAI CLI | Claude CLI |
| --- | --- | --- | --- |
| API 覆盖广度 | ⭐⭐⭐⭐⭐ (18域) | ⭐⭐⭐⭐ (8域) | ⭐ (0域) |
| 命令一致性 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 错误处理 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| 输出美化 | ⭐⭐⭐⭐ (Rich) | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 脚本集成 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 交互体验 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Agent 能力 | ❌ | ❌ | ⭐⭐⭐⭐⭐ |
| 异步任务管理 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ |
| 多模态支持 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| RL/微调 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ |
| Shell 补全 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 文档/示例 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 8. DashScope CLI 改进建议

### 8.1 短期（P0）— 提升现有体验

| # | 改进 | 说明 | 参考 |
| --- | --- | --- | --- |
| 1 | 添加 `dashscope auth login` | 交互式认证，保存 Key 到本地 | OpenAI `openai auth login` |
| 2 | 启用 Shell 补全 | add_completion=True + 安装脚本 | Typer 原生支持 |
| 3 | stdin JSON 输入 | `echo '...' \| dashscope generation create --stdin` | OpenAI CLI |
| 4 | `--output-format` 选项 | json / text / table 可选 | Claude CLI |
| 5 | 流式输出纯文本 | 流式时只输出 content 文本，不输出完整 dict | Claude CLI |

### 8.2 中期（P1）— 补齐能力

| # | 改进 | 说明 | 参考 |
| --- | --- | --- | --- |
| 6 | 管道模式 | `cat file.py \| dashscope generation create -m qwen-plus --stdin-prompt "Review this"` | Claude CLI |
| 7 | 批量处理 | `dashscope embeddings create --batch-file inputs.txt` | 自研 |
| 8 | 任务等待 + 进度 | 所有异步任务支持 `--wait` + Rich Progress | 已有（rl），扩展到所有 |
| 9 | `dashscope config` | 配置管理（默认模型、默认 Key、输出格式） | aws configure |
| 10 | JSON Schema 验证 | 输入参数本地验证，不发请求就报错 | OpenAI SDK |

### 8.3 长期（P2）— 差异化竞争

| # | 改进 | 说明 | 参考 |
| --- | --- | --- | --- |
| 11 | 交互式 REPL | `dashscope chat` 进入对话模式 | Claude CLI |
| 12 | Agent 模式 | 集成 acli 的 Agent 能力到 CLI | Claude CLI + 自有 acli |
| 13 | MCP 支持 | `dashscope --mcp-server` 连接外部工具 | Claude CLI |
| 14 | 会话管理 | `--continue` / `--resume` 多轮对话 | Claude CLI |
| 15 | 权限系统 | 工具调用分级确认 | Claude CLI |

### 8.4 差异化定位建议

DashScope CLI 不应试图复制 Claude CLI（那是 Agent 赛道），而应强化自身优势：

```
DashScope CLI 定位：阿里云 AI 平台的完整管理工具
├── 模型管理（list / get / deploy / scale）
├── 微调全流程（upload → create → monitor → deploy）
├── RL 训练全流程（create → logs → monitor）
├── 多模态全栈（图像/视频/语音/文本）
└── 与阿里云生态集成（OSS / RAM / 工作空间）
```

**独特价值**：

- OpenAI CLI 不支持部署管理 → DashScope 有
- Claude CLI 不做 API 管理 → DashScope 有
- 两者都不支持 RL 训练 → DashScope 有
- 两者都不支持视频生成 → DashScope 有

---

## 9. 命令设计最佳实践对比

### 9.1 命名规范

| 方面 | DashScope | OpenAI | 建议 |
| --- | --- | --- | --- |
| 资源名 | fine-tunes (kebab) | fine_tuning.jobs (dot) | DashScope 更好（shell 友好） |
| 动作名 | create / list / get / delete | create / list / retrieve / delete | 统一用 get 比 retrieve 更直觉 |
| 缩写 | ft = fine-tunes | 无 | DashScope 好 |
| 隐藏命令 | ✅ hidden=True | ✅ | 两者都好 |

### 9.2 参数设计

| 方面 | DashScope | OpenAI | Claude |
| --- | --- | --- | --- |
| 模型 | `-m` / `--model` | `-m` / `--model` | `--model` |
| 输入 | `-p` / `--prompt` | `-g` / `--message` (role + content) | 直接输入 |
| 流式 | `-s` / `--stream` | `--stream` | `--output-format stream-json` |
| 全局 Key | `-k` / `--api-key` | 环境变量 | OAuth |
| 输出格式 | ❌ 无 | ❌ 无 | `--output-format` |

### 9.3 退出码

| 退出码 | DashScope | OpenAI | Claude |
| --- | --- | --- | --- |
| 0 | 成功 | 成功 | 成功 |
| 1 | API 错误 / SDK 异常 | API 错误 | 执行失败 |
| 2 | 参数错误 | 参数错误 | 权限拒绝 |

---

## 参考

- DashScope CLI 源码：`dashscope/cli/`（24 个文件）
- OpenAI CLI 文档
- Claude CLI 文档
