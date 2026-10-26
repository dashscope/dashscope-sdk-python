# CLI 源码级对比（仅 CLI，不含 SDK）：DashScope CLI vs OpenAI CLI vs Anthropic CLI

> 对比维度：1) 用户接口友好 2) 错误码/错误契约 3) 流式/非流式、同步/异步
> 源码/文档依据：生成时间：2026-08
> 来源：钉钉文档（alidocs YMyQA2dXW7gYo6MzcZR0MPdZWzlwrZgb）

---

## 0. 首先必须讲清楚的事实：三家"CLI"已经不是同一物种

### 0.1 OpenAI 把 API CLI 删了

openai-python v2.35.0（2026-05-06）CHANGELOG 明确记录：

```
### Chores
* remove legacy python cli (32f36e4)
* rename legacy python cli entrypoint (a3b182d)
```

源码验证：v2.52.0 的 pyproject.toml 已无 CLI 入口点、无 cli extra、无 cli 模块。也就是说——

OpenAI 曾经有和 DashScope CLI 同类的东西（`openai api chat.completions.create ...`，随 pip 包分发），但在 2026 年 5 月主动删除了它。

OpenAI 现在的官方 CLI 是 Codex CLI：Rust 编写的独立二进制，`curl ... install.sh | sh` / npm / brew 安装，定位是编程 Agent（对标 Claude Code），不再是 API 管理工具。

### 0.2 Anthropic 从未在 Python SDK 里放过 CLI

anthropic-sdk-python 源码中没有任何 CLI 模块/入口点。Anthropic 的官方 CLI 是 Claude Code（npm 包 `@anthropic-ai/claude-code`），同样是编程 Agent。

### 0.3 DashScope 是三家中唯一"SDK 内嵌 API 管理 CLI"的

| | DashScope | OpenAI | Anthropic |
| --- | --- | --- | --- |
| SDK 内嵌 CLI | ✅ dashscope 命令（Typer） | ❌ v2.35.0 已删除 | ❌ 从未有过 |
| 独立 Agent CLI | ⚠️ acli（独立项目） | ✅ Codex CLI（Rust 二进制） | ✅ Claude Code（npm） |
| API 管理类 CLI | ✅ 仍在维护 | ❌ 放弃该品类 | ❌ 放弃该品类 |

**战略解读**：OpenAI/Anthropic 的战略是"API 调用请用 SDK，终端交互请用 Agent"，中间的"API 管理 CLI"品类被两家同时放弃。DashScope CLI 因此有了双重身份：

1. **机会**：该品类只剩 DashScope 一家在做，aws cli 式的平台管理工具有独立价值；
2. **警示**：两家头部厂商都判断"API CLI"不值得维护，DashScope 需要想清楚这个 CLI 的独特价值（答案大概率是：微调/部署/RL 等长流程管理，而不是 `generation create` 这种一条 curl 就能替代的调用）。

---

## 1. 用户接口友好

### 1.1 安装与认证

| | DashScope CLI | Codex CLI | Claude Code CLI |
| --- | --- | --- | --- |
| 安装 | `pip install dashscope`（SDK 附赠，零额外步骤） | `curl install.sh \| sh` / `npm i -g @openai/codex` / `brew install --cask codex`，原生二进制 | `npm i -g @anthropic-ai/claude-code` |
| 认证方式 | 环境变量 `DASHSCOPE_API_KEY` 或 `-k` 参数 | `codex login`（浏览器 OAuth，绑定 ChatGPT 订阅）或 API key；`codex login status` 退出码 0/1 可脚本化探测 | claude 首次运行引导 OAuth |
| 凭证管理命令 | ❌ 无 | ✅ codex login/logout/login status | ✅ /login、/logout |
| 多账号/多环境 | ❌ | ✅ ~/.codex/config.toml + `-c key=value` 单次覆盖 | ✅ settings 分层（企业/用户/项目） |

**差距点**：DashScope CLI 连 `dashscope auth whoami`（验证当前 key 有效性）都没有。Codex 的 `login status` 用退出码做脚本探测是 CI 友好的标准做法，值得直接抄。

### 1.2 交互模型

| | DashScope CLI | Codex CLI | Claude Code CLI |
| --- | --- | --- | --- |
| 默认形态 | 一次性命令 | 交互式 TUI（codex 直接进入） | 交互式 REPL（claude 直接进入） |
| 脚本模式 | 所有命令天生可脚本化 | `codex exec`（短写 `codex e`） | `claude -p "..."` |
| 会话恢复 | ❌ | ✅ `codex resume [--last/--all]`、`codex exec resume` | ✅ `claude --continue/--resume` |
| 斜杠命令 | ❌ | ✅ /resume /goal /login 等 20+ | ✅ /help /clear /compact 等 |
| 图片/文件输入 | 参数传 URL/路径 | TUI 直接粘图 | 管道 `cat x \| claude -p` |

### 1.3 帮助系统与可发现性

| | DashScope CLI | Codex CLI | Claude Code CLI |
| --- | --- | --- | --- |
| 帮助 | Typer 自动生成，no_args_is_help=True | 每命令独立参考页 + 成熟度标注（Stable/Beta/Experimental） | claude --help + /help |
| Shell 补全 | ❌ 源码确认 add_completion=False（cli/__init__.py:244） | ✅（Rust 端 clap 支持） | ✅ |
| 示例 | ❌ help 里无 example | ✅ 文档含场景化示例 | ✅ |
| 配置系统 | ❌ 无（全靠命令行参数） | ✅ ~/.codex/config.toml，命令行 -c k=v 覆盖，优先级明确 | ✅ settings.json 分层 |

**差距点**：`add_completion=False` 是 DashScope CLI 里一个一行就能修的短板（Typer 原生支持生成 bash/zsh/fish 补全脚本）。

### 1.4 用户接口友好小结

- **Codex / Claude Code**：OAuth 登录引导 + 交互 TUI + 会话恢复 + 分层配置 + Shell 补全——把 CLI 当"产品"打磨；
- **DashScope CLI**：停留在"SDK 的命令行薄封装"水平，但有一个别人没有的优点——pip 装 SDK 即得，零额外安装，且命令覆盖面（18 个资源域）远超对手；
- **最痛的三个缺口**：无 auth 命令、无 shell 补全、无配置文件。

---

## 2. 错误码与错误契约

### 2.1 DashScope CLI：三家里错误信息最"全"，但契约最"弱"

源码（cli/common.py）显示 DashScope CLI 的错误处理其实是认真设计过的：

```python
# common.py:35 — 错误输出带四要素
def print_failed_message(rsp):
    err_console.print(
        f"[red]Failed[/red] request_id: {rsp.request_id}, "
        f"status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}")

# common.py:44 — 双层错误检查（HTTP 层 + 业务层）
def ensure_ok(rsp, check_business_error=True):
    if rsp.status_code != HTTPStatus.OK: ...          # HTTP 错误
    ...
    if error_code and error_code != "": ...           # HTTP 200 但业务错误
```

- **四要素齐全**：request_id / status_code / code / message —— Codex 和 Claude Code 的错误输出都没有这么全的排查信息（Agent 类 CLI 面向人类，不需要）；
- **双层检查**：能识别"HTTP 200 但业务失败"——这是 DashScope 网关特性，另两家不存在此问题；
- **装饰器兜底**：`handle_sdk_error("action")` 把 SDK 异常统一转成红色友好错误。

但契约层面有三个问题：

1. **退出码单一**：所有失败都是 exit 1（`raise typer.Exit(1)`），无法像 `codex login status` 那样用不同退出码区分错误类别。脚本只能靠解析 stderr 文本判断错误类型。
2. **流式分支的错误不退出**（源码实锤的 bug，cli/generation.py:172-189）：

```python
if stream:
    for rsp in response:
        if rsp.status_code == HTTPStatus.OK:
            typer.echo(rsp.output)
        else:
            print_failed_message(rsp)   # ← 只打印！循环继续，命令最终 exit 0
else:
    ...
        print_failed_message(response)
        raise typer.Exit(1)             # ← 非流式分支有退出
```

流式中途出错时命令以 0 退出，CI 里会被当成成功——这是与另两家对比后暴露出的真实缺陷。

3. **错误不可机器解析**：没有 `--output-format json` 的错误输出，stderr 是 Rich 着色的自然语言。

### 2.2 Codex CLI：为自动化设计的错误契约

- `--json` 模式下所有状态变化（含错误）是 NDJSON 事件流，错误是结构化事件而非文本；
- `codex login status` 用退出码表达状态（0=有凭证）；
- `codex cloud` 列表 plain-text 打印任务状态，`--json` 时给出 tasks 数组 + cursor 分页——官方明确建议"Use --json for automation"；
- 权限/沙箱拒绝是显式审批流（approval policy），不是事后报错：untrusted / on-failure / on-request / never 四级，越界动作先问人。

### 2.3 Claude Code CLI

- `--output-format json` 时错误进 JSON 结构，退出码非 0；
- 权限拒绝走 permission-mode 交互（acceptEdits/plan/bypassPermissions），同样把"错误"前置为"审批"。

### 2.4 错误契约对比表

| | DashScope CLI | Codex CLI | Claude Code |
| --- | --- | --- | --- |
| 错误信息完整度（request_id 等） | ⭐⭐⭐⭐⭐ 四要素 | ⭐⭐⭐ 结构化事件 | ⭐⭐⭐ |
| 退出码区分度 | ⭐ 只有 0/1 | ⭐⭐⭐⭐ 分命令设计 | ⭐⭐⭐ |
| 机器可解析错误 | ❌ 着色文本 | ✅ NDJSON | ✅ JSON |
| 流式中途出错行为 | ❌ exit 0（bug） | ✅ 错误事件 | ✅ 错误事件 |
| 危险操作防护 | N/A（只读多） | ✅ 沙箱+审批四级 | ✅ 权限模式 |
| 双层错误（HTTP+业务）检查 | ✅ 独有 | N/A | N/A |

---

## 3. 流式 / 非流式、同步 / 异步

### 3.1 流式输出

| | DashScope CLI | Codex CLI | Claude Code |
| --- | --- | --- | --- |
| 开关 | `--stream` / `-s`（逐命令） | 交互 TUI 天然流式；exec 默认格式化流式输出 | 交互天然流式；`-p` 配 `--output-format stream-json` |
| 流的内容 | 原始 dict：`typer.echo(rsp.output)` + 单独一行 usage（generation.py:173-178） | NDJSON 事件（--json）或格式化渲染 | JSON 事件（assistant/tool_use/result）或渲染文本 |
| 增量控制 | 底层 incremental_output 参数，CLI 未暴露语义化选择 | N/A | N/A |
| 人类可读性 | ❌ 流出的是 Python dict 的 repr，不是纯文本 | ✅ 默认渲染 | ✅ Markdown 渲染 |
| 机器可读性 | ⚠️ 是 dict repr 不是 JSON，管道里没法 jq | ✅ NDJSON | ✅ JSON |

DashScope 流式输出的具体问题（generation.py:175）：`typer.echo(rsp.output)` 打印的是 dict 的字符串表示（单引号、Python 字面量），既不是增量文本也不是合法 JSON——人看不懂、机器没法解析，是三家里最弱的流式实现。改进方向很明确：

- **人类模式**：只打印增量 content 文本；
- **机器模式**：`--output-format json` 时每 chunk 一行合法 JSON。

### 3.2 非流式

三家都是一条命令拿完整结果，差别在输出形态：

| | DashScope CLI | Codex CLI | Claude Code |
| --- | --- | --- | --- |
| 默认输出 | output dict + usage（两行 Python repr） | 格式化文本 | 渲染文本 |
| JSON 输出选项 | ❌ 无（默认就不是 JSON） | ✅ --json | ✅ --output-format json |

### 3.3 同步 vs 异步 —— DashScope 的主场

这是维度反转的一节：Agent 类 CLI 没有"异步任务"概念（任务在会话内同步完成），而 DashScope CLI 服务的 API 有大量异步长任务（微调、RL、视频生成、部署），CLI 为此建了完整的异步设施（cli/common.py:19-21、cli/agentic_rl.py）：

| 异步能力 | DashScope CLI | Codex CLI | Claude Code |
| --- | --- | --- | --- |
| 任务提交 | ✅ ft create / rl create / 异步图像视频任务 | ❌ | ❌ |
| 状态轮询 | ✅ 内置 POLL_INTERVAL=30s 轮询 + Rich Progress 进度条 | ❌ | ❌ |
| 日志跟随 | ✅ rl logs --follow（LOG_PAGE_SIZE=1000 分页拉取） | ❌ | ❌ |
| 任务列表/详情 | ✅ ft list/get、rl list/get | ❌ | ❌ |
| 取消 | ✅ ft cancel | ❌ | ❌ |
| 云端并行任务 | ❌ | ✅ codex cloud（云端环境并行跑任务，--json 输出 tasks 数组+cursor 分页，可从 Web/GitHub/Linear/Slack 发起） | ❌ |

**值得注意**：Codex cloud 本质上是 OpenAI 版的"异步任务管理"，只是对象从"微调任务"变成了"编程任务"——提交、状态查询、分页列表、JSON 输出，形态与 DashScope 的 ft/rl 管理惊人相似。这说明"长任务管理 CLI"这个品类没有消失，只是换了载体。

### 3.4 同步/异步维度小结

- DashScope CLI 的同步调用（generation create 等）体验弱于对手的 Agent 交互，但这不是它的核心价值；
- DashScope CLI 的异步任务管理（提交→轮询→日志→取消）是三家中唯一面向 AI 训练/生成流水线的，配合 Rich Progress 有不错的体验；
- 对标 Codex cloud 后，DashScope 的异步管理可以补的一课是：任务列表的 `--json` 机器输出 + cursor 分页。

---

## 4. 总评分

| 维度 | DashScope CLI | Codex CLI | Claude Code |
| --- | --- | --- | --- |
| 安装便捷（pip 附赠） | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 认证体验 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 交互模式 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 配置系统 | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 错误信息完整度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 错误契约（退出码/机器可读） | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 流式输出质量 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 异步任务管理 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐（cloud） | ⭐ |
| API 资源覆盖 | ⭐⭐⭐⭐⭐（18 域） | ⭐（Agent 域） | ⭐（Agent 域） |

---

## 5. 给 DashScope CLI 的建议（纯 CLI 视角）

### 立即修（小改动，高回报）

1. **修流式错误退出码 bug**：generation.py 流式分支出错后补 `raise typer.Exit(1)`（§2.1 实锤）；
2. **开 Shell 补全**：`add_completion=True`（一行改动，cli/__init__.py:244）；
3. **流式输出改为增量纯文本**：默认打 content 增量，加 `--output-format json` 时输出每行合法 JSON；
4. **加 `dashscope auth whoami/status`**：验证 key 有效性，退出码区分（抄 Codex login status）。

### 中期

5. **错误退出码分级**：认证错=2、参数错=3、限流=4、服务端=5，脚本可免解析 stderr；
6. **`--output-format json` 全局化**：所有命令（尤其 ft list / rl list）支持 JSON + cursor 分页输出；
7. **配置文件**：`~/.dashscope/config.toml` 存默认 model/workspace/输出格式，`-c` 覆盖（抄 Codex 的 config 优先级设计）。

### 定位建议

8. 不要再和 Codex/Claude Code 比"对话体验"——那是 Agent 赛道，acli 在跟；
9. DashScope CLI 的对标物其实是 Codex cloud + aws cli 的混合体：把"AI 平台长流程管理"（微调→部署→RL→视频批处理）做成三家中无可替代的那一个，这才是 OpenAI/Anthropic 删除 API CLI 之后留下的空位。

---

## 附：源码证据索引

| 结论 | 证据 |
| --- | --- |
| OpenAI 删除 API CLI | openai-python/CHANGELOG.md v2.35.0 "remove legacy python cli"；v2.52.0 无 cli 入口点/extra |
| Anthropic SDK 无 CLI | anthropic-sdk-python 全仓无 cli 模块/入口点 |
| DashScope 关闭补全 | dashscope/cli/__init__.py:244 add_completion=False |
| 错误四要素+双层检查 | dashscope/cli/common.py:35-91 |
| 流式错误 exit 0 bug | dashscope/cli/generation.py:172-189 |
| 流式输出 dict repr | dashscope/cli/generation.py:175 typer.echo(rsp.output) |
| 异步轮询常量 | dashscope/cli/common.py:19-21 POLL_INTERVAL=30 |
| RL 进度条/日志 | dashscope/cli/agentic_rl.py Rich Progress、logs follow |
| Codex exec/--json/login status/resume/cloud | developers.openai.com/codex Developer commands 参考页 |
| Codex 沙箱+审批 | developers.openai.com/codex/security |
