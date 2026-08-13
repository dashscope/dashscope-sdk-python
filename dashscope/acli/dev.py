# -*- coding: utf-8 -*-
"""Developer / extension commands.

`/dev` 提供两类能力：

1. **运行时操作**（直接生效并落盘到 workspace 配置）
   - `model list`           列出所有 provider 的可选模型
   - `model add <p> <m>`    给某 provider 注册新模型（持久化到 workspace 的 custom_models）
   - `model remove  <p> <m>`    移除自定义模型

2. **扩展指南**（打印 Markdown 步骤，让你知道在哪些文件里加什么代码）
   - `provider`             新增一个 LLM Provider
   - `platform`             新增一个云端 Platform 能力
   - `tool`                 新增一个本地工具
   - `skill`                新增一个预置 Skill
"""
# pylint: disable=wrong-import-position,wrong-import-order,unused-argument
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown

from dashscope.acli.config import (
    PROVIDER_MODELS,
    Config,
    normalize_model_name,
    register_custom_model,
)

console = Console()


# ===== Runtime: model registration =====


def _format_models() -> str:
    """Render PROVIDER_MODELS as a markdown table snapshot."""
    lines = ["## 当前可选模型", "", "| Provider | 模型 |", "|---|---|"]
    for prov, models in PROVIDER_MODELS.items():
        lines.append(f"| `{prov}` | {', '.join(f'`{m}`' for m in models)} |")
    return "\n".join(lines)


# Model-name prefixes that unambiguously map to a built-in provider.
# Used when the user types `/dev model add qwen-max` without specifying
# the provider explicitly.
_MODEL_PROVIDER_HINTS: dict[str, str] = {
    "qwen": "tongyi",
    "claude": "anthropic",
    "gpt": "openai",
}


def _infer_provider_from_model(model: str) -> str | None:
    """Infer a built-in provider from a model name prefix.

    Returns None if the prefix is unknown or ambiguous.
    """
    model_lower = model.lower()
    for prefix, provider in sorted(
        _MODEL_PROVIDER_HINTS.items(),
        key=lambda x: -len(x[0]),
    ):
        if model_lower.startswith(prefix):
            return provider
    return None


def _apply_custom_models(config: Config) -> None:
    """Merge config.custom_models into the in-process PROVIDER_MODELS dict.

    Format of each entry: `provider:model`. Unknown providers are skipped
    silently.
    """
    for entry in getattr(config, "custom_models", []) or []:
        if ":" not in entry:
            continue
        prov, model = entry.split(":", 1)
        prov, model = prov.strip(), model.strip()
        if not prov or not model or prov not in PROVIDER_MODELS:
            continue
        if model not in PROVIDER_MODELS[prov]:
            PROVIDER_MODELS[prov].append(model)


def _model_add(config: Config, provider: str, model: str) -> None:
    if provider not in PROVIDER_MODELS:
        console.print(f"[red]未知 Provider: {provider}[/red]")
        console.print(f"[dim]可选: {', '.join(PROVIDER_MODELS.keys())}[/dim]")
        return
    normalized = normalize_model_name(model)
    if normalized != model.strip():
        console.print(
            f"[dim]模型名归一化为 {normalized} (API 模型 ID 大小写敏感)[/dim]",
        )
    if normalized in {m.lower() for m in PROVIDER_MODELS[provider]}:
        console.print(f"[dim]{provider}/{normalized} 已存在[/dim]")
        return
    register_custom_model(config, provider, normalized)
    console.print(f"[green]✓ 已注册 {provider}/{normalized}[/green]")
    console.print("[dim]切到该模型: /provider[/dim]")


def _model_remove(config: Config, provider: str, model: str) -> None:
    # add 时存的是归一化小写名，remove 必须同样归一化才能命中。
    model = normalize_model_name(model)
    entry = f"{provider}:{model}"
    if entry not in (config.custom_models or []):
        console.print(f"[yellow]{entry} 不是自定义模型（无法移除内置项）[/yellow]")
        return
    config.custom_models.remove(entry)
    if provider in PROVIDER_MODELS and model in PROVIDER_MODELS[provider]:
        PROVIDER_MODELS[provider].remove(model)
    config.save_workspace()
    console.print(f"[yellow]✗ 已移除 {provider}/{model}[/yellow]")


def _model_list(config: Config) -> None:
    console.print(Markdown(_format_models()))
    if config.custom_models:
        console.print()
        console.print(
            f"[dim]Workspace 自定义: {', '.join(config.custom_models)}[/dim]",
        )


# ===== Guides =====


_GUIDE_PROVIDER = """\
## 新增 LLM Provider

> 把一家新的 LLM 接入 acli 的 chat / stream / tool-call 闭环。
> 绝大多数场景**不用写代码**——填一份 TOML 即可。

acli 只内置 3 个协议实现，所有 provider（含内置 tongyi/anthropic/openai/
deepseek/zhipu/ideatalk/ollama）都通过 `custom-extensions.toml` 配置：

| 协议字段    | 实现类              | 适用场景                   |
|-------------|---------------------|----------------------------|
| `openai`    | `OpenAIProvider`    | OpenAI 兼容端点（Moonshot/ |
|             |                     | Yi/Step/Deepseek/Zhipu/Ollama…） |
| `anthropic` | `AnthropicProvider` | Anthropic Messages API（Claude / 代理端） |
| `dashscope` | `TongyiProvider`    | DashScope OpenAI-compat 端点（通义千问） |

**步骤**（Layer-1，推荐）

1. 运行 `/dev provider add`，交互式填 name / base_url / api_key / model / protocol
2. 或手动在 `~/.acli/custom-extensions.toml`（全局）或
   `./.acli/custom-extensions.toml`（workspace）加一段：

```toml
# Moonshot / Kimi —— OpenAI 兼容
[[providers]]
name = "moonshot"
base_url = "https://api.moonshot.cn/v1"
api_key_env = "MOONSHOT_API_KEY"
default_model = "kimi-k2"
models = ["kimi-k1"]
protocol = "openai"

# 通过 Anthropic 协议代理访问 Qwen
[[providers]]
name = "bailian-anthropic"
base_url = "https://example.com/apps/anthropic"
api_key_env = "DASHSCOPE_API_KEY"
default_model = "qwen3.7-max"
protocol = "anthropic"

# Ollama 本地无 auth
[[providers]]
name = "ollama"
base_url = "http://localhost:11434/v1"
default_model = "llama3"
protocol = "openai"
auth = false
```

**步骤**（Layer-2，仅当协议不在上表内时）

只有当新 LLM 用了非标准协议（非 OpenAI/Anthropic/DashScope 兼容）才需要写
Python：在 `src/acli/providers/<name>.py` 实现完整的 `LLMProvider` 协议
（`chat` / `stream_chat` + tool-call 透传，见 `providers/base.py`），然后在
`providers/__init__.py` 的 `_create_provider` 加一个 `if proto == "<name>"`
分支。

**验证**

```
acli> /provider   # 问答式切换 Provider / 写入 API Key（auth=false 的 provider 不需要）
acli> 你好         # 触发一次 chat
```
"""


_GUIDE_PLATFORM = """\
## 新增 Platform 能力

> Platform = 云端 SDK 包装层。一个 Platform 可暴露多个 capability（如 bailian → memory + mcp）。

**步骤**

1. 在 `src/acli/platforms/<vendor>/` 下新建包，写一个客户端类，按
   `platforms/base.py` 的 Protocol 之一实现接口（`MemoryProvider` /
   `KBProvider` / `SearchProvider` / `ContextProvider` / `DataProvider` /
   `PromptProvider`）
   - 若是全新能力类别，先去 `platforms/base.py` 增加新的 Protocol + dataclass
2. 在 `src/acli/platforms/__init__.py` 加 `get_<cap>_provider(config)`
   工厂；缺凭证时返回 `None`
3. 在 `src/acli/cli.py` `CAPABILITY_CATALOG` 加一行：
   ```python
   {"key": "<vendor>.<cap>", "name": "...", "platform": "<vendor>",
   "cap": "<cap>", "requires": ["..."]}
   ```
4. 在 `src/acli/tools/platform.py` `register_platform_tools()` 加分支，
   在能力启用时把 client 包成若干工具注册到 `registry`
5. 如需 `/<cap>` 子命令，在 `cli.py` 加 `_handle_<cap>_command` 并在
   `_handle_slash_command` + `_run_loop` 路由里挂载（注意走
   `_require_capability` 网关）
6. 如果想被 `/update` 同步资源，在 `cli.py` `_get_update_targets()` 注册一个 `_sync_<vendor>`

**验证**

```
acli> /capability enable <vendor>.<cap>
acli> /update <vendor>
acli> /<cap> list                # 走自家子命令
acli> 帮我查一下 ...              # LLM 自动选用新工具
```
"""


_GUIDE_TOOL = """\
## 新增本地工具

> 本地工具 = LLM 在 agent loop 里能直接调用的能力，不依赖云端。

**步骤**

1. 在 `src/acli/tools/<name>.py` 写一个普通 `async`/同步函数，加 `@tool(...)` 装饰器：
   ```python
   from dashscope.acli.tools.registry import PermissionLevel, tool

   @tool(
       name="my_tool",
       description="一句话告诉 LLM 这个工具干嘛、何时该调",
       permission=PermissionLevel.AUTO,  # AUTO / CONFIRM / DANGEROUS
   )
   def my_tool(path: str, limit: int = 10) -> str:
       ...
   ```
   - 参数 schema 由 `_build_parameters_schema` 从类型注解自动推导
     （`str/int/float/bool/list/dict` + `Optional`）
   - 返回字符串作为 tool result 喂回 LLM
2. 在 `src/acli/cli.py` 顶部加一行 `import acli.tools.<name>`
   触发注册（看 filesystem/shell 的写法）
3. 写好 description——它是 LLM 决定要不要调你这个工具的唯一依据，要写清楚 **何时该调 / 何时不该调 / 输入输出**

**权限分级**

- `AUTO`      读操作 / 幂等查询，直接执行
- `CONFIRM`   可逆写操作（写文件、上传），逐次让用户确认
- `DANGEROUS` 不可逆（删除、`rm -rf`），双重警告

**验证**

```
acli> 用 my_tool 帮我把 X 处理一下     # 看 LLM 是否自主选中
```
"""


_GUIDE_SKILL = """\
## 新增 Skill

> Skill = 预置 Prompt 模板，`/skill <name> <args...>` 一行替代手敲提示词，可选依赖某个 MCP 服务。

**步骤**

1. 在 `src/acli/skills/<name>.py` 注册：
   ```python
   from dashscope.acli.skills.base import Skill, register

   register(Skill(
       name="my-skill",
       description="一句话说这个 Skill 干嘛",
       mcp_service="",  # 若依赖 MCP 服务则填 service 名，会自动尝试连接
       prompt_template="请... {arg1} ... {arg2}",  # f-string 风格的占位符
       arguments=["arg1", "arg2"],              # 最后一个会贪婪匹配剩余 token
   ))
   ```
2. 在 `src/acli/skills/__init__.py` 的 `from acli.skills import ...`
   行追加你的模块名，触发注册副作用
3. 调用：`/skill my-skill 值1 值2`；不带参数会打印 usage

**依赖 MCP 服务的写法**

- `mcp_service="time"` 这种填了名字的，`/skill` 会先尝试 auto-connect；
  要让它在 `/mcp services` 里可见，去 `skills/base.py` 的
  `KNOWN_MCP_SERVICES` 加描述
"""


_GUIDE_INDEX = """\
## /dev — 开发/扩展指引

**运行时（直接生效）**
- `dev model list`                       查看所有 provider 的可选模型
- `dev model add <p> <m>`                给 provider 注册新模型（持久化到 workspace）
- `dev model remove  <p> <m>`                移除自定义模型

**Layer-1 扩展：配置即程序**（写入 `custom-extensions.toml`，无需改代码）
- `dev provider add`                     交互式注册扩展 LLM Provider（OpenAI 兼容）
- `dev provider list` / `remove <name>`      查看 / 删除
- `dev capability add`                   生成 capability scaffold 模板供编辑
- `dev capability list` / `remove <key>`     查看 / 删除
- `dev skill add`                        交互式新增自定义 Skill（Prompt 模板）
- `dev skill list` / `remove <name>`         查看 / 删除
- `dev tool add`                         交互式新增 Shell 工具（命令包装为 LLM 工具）
- `dev tool list` / `remove <name>`          查看 / 删除

**调试 / 测试**
- `dev debug tools`                      列出所有已注册工具（名称、权限、描述）
- `dev debug schema <name>`              查看工具的参数 JSON Schema
- `dev debug call <name> {"arg":"val"}`  手动调用工具测试（不经过 LLM）
- `dev debug prompt`                     查看当前完整 system prompt
- `dev test provider <name>`             测试 provider 连通性（发送 hello）
- `dev reload`                           热重载 custom-extensions.toml（无需重启）
- `dev log`                              查看工具注册统计

**扩展指南**（打印步骤 + 文件 + 代码片段；Layer-2，写真实 Python 模块）
- `dev provider`                         新增一个 LLM Provider（写 Python）
- `dev platform`                         新增一个云端 Platform 能力
- `dev tool`                             新增一个本地工具（写 Python）
- `dev skill`                            新增一个预置 Skill（写 Python）

**插件目录**
- `.acli/plugins/*.py` 文件启动时自动加载（可在其中注册 tool/skill）

> 多数新 LLM 是 OpenAI 兼容协议，用 Layer-1 (`dev provider add`) 即可，30 秒接入；
> 自定义 Skill 用 `dev skill add`，Shell 工具用 `dev tool add`，都不需要写代码；
> 协议异常的 / 想做复杂本地工具的，走 Layer-2 指南。
"""


_GUIDES = {
    "provider": _GUIDE_PROVIDER,
    "platform": _GUIDE_PLATFORM,
    "tool": _GUIDE_TOOL,
    "skill": _GUIDE_SKILL,
}


# ===== Runtime: extension provider / capability management =====

import getpass  # noqa: E402
import os  # noqa: E402


def _prompt(label: str, default: str = "", secret: bool = False) -> str:
    """Small input wrapper. Returns default when user just hits enter.
    secret=True suppresses echo (for ENC fallback paths only)."""
    hint = f" [{default}]" if default else ""
    if secret:
        val = getpass.getpass(f"  {label}{hint}: ").strip()
    else:
        val = input(f"  {label}{hint}: ").strip()
    return val or default


def _choose_target(default_global: bool = True):
    """Ask the user whether to write to global or workspace toml file."""
    from dashscope.acli.extensions import (
        GLOBAL_EXTENSIONS_FILE,
        WORKSPACE_EXTENSIONS_FILE,
    )

    console.print()
    console.print(
        f"  [1] 全局: {GLOBAL_EXTENSIONS_FILE}  [dim](跨 workspace 共享)[/dim]",
    )
    console.print(f"  [2] 当前 workspace: {WORKSPACE_EXTENSIONS_FILE}")
    raw = input("  写入位置 [1]: ").strip() or "1"
    return WORKSPACE_EXTENSIONS_FILE if raw == "2" else GLOBAL_EXTENSIONS_FILE


def _hot_reload(config: Config | None = None) -> None:
    """Re-apply extensions after a /dev write so the change is visible
    without restart. Mutates PROVIDER_MODELS in place, refreshes the
    extensions.current() singleton, AND folds new/removed extension caps
    into cli.CAPABILITY_CATALOG so /capability enable can find them
    without a restart. When config is given, enabled extension caps are
    re-registered so their HTTP-tool closures rebind to the new toml."""
    from dashscope.acli.extensions import apply_extensions

    ext = apply_extensions(PROVIDER_MODELS)
    # Lazy import to avoid the dev.py ↔ cli.py cycle at module load
    from dashscope.acli.cli import sync_extensions_into_catalog

    sync_extensions_into_catalog(ext)
    if config is not None:
        from dashscope.acli.tools.platform import (
            refresh_extension_capability_tools,
        )

        refresh_extension_capability_tools(config)


def _provider_add(config: Config) -> None:
    """Interactive flow that ends with a [[providers]] block in toml."""
    from dashscope.acli.extensions import (
        CustomProvider,
        append_provider,
        encrypt_for_toml,
        load_extensions,
    )

    console.print("\n[bold]新增 Provider[/bold]")
    name = _prompt("Provider 名 (如 dashscope)")
    if not name:
        console.print("[dim]已取消[/dim]")
        return
    existing = [p for p in load_extensions().providers if p.name == name]
    if existing or name in PROVIDER_MODELS:
        console.print(f"[red]名称冲突: {name} 已存在（内置或扩展）[/red]")
        return
    base_url = _prompt(
        "API base URL (如 https://dashscope.aliyuncs.com/compatible-mode/v1)",
    )
    if not base_url:
        console.print("[red]base_url 不可为空[/red]")
        return
    default_model = _prompt("默认模型 (如 qwen-max)")
    models_raw = _prompt("其他模型 (逗号分隔，可空)")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    protocol_raw = (
        _prompt("协议 (openai/anthropic/dashscope) [openai]").strip() or "openai"
    )
    protocol = protocol_raw.lower()
    if protocol not in ("openai", "anthropic", "dashscope"):
        console.print("[red]协议必须是 openai / anthropic / dashscope[/red]")
        return

    auth_choice = _prompt("需要 API Key? (y/n) [y]").strip().lower() or "y"
    needs_auth = auth_choice != "n"

    # API key handling — env preferred, ENC fallback, plaintext refused
    console.print()
    api_key_env = ""
    api_key_enc = ""
    if needs_auth:
        console.print("[bold]API Key 来源[/bold]:")
        console.print("  [1] 环境变量 (推荐，toml 只存 env var 名)")
        console.print("  [2] 加密嵌入 toml (XOR + 机器指纹，仅本机可解)")
        choice = input("  选择 [1]: ").strip() or "1"
        if choice == "2":
            secret = _prompt("API key (输入隐藏，加密后写入)", secret=True)
            if not secret:
                console.print("[red]空 key，已取消[/red]")
                return
            api_key_enc = encrypt_for_toml(secret)
        else:
            api_key_env = _prompt("环境变量名 (如 MOONSHOT_API_KEY)")
            if not api_key_env:
                console.print("[red]env var 名不可为空[/red]")
                return
            if not os.environ.get(api_key_env):
                console.print(
                    f"[yellow]提示: {api_key_env} 当前未导出，本会话不会生效，"
                    f"export 后重启或下次启动可用。[/yellow]",
                )

    target = _choose_target()
    p = CustomProvider(
        name=name,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key_enc=api_key_enc,
        default_model=default_model,
        models=models,
        protocol=protocol,
        auth=needs_auth,
    )
    append_provider(target, p)
    _hot_reload()
    console.print(f"\n[green]✓ provider {name} 已写入 {target}[/green]")
    console.print(
        f"[dim]立即可用: /provider 切换到 {name}/{default_model or '<model>'}[/dim]",
    )


def _provider_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.providers:
        console.print("[dim]当前未注册扩展 provider；可用 /dev provider add 添加[/dim]")
        return
    console.print("[bold]扩展 Provider:[/bold]")
    for p in ext.providers:
        auth = (
            f"env={p.api_key_env}"
            if p.api_key_env
            else "ENC(本机加密)"
            if p.api_key_enc
            else "?"
        )
        console.print(
            f"  • [cyan]{p.name}[/cyan] → {p.base_url} "
            f"[dim](default={p.default_model}, "
            f"protocol={p.resolved_protocol()}, "
            f"auth={auth}, source={p.source})[/dim]",
        )


def _provider_remove(name: str) -> None:
    from dashscope.acli.extensions import (
        GLOBAL_EXTENSIONS_FILE,
        WORKSPACE_EXTENSIONS_FILE,
        remove_provider,
    )

    removed = False
    for target in (WORKSPACE_EXTENSIONS_FILE, GLOBAL_EXTENSIONS_FILE):
        if remove_provider(target, name):
            console.print(f"[yellow]✗ 已从 {target} 删除 provider {name}[/yellow]")
            removed = True
    if not removed:
        console.print(f"[dim]未找到扩展 provider: {name}[/dim]")
        return
    _hot_reload()


def _capability_add(config: Config) -> None:
    """Scaffold a [[capabilities]] block in toml the user then edits in their
    editor — tool definitions are too complex for a smooth one-shot prompt."""
    from dashscope.acli.extensions import (
        append_capability_scaffold,
        load_extensions,
    )

    console.print("\n[bold]新增 Capability (HTTP tool group)[/bold]")
    console.print("[dim](会写入 toml 模板，编辑器里补全字段更顺手)[/dim]")
    key = _prompt("Capability key (vendor.feature 格式，如 dashscope.web)")
    if not key or "." not in key:
        console.print("[red]key 必须为 vendor.feature 格式[/red]")
        return
    if any(c.key == key for c in load_extensions().capabilities):
        console.print(f"[red]{key} 已存在[/red]")
        return
    display = _prompt("显示名 (可空，默认用 key)")
    target = _choose_target()
    append_capability_scaffold(target, key, display)
    _hot_reload(config)
    console.print(f"\n[green]✓ 模板已写入 {target}[/green]")
    console.print(
        f"[yellow]下一步:[/yellow] 用编辑器打开 {target} 修改 [[capabilities.tools]] "
        f"段填实际 endpoint/params/body_template，然后:\n"
        f"  acli> /capability enable {key}",
    )


def _capability_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.capabilities:
        console.print(
            "[dim]当前未注册扩展 capability；可用 /dev capability add 添加[/dim]",
        )
        return
    console.print("[bold]扩展 Capability:[/bold]")
    for c in ext.capabilities:
        console.print(
            f"  • [cyan]{c.key}[/cyan] — {c.display} "
            f"[dim]({len(c.tools)} 工具)[/dim]",
        )
        for t in c.tools:
            console.print(
                f"      [dim]· {t.name} → {t.http_method} {t.endpoint}[/dim]",
            )


def _capability_remove(key: str, config: Config) -> None:
    from dashscope.acli.extensions import (
        GLOBAL_EXTENSIONS_FILE,
        WORKSPACE_EXTENSIONS_FILE,
        remove_capability,
    )

    removed = False
    for target in (WORKSPACE_EXTENSIONS_FILE, GLOBAL_EXTENSIONS_FILE):
        if remove_capability(target, key):
            console.print(
                f"[yellow]✗ 已从 {target} 删除 capability {key}[/yellow]",
            )
            removed = True
    if not removed:
        console.print(f"[dim]未找到扩展 capability: {key}[/dim]")
        return
    _hot_reload(config)


# ===== Runtime: skill registration =====


def _skill_add() -> None:
    from dashscope.acli.extensions import (
        CustomSkill,
        append_skill,
        load_extensions,
    )
    from dashscope.acli.skills.base import BUILTIN_SKILLS, Skill, register

    console.print("\n[bold]新增 Skill (Prompt 模板)[/bold]")
    name = _prompt("Skill 名 (如 code-review)")
    if not name:
        console.print("[dim]已取消[/dim]")
        return
    existing = load_extensions()
    if name in BUILTIN_SKILLS or any(s.name == name for s in existing.skills):
        console.print(f"[red]名称冲突: {name} 已存在[/red]")
        return
    description = _prompt("描述 (一句话说明用途)")
    prompt_template = _prompt("Prompt 模板 (用 {arg} 做占位符)")
    if not prompt_template:
        console.print("[red]模板不可为空[/red]")
        return
    args_raw = _prompt("参数列表 (逗号分隔，如 city,lang；可空)")
    arguments = (
        [a.strip() for a in args_raw.split(",") if a.strip()]
        if args_raw
        else []
    )
    mcp_service = _prompt("依赖的 MCP 服务 (可空)")

    target = _choose_target()
    s = CustomSkill(
        name=name,
        description=description,
        prompt_template=prompt_template,
        arguments=arguments,
        mcp_service=mcp_service,
    )
    append_skill(target, s)
    # Register into runtime immediately
    register(
        Skill(
            name=name,
            description=description,
            mcp_service=mcp_service,
            prompt_template=prompt_template,
            arguments=arguments,
        ),
    )
    _hot_reload()
    console.print(f"\n[green]✓ skill {name} 已写入 {target}[/green]")
    console.print(
        f"[dim]立即可用: /skill {name} "
        f"{' '.join(f'<{a}>' for a in arguments)}[/dim]",
    )


def _skill_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.skills:
        console.print("[dim]当前未注册自定义 skill；可用 /dev skill add 添加[/dim]")
        return
    console.print("[bold]自定义 Skill:[/bold]")
    for s in ext.skills:
        args = " ".join(f"<{a}>" for a in s.arguments)
        mcp = f" [MCP: {s.mcp_service}]" if s.mcp_service else ""
        console.print(f"  • [cyan]{s.name}[/cyan] {args}{mcp}")
        console.print(f"    {s.description}")
        console.print(
            f"    [dim]模板: {s.prompt_template[:60]}"
            f"{'...' if len(s.prompt_template) > 60 else ''}[/dim]",
        )


def _skill_remove(name: str) -> None:
    from dashscope.acli.extensions import (
        GLOBAL_EXTENSIONS_FILE,
        WORKSPACE_EXTENSIONS_FILE,
        remove_skill,
    )

    removed = False
    for target in (WORKSPACE_EXTENSIONS_FILE, GLOBAL_EXTENSIONS_FILE):
        if remove_skill(target, name):
            console.print(f"[yellow]✗ 已从 {target} 删除 skill {name}[/yellow]")
            removed = True
    if not removed:
        console.print(f"[dim]未找到自定义 skill: {name}[/dim]")
        return
    _hot_reload()


# ===== Runtime: shell tool registration =====


def _tool_add() -> None:
    from dashscope.acli.extensions import (
        CustomShellTool,
        append_shell_tool,
        load_extensions,
    )

    console.print("\n[bold]新增 Shell 工具[/bold]")
    console.print("[dim](把一个 shell 命令包装成 LLM 可调用的工具)[/dim]")
    name = _prompt("工具名 (如 check_port)")
    if not name:
        console.print("[dim]已取消[/dim]")
        return
    existing = load_extensions()
    if any(t.name == name for t in existing.shell_tools):
        console.print(f"[red]名称冲突: {name} 已存在[/red]")
        return
    description = _prompt("描述 (告诉 LLM 何时调用)")
    command_template = _prompt("命令模板 (用 {{arg}} 做占位符，如 curl -s {{url}})")
    if not command_template:
        console.print("[red]命令模板不可为空[/red]")
        return

    # Parse params from template
    import re

    param_names = re.findall(r"\{\{(\w+)\}\}", command_template)
    params = []
    if param_names:
        console.print(f"[dim]检测到参数: {', '.join(param_names)}[/dim]")
        for pn in param_names:
            desc = _prompt(f"  参数 {pn} 的描述 (可空)")
            params.append(
                {
                    "name": pn,
                    "type": "string",
                    "required": True,
                    "description": desc or pn,
                },
            )

    console.print("\n[bold]权限级别[/bold]:")
    console.print("  [1] auto     — 自动执行，不问用户")
    console.print("  [2] confirm  — 执行前确认 (推荐)")
    console.print("  [3] dangerous — 双重确认")
    perm_choice = input("  选择 [2]: ").strip() or "2"
    perm_map = {"1": "auto", "2": "confirm", "3": "dangerous"}
    permission = perm_map.get(perm_choice, "confirm")

    target = _choose_target()
    t = CustomShellTool(
        name=name,
        description=description,
        command_template=command_template,
        params=params,
        permission=permission,
    )
    append_shell_tool(target, t)
    _hot_reload()
    console.print(f"\n[green]✓ shell tool {name} 已写入 {target}[/green]")
    console.print(
        f"[dim]LLM 可直接调用: {name}({', '.join(pn for pn in param_names)})[/dim]",
    )


def _tool_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.shell_tools:
        console.print("[dim]当前未注册自定义 shell tool；可用 /dev tool add 添加[/dim]")
        return
    console.print("[bold]自定义 Shell 工具:[/bold]")
    for t in ext.shell_tools:
        console.print(f"  • [cyan]{t.name}[/cyan] [{t.permission}]")
        console.print(f"    {t.description}")
        console.print(f"    [dim]命令: {t.command_template}[/dim]")


def _tool_remove(name: str) -> None:
    from dashscope.acli.extensions import (
        GLOBAL_EXTENSIONS_FILE,
        WORKSPACE_EXTENSIONS_FILE,
        remove_shell_tool,
    )

    removed = False
    for target in (WORKSPACE_EXTENSIONS_FILE, GLOBAL_EXTENSIONS_FILE):
        if remove_shell_tool(target, name):
            console.print(
                f"[yellow]✗ 已从 {target} 删除 shell tool {name}[/yellow]",
            )
            removed = True
    if not removed:
        console.print(f"[dim]未找到自定义 shell tool: {name}[/dim]")
        return
    _hot_reload()


# ===== Debug / Test / Reload =====


def _debug_tools() -> None:
    from dashscope.acli.tools.registry import registry

    tools = registry.list_tools()
    if not tools:
        console.print("[dim]无已注册工具[/dim]")
        return
    console.print(f"[bold]已注册工具 ({len(tools)} 个):[/bold]")
    for t in sorted(tools, key=lambda x: x.name):
        console.print(
            f"  [cyan]{t.name:30s}[/cyan] "
            f"[{t.permission.value:8s}]  {t.description[:50]}",
        )


def _debug_schema(name: str) -> None:
    import json as _json

    from dashscope.acli.tools.registry import registry

    tool = registry.get(name)
    if not tool:
        console.print(f"[red]工具 {name} 不存在[/red]")
        return
    console.print(f"[bold]{tool.name}[/bold] [{tool.permission.value}]")
    console.print(f"  {tool.description}")
    console.print("\n[bold]Parameters JSON Schema:[/bold]")
    console.print(_json.dumps(tool.parameters, ensure_ascii=False, indent=2))


def _debug_prompt(config: Config) -> None:
    from dashscope.acli.agent import SYSTEM_PROMPT
    from dashscope.acli.skills import skills_summary_for_llm
    from dashscope.acli.tools.platform import disabled_capabilities_hint

    prompt = SYSTEM_PROMPT
    skills = skills_summary_for_llm(set())
    if skills:
        prompt += "\n\n可用 Skill 模板:\n" + skills
    disabled = disabled_capabilities_hint(config)
    if disabled:
        prompt += disabled
    if config.user_directives:
        prompt += "\n\n## 用户长效操作规则\n"
        for i, r in enumerate(config.user_directives, 1):
            prompt += f"{i}. {r}\n"

    console.print(f"[bold]当前 System Prompt ({len(prompt)} 字符):[/bold]\n")
    console.print(prompt)


async def _debug_call(cmd_parts: list[str]) -> None:
    import json as _json

    from dashscope.acli.tools.registry import registry

    if len(cmd_parts) < 4:
        console.print(
            '[dim]用法: /dev debug call <tool_name> {"arg": "val"}[/dim]',
        )
        return
    name = cmd_parts[3]
    tool = registry.get(name)
    if not tool:
        console.print(f"[red]工具 {name} 不存在[/red]")
        return
    args_str = " ".join(cmd_parts[4:]) if len(cmd_parts) > 4 else "{}"
    try:
        kwargs = _json.loads(args_str)
    except _json.JSONDecodeError as e:
        console.print(f"[red]参数 JSON 解析失败: {e}[/red]")
        return
    console.print(f"[dim]调用 {name}({kwargs})...[/dim]")
    try:
        import asyncio

        if asyncio.iscoroutinefunction(tool.func):
            result = await tool.func(**kwargs)
        else:
            result = tool.func(**kwargs)
        console.print(f"[green]结果:[/green]\n{result}")
    except Exception as e:
        console.print(f"[red]调用失败: {type(e).__name__}: {e}[/red]")


async def _test_provider(name: str, config: Config) -> None:
    import copy as _copy

    from dashscope.acli.extensions import find_provider
    from dashscope.acli.providers import (
        _create_provider,
        build_profiles_from_config,
    )

    console.print(f"[dim]测试 provider {name}...[/dim]")
    try:
        ext = find_provider(name)
        if ext is not None:
            model = PROVIDER_MODELS.get(name, [""])[0] or (
                ext.default_model or ext.resolved_models()[0]
            )
        else:
            model = PROVIDER_MODELS.get(name, [""])[0]

        # Resolve the provider exactly like the runtime does: probe config
        # with `name` as primary, then the standard profile pipeline (which
        # folds in extension-toml protocol / base_url / encrypted keys).
        # base_url / protocol describe the session's primary provider, so
        # clear them to let `name` resolve its own.
        probe = _copy.copy(config)
        probe.provider = name
        probe.model = model
        probe.base_url = ""
        probe.protocol = ""
        profiles = build_profiles_from_config(probe)
        profile = profiles[0] if profiles else None
        api_key = profile.api_key if profile else ""
        model = profile.model if profile else model

        if not model:
            console.print(f"[red]✗ {name} 没有配置模型[/red]")
            return
        if not api_key:
            if ext is not None:
                env_hint = ext.api_key_env or "(无)"
                console.print(
                    f"[red]✗ {name} 没有配置 API Key[/red]\n"
                    f"  [dim]已检查: ~/.acli/config.toml 的 {name}_api_key、"
                    f"环境变量 {env_hint}、custom-extensions.toml 加密 key[/dim]",
                )
            else:
                console.print(f"[red]✗ {name} 没有配置 API Key[/red]")
            return

        provider = _create_provider(profile)
        messages = [
            {"role": "system", "content": "回复 OK 即可"},
            {"role": "user", "content": "hello"},
        ]
        resp = await provider.chat(messages, tools=[])
        console.print(f"[green]✓ {name} 连通正常[/green]")
        console.print(f"  响应: {resp.content[:100]}")
        if resp.usage:
            console.print(
                f"  Token: input={resp.usage.get('input_tokens', '?')}, "
                f"output={resp.usage.get('output_tokens', '?')}",
            )
    except Exception as e:
        console.print(f"[red]✗ {name} 连接失败: {type(e).__name__}: {e}[/red]")


def _dev_reload(config: Config) -> None:
    """Re-apply extensions from toml files without restart."""
    _hot_reload(config)
    console.print("[green]✓ 扩展已重新加载[/green]")
    from dashscope.acli.extensions import current

    ext = current()
    console.print(
        f"  providers: {len(ext.providers)}, "
        f"capabilities: {len(ext.capabilities)}, "
        f"skills: {len(ext.skills)}, shell_tools: {len(ext.shell_tools)}",
    )


def _dev_log(config: Config) -> None:
    """Show recent API call log from executor stats."""
    from dashscope.acli.tools.registry import registry

    console.print("[bold]最近 API 调用日志:[/bold]")
    console.print(f"  Provider: {config.provider}")
    console.print(f"  Model: {config.model}")
    console.print("\n[bold]已注册工具统计:[/bold]")
    tools = registry.list_tools()
    by_perm = {}
    for t in tools:
        by_perm.setdefault(t.permission.value, []).append(t.name)
    for perm, names in sorted(by_perm.items()):
        console.print(f"  [{perm}] {len(names)} 个工具")
    console.print("\n[dim]详细 token 统计请使用 /stats 命令[/dim]")


# ===== Entry =====


def handle_dev_command(cmd: str, config: Config) -> None:
    parts = cmd.strip().split()
    if len(parts) <= 1:
        console.print(Markdown(_GUIDE_INDEX))
        return

    sub = parts[1]

    if sub == "model":
        action = parts[2] if len(parts) > 2 else "list"
        if action == "list":
            _model_list(config)
        elif action == "add":
            if len(parts) >= 5:
                _model_add(config, parts[3], parts[4])
            elif len(parts) == 4:
                # Allow shorthand: /dev model add glm-image
                model = parts[3]
                provider = _infer_provider_from_model(model)
                if provider is None:
                    console.print(
                        f"[red]无法从模型名 '{model}' 推断 Provider，"
                        f"请使用完整语法: /dev model add <provider> <name>[/red]",
                    )
                    console.print(
                        "[dim]可识别前缀: "
                        f"{', '.join(_MODEL_PROVIDER_HINTS.keys())}[/dim]",
                    )
                    return
                _model_add(config, provider, model)
            else:
                console.print(
                    "[dim]用法:\n"
                    "  /dev model list                         — 列出可选模型\n"
                    "  /dev model add <provider> <name>        — 注册新模型\n"
                    "  /dev model add <name>                   — "
                    "简写，按模型名前缀推断 provider\n"
                    "  /dev model remove <provider> <name>     — "
                    "移除自定义模型[/dim]",
                )
        elif action in ("remove", "rm") and len(parts) >= 5:
            _model_remove(config, parts[3], parts[4])
        else:
            console.print(
                "[dim]用法:\n"
                "  /dev model list                         — 列出可选模型\n"
                "  /dev model add <provider> <name>        — 注册新模型\n"
                "  /dev model add <name>                   — "
                "简写，按模型名前缀推断 provider\n"
                "  /dev model remove <provider> <name>     — 移除自定义模型[/dim]",
            )
        return

    # provider/capability: action form first ("add"/"list"/"rm"), then
    # bare form falls through to the docs guide.
    if sub == "provider" and len(parts) >= 3:
        action = parts[2]
        if action == "add":
            _provider_add(config)
        elif action == "list":
            _provider_list()
        elif action in ("remove", "rm") and len(parts) >= 4:
            _provider_remove(parts[3])
        else:
            console.print(
                "[dim]用法:\n"
                "  /dev provider add          — 交互式新增扩展 provider\n"
                "  /dev provider list         — 列出扩展 provider\n"
                "  /dev provider remove <name>    — 删除扩展 provider[/dim]",
            )
        return

    if sub == "capability" and len(parts) >= 3:
        action = parts[2]
        if action == "add":
            _capability_add(config)
        elif action == "list":
            _capability_list()
        elif action in ("remove", "rm") and len(parts) >= 4:
            _capability_remove(parts[3], config)
        else:
            console.print(
                "[dim]用法:\n"
                "  /dev capability add        — 写入 capability scaffold 模板\n"
                "  /dev capability list       — 列出扩展 capability\n"
                "  /dev capability remove <key>   — 删除扩展 capability[/dim]",
            )
        return

    if sub == "skill" and len(parts) >= 3:
        action = parts[2]
        if action == "add":
            _skill_add()
        elif action == "list":
            _skill_list()
        elif action in ("remove", "rm") and len(parts) >= 4:
            _skill_remove(parts[3])
        else:
            console.print(
                "[dim]用法:\n"
                "  /dev skill add             — 交互式新增自定义 Skill\n"
                "  /dev skill list            — 列出自定义 Skill\n"
                "  /dev skill remove <name>       — 删除自定义 Skill[/dim]",
            )
        return

    if sub == "tool" and len(parts) >= 3:
        action = parts[2]
        if action == "add":
            _tool_add()
        elif action == "list":
            _tool_list()
        elif action in ("remove", "rm") and len(parts) >= 4:
            _tool_remove(parts[3])
        else:
            console.print(
                "[dim]用法:\n"
                "  /dev tool add              — 交互式新增 Shell 工具\n"
                "  /dev tool list             — 列出自定义 Shell 工具\n"
                "  /dev tool remove <name>        — 删除自定义 Shell 工具[/dim]",
            )
        return

    if sub == "debug":
        action = parts[2] if len(parts) > 2 else "tools"
        if action == "tools":
            _debug_tools()
        elif action == "schema" and len(parts) >= 4:
            _debug_schema(parts[3])
        elif action == "call":
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop (script/worker-thread context): safe to
                # drive the coroutine with a private loop.
                asyncio.run(_debug_call(parts))
            else:
                # Inside the REPL/TUI loop: nesting a second loop crashes, so
                # schedule on the running one instead.
                loop.create_task(_debug_call(parts))
        elif action == "prompt":
            _debug_prompt(config)
        else:
            console.print(
                "[dim]用法:\n"
                "  /dev debug tools           — 列出所有已注册工具\n"
                "  /dev debug schema <name>   — 查看工具的 JSON Schema\n"
                "  /dev debug call <name> {}  — 手动调用工具测试\n"
                "  /dev debug prompt          — 查看当前 system prompt[/dim]",
            )
        return

    if sub == "test":
        if len(parts) >= 4 and parts[2] == "provider":
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop (script/worker-thread context): safe to
                # drive the coroutine with a private loop.
                asyncio.run(_test_provider(parts[3], config))
            else:
                # Inside the REPL/TUI loop: run_until_complete on a running
                # loop crashes, so schedule on the running one instead.
                loop.create_task(_test_provider(parts[3], config))
        else:
            console.print(
                "[dim]用法:\n"
                "  /dev test provider <name>  — 测试 provider 连通性[/dim]",
            )
        return

    if sub == "reload":
        _dev_reload(config)
        return

    if sub == "log":
        _dev_log(config)
        return

    if sub in _GUIDES:
        console.print(Markdown(_GUIDES[sub]))
        return

    console.print(f"[red]未知 /dev 子命令: {sub}[/red]")
    console.print(Markdown(_GUIDE_INDEX))
