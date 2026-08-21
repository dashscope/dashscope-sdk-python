# -*- coding: utf-8 -*-
"""Developer / extension commands.

`/dev` provides two kinds of capabilities:

1. **Runtime operations** (take effect immediately and persist to the
   workspace config)
   - `model list`              list selectable models for all providers
   - `model add <p> <m>`       register a new model for a provider
     (persisted to the workspace's custom_models)
   - `model remove  <p> <m>`   remove a custom model

2. **Extension guides** (print Markdown steps showing which files to
   edit and what code to add)
   - `provider`                add an LLM Provider
   - `platform`                add a cloud Platform capability
   - `tool`                    add a local tool
   - `skill`                   add a preset Skill
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
    lines = ["## Available Models", "", "| Provider | Model |", "|---|---|"]
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
        console.print(f"[red]Unknown provider: {provider}[/red]")
        console.print(
            f"[dim]Available: {', '.join(PROVIDER_MODELS.keys())}[/dim]",
        )
        return
    normalized = normalize_model_name(model)
    if normalized != model.strip():
        console.print(
            f"[dim]Model name normalized to {normalized} "
            f"(API model IDs are case-sensitive)[/dim]",
        )
    if normalized in {m.lower() for m in PROVIDER_MODELS[provider]}:
        console.print(f"[dim]{provider}/{normalized} already exists[/dim]")
        return
    register_custom_model(config, provider, normalized)
    console.print(f"[green]✓ Registered {provider}/{normalized}[/green]")
    console.print("[dim]Switch to it: /provider[/dim]")


def _model_remove(config: Config, provider: str, model: str) -> None:
    # add stores the normalized lowercase name, so remove must normalize
    # the same way to hit it.
    model = normalize_model_name(model)
    entry = f"{provider}:{model}"
    if entry not in (config.custom_models or []):
        console.print(
            f"[yellow]{entry} is not a custom model "
            f"(built-ins cannot be removed)[/yellow]",
        )
        return
    config.custom_models.remove(entry)
    if provider in PROVIDER_MODELS and model in PROVIDER_MODELS[provider]:
        PROVIDER_MODELS[provider].remove(model)
    config.save_workspace()
    console.print(f"[yellow]✗ Removed {provider}/{model}[/yellow]")


def _model_list(config: Config) -> None:
    console.print(Markdown(_format_models()))
    if config.custom_models:
        console.print()
        console.print(
            f"[dim]Workspace custom: {', '.join(config.custom_models)}[/dim]",
        )


# ===== Guides =====


_GUIDE_PROVIDER = """\
## Add an LLM Provider

> Wire a new LLM into acli's chat / stream / tool-call loop.
> In most scenarios **no code is needed** — just fill in a TOML block.

acli ships only 3 protocol implementations; every provider (including
built-ins tongyi/anthropic/openai/deepseek/zhipu/ideatalk/ollama) is
configured via `custom-extensions.toml`:

| Protocol    | Implementation      | Use case                        |
|-------------|---------------------|---------------------------------|
| `openai`    | `OpenAIProvider`    | OpenAI-compatible endpoints     |
|             |                     | (Moonshot/Yi/Step/Deepseek/     |
|             |                     | Zhipu/Ollama…)                  |
| `anthropic` | `AnthropicProvider` | Anthropic Messages API (Claude  |
|             |                     | / proxied endpoints)            |
| `dashscope` | `TongyiProvider`    | DashScope OpenAI-compat         |
|             |                     | endpoint (Qwen)                 |

**Steps** (Layer-1, recommended)

1. Run `/dev provider add` and fill in name / base_url / api_key /
   model / protocol interactively
2. Or manually add a block to `~/.acli/custom-extensions.toml`
   (global) or `./.acli/custom-extensions.toml` (workspace):

```toml
# Moonshot / Kimi — OpenAI compatible
[[providers]]
name = "moonshot"
base_url = "https://api.moonshot.cn/v1"
api_key_env = "MOONSHOT_API_KEY"
default_model = "kimi-k2"
models = ["kimi-k1"]
protocol = "openai"

# Access Qwen via an Anthropic-protocol proxy
[[providers]]
name = "bailian-anthropic"
base_url = "https://example.com/apps/anthropic"
api_key_env = "DASHSCOPE_API_KEY"
default_model = "qwen3.7-max"
protocol = "anthropic"

# Local Ollama, no auth
[[providers]]
name = "ollama"
base_url = "http://localhost:11434/v1"
default_model = "llama3"
protocol = "openai"
auth = false
```

**Steps** (Layer-2, only when the protocol is not in the table above)

Only needed when the new LLM speaks a non-standard protocol (not
OpenAI/Anthropic/DashScope compatible): implement the full
`LLMProvider` protocol in `src/acli/providers/<name>.py` (`chat` /
`stream_chat` + tool-call passthrough, see `providers/base.py`), then
add an `if proto == "<name>"` branch to `_create_provider` in
`providers/__init__.py`.

**Verify**

```
acli> /provider   # interactively switch Provider / write API Key
                  # (auth=false providers don't need one)
acli> hello       # trigger a chat
```
"""


_GUIDE_PLATFORM = """\
## Add a Platform Capability

> Platform = cloud SDK wrapper layer. One Platform can expose multiple
> capabilities (e.g. bailian → memory + mcp).

**Steps**

1. Create a package under `src/acli/platforms/<vendor>/` with a client
   class implementing one of the Protocols in `platforms/base.py`
   (`MemoryProvider` / `KBProvider` / `SearchProvider` /
   `ContextProvider` / `DataProvider` / `PromptProvider`)
   - For a brand-new capability category, first add a new Protocol +
     dataclass to `platforms/base.py`
2. Add a `get_<cap>_provider(config)` factory in
   `src/acli/platforms/__init__.py`; return `None` when credentials
   are missing
3. Add a line to `CAPABILITY_CATALOG` in `src/acli/cli.py`:
   ```python
   {"key": "<vendor>.<cap>", "name": "...", "platform": "<vendor>",
   "cap": "<cap>", "requires": ["..."]}
   ```
4. Add a branch in `register_platform_tools()` in
   `src/acli/tools/platform.py` that wraps the client into tools and
   registers them with `registry` when the capability is enabled
5. If you need a `/<cap>` subcommand, add `_handle_<cap>_command` in
   `cli.py` and mount it in the `_handle_slash_command` + `_run_loop`
   routing (remember the `_require_capability` gate)
6. To be synced by `/update`, register a `_sync_<vendor>` in `cli.py`
   `_get_update_targets()`

**Verify**

```
acli> /capability enable <vendor>.<cap>
acli> /update <vendor>
acli> /<cap> list                # via its own subcommand
acli> look up ... for me         # LLM picks the new tool itself
```
"""


_GUIDE_TOOL = """\
## Add a Local Tool

> Local tool = a capability the LLM can call directly in the agent
> loop, with no cloud dependency.

**Steps**

1. Write a plain `async`/sync function in `src/acli/tools/<name>.py`
   with the `@tool(...)` decorator:
   ```python
   from dashscope.acli.tools.registry import PermissionLevel, tool

   @tool(
       name="my_tool",
       description="one line telling the LLM what it does and when",
       permission=PermissionLevel.AUTO,  # AUTO / CONFIRM / DANGEROUS
   )
   def my_tool(path: str, limit: int = 10) -> str:
       ...
   ```
   - The parameter schema is auto-derived from type annotations by
     `_build_parameters_schema` (`str/int/float/bool/list/dict` +
     `Optional`)
   - Return a string as the tool result fed back to the LLM
2. Add `import acli.tools.<name>` at the top of `src/acli/cli.py` to
   trigger registration (see filesystem/shell for the pattern)
3. Write a good description — it is the ONLY signal the LLM has for
   deciding whether to call your tool; spell out **when to call / when
   not to call / inputs & outputs**

**Permission levels**

- `AUTO`      read ops / idempotent queries, run directly
- `CONFIRM`   reversible writes (write file, upload), confirm each time
- `DANGEROUS` irreversible (delete, `rm -rf`), double warning

**Verify**

```
acli> use my_tool to process X     # does the LLM pick it on its own?
```
"""


_GUIDE_SKILL = """\
## Add a Skill

> Skill = preset Prompt template. `/skill <name> <args...>` replaces
> hand-typing a prompt; may optionally depend on an MCP service.

**Steps**

1. Register in `src/acli/skills/<name>.py`:
   ```python
   from dashscope.acli.skills.base import Skill, register

   register(Skill(
       name="my-skill",
       description="one line on what this Skill does",
       mcp_service="",  # MCP service name if required; auto-connects
       prompt_template="please... {arg1} ... {arg2}",  # placeholders
       arguments=["arg1", "arg2"],  # last greedily eats the rest
   ))
   ```
2. Append your module name to the `from acli.skills import ...` line
   in `src/acli/skills/__init__.py` to trigger the registration side
   effect
3. Call it: `/skill my-skill val1 val2`; with no args it prints usage

**Depending on an MCP service**

- With a name like `mcp_service="time"` filled in, `/skill` tries to
  auto-connect first; to make it visible under `/mcp services`, add a
  description to `KNOWN_MCP_SERVICES` in `skills/base.py`
"""


_GUIDE_INDEX = """\
## /dev — developer / extension guide

**Runtime (takes effect immediately)**
- `dev model list`                        list models per provider
- `dev model add <p> <m>`                 register a new model for a
  provider (persisted to workspace)
- `dev model remove  <p> <m>`             remove a custom model

**Layer-1 extensions: config as program** (written to
`custom-extensions.toml`, no code changes needed)
- `dev provider add`                      interactively register an
  extension LLM Provider (OpenAI compatible)
- `dev provider list` / `remove <name>`   view / delete
- `dev capability add`                    write a capability scaffold
  template for editing
- `dev capability list` / `remove <key>`  view / delete
- `dev skill add`                         interactively add a custom
  Skill (Prompt template)
- `dev skill list` / `remove <name>`      view / delete
- `dev tool add`                          interactively add a Shell
  tool (wrap a command as an LLM tool)
- `dev tool list` / `remove <name>`       view / delete

**Debug / test**
- `dev debug tools`                       list all registered tools
  (name, permission, description)
- `dev debug schema <name>`               show a tool's parameter
  JSON Schema
- `dev debug call <name> {"arg":"val"}`   invoke a tool manually (no
  LLM)
- `dev debug prompt`                      show the current full system
  prompt
- `dev test provider <name>`              test provider connectivity
  (sends hello)
- `dev reload`                            hot-reload
  custom-extensions.toml (no restart)
- `dev log`                               show tool registration
  stats

**Extension guides** (print steps + files + code snippets; Layer-2,
real Python modules)
- `dev provider`                          add an LLM Provider (write
  Python)
- `dev platform`                          add a cloud Platform
  capability
- `dev tool`                              add a local tool (write
  Python)
- `dev skill`                             add a preset Skill (write
  Python)

**Plugin directory**
- `.acli/plugins/*.py` files auto-load at startup (can register
  tools/skills)

> Most new LLMs speak an OpenAI-compatible protocol — Layer-1
> (`dev provider add`) gets you connected in 30 seconds; custom Skills
> use `dev skill add`, Shell tools use `dev tool add`, no code needed;
> for unusual protocols or complex local tools, follow the Layer-2
> guides.
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
        f"  [1] Global: {GLOBAL_EXTENSIONS_FILE}  "
        f"[dim](shared across workspaces)[/dim]",
    )
    console.print(f"  [2] Current workspace: {WORKSPACE_EXTENSIONS_FILE}")
    raw = input("  Write to [1]: ").strip() or "1"
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

    console.print("\n[bold]Add Provider[/bold]")
    name = _prompt("Provider name (e.g. dashscope)")
    if not name:
        console.print("[dim]Cancelled[/dim]")
        return
    existing = [p for p in load_extensions().providers if p.name == name]
    if existing or name in PROVIDER_MODELS:
        console.print(
            f"[red]Name conflict: {name} already exists "
            f"(built-in or extension)[/red]",
        )
        return
    base_url = _prompt(
        "API base URL "
        "(e.g. https://dashscope.aliyuncs.com/compatible-mode/v1)",
    )
    if not base_url:
        console.print("[red]base_url must not be empty[/red]")
        return
    default_model = _prompt("Default model (e.g. qwen-max)")
    models_raw = _prompt("Other models (comma-separated, optional)")
    models = [m.strip() for m in models_raw.split(",") if m.strip()]
    protocol_raw = (
        _prompt("Protocol (openai/anthropic/dashscope) [openai]").strip()
        or "openai"
    )
    protocol = protocol_raw.lower()
    if protocol not in ("openai", "anthropic", "dashscope"):
        console.print(
            "[red]Protocol must be openai / anthropic / dashscope[/red]",
        )
        return

    auth_choice = _prompt("Requires API Key? (y/n) [y]").strip().lower() or "y"
    needs_auth = auth_choice != "n"

    # API key handling — env preferred, ENC fallback, plaintext refused
    console.print()
    api_key_env = ""
    api_key_enc = ""
    if needs_auth:
        console.print("[bold]API Key source[/bold]:")
        console.print("  [1] Env var (recommended; toml stores the name)")
        console.print(
            "  [2] Encrypted into toml (XOR + machine fingerprint, "
            "this machine only)",
        )
        choice = input("  Choose [1]: ").strip() or "1"
        if choice == "2":
            secret = _prompt(
                "API key (hidden input, stored encrypted)",
                secret=True,
            )
            if not secret:
                console.print("[red]Empty key, cancelled[/red]")
                return
            api_key_enc = encrypt_for_toml(secret)
        else:
            api_key_env = _prompt("Env var name (e.g. MOONSHOT_API_KEY)")
            if not api_key_env:
                console.print("[red]Env var name must not be empty[/red]")
                return
            if not os.environ.get(api_key_env):
                console.print(
                    f"[yellow]Note: {api_key_env} is not exported, so "
                    f"it won't work in this session; export it and "
                    f"restart, or it will apply on next launch.[/yellow]",
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
    console.print(f"\n[green]✓ Provider {name} written to {target}[/green]")
    console.print(
        f"[dim]Ready now: /provider switch to "
        f"{name}/{default_model or '<model>'}[/dim]",
    )


def _provider_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.providers:
        console.print(
            "[dim]No extension providers registered; "
            "use /dev provider add[/dim]",
        )
        return
    console.print("[bold]Extension providers:[/bold]")
    for p in ext.providers:
        auth = (
            f"env={p.api_key_env}"
            if p.api_key_env
            else "ENC(machine-local)"
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
            console.print(
                f"[yellow]✗ Removed provider {name} from {target}[/yellow]",
            )
            removed = True
    if not removed:
        console.print(f"[dim]Extension provider not found: {name}[/dim]")
        return
    _hot_reload()


def _capability_add(config: Config) -> None:
    """Scaffold a [[capabilities]] block in toml the user then edits in their
    editor — tool definitions are too complex for a smooth one-shot prompt."""
    from dashscope.acli.extensions import (
        append_capability_scaffold,
        load_extensions,
    )

    console.print("\n[bold]Add Capability (HTTP tool group)[/bold]")
    console.print(
        "[dim](writes a toml template; fill in fields in your editor)[/dim]",
    )
    key = _prompt("Capability key (vendor.feature, e.g. dashscope.web)")
    if not key or "." not in key:
        console.print("[red]key must be in vendor.feature format[/red]")
        return
    if any(c.key == key for c in load_extensions().capabilities):
        console.print(f"[red]{key} already exists[/red]")
        return
    display = _prompt("Display name (optional, defaults to key)")
    target = _choose_target()
    append_capability_scaffold(target, key, display)
    _hot_reload(config)
    console.print(f"\n[green]✓ Template written to {target}[/green]")
    console.print(
        f"[yellow]Next:[/yellow] open {target} in your editor and fill "
        f"in the [[capabilities.tools]] section with the real "
        f"endpoint/params/body_template, then:\n"
        f"  acli> /capability enable {key}",
    )


def _capability_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.capabilities:
        console.print(
            "[dim]No extension capabilities registered; "
            "use /dev capability add[/dim]",
        )
        return
    console.print("[bold]Extension capabilities:[/bold]")
    for c in ext.capabilities:
        console.print(
            f"  • [cyan]{c.key}[/cyan] — {c.display} "
            f"[dim]({len(c.tools)} tools)[/dim]",
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
                f"[yellow]✗ Removed capability {key} from {target}[/yellow]",
            )
            removed = True
    if not removed:
        console.print(f"[dim]Extension capability not found: {key}[/dim]")
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

    console.print("\n[bold]Add Skill (Prompt template)[/bold]")
    name = _prompt("Skill name (e.g. code-review)")
    if not name:
        console.print("[dim]Cancelled[/dim]")
        return
    existing = load_extensions()
    if name in BUILTIN_SKILLS or any(s.name == name for s in existing.skills):
        console.print(f"[red]Name conflict: {name} already exists[/red]")
        return
    description = _prompt("Description (one line on what it's for)")
    prompt_template = _prompt("Prompt template (use {arg} placeholders)")
    if not prompt_template:
        console.print("[red]Template must not be empty[/red]")
        return
    args_raw = _prompt("Arguments (comma-separated, e.g. city,lang; optional)")
    arguments = (
        [a.strip() for a in args_raw.split(",") if a.strip()]
        if args_raw
        else []
    )
    mcp_service = _prompt("Required MCP service (optional)")

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
    console.print(f"\n[green]✓ Skill {name} written to {target}[/green]")
    console.print(
        f"[dim]Ready now: /skill {name} "
        f"{' '.join(f'<{a}>' for a in arguments)}[/dim]",
    )


def _skill_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.skills:
        console.print(
            "[dim]No custom skills registered; use /dev skill add[/dim]",
        )
        return
    console.print("[bold]Custom skills:[/bold]")
    for s in ext.skills:
        args = " ".join(f"<{a}>" for a in s.arguments)
        mcp = f" [MCP: {s.mcp_service}]" if s.mcp_service else ""
        console.print(f"  • [cyan]{s.name}[/cyan] {args}{mcp}")
        console.print(f"    {s.description}")
        console.print(
            f"    [dim]Template: {s.prompt_template[:60]}"
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
            console.print(
                f"[yellow]✗ Removed skill {name} from {target}[/yellow]",
            )
            removed = True
    if not removed:
        console.print(f"[dim]Custom skill not found: {name}[/dim]")
        return
    _hot_reload()


# ===== Runtime: shell tool registration =====


def _tool_add() -> None:
    from dashscope.acli.extensions import (
        CustomShellTool,
        append_shell_tool,
        load_extensions,
    )

    console.print("\n[bold]Add Shell Tool[/bold]")
    console.print(
        "[dim](wraps a shell command as an LLM-callable tool)[/dim]",
    )
    name = _prompt("Tool name (e.g. check_port)")
    if not name:
        console.print("[dim]Cancelled[/dim]")
        return
    existing = load_extensions()
    if any(t.name == name for t in existing.shell_tools):
        console.print(f"[red]Name conflict: {name} already exists[/red]")
        return
    description = _prompt("Description (tells the LLM when to call it)")
    command_template = _prompt(
        "Command template (use {{arg}} placeholders, e.g. curl -s {{url}})",
    )
    if not command_template:
        console.print("[red]Command template must not be empty[/red]")
        return

    # Parse params from template
    import re

    param_names = re.findall(r"\{\{(\w+)\}\}", command_template)
    params = []
    if param_names:
        console.print(f"[dim]Detected params: {', '.join(param_names)}[/dim]")
        for pn in param_names:
            desc = _prompt(f"  Description of param {pn} (optional)")
            params.append(
                {
                    "name": pn,
                    "type": "string",
                    "required": True,
                    "description": desc or pn,
                },
            )

    console.print("\n[bold]Permission level[/bold]:")
    console.print("  [1] auto      — run without asking the user")
    console.print("  [2] confirm   — confirm before running (recommended)")
    console.print("  [3] dangerous — double confirmation")
    perm_choice = input("  Choose [2]: ").strip() or "2"
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
    console.print(f"\n[green]✓ Shell tool {name} written to {target}[/green]")
    console.print(
        f"[dim]LLM can call directly: "
        f"{name}({', '.join(pn for pn in param_names)})[/dim]",
    )


def _tool_list() -> None:
    from dashscope.acli.extensions import current

    ext = current()
    if not ext.shell_tools:
        console.print(
            "[dim]No custom shell tools registered; "
            "use /dev tool add[/dim]",
        )
        return
    console.print("[bold]Custom shell tools:[/bold]")
    for t in ext.shell_tools:
        console.print(f"  • [cyan]{t.name}[/cyan] [{t.permission}]")
        console.print(f"    {t.description}")
        console.print(f"    [dim]Command: {t.command_template}[/dim]")


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
                f"[yellow]✗ Removed shell tool {name} from {target}[/yellow]",
            )
            removed = True
    if not removed:
        console.print(f"[dim]Custom shell tool not found: {name}[/dim]")
        return
    _hot_reload()


# ===== Debug / Test / Reload =====


def _debug_tools() -> None:
    from dashscope.acli.tools.registry import registry

    tools = registry.list_tools()
    if not tools:
        console.print("[dim]No tools registered[/dim]")
        return
    console.print(f"[bold]Registered tools ({len(tools)}):[/bold]")
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
        console.print(f"[red]Tool {name} does not exist[/red]")
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
        prompt += "\n\nAvailable Skill templates:\n" + skills
    disabled = disabled_capabilities_hint(config)
    if disabled:
        prompt += disabled
    if config.user_directives:
        prompt += "\n\n## Long-term user directives\n"
        for i, r in enumerate(config.user_directives, 1):
            prompt += f"{i}. {r}\n"

    console.print(
        f"[bold]Current system prompt ({len(prompt)} chars):[/bold]\n",
    )
    console.print(prompt)


async def _debug_call(cmd_parts: list[str]) -> None:
    import json as _json

    from dashscope.acli.tools.registry import registry

    if len(cmd_parts) < 4:
        console.print(
            '[dim]Usage: /dev debug call <tool_name> {"arg": "val"}[/dim]',
        )
        return
    name = cmd_parts[3]
    tool = registry.get(name)
    if not tool:
        console.print(f"[red]Tool {name} does not exist[/red]")
        return
    args_str = " ".join(cmd_parts[4:]) if len(cmd_parts) > 4 else "{}"
    try:
        kwargs = _json.loads(args_str)
    except _json.JSONDecodeError as e:
        console.print(f"[red]Failed to parse args JSON: {e}[/red]")
        return
    console.print(f"[dim]Calling {name}({kwargs})...[/dim]")
    try:
        import asyncio

        if asyncio.iscoroutinefunction(tool.func):
            result = await tool.func(**kwargs)
        else:
            result = tool.func(**kwargs)
        console.print(f"[green]Result:[/green]\n{result}")
    except Exception as e:
        console.print(f"[red]Call failed: {type(e).__name__}: {e}[/red]")


async def _test_provider(name: str, config: Config) -> None:
    import copy as _copy

    from dashscope.acli.extensions import find_provider
    from dashscope.acli.providers import (
        _create_provider,
        build_profiles_from_config,
    )

    console.print(f"[dim]Testing provider {name}...[/dim]")
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
            console.print(f"[red]✗ {name} has no model configured[/red]")
            return
        if not api_key:
            if ext is not None:
                env_hint = ext.api_key_env or "(none)"
                console.print(
                    f"[red]✗ {name} has no API Key configured[/red]\n"
                    f"  [dim]Checked: {name}_api_key in "
                    f"~/.acli/config.toml, env var {env_hint}, "
                    f"encrypted key in custom-extensions.toml[/dim]",
                )
            else:
                console.print(
                    f"[red]✗ {name} has no API Key configured[/red]",
                )
            return

        provider = _create_provider(profile)
        messages = [
            {"role": "system", "content": "Reply with OK"},
            {"role": "user", "content": "hello"},
        ]
        resp = await provider.chat(messages, tools=[])
        console.print(f"[green]✓ {name} is reachable[/green]")
        console.print(f"  Response: {resp.content[:100]}")
        if resp.usage:
            console.print(
                f"  Token: input={resp.usage.get('input_tokens', '?')}, "
                f"output={resp.usage.get('output_tokens', '?')}",
            )
    except Exception as e:
        console.print(
            f"[red]✗ {name} connection failed: "
            f"{type(e).__name__}: {e}[/red]",
        )


def _dev_reload(config: Config) -> None:
    """Re-apply extensions from toml files without restart."""
    _hot_reload(config)
    console.print("[green]✓ Extensions reloaded[/green]")
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

    console.print("[bold]Recent API call log:[/bold]")
    console.print(f"  Provider: {config.provider}")
    console.print(f"  Model: {config.model}")
    console.print("\n[bold]Registered tool stats:[/bold]")
    tools = registry.list_tools()
    by_perm = {}
    for t in tools:
        by_perm.setdefault(t.permission.value, []).append(t.name)
    for perm, names in sorted(by_perm.items()):
        console.print(f"  [{perm}] {len(names)} tools")
    console.print("\n[dim]For detailed token stats use /stats[/dim]")


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
                        f"[red]Cannot infer provider from model name "
                        f"'{model}'; use the full syntax: "
                        f"/dev model add <provider> <name>[/red]",
                    )
                    console.print(
                        "[dim]Recognized prefixes: "
                        f"{', '.join(_MODEL_PROVIDER_HINTS.keys())}[/dim]",
                    )
                    return
                _model_add(config, provider, model)
            else:
                console.print(
                    "[dim]Usage:\n"
                    "  /dev model list                       "
                    "— list available models\n"
                    "  /dev model add <provider> <name>      "
                    "— register a new model\n"
                    "  /dev model add <name>                 "
                    "— shorthand; infer provider from name prefix\n"
                    "  /dev model remove <provider> <name>   "
                    "— remove a custom model[/dim]",
                )
        elif action in ("remove", "rm") and len(parts) >= 5:
            _model_remove(config, parts[3], parts[4])
        else:
            console.print(
                "[dim]Usage:\n"
                "  /dev model list                       "
                "— list available models\n"
                "  /dev model add <provider> <name>      "
                "— register a new model\n"
                "  /dev model add <name>                 "
                "— shorthand; infer provider from name prefix\n"
                "  /dev model remove <provider> <name>   "
                "— remove a custom model[/dim]",
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
                "[dim]Usage:\n"
                "  /dev provider add            "
                "— interactively add an extension provider\n"
                "  /dev provider list           "
                "— list extension providers\n"
                "  /dev provider remove <name>  "
                "— delete an extension provider[/dim]",
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
                "[dim]Usage:\n"
                "  /dev capability add            "
                "— write a capability scaffold template\n"
                "  /dev capability list           "
                "— list extension capabilities\n"
                "  /dev capability remove <key>   "
                "— delete an extension capability[/dim]",
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
                "[dim]Usage:\n"
                "  /dev skill add            "
                "— interactively add a custom Skill\n"
                "  /dev skill list           — list custom Skills\n"
                "  /dev skill remove <name>  — delete a custom Skill[/dim]",
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
                "[dim]Usage:\n"
                "  /dev tool add            "
                "— interactively add a Shell tool\n"
                "  /dev tool list           — list custom Shell tools\n"
                "  /dev tool remove <name>  "
                "— delete a custom Shell tool[/dim]",
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
                "[dim]Usage:\n"
                "  /dev debug tools           — list all registered tools\n"
                "  /dev debug schema <name>   "
                "— show a tool's JSON Schema\n"
                "  /dev debug call <name> {}  — invoke a tool manually\n"
                "  /dev debug prompt          "
                "— show the current system prompt[/dim]",
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
                "[dim]Usage:\n"
                "  /dev test provider <name>  "
                "— test provider connectivity[/dim]",
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

    console.print(f"[red]Unknown /dev subcommand: {sub}[/red]")
    console.print(Markdown(_GUIDE_INDEX))
