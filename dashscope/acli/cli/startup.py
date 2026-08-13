# -*- coding: utf-8 -*-
"""Startup functions: banner, system prompt loading, provider debug."""

from __future__ import annotations

from rich.console import Console

from dashscope.acli import __version__
from dashscope.acli.cli.constants import ALL_CAPABILITY_KEYS
from dashscope.acli.config import WORKSPACE_DIR, Config
from dashscope.acli.tools.registry import registry
from dashscope.acli.utils import mask_secret

console = Console()


def _load_system_prompt() -> str | None:
    """Load system prompt from .acli/system-prompt.md (workspace then global).

    Project-specific instructions (.acli/rules.jsonl, .cursorrules, etc.)
    are discovered separately by Agent.__init__ and passed to the prompt
    pipeline for proper stable/ephemeral separation.
    """
    from dashscope.acli.config import CONFIG_DIR

    base: str | None = None
    for d in (WORKSPACE_DIR, CONFIG_DIR):
        path = d / "system-prompt.md"
        if path.is_file():
            try:
                base = path.read_text(encoding="utf-8").strip() or None
            except OSError:
                pass
            break

    return base


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            return text[end + 3 :].strip()
    return text.strip()


def _load_references() -> str | None:
    """Load .acli/references/*.md (global then workspace) as always-on context.

    Unlike skills (invoked on demand), references are knowledge docs that must
    always be in the system prompt — e.g. generated SDK API indexes.
    """
    from dashscope.acli.config import CONFIG_DIR

    parts: dict[str, str] = {}
    for d in (CONFIG_DIR, WORKSPACE_DIR):
        ref_dir = d / "references"
        if not ref_dir.is_dir():
            continue
        for path in sorted(ref_dir.glob("*.md")):
            try:
                body = _strip_frontmatter(path.read_text(encoding="utf-8"))
            except OSError:
                continue
            if body:
                parts[
                    path.name
                ] = body  # workspace overrides global by file name
    return "\n\n---\n\n".join(parts.values()) if parts else None


def _compose_system_prompt(base: str | None) -> str | None:
    """Append .acli/references/ content to the resolved base prompt.

    Applied at every consumption site so embedded prompts (which bypass
    _load_system_prompt) still get references.
    """
    refs = _load_references()
    if not refs:
        return base
    section = f"## References (auto-loaded)\n\n{refs}"
    return f"{base}\n\n---\n\n{section}" if base else section


def _print_banner(config: Config | None = None):
    logo = (
        "     _                    _   _       ____ _     ___\n"
        "    / \\   __ _  ___ _ __ | |_(_) ___ / ___| |   |_ _|\n"
        "   / _ \\ / _` |/ _ \\ '_ \\| __| |/ __| |   | |    | |\n"
        "  / ___ \\ (_| |  __/ | | | |_| | (__| |___| |___ | |\n"
        " /_/   \\_\\__, |\\___|_| |_|\\__|_|\\___|\\____|_____|___|\n"
        "         |___/"
    )
    console.print(f"[bold cyan]{logo}[/bold cyan]", highlight=False)
    console.print(
        f"  [bold]AgenticCLI[/bold] [dim]v{__version__}[/dim] — 用自然语言驱动一切",
    )
    console.print(f"  [dim]Workspace: {WORKSPACE_DIR}[/dim]\n")

    # ── 启动信息：provider / model / 用户 / 能力 / 工具 ──
    if config is not None:
        user_display = config.user_name or "(未设置)"
        console.print(
            f"  [bold]Provider:[/bold] [cyan]{config.provider}[/cyan]  "
            f"[bold]Model:[/bold] [cyan]{config.model}[/cyan]  "
            f"[bold]User:[/bold] [cyan]{user_display}[/cyan]",
        )

        # API Key 状态（当前 provider）
        from dashscope.acli.cli.handlers_key import all_key_targets

        targets = all_key_targets(config)
        key_info = targets.get(config.provider)
        if key_info:
            key_val = getattr(config, key_info["field"], "")
            if key_val:
                console.print(
                    f"  [bold]API Key:[/bold] "
                    f"[green]✓ {mask_secret(key_val)}[/green] "
                    f"[dim]({key_info['env']})[/dim]",
                )
            else:
                console.print(
                    f"  [bold]API Key:[/bold] [red]✗ 未设置[/red] "
                    f"[dim]({key_info['env']}, 用 /provider 设置)[/dim]",
                )

        # 已启用能力
        caps_display = (
            ALL_CAPABILITY_KEYS
            if config.enabled_capabilities is None
            else config.enabled_capabilities
        )
        caps_str = ", ".join(caps_display) if caps_display else "无"
        console.print(f"  [bold]能力:[/bold]   [dim]{caps_str}[/dim]")

        # 已注册工具数量
        tool_count = (
            len(registry.list_tools())
            if hasattr(registry, "list_tools")
            else "?"
        )
        console.print(f"  [bold]工具:[/bold]   [dim]{tool_count} 个已注册[/dim]")

        # SDK 知识库（仅 embedded 模式）
        sdk_index = getattr(config, "_embedded_sdk_index", None)
        if sdk_index:
            console.print(
                f"  [bold]SDK:[/bold]    [green]✓[/green] "
                f"[dim]{', '.join(sdk_index)}[/dim]",
            )

        console.print()

    console.print("  [dim]会话: /help /clear /exit[/dim]")
    console.print(
        "  [dim]配置: /setup /capability /subagents /provider /trust "
        "/rule[/dim]",
    )
    console.print("  [dim]能力: /profile /mcp /skill[/dim]")
    console.print(
        "  [dim]开发: /dev model|provider|capability (运行时) "
        "/dev platform|tool|skill (扩展指南)[/dim]",
    )
    console.print(
        "  [dim]输入: Enter 提交 / Ctrl+J 换行 / 粘贴多行自动保留 / "
        "@文件 展开内容 / /v 语音输入[/dim]",
    )
    console.print()


def _print_provider_debug(provider):
    """Print provider endpoint and key info for debugging."""
    if hasattr(provider, "client") and hasattr(provider.client, "base_url"):
        # OpenAIProvider
        base_url = str(provider.client.base_url)
        key = provider.client.api_key or ""
        console.print(f"[dim]  endpoint: {base_url}[/dim]")
        console.print(f"[dim]  api_key: {mask_secret(key)}[/dim]")
    elif hasattr(provider, "api_key"):
        # TongyiProvider
        key = provider.api_key or ""
        endpoint = getattr(provider, "base_url", "dashscope SDK")
        console.print(f"[dim]  endpoint: {endpoint}[/dim]")
        console.print(f"[dim]  api_key: {mask_secret(key)}[/dim]")
