from __future__ import annotations

import asyncio
import sys
import warnings

warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")

from rich.console import Console

import dashscope.acli.tools.browser  # noqa: F401
import dashscope.acli.tools.camera  # noqa: F401

# Ensure tools are registered by importing
import dashscope.acli.tools.filesystem  # noqa: F401
import dashscope.acli.tools.shell  # noqa: F401
import dashscope.acli.tools.web_search  # noqa: F401
from dashscope.acli import __version__
from dashscope.acli.config import PROVIDER_MODELS, Config
from dashscope.acli.dev import handle_dev_command
from dashscope.acli.skills import load_skill_files

load_skill_files()
from dashscope.acli.cli.completer import _get_arg_hint, _is_dir_safe

# Import constants and multimodal handling from submodules
from dashscope.acli.cli.constants import (
    _AT_AUDIO_MAX_BYTES,
    _AT_IMAGE_MAX_BYTES,
    _AT_PATH_AT_CURSOR_RE,
    _DEV_SUBCOMMANDS,
    _PATH_COMPLETION_LIMIT,
    _SUBCOMMANDS,
    _TOP_LEVEL_COMMANDS,
    ALL_CAPABILITY_KEYS,
    CAPABILITY_CATALOG,
)
from dashscope.acli.cli.multimodal import _expand_at_references, _to_multimodal_content

console = Console()
from dashscope.acli.cli.dispatch import (
    _handle_skill_continue,
    _handle_slash_command,
    dispatch_async_command,
)
from dashscope.acli.cli.examples import _handle_example_command
from dashscope.acli.cli.handlers_capability import _cap_enabled, sync_extensions_into_catalog
from dashscope.acli.cli.handlers_misc import _handle_report_command
from dashscope.acli.cli.handlers_setup import _handle_setup

# Import MCP management from submodule
from dashscope.acli.cli.mcp import _connect_mcp, _mcp_clients
from dashscope.acli.cli.repl import _run_loop
from dashscope.acli.cli.runners import _run_dry_run, _run_oneshot, _run_tui_mode
from dashscope.acli.cli.startup import _compose_system_prompt, _load_system_prompt
from dashscope.acli.cli.streaming import _do_compress, _do_summarize

# Cron scheduler
_scheduler = None


def main():
    # Parse --cli / --tui / --dry-run flags (can appear anywhere in argv)
    use_cli = "--cli" in sys.argv
    use_tui = "--tui" in sys.argv
    use_dry_run = "--dry-run" in sys.argv
    sys.argv = [a for a in sys.argv if a not in ("--cli", "--tui", "--dry-run")]

    # Parse --protocol flag (can appear anywhere in argv)
    protocol_override = None
    max_turns_override = None
    new_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--protocol" and i + 1 < len(sys.argv):
            protocol_override = sys.argv[i + 1]
            i += 2
            continue
        if sys.argv[i] == "--max-turns" and i + 1 < len(sys.argv):
            max_turns_override = sys.argv[i + 1]
            i += 2
            continue
        new_argv.append(sys.argv[i])
        i += 1
    sys.argv = new_argv

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("--version", "-v", "-V"):
            print(f"acli {__version__}")
            return
        elif arg in ("--help", "-h"):
            print(
                "Usage: acli [--version] [--help] [-c <prompt>] [--cli | --tui] [--dry-run]"
            )
            print()
            print("对话即操作——你说需求，AI 来执行。")
            print()
            print("Options:")
            print("  -c, --command <prompt>  执行单次对话后退出（适合脚本/管道）")
            print("  --cli                   使用传统 readline REPL 模式")
            print(
                "  --tui                   使用 Textual 富 UI 模式（默认从 config.toml 读取）"
            )
            print("  --protocol <name>       指定 API 协议（openai | anthropic）")
            print("  --max-turns <n>         覆盖最大对话轮数（默认 1000）")
            print(
                "  --dry-run               预览当前配置（加载的 skill、MCP、tool 等）而不启动"
            )
            print()
            print("Subcommands:")
            print("  example                         列出可用示例")
            print(
                "  example download <name>         平铺合并示例到当前目录（冲突自动备份）"
            )
            print(
                "  mcp-server                      启动 MCP Server（stdio 对外暴露工具）"
            )
            print()
            print("启动后输入 /help 查看内置命令。")
            return
        elif arg in ("-c", "--command"):
            prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            if not prompt:
                print("错误: -c 需要提供 prompt 参数")
                print('用法: acli -c "你的需求"')
                sys.exit(1)
            config = Config.load()
            asyncio.run(_run_oneshot(config, prompt))
            return
        elif arg in ("example", "examples"):
            _handle_example_command(sys.argv[2:])
            return
        elif arg == "mcp-server":
            from dashscope.acli.mcp_server import main as _mcp_server_main

            _mcp_server_main()
            return
    config = Config.load()
    # Apply --protocol override
    if protocol_override:
        if protocol_override.lower() in ("openai", "anthropic"):
            config.protocol = protocol_override.lower()
        else:
            print(f"错误: 未知协议 '{protocol_override}'（可选: openai, anthropic）")
            sys.exit(1)

    # Apply --max-turns override
    if max_turns_override is not None:
        try:
            config.max_turns = int(max_turns_override)
        except ValueError:
            print(f"错误: --max-turns 必须是整数，收到 '{max_turns_override}'")
            sys.exit(1)

    # Handle --dry-run: preview configuration without starting the agent
    if use_dry_run:
        _run_dry_run(config)
        return

    try:
        if use_cli:
            asyncio.run(_run_loop(config))
        elif use_tui:
            _run_tui_mode(config)
        elif config.tui:
            _run_tui_mode(config)
        else:
            asyncio.run(_run_loop(config))
    except (KeyboardInterrupt, SystemExit):
        pass
