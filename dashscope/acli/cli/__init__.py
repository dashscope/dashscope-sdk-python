# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position,unused-import
# pylint: disable=too-many-branches,too-many-statements
from __future__ import annotations

import asyncio
import sys
import warnings

warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")

from rich.console import Console  # noqa: E402

import dashscope.acli.tools.browser  # noqa: F401,E402
import dashscope.acli.tools.camera  # noqa: F401,E402

# Ensure tools are registered by importing
import dashscope.acli.tools.filesystem  # noqa: F401,E402
import dashscope.acli.tools.shell  # noqa: F401,E402
import dashscope.acli.tools.web_search  # noqa: F401,E402
from dashscope.acli import __version__  # noqa: E402
from dashscope.acli.config import PROVIDER_MODELS, Config  # noqa: F401,E402
from dashscope.acli.dev import handle_dev_command  # noqa: F401,E402
from dashscope.acli.skills import load_skill_files  # noqa: E402

load_skill_files()
from dashscope.acli.cli.completer import (  # noqa: F401,E402
    _get_arg_hint,
    _is_dir_safe,
)

# Import constants and multimodal handling from submodules
from dashscope.acli.cli.constants import (  # noqa: F401,E402
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
from dashscope.acli.cli.multimodal import (  # noqa: F401,E402
    _expand_at_references,
    _to_multimodal_content,
)

console = Console()
from dashscope.acli.cli.dispatch import (  # noqa: F401,E402
    _handle_skill_continue,
    _handle_slash_command,
    dispatch_async_command,
)
from dashscope.acli.cli.examples import (  # noqa: E402
    _handle_example_command,
)
from dashscope.acli.cli.handlers_capability import (  # noqa: F401,E402
    _cap_enabled,
    sync_extensions_into_catalog,
)
from dashscope.acli.cli.handlers_misc import (  # noqa: F401,E402
    _handle_report_command,
)
from dashscope.acli.cli.handlers_setup import (  # noqa: F401,E402
    _handle_setup,
)

# Import MCP management from submodule
from dashscope.acli.cli.mcp import (  # noqa: F401,E402
    _connect_mcp,
    _mcp_clients,
)
from dashscope.acli.cli.repl import _run_loop  # noqa: E402
from dashscope.acli.cli.runners import (  # noqa: E402
    _run_dry_run,
    _run_oneshot,
    _run_tui_mode,
)
from dashscope.acli.cli.startup import (  # noqa: F401,E402
    _compose_system_prompt,
    _load_system_prompt,
)
from dashscope.acli.cli.streaming import (  # noqa: F401,E402
    _do_compress,
    _do_summarize,
)

# Cron scheduler
_scheduler = None


def main():
    # Parse --cli / --tui / --dry-run flags (can appear anywhere in argv)
    use_cli = "--cli" in sys.argv
    use_tui = "--tui" in sys.argv
    use_dry_run = "--dry-run" in sys.argv
    sys.argv = [
        a for a in sys.argv if a not in ("--cli", "--tui", "--dry-run")
    ]

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
                "Usage: acli [--version] [--help] [-c <prompt>] "
                "[--cli | --tui] [--dry-run]",
            )
            print()
            print("Chat is action — describe what you need, AI executes it.")
            print()
            print("Options:")
            print(
                "  -c, --command <prompt>  Run a single prompt, then "
                "exit (good for scripts/pipes)",
            )
            print(
                "  --cli                   Use the classic readline "
                "REPL mode",
            )
            print(
                "  --tui                   "
                "Use the Textual rich UI mode (default read from "
                "config.toml)",
            )
            print(
                "  --protocol <name>       API protocol "
                "(openai | anthropic)",
            )
            print(
                "  --max-turns <n>         Override max conversation "
                "turns (default 1000)",
            )
            print(
                "  --dry-run               Preview current config (loaded "
                "skills, MCP, tools, etc.) without starting",
            )
            print()
            print("Subcommands:")
            print(
                "  example                         List available examples",
            )
            print(
                "  example download <name>         Flat-merge an example "
                "into the current directory (conflicts auto-backed-up)",
            )
            print(
                "  mcp-server                      "
                "Start an MCP Server (exposes tools over stdio)",
            )
            print()
            print("Type /help after startup to see built-in commands.")
            return
        elif arg in ("-c", "--command"):
            prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
            if not prompt:
                print("Error: -c requires a prompt argument")
                print('Usage: acli -c "your request"')
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
            print(
                f"Error: unknown protocol '{protocol_override}' "
                f"(choices: openai, anthropic)",
            )
            sys.exit(1)

    # Apply --max-turns override
    if max_turns_override is not None:
        try:
            config.max_turns = int(max_turns_override)
        except ValueError:
            print(
                f"Error: --max-turns must be an integer, got "
                f"'{max_turns_override}'",
            )
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
