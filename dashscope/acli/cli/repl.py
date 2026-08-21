# -*- coding: utf-8 -*-
"""REPL main loop."""
# pylint: disable=protected-access,too-many-branches,too-many-statements

from __future__ import annotations

import os
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel

from dashscope.acli.agent import Agent
from dashscope.acli.cli.completer import (
    AcliCompleter,
    SafeFileHistory,
    _HintProcessor,
)
from dashscope.acli.cli.dispatch import (
    _handle_skill_continue,
    _handle_slash_command,
    dispatch_async_command,
)
from dashscope.acli.cli.handlers_capability import sync_extensions_into_catalog
from dashscope.acli.cli.handlers_key import ensure_provider_key
from dashscope.acli.cli.handlers_profile import set_memory_client
from dashscope.acli.cli.handlers_setup import _handle_setup
from dashscope.acli.cli.mcp import (
    _connect_mcp,
    _disconnect_mcp,
    _init_mcp_servers,
    _mcp_clients,
)
from dashscope.acli.cli.multimodal import (
    _expand_at_references,
    _to_multimodal_content,
)
from dashscope.acli.cli.startup import (
    _compose_system_prompt,
    _load_system_prompt,
    _print_banner,
)
from dashscope.acli.cli.streaming import (
    _do_compress,
    _do_summarize,
    _stream_response,
)
from dashscope.acli.config import (
    PROVIDER_MODELS,
    WORKSPACE_CONFIG_FILE,
    WORKSPACE_DIR,
    Config,
    is_audio_model,
    is_vision_model,
)
from dashscope.acli.dev import _apply_custom_models
from dashscope.acli.executor import Executor
from dashscope.acli.providers import get_provider, get_provider_chain
from dashscope.acli.session import get_session_manager
from dashscope.acli.skills import list_known_services
from dashscope.acli.utils import UserAbortedTurn, UserSupplement

console = Console()


def _reset_terminal_modes() -> None:
    # A force-killed TUI leaves mouse-reporting modes on, which makes iTerm2
    # capture the scroll wheel so the scrollback region can't be scrolled.
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l")
        sys.stdout.flush()


async def _run_loop(config: Config):
    _reset_terminal_modes()
    _apply_custom_models(config)
    # Re-arm audit redaction from persisted config (restart must not drop it).
    from dashscope.acli.audit import configure_audit_logger

    configure_audit_logger(config)
    from dashscope.acli.debuglog import configure_debug_log

    configure_debug_log(config)
    from dashscope.acli.permission import configure_permission_policy

    configure_permission_policy()
    # Layer-1 extensions: load custom-extensions.toml entries into
    # PROVIDER_MODELS and stash the spec for get_provider / tool registry.
    from dashscope.acli.extensions import apply_extensions

    _ext = apply_extensions(PROVIDER_MODELS)
    if _ext.errors:
        for err in _ext.errors:
            console.print(
                f"[yellow]Extension load warning:[/yellow] "
                f"[dim]{err}[/dim]",
            )
    sync_extensions_into_catalog(_ext)
    # _load_plugins() — deprecated; plugins load via the hooks mechanism
    provider = get_provider_chain(config)
    executor = Executor(auto_approve=config.auto_approve)

    # Initialize memory
    # Memory client managed by handlers_profile
    memory = None
    if config.memory_enabled:
        from dashscope.acli.platforms import get_memory_provider

        memory = get_memory_provider(config)
        if memory:
            set_memory_client(memory)

    # user_name is display-only; no longer derived from memory_user_id.
    user_name = config.user_name
    session_manager = get_session_manager()
    session_path = (
        session_manager.get_history_path() if config.session_persist else None
    )
    # Closure captures `config` — when /capability toggles mutate
    # enabled_capabilities, the next turn's system prompt sees the new state.
    # Shared hook bus for agent lifecycle and skill packages.
    from dashscope.acli.hooks import create_hook_bus
    from dashscope.acli.tools.platform import disabled_capabilities_hint

    hook_bus = create_hook_bus()

    agent = Agent(
        provider=provider,
        executor=executor,
        max_turns=config.max_turns,
        memory=memory,
        user_name=user_name,
        provider_name=config.provider,
        model_name=config.model,
        session_path=session_path,
        disabled_caps_provider=lambda: disabled_capabilities_hint(config),
        directives_provider=lambda: config.user_directives,
        system_prompt=_compose_system_prompt(
            getattr(config, "_embedded_system_prompt", None)
            or _load_system_prompt(),
        ),
        hook_bus=hook_bus,
    )

    # Pin parent agent ref for local.subagent / local.delegate BEFORE platform
    # tool registration so register_one_capability finds a parent to attach to.
    from dashscope.acli.agents.delegate import (
        set_config as set_delegate_config,
    )
    from dashscope.acli.agents.delegate import (
        set_parent_agent as set_delegate_parent,
    )
    from dashscope.acli.agents.subagent import (
        set_config as set_subagent_config,
    )
    from dashscope.acli.agents.subagent import (
        set_parent_agent as set_subagent_parent,
    )

    set_subagent_parent(agent)
    set_subagent_config(config)
    set_delegate_parent(agent)
    set_delegate_config(config)

    # Register platform tools (memory, kb, data, prompt, search, context,
    # mcp, subagent, delegate)
    from dashscope.acli.tools.platform import register_platform_tools

    register_platform_tools(config, connect_mcp_fn=_connect_mcp)

    # Load skill packages (project .acli/skills/<name>/ and global
    # ~/.acli/skills/<name>/).
    from dashscope.acli.skills import get_skill_manager

    skill_manager = get_skill_manager()
    skill_manager._registry_url = getattr(config, "skill_registry", "")
    skill_manager._global = False
    skill_manager.load(hook_bus=hook_bus)

    # Register session tools (switch model/provider, capability, mcp
    # management)
    from dashscope.acli.tools.session import register_session_tools

    register_session_tools(
        config,
        get_agent=lambda: agent,
        get_provider_fn=get_provider,
        connect_mcp_fn=_connect_mcp,
        disconnect_mcp_fn=_disconnect_mcp,
        list_mcp_services_fn=list_known_services,
        get_mcp_clients_fn=lambda: _mcp_clients,
    )

    from dashscope.acli.tools.evolution import register_evolution_tools

    register_evolution_tools(get_agent=lambda: agent)

    from dashscope.acli.tools.skills import register_skill_tools

    register_skill_tools(get_agent=lambda: agent)

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    _embedded_name = getattr(config, "_embedded_app_name", None)
    if _embedded_name:
        from dashscope.acli import __version__

        console.print(
            f"[bold green]{_embedded_name}[/bold green] "
            f"[dim]v{__version__} ({config.provider}/{config.model})[/dim]",
        )
        console.print(f"[dim]Workspace: {WORKSPACE_DIR}[/dim]")
        # Show SDK index if available
        sdk_index = getattr(config, "_embedded_sdk_index", None)
        if sdk_index:
            console.print(f"[dim]SDK Index: {', '.join(sdk_index)}[/dim]")
        console.print("[dim]Type /help for commands, /exit to quit[/dim]\n")
    else:
        _print_banner(config)

    # Restore the prior conversation if any
    try:
        restored = agent.load_session()
        if restored:
            console.print(
                f"  [dim]Restored {restored} history messages "
                f"({session_path})[/dim]\n",
            )
        else:
            console.print(
                f"  [dim]No history messages ({session_path})[/dim]\n",
            )
    except Exception:
        console.print(
            f"  [dim]History load failed ({session_path})[/dim]\n",
        )

    # First-run: auto-trigger /setup if workspace config doesn't exist
    # Skip if essential config (provider/model/API key) is already set globally
    if not WORKSPACE_CONFIG_FILE.exists():
        has_api_key = bool(config.api_key)
        using_defaults = (
            config.provider == "tongyi" and config.model == "qwen3.7-plus"
        )
        if not has_api_key or using_defaults:
            await _handle_setup(config, agent)

    # Check API key after banner is shown
    ensure_provider_key(config, agent)

    history_file = session_manager.get_input_history_path()
    # Context-aware: reads live config + extension registry on each keystroke
    # so /model lists models for the current provider, /capability enable
    # only shows currently-disabled caps, /dev provider rm only shows
    # user-added providers, etc.
    _completer = AcliCompleter(config)

    _kb = KeyBindings()

    def _selected_is_dir(buf) -> bool:
        """True if the highlighted completion inserts a path ending in `/`
        — used to decide whether Tab/Enter should drill in or submit."""
        state = buf.complete_state
        return bool(
            state
            and state.current_completion
            and state.current_completion.text.endswith("/"),
        )

    @_kb.add("enter", filter=has_completions)
    def _accept_completion(event):
        buf = event.app.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            is_dir = _selected_is_dir(buf)
            buf.apply_completion(buf.complete_state.current_completion)
            if is_dir:
                buf.start_completion(select_first=False)
                return
        buf.validate_and_handle()

    # Tab with menu open: accept + drill into next level.
    @_kb.add("tab", filter=has_completions)
    def _tab_completion(event):
        buf = event.app.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            is_dir = _selected_is_dir(buf)
            buf.apply_completion(buf.complete_state.current_completion)
            if not is_dir:
                buf.insert_text(" ")
            buf.start_completion(select_first=False)

    # Tab without menu: open completion. If only one match, apply + drill in.
    @_kb.add("tab", filter=~has_completions)
    def _tab_start_completion(event):
        buf = event.app.current_buffer
        buf.start_completion(select_first=False)
        if buf.complete_state:
            if len(buf.complete_state.completions) == 1:
                buf.go_to_completion(0)
                buf.apply_completion(buf.complete_state.current_completion)
                buf.insert_text(" ")
                buf.start_completion(select_first=False)

    # Right arrow: apply selected completion and drill into next-level menu.
    @_kb.add("right", filter=has_completions)
    def _right_drill_in(event):
        buf = event.app.current_buffer
        if buf.complete_state and buf.complete_state.current_completion:
            is_dir = _selected_is_dir(buf)
            buf.apply_completion(buf.complete_state.current_completion)
            if not is_dir:
                buf.insert_text(" ")
            buf.start_completion(select_first=False)

    # Left arrow: go back one level (remove last token) and re-show menu.
    @_kb.add("left", filter=has_completions)
    def _left_go_back(event):
        buf = event.app.current_buffer
        buf.cancel_completion()
        text = buf.text
        # Remove trailing space + last token to go back one level
        stripped = text.rstrip()
        if " " in stripped:
            prev = stripped[: stripped.rfind(" ") + 1]
            buf.text = prev
            buf.cursor_position = len(prev)
            buf.start_completion(select_first=False)
        else:
            buf.text = ""
            buf.cursor_position = 0

    # Plain Enter (no active completion) = submit. With multiline=True this
    # overrides prompt_toolkit's default of inserting a newline.
    @_kb.add("enter", filter=~has_completions)
    def _submit(event):
        event.current_buffer.validate_and_handle()

    # Multi-line input: Esc+Enter / Ctrl+J / Alt+Enter all insert newline.
    # Useful for typing multi-line prompts by hand.
    @_kb.add("escape", "enter")
    def _esc_enter(event):
        event.current_buffer.insert_text("\n")

    @_kb.add("c-j")
    def _ctrl_j(event):
        event.current_buffer.insert_text("\n")

    @_kb.add("c-t")
    def _voice_input_key(event):
        event.app.exit(result="__VOICE_INPUT__")

    # Bracketed paste: terminals signal start/end of paste so prompt_toolkit
    # treats the whole block as data — newlines inside the paste become
    # literal newlines in the buffer instead of submitting. Default is on;
    # we just need multiline=True for newlines to survive.
    from prompt_toolkit.styles import Style as PTStyle

    session: PromptSession = PromptSession(
        history=SafeFileHistory(str(history_file)),
        completer=_completer,
        complete_while_typing=True,
        key_bindings=_kb,
        multiline=True,
        enable_open_in_editor=True,
        input_processors=[_HintProcessor()],
        style=PTStyle.from_dict({"hint": "#888888 italic"}),
    )

    # Connect MCP servers from config
    if config.mcp_servers:
        await _init_mcp_servers(config)

    # Initialize cron scheduler
    import dashscope.acli.cli as _pkg
    from dashscope.acli.scheduler import Scheduler

    _pkg._scheduler = Scheduler(config, agent)
    await _pkg._scheduler.load_and_start()

    # --- CLI confirm callback: use prompt_toolkit to avoid double-Enter ---
    # Rich's Prompt.ask() uses blocking input(), which conflicts with
    # prompt_toolkit's terminal raw mode — the first Enter gets swallowed.
    # Using the same PromptSession for confirmations fixes this.
    async def _cli_confirm_callback(
        tool_def,
        arguments: dict,
        is_dangerous: bool,
    ) -> str:
        from dashscope.acli.utils.text import truncate_value

        title = "⚠️  Dangerous operation" if is_dangerous else "Confirm"
        border_style = "red bold" if is_dangerous else "yellow"
        args_display = "\n".join(
            f"  {k}: {truncate_value(v)}" for k, v in arguments.items()
        )
        content = f"Tool: {tool_def.name}\nArgs:\n{args_display}"
        console.print(Panel(content, title=title, border_style=border_style))

        if is_dangerous:
            prompt_sym = "Execute? [y]es / [n]o  [y]: "
        else:
            prompt_sym = (
                "Execute? [y]es / [n]o / [u]pdate (add info, re-plan) "
                "/ [a]lways / [s]top  [y]: "
            )

        try:
            with patch_stdout():
                raw = await session.prompt_async(prompt_sym)
        except (KeyboardInterrupt, EOFError):
            return "n"

        choice = raw.strip().lower() or "y"
        if is_dangerous:
            # Dangerous ops accept only y/n, matching the sync path
            # (no always-trust grant).
            if choice not in ("y", "n"):
                choice = "n"
        elif choice not in ("y", "n", "u", "a", "s"):
            choice = "y"
        if choice == "u":
            try:
                with patch_stdout():
                    supplement = await session.prompt_async(
                        "[yellow]Enter supplementary info:[/yellow] ",
                    )
            except (KeyboardInterrupt, EOFError):
                supplement = ""
            if not supplement.strip():
                raise UserAbortedTurn("No supplementary info provided")
            raise UserSupplement(supplement.strip())
        return choice

    agent.executor._confirm_callback = _cli_confirm_callback

    import time as _time

    _last_ctrl_c = 0.0

    while True:
        try:
            with patch_stdout():
                _prompt_sym = (
                    getattr(config, "_embedded_prompt_symbol", None)
                    or "acli> "
                )
                user_input = await session.prompt_async(_prompt_sym)
            _last_ctrl_c = 0.0
        except KeyboardInterrupt:
            now = _time.time()
            if now - _last_ctrl_c < 2.0:
                console.print("\n[dim]Bye![/dim]")
                break
            _last_ctrl_c = now
            console.print("[dim]Press Ctrl+C again to exit[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]Bye![/dim]")
            break
        user_input = user_input.strip()

        # Process pending cron prompts (non-subagent mode)
        if _pkg._scheduler:
            pending = _pkg._scheduler.get_pending_prompts()
            if pending:
                for prompt in pending:
                    console.print("\n[cyan][CRON → main chat][/cyan]")
                    await _stream_response(agent, config, prompt)
                    console.print()
                continue

        # Voice input via Option+V keybinding
        if user_input == "__VOICE_INPUT__":
            from dashscope.acli.ui.voice import voice_input as _voice_input

            text = await _voice_input(
                config.tongyi_api_key,
                model=config.asr_model,
                silence_threshold=config.voice_silence_threshold,
                silence_duration=config.voice_silence_duration,
                max_recording_seconds=config.voice_max_seconds,
            )
            if text:
                user_input = text
            else:
                continue

        if not user_input:
            if (
                agent.messages
                and agent.messages[-1].get("role") == "assistant"
            ):
                user_input = "1"
            else:
                continue

        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                import subprocess as _sp

                env = os.environ.copy()
                env["ACLI_CLI"] = "1"
                proc = _sp.run(shell_cmd, shell=True, env=env, check=False)
                if proc.returncode != 0:
                    print(f"(exit code {proc.returncode})")
            continue

        if user_input.startswith("/"):
            # Do NOT add patch_stdout here: with no prompt app running it
            # replaces every ESC in the output with '?', garbling slash
            # commands' Rich output.
            try:
                result = _handle_slash_command(user_input, agent, config)
            except Exception as e:
                # A buggy handler must not kill the session loop.
                console.print(f"[red]Command error: {e}[/red]")
                continue
            if result == "voice":
                from dashscope.acli.ui.voice import voice_input as _voice_input

                text = await _voice_input(
                    config.tongyi_api_key,
                    model=config.asr_model,
                    silence_threshold=config.voice_silence_threshold,
                    silence_duration=config.voice_silence_duration,
                    max_recording_seconds=config.voice_max_seconds,
                )
                if text:
                    user_input = text
                else:
                    continue
            elif result == "async":
                await dispatch_async_command(user_input, config, agent)
                continue
            elif result == "compress":
                await _do_compress(agent)
                continue
            elif result == "summarize":
                await _do_summarize(agent)
                continue
            elif result == "skill":
                skill_result = await _handle_skill_continue(
                    user_input,
                    config,
                    agent,
                )
                if skill_result:
                    user_input = skill_result
                else:
                    continue
            elif result:
                continue

        # Expand @path tokens to file contents before sending to the agent.
        # Done here (post slash-command handling) so slash commands themselves
        # aren't affected — /key foo@bar still sees the literal "foo@bar".
        expanded_text, images, audio_clips = _expand_at_references(user_input)
        if images and not is_vision_model(config.model):
            console.print(
                f"[yellow]Model {config.model} does not support images; "
                f"{len(images)} image(s) ignored. Switch to a vision model "
                f"(e.g. qwen-vl-max) and retry.[/yellow]",
            )
            images = []
        if audio_clips and not is_audio_model(config.model):
            console.print(
                f"[yellow]Model {config.model} does not support audio; "
                f"{len(audio_clips)} audio clip(s) ignored. Switch to an "
                f"audio model (e.g. qwen-omni-turbo) and retry.[/yellow]",
            )
            audio_clips = []
        user_input = _to_multimodal_content(expanded_text, images, audio_clips)

        messages_before = len(agent.messages)
        try:
            console.print()
            await _stream_response(agent, config, user_input)
            console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted[/dim]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")

        # Auto-summarize long tasks (>= 6 new messages = user + assistant
        # + tool calls)
        messages_added = len(agent.messages) - messages_before
        if messages_added >= 6:
            console.print(
                f"[dim]Long task detected ({messages_added} messages), "
                f"auto-summarizing...[/dim]",
            )
            await _do_summarize(agent, silent=False)

    # Shutdown scheduler on exit
    if _pkg._scheduler:
        await _pkg._scheduler.shutdown()
