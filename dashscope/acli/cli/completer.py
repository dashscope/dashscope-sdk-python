# -*- coding: utf-8 -*-
"""Prompt completion and history management."""
# pylint: disable=too-many-branches,too-many-return-statements,unused-argument

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.layout.processors import Processor, Transformation

from dashscope.acli.cli.constants import (
    _ARG_HINTS,
    _AT_PATH_AT_CURSOR_RE,
    _DEV_SUBCOMMANDS,
    _PATH_COMPLETION_LIMIT,
    _SECRET_PATTERN,
    _SUBCOMMANDS,
    _TOP_LEVEL_COMMANDS,
    CAPABILITY_CATALOG,
)
from dashscope.acli.config import Config
from dashscope.acli.skills import BUILTIN_SKILLS, KNOWN_MCP_SERVICES
from dashscope.acli.utils import sanitize_text

if TYPE_CHECKING:
    pass


class SafeFileHistory(FileHistory):
    """FileHistory that redacts API keys before saving."""

    def store_string(self, string: str) -> None:
        sanitized = _SECRET_PATTERN.sub(r"\1***", string)
        sanitized = sanitize_text(sanitized)
        # Ensure parent directory exists (FileHistory silently drops on
        # failure)
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        super().store_string(sanitized)


def _get_arg_hint(text: str) -> str:
    """Given current input text, return the ghost hint to show (or empty)."""
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return ""
    if not text.endswith(" "):
        return ""
    tokens = stripped.split()
    arg_index = len(tokens)
    if not tokens:
        return ""
    cmd = tokens[0]
    sub = tokens[1] if len(tokens) >= 2 else None
    hint = _ARG_HINTS.get((cmd, sub, arg_index), "")
    if not hint:
        hint = _ARG_HINTS.get((cmd, None, arg_index), "")
    return hint


class _HintProcessor(Processor):
    """Appends dim ghost text at end-of-line showing expected arguments."""

    def apply_transformation(self, transformation_input):
        doc = transformation_input.document
        lineno = transformation_input.lineno
        if lineno != 0:
            return Transformation(transformation_input.fragments)

        text = doc.text_before_cursor
        hint = _get_arg_hint(text)
        if hint:
            fragments: StyleAndTextTuples = transformation_input.fragments + [
                ("class:hint", hint),
            ]
            return Transformation(fragments)
        return Transformation(transformation_input.fragments)


def _is_dir_safe(entry) -> bool:
    """is_dir() with PermissionError / broken-symlink swallowed."""
    try:
        return entry.is_dir()
    except OSError:
        return False


class AcliCompleter(Completer):
    """Context-aware completer for the REPL prompt.

    Two completion modes share one entry:

    - ``/`` at line start triggers slash-command completion (commands,
      sub-commands, arg-slot values like model / capability / mcp service).
      Reads live state (``config``, ``CAPABILITY_CATALOG``, extension
      registry, ``_mcp_clients`` …) on every keystroke.

    - ``@<partial-path>`` anywhere in the line triggers filesystem
      completion against that path's directory portion — typing ``@`` alone
      lists cwd, ``@../`` lists the parent, ``@~/`` lists home, etc.
      Directories get a trailing ``/`` so you can keep drilling in.

    Plain natural-language input gets nothing — the popup never gets in
    the way of normal prompts.
    """

    def __init__(self, config: Config):
        self.config = config

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # @path completion takes priority — works even inside natural
        # language (e.g., `check @src/acli/cli.py`).
        m = _AT_PATH_AT_CURSOR_RE.search(text)
        if m:
            yield from self._path_completions(m.group(1))
            return

        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return

        # Distinguish "user is mid-word" vs "user just hit space, starting a
        # fresh arg slot". split() loses the trailing-whitespace signal, so
        # check the raw text first.
        ends_with_space = text != "" and text[-1] in (" ", "\t")
        tokens = stripped.split()

        if ends_with_space:
            current = ""
            arg_index = len(tokens)
        else:
            current = tokens[-1] if tokens else ""
            arg_index = max(0, len(tokens) - 1)

        candidates = self._slot_candidates(tokens, arg_index)
        for c in candidates:
            if c.startswith(current):
                yield Completion(c, start_position=-len(current))

    def _path_completions(self, raw_path: str):
        """List entries of the directory implied by ``raw_path`` (the
        substring captured after ``@``). Returns ``Completion`` objects
        whose insertion text is just the filename — prompt_toolkit
        replaces the prefix-before-cursor, so the rest of the typed path
        stays put. Directories get a trailing ``/``."""
        # Decompose raw_path into (directory_to_list, name_prefix_to_match).
        # Cases handled:
        #   ""        -> cwd, no prefix
        #   "src/"    -> src/, no prefix
        #   "src/a"   -> src/, prefix="a"
        #   "../"     -> parent dir, no prefix
        #   ".."      -> cwd, prefix=".."  (so `..` and `../` get suggested)
        #   "~"       -> home, no prefix
        #   "~/x"     -> home, prefix="x"
        sep = os.sep
        if raw_path == "":
            dir_str, prefix = ".", ""
        elif raw_path.endswith("/") or raw_path.endswith(sep):
            dir_str, prefix = raw_path, ""
        elif "/" in raw_path or sep in raw_path:
            last = max(raw_path.rfind("/"), raw_path.rfind(sep))
            dir_str, prefix = raw_path[: last + 1], raw_path[last + 1 :]
        elif raw_path in (".", "..", "~"):
            # Bare `.`, `..` or `~` — list that directory directly so
            # `@..` shows the parent immediately without needing the user
            # to add the trailing `/`.
            dir_str, prefix = raw_path + sep, ""
        else:
            dir_str, prefix = ".", raw_path

        directory = Path(dir_str).expanduser()
        if not directory.is_dir():
            return

        try:
            # Materialize but cap size — directories like node_modules
            # would otherwise lag the popup. iterdir() is unsorted; we
            # sort dirs-first / case-insensitive-by-name for stable UX.
            entries = []
            for entry in directory.iterdir():
                entries.append(entry)
                if len(entries) >= _PATH_COMPLETION_LIMIT * 4:
                    break
        except (OSError, PermissionError):
            return
        entries.sort(key=lambda e: (not _is_dir_safe(e), e.name.lower()))

        shown = 0
        for entry in entries:
            if shown >= _PATH_COMPLETION_LIMIT:
                break
            name = entry.name
            # Hide dotfiles unless the user is explicitly typing a `.` prefix.
            if name.startswith(".") and not prefix.startswith("."):
                continue
            if not name.startswith(prefix):
                continue
            is_dir = _is_dir_safe(entry)
            insertion = name + sep if is_dir else name
            yield Completion(
                insertion,
                start_position=-len(prefix),
                display=insertion,
                display_meta="dir" if is_dir else "file",
            )
            shown += 1

    def _slot_candidates(self, tokens: list[str], arg_index: int) -> list[str]:
        # Import runtime state here to avoid circular imports
        from dashscope.acli.cli import _mcp_clients, _scheduler

        if arg_index == 0:
            return _TOP_LEVEL_COMMANDS

        cmd = tokens[0]

        if cmd == "/skill" and arg_index == 1:
            return _SUBCOMMANDS["/skill"] + list(BUILTIN_SKILLS.keys())

        if cmd == "/capability":
            if arg_index == 1:
                return _SUBCOMMANDS["/capability"]
            if arg_index == 2:
                sub = tokens[1]
                all_keys = [c["key"] for c in CAPABILITY_CATALOG]
                enabled = (
                    set(self.config.enabled_capabilities)
                    if self.config.enabled_capabilities is not None
                    else set(all_keys)
                )
                if sub == "enable":
                    return [k for k in all_keys if k not in enabled]
                if sub == "disable":
                    return [k for k in all_keys if k in enabled]
                if sub == "config":
                    return list(self.config.subagents.keys())
            if arg_index == 3 and tokens[1] == "config":
                return ["model", "temperature", "max_turns"]
            return []

        if cmd == "/subagents":
            from dashscope.acli.agents.subagents import (
                SUBAGENT_CAPABILITY_KEYS,
            )

            if arg_index == 1:
                return _SUBCOMMANDS["/subagents"]
            if arg_index == 2:
                sub = tokens[1]
                if sub in ("enable", "disable", "config"):
                    return sorted(SUBAGENT_CAPABILITY_KEYS)
            if arg_index == 3 and tokens[1] == "config":
                return ["model", "temperature", "max_turns"]
            return []

        if cmd == "/mcp":
            if arg_index == 1:
                return _SUBCOMMANDS["/mcp"]
            if arg_index == 2:
                if tokens[1] == "add":
                    return list(KNOWN_MCP_SERVICES.keys())
                if tokens[1] == "remove":
                    return list(_mcp_clients.keys())
            return []

        if cmd == "/dev":
            if arg_index == 1:
                return _SUBCOMMANDS["/dev"]
            if arg_index == 2:
                return _DEV_SUBCOMMANDS.get(tokens[1], [])
            if arg_index == 3 and tokens[2] == "rm":
                from dashscope.acli.extensions import current as ext_current

                if tokens[1] == "model":
                    return list(self.config.custom_models)
                if tokens[1] == "provider":
                    return [p.name for p in ext_current().providers]
                if tokens[1] == "capability":
                    return [c.key for c in ext_current().capabilities]
            return []

        if cmd == "/cron":
            if arg_index == 1:
                return _SUBCOMMANDS["/cron"]
            sub = tokens[1] if len(tokens) > 1 else ""
            if sub in ("remove", "pause", "resume") and arg_index == 2:
                if _scheduler:
                    return list(_scheduler.jobs.keys())
                return []
            if sub == "add":
                prev = tokens[arg_index - 1] if arg_index > 1 else ""
                if prev == "skill":
                    return list(BUILTIN_SKILLS.keys())
                if prev in ("every", "at", "cron"):
                    return []
                return [
                    "every",
                    "at",
                    "cron",
                    "skill",
                    "condition",
                    "no-subagent",
                ]
            return []

        # Generic fallback: any command listed in _SUBCOMMANDS gets its
        # second-level keywords offered at arg_index 1.
        if arg_index == 1 and cmd in _SUBCOMMANDS:
            return _SUBCOMMANDS[cmd]
        return []
