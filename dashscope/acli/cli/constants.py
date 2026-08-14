# -*- coding: utf-8 -*-
"""Constants and configuration data for acli CLI."""

from __future__ import annotations

import re
from typing import Any

# API key target configurations
KEY_TARGETS = {
    "tongyi": {
        "field": "tongyi_api_key",
        "env": "DASHSCOPE_API_KEY",
        "scope": "global",
        "desc": "Tongyi Qwen / Bailian",
    },
    "anthropic": {
        "field": "anthropic_api_key",
        "env": "ANTHROPIC_API_KEY",
        "scope": "global",
        "desc": "Anthropic Claude",
    },
    "openai": {
        "field": "openai_api_key",
        "env": "OPENAI_API_KEY",
        "scope": "global",
        "desc": "OpenAI",
    },
}

# ASR model options
ASR_MODELS = [
    "paraformer-realtime-v2",
    "paraformer-realtime-v1",
    "sensevoice-v1",
]

# Theme presets
THEME_PRESETS = {
    "dark": {
        "background": "#1e1e1e",
        "text": "#d4d4d4",
        "border": "ansi_bright_blue",
        "border_style": "solid",
        "accent": "#569cd6",
        "muted": "#6a6a6a",
        "panel_border": "bright_blue",
    },
    "light": {
        "background": "#ffffff",
        "text": "#1e1e1e",
        "border": "ansi_blue",
        "border_style": "solid",
        "accent": "#007acc",
        "muted": "#595959",
        "panel_border": "blue",
    },
    "monokai": {
        "background": "#272822",
        "text": "#f8f8f2",
        "border": "#a6e22e",
        "border_style": "solid",
        "accent": "#a6e22e",
        "muted": "#75715e",
        "panel_border": "#a6e22e",
    },
    "dracula": {
        "background": "#282a36",
        "text": "#f8f8f2",
        "border": "#bd93f9",
        "border_style": "solid",
        "accent": "#bd93f9",
        "muted": "#6272a4",
        "panel_border": "#bd93f9",
    },
    "nord": {
        "background": "#2e3440",
        "text": "#d8dee9",
        "border": "#88c0d0",
        "border_style": "solid",
        "accent": "#88c0d0",
        "muted": "#4c566a",
        "panel_border": "#88c0d0",
    },
    "solarized": {
        "background": "#002b36",
        "text": "#839496",
        "border": "#268bd2",
        "border_style": "solid",
        "accent": "#268bd2",
        "muted": "#586e75",
        "panel_border": "#268bd2",
    },
}

# Capability catalog
CAPABILITY_CATALOG: list[dict[str, Any]] = [
    {
        "key": "bailian.mcp",
        "name": "Cloud tool extensions",
        "platform": "bailian",
        "cap": "mcp",
        "requires": ["tongyi_api_key"],
    },
    {
        "key": "bailian.cli",
        "name": "Bailian CLI suite (bl)",
        "platform": "bailian",
        "cap": "cli",
        "requires": [],
    },
    {
        "key": "local.subagent",
        "name": "Subagent (isolated context)",
        "platform": "local",
        "cap": "subagent",
        "requires": [],
    },
    {
        "key": "local.delegate",
        "name": "Delegate (parallel subagents)",
        "platform": "local",
        "cap": "delegate",
        "requires": [],
    },
    {
        "key": "local.memory",
        "name": "Long-term memory (memory_search/store)",
        "platform": "local",
        "cap": "memory",
        "requires": [],
    },
]

ALL_CAPABILITY_KEYS = [c["key"] for c in CAPABILITY_CATALOG]

# Regex patterns for sanitization and parsing
_SECRET_PATTERN = re.compile(
    r"(/(?:provider|key)\s+\w+\s+)\S+",
)

# Matches @<path> where path is anything that looks like a file
_AT_FILE_PATTERN = re.compile(r"@([^\s,;'\"()<>{}\[\]]+)")

# Cap per-file content inlined into the prompt
_AT_FILE_MAX_CHARS = 20000

# Cap per-image bytes before base64
_AT_IMAGE_MAX_BYTES = 5 * 1024 * 1024  # 5 MB

# Image extensions → MIME type
_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Audio extensions → MIME type and OpenAI input_audio format name
_AUDIO_MIME = {
    ".mp3": ("audio/mp3", "mp3"),
    ".wav": ("audio/wav", "wav"),
    ".m4a": ("audio/m4a", "mp3"),
    ".ogg": ("audio/ogg", "ogg"),
    ".webm": ("audio/webm", "webm"),
    ".flac": ("audio/flac", "flac"),
}

# Cap per-audio bytes before base64
_AT_AUDIO_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

# Matches image or audio sentinels inside expanded text
_MEDIA_SENTINEL_RE = re.compile(r"<<__ACLI_(IMG|AUDIO)_(\d+)__>>")

# Top-level slash commands
_TOP_LEVEL_COMMANDS = [
    "/help",
    "/clear",
    "/info",
    "/copy",
    "/compress",
    "/summarize",
    "/stats",
    "/report",
    "/feedback",
    "/history",
    "/json",
    "/save",
    "/privacy",
    "/audit",
    "/directives",
    "/exit",
    "/voice",
    "/tts",
    "/camera",
    "/provider",
    "/setup",
    "/capability",
    "/subagents",
    "/trust",
    "/rule",
    "/skill",
    "/profile",
    "/memory",
    "/session",
    "/mcp",
    "/cron",
    "/dev",
    "/debug",
    "/log",
    "/trace",
    "/theme",
    "/example",
]

# Static second-level subcommands
_SUBCOMMANDS: dict[str, list[str]] = {
    "/example": ["list", "download", "restore"],
    "/capability": ["list", "enable", "disable", "reload", "config"],
    "/subagents": ["list", "reload", "enable", "disable", "config"],
    "/trust": ["list", "clear", "allow", "deny"],
    "/rule": ["list", "add", "remove", "edit", "clear"],
    "/profile": ["list", "search", "add", "remove", "clear"],
    "/memory": ["list", "search", "remove", "clear"],
    "/session": ["new", "list", "switch", "rename", "remove"],
    "/mcp": ["list", "add", "remove"],
    "/cron": ["add", "list", "remove", "pause", "resume"],
    "/feedback": ["good", "bad"],
    "/history": ["stats", "list", "export", "clear"],
    "/json": ["on", "off"],
    "/privacy": ["on", "off", "status"],
    "/audit": ["recent", "query", "clear"],
    "/directives": ["add", "rm", "clear", "proposals", "accept", "reject"],
    "/skill": [
        "list",
        "add",
        "remove",
        "install",
        "uninstall",
        "enable",
        "disable",
        "update",
        "search",
        "publish",
        "link",
    ],
    "/dev": [
        "model",
        "provider",
        "capability",
        "platform",
        "tool",
        "skill",
        "debug",
        "test",
        "reload",
        "log",
    ],
    "/camera": ["capture", "record"],
    "/debug": ["on", "off", "status"],
    "/log": ["tail", "search", "clear"],
    "/trace": ["tail", "search", "clear"],
    "/voice": ["on", "off", "status", "model", "silence", "max", "threshold"],
    "/tts": ["on", "off", "status", "model", "voice", "speed", "say", "last"],
    "/theme": [
        "list",
        "set",
        "background",
        "text",
        "border",
        "accent",
        "muted",
        "border_style",
    ],
}

_DEV_SUBCOMMANDS: dict[str, list[str]] = {
    "model": ["list", "add", "remove"],
    "provider": ["list", "add", "remove"],
    "capability": ["list", "add", "remove"],
    "skill": ["list", "add", "remove"],
    "tool": ["list", "add", "remove"],
    "debug": ["tools", "schema", "call", "prompt"],
    "test": ["provider"],
}

# Completion-related patterns
_AT_PATH_AT_CURSOR_RE = re.compile(r"@([^\s]*)$")
_PATH_COMPLETION_LIMIT = 300
# Argument hints: dim ghost text appended after cursor to guide
# free-form input.
# Key = (command, subcommand_or_None, arg_index) → hint text.
_ARG_HINTS: dict[tuple, str] = {
    ("/camera", "capture", 2): " <filename.jpg>",
    ("/camera", "record", 2): " <duration> <filename.mp4>",
    ("/camera", "record", 3): " <filename.mp4>",
    ("/voice", "model", 2): " <model_name>",
    ("/voice", "silence", 2): " <seconds>",
    ("/voice", "max", 2): " <seconds>",
    ("/voice", "threshold", 2): " <rms>",
    ("/tts", "model", 2): " <model_name>",
    ("/tts", "voice", 2): " <voice_name>",
    ("/tts", "speed", 2): " <0.5-2.0>",
    ("/tts", "say", 2): " <text>",
    ("/rule", "add", 2): ' "<rule text>"',
    ("/rule", "remove", 2): " <number>",
    ("/rule", "edit", 2): " <number> <new content>",
    ("/profile", "search", 2): " <keyword>",
    ("/profile", "add", 2): " <content>",
    ("/profile", "remove", 2): " <number>",
    ("/memory", "search", 2): " <keyword>",
    ("/memory", "remove", 2): " <id|num>",
    ("/mcp", "add", 2): " <svc>",
    ("/mcp", "remove", 2): " <svc>",
    ("/skill", None, 2): " <args...>",
}
