"""Shared command definitions and utilities used by both CLI and TUI.

This module contains UI-agnostic pieces of slash-command handling so that
command behavior only needs to be updated in one place.
"""

from __future__ import annotations

import os
import subprocess as _sp
from typing import Any

# Single source of truth for the /help menu. CLI and TUI both render from
# this structure so that adding or renaming a command only requires one edit.
HELP_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "会话",
        [
            ("/help", "显示帮助"),
            ("/clear", "清空对话历史"),
            ("/info", "查看当前运行信息（provider、model、配置等）"),
            ("/stats", "查看会话统计（工具调用次数、模型信息）"),
            ("/voice", "语音输入控制 (on/off/model/silence/max/threshold)"),
            ("/tts", "语音输出控制 (on/off/status/model/voice/speed/say/last)"),
            ("/camera capture [file]", "摄像头拍照"),
            ("/camera record [duration] [file]", "摄像头录像(默认5s)"),
            ("/copy", "复制最近一条回复到剪贴板"),
            ("/save [path]", "保存最近一条回复到文件（默认 acli_output_<时间>.md）"),
            ("/json on|off", "JSON 输出模式开关（开启后 Agent 回复强制 JSON 格式）"),
            ("/compress", "压缩对话上下文（LLM 摘要后替换历史）"),
            ("/history", "对话历史管理 (stats/list/export/clear)"),
            ("/feedback good|bad", "标注任务满意度（存入经验记忆）"),
            ("/report", "生成 trace 性能报告"),
            ("/log", "查看调试模式记录的 LLM prompt (tail [N]/search <关键词>/clear)，支持翻页"),
            ("/trace", "查看执行 trace（函数调用/耗时/数据流）(tail [N]/search <关键词>/clear)，支持翻页"),
            ("/exit", "退出"),
        ],
    ),
    (
        "配置",
        [
            ("/setup", "初始化 Workspace（用户名、Provider、模型、能力）"),
            ("/capability", "能力开关 (list/enable/disable/reload/config)"),
            ("/subagents", "子代理管理 (list/reload/enable/disable/config)"),
            (
                "/provider",
                "依次配置 Provider / Key / 模型 / 协议（问答式）",
            ),
            ("/trust", "本次对话工具授信/拒绝缓存 (list/clear/allow/deny)"),
            (
                "/rule",
                "用户长效操作规则 (list/add/remove/edit/clear)，每轮注入 system prompt",
            ),
            ("/privacy", "隐私模式 (on/off/status)，启用后数据本地化"),
            ("/debug", "调试模式 (on/off/status)，开启后最终 LLM prompt 写入日志"),
            ("/theme", "主题设置 (list/set/自定义颜色)"),
            ("/directives", "Directives 自动学习提议 (proposals/accept/reject)"),
        ],
    ),
    (
        "能力",
        [
            ("/profile", "用户档案 (list/search/add/remove/clear)"),
            ("/memory", "对话历史 (list/search/remove <id|num>/clear)"),
            ("/session", "会话管理 (new/list/switch/rename/remove)"),
            ("/summarize", "总结当前任务，记录关键步骤和教训"),
            ("/mcp", "MCP 服务 (list/add/remove)"),
            (
                "/skill",
                "技能调用与管理 (list/add/remove/install/uninstall/enable/disable/update)",
            ),
            ("/cron", "定时任务 (add/list/remove/pause/resume)"),
            ("/audit", "审计日志 (recent [N]/query/clear)"),
        ],
    ),
    (
        "开发 / 扩展",
        [
            ("/dev", "总览（含运行时模型注册 + 扩展指南）"),
            (
                "/dev model add|list|remove <provider> [<name>]",
                "给 provider 注册/列出/删除模型（持久化到 workspace）",
            ),
            (
                "/dev provider add|list|remove [name]",
                "Layer-1 扩展 LLM Provider（OpenAI 兼容），写入 custom-extensions.toml",
            ),
            (
                "/dev capability add|list|remove [key]",
                "Layer-1 扩展 HTTP 工具能力，scaffold + 编辑",
            ),
            (
                "/dev skill add|list|remove [name]",
                "Layer-1 自定义 Skill（Prompt 模板），写入 custom-extensions.toml",
            ),
            (
                "/dev tool add|list|remove [name]",
                "Layer-1 自定义 Shell 工具（命令包装为 LLM 工具）",
            ),
            (
                "/dev debug tools|schema|call|prompt",
                "调试：已注册工具 / 参数 Schema / 手动调用 / system prompt",
            ),
            (
                "/dev test provider <name> | reload | log",
                "测试 provider 连通性 / 热重载扩展 / 工具注册统计",
            ),
            (
                "/dev platform | tool | skill",
                "Layer-2 写真实 Python 模块的扩展指南（打印步骤）",
            ),
            ("/example", "列出可用示例项目"),
            (
                "/example download <name>",
                "合并示例到 ./.acli/（冲突自动备份，restore 可撤销）",
            ),
            ("/example restore", "恢复 .acli/backup/ 备份（撤销最近一次合并）"),
        ],
    ),
]

# Examples shown at the bottom of /help.
_HELP_EXAMPLES = [
    "列出当前目录的文件",
    "创建一个 test.txt，内容为 hello world",
    "/mcp add code-interpreter",
    "/cron add every 5m /skill my-skill arg1",
    "/history export history.json --format json",
    "/json on",
    "/save output.md",
]


def render_help_text() -> str:
    """Return the /help content as Rich-tagged text."""
    lines = ["[bold]可用命令[/bold]"]
    for title, items in HELP_SECTIONS:
        lines.append(f"\n[bold yellow]{title}[/bold yellow]")
        for cmd_text, desc in items:
            lines.append(f"  [cyan]{cmd_text}[/cyan] [dim]—[/dim] {desc}")
    lines.append("\n[bold]使用示例[/bold]")
    for ex in _HELP_EXAMPLES:
        lines.append(f"  [dim]·[/dim] {ex}")
    return "\n".join(lines)


def handle_shell_escape(shell_cmd: str) -> tuple[str, str, int]:
    """Execute a shell escape command and return (stdout, stderr, rc).

    Used by the TUI where output must be captured and rendered into the
    RichLog. The CLI keeps its own direct-terminal implementation for
    streaming interactive commands.
    """
    env = os.environ.copy()
    env["ACLI_CLI"] = "1"
    try:
        proc = _sp.run(
            shell_cmd,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except Exception as e:
        return "", str(e), 1
