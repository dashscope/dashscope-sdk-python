# -*- coding: utf-8 -*-
"""
Tool chain composition — common workflow patterns for effective tool usage.
Provides guidance on combining tools for multi-step operations.
"""

from __future__ import annotations

from pathlib import Path

from dashscope.acli.utils.keywords import (
    expand_scoring_terms,
    extract_keywords,
)

# Common tool chain patterns with examples
TOOL_CHAINS = {
    "file_edit": {
        "description": "编辑文件的完整流程",
        "steps": [
            "1. read_file 读取当前内容",
            "2. write_file 写入修改后的完整内容",
            "3. 可选: run_command 验证（如 python -m py_compile 检查语法）",
        ],
        "example": (
            "用户: '修改 config.py 中的端口为 8080'\n"
            "→ read_file config.py → 修改内容 → write_file config.py"
        ),
    },
    "code_refactor": {
        "description": "重构代码的完整流程",
        "steps": [
            "1. search_files 找到所有使用位置",
            "2. read_file 阅读相关代码",
            "3. create_plan 制定重构计划",
            "4. write_file 逐个修改",
            "5. run_command 运行测试验证",
        ],
        "example": (
            "用户: '重构 auth 模块，提取公共函数'\n"
            "→ search_files 'def auth' → read_file → create_plan "
            "→ write_file → run_command pytest"
        ),
    },
    "bug_fix": {
        "description": "修复 bug 的完整流程",
        "steps": [
            "1. read_file 阅读相关代码",
            "2. search_files 搜索相关上下文",
            "3. write_file 修复代码",
            "4. run_command 运行测试验证",
        ],
        "example": (
            "用户: '修复登录时的空指针错误'\n"
            "→ read_file auth.py → search_files 'login' "
            "→ write_file 修复 → run_command pytest tests/test_auth.py"
        ),
    },
    "project_setup": {
        "description": "设置新项目的完整流程",
        "steps": [
            "1. create_directory 创建目录结构",
            "2. write_file 创建配置文件（pyproject.toml, .gitignore 等）",
            "3. write_file 创建基础代码文件",
            "4. run_command 初始化（git init, pip install 等）",
        ],
        "example": (
            "用户: '创建一个 FastAPI 项目'\n"
            "→ create_directory → write_file pyproject.toml "
            "→ write_file main.py → run_command git init"
        ),
    },
    "code_review": {
        "description": "审查代码的完整流程",
        "steps": [
            "1. list_directory 查看项目结构",
            "2. read_file 阅读关键文件",
            "3. search_files 搜索潜在问题模式",
            "4. 输出审查报告",
        ],
        "example": (
            "用户: '审查这个项目的安全性'\n"
            "→ list_directory → read_file 关键文件 → search_files 危险模式 "
            "→ 输出报告"
        ),
    },
}

# Trigger phrases that directly activate a chain.
_CHAIN_TRIGGERS = {
    "file_edit": ("修改", "编辑", "更新一下", "改一下", "改动", "edit", "modify"),
    "code_refactor": ("重构", "refactor"),
    "bug_fix": ("修复", "修一下", "fix", "bug", "报错", "错误", "崩溃", "error"),
    "project_setup": ("初始化", "创建项目", "新项目", "搭建", "setup", "scaffold"),
    "code_review": ("审查", "评审", "review"),
}


def get_relevant_chains(query: str = "", limit: int = 2) -> str:
    """Return at most ``limit`` chain patterns relevant to the query.

    Matching: trigger phrases first, then multi-char keyword overlap with
    the chain text (requires >=2 distinct hits so generic words like
    "文件"/"file" can't match every chain). Steps only — examples are
    omitted to keep the injected section small. Empty query injects nothing.
    """
    if not query.strip():
        return ""
    lowered = query.lower()
    keywords = expand_scoring_terms(extract_keywords(query))

    matched: list[str] = []
    for name, chain in TOOL_CHAINS.items():
        triggers = _CHAIN_TRIGGERS.get(name, ())
        if any(t in lowered for t in triggers):
            matched.append(name)
            continue
        haystack = (
            str(chain["description"]) + " " + " ".join(chain["steps"])
        ).lower()
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits >= 2:
            matched.append(name)

    if not matched:
        return ""
    lines = ["\n\n## 相关工具链模式"]
    for name in matched[:limit]:
        chain = TOOL_CHAINS[name]
        lines.append(f"\n### {chain['description']}")
        for step in chain["steps"]:
            lines.append(f"  {step}")
    return "\n".join(lines)


def get_fallback_hints(tool_name: str) -> str:
    """Get fallback tool suggestions when a tool fails."""
    fallbacks = {
        "read_file": "如果 read_file 失败，先用 list_directory 确认路径是否正确",
        "search_files": "如果 search_files 失败，尝试 run_command grep 或 rg 命令",
        "list_directory": "如果 list_directory 失败，尝试 run_command ls 命令",
        "run_command": "如果 run_command 失败，检查命令语法或使用其他工具",
    }
    hint = fallbacks.get(tool_name, "")
    if hint:
        return f"\n**备选方案**: {hint}"
    return ""


class ToolChainLibrary:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir

    def get_relevant_chains(self, query: str = "") -> str:
        return get_relevant_chains(query)

    def get_fallback_hints(self, tool_name: str) -> str:
        return get_fallback_hints(tool_name)
