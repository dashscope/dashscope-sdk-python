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
        "description": "Full flow for editing a file",
        "steps": [
            "1. read_file to read the current content",
            "2. write_file with the full updated content",
            "3. Optional: run_command to verify (e.g. python -m py_compile "
            "to check syntax)",
        ],
        "example": (
            "User: 'change the port in config.py to 8080'\n"
            "→ read_file config.py → edit content → write_file config.py"
        ),
    },
    "code_refactor": {
        "description": "Full flow for refactoring code",
        "steps": [
            "1. search_files to find all usages",
            "2. read_file to read the relevant code",
            "3. create_plan for the refactor plan",
            "4. write_file to edit files one by one",
            "5. run_command to run tests and verify",
        ],
        "example": (
            "User: 'refactor the auth module, extract shared functions'\n"
            "→ search_files 'def auth' → read_file → create_plan "
            "→ write_file → run_command pytest"
        ),
    },
    "bug_fix": {
        "description": "Full flow for fixing a bug",
        "steps": [
            "1. read_file to read the relevant code",
            "2. search_files for related context",
            "3. write_file to apply the fix",
            "4. run_command to run tests and verify",
        ],
        "example": (
            "User: 'fix the null pointer error on login'\n"
            "→ read_file auth.py → search_files 'login' "
            "→ write_file fix → run_command pytest tests/test_auth.py"
        ),
    },
    "project_setup": {
        "description": "Full flow for setting up a new project",
        "steps": [
            "1. create_directory for the directory layout",
            "2. write_file for config files (pyproject.toml, .gitignore, "
            "etc.)",
            "3. write_file for starter code files",
            "4. run_command to initialize (git init, pip install, etc.)",
        ],
        "example": (
            "User: 'create a FastAPI project'\n"
            "→ create_directory → write_file pyproject.toml "
            "→ write_file main.py → run_command git init"
        ),
    },
    "code_review": {
        "description": "Full flow for reviewing code",
        "steps": [
            "1. list_directory to view the project structure",
            "2. read_file to read key files",
            "3. search_files for potential problem patterns",
            "4. Output the review report",
        ],
        "example": (
            "User: 'review this project's security'\n"
            "→ list_directory → read_file key files → search_files "
            "risky patterns → output report"
        ),
    },
}

# Trigger phrases that directly activate a chain. Bilingual on purpose:
# they are substring-matched against the raw user query.
_CHAIN_TRIGGERS = {
    "file_edit": (
        "修改",
        "编辑",
        "更新一下",
        "改一下",
        "改动",
        "edit",
        "modify",
        "update",
        "change",
    ),
    "code_refactor": ("重构", "refactor"),
    "bug_fix": (
        "修复",
        "修一下",
        "fix",
        "bug",
        "报错",
        "错误",
        "崩溃",
        "error",
        "crash",
    ),
    "project_setup": (
        "初始化",
        "创建项目",
        "新项目",
        "搭建",
        "setup",
        "scaffold",
        "init",
        "new project",
        "create project",
    ),
    "code_review": ("审查", "评审", "review"),
}


def get_relevant_chains(query: str = "", limit: int = 2) -> str:
    """Return at most ``limit`` chain patterns relevant to the query.

    Matching: trigger phrases first, then multi-char keyword overlap with
    the chain text (requires >=2 distinct hits so generic words like
    "file" can't match every chain). Steps only — examples are
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
    lines = ["\n\n## Relevant tool chain patterns"]
    for name in matched[:limit]:
        chain = TOOL_CHAINS[name]
        lines.append(f"\n### {chain['description']}")
        for step in chain["steps"]:
            lines.append(f"  {step}")
    return "\n".join(lines)


def get_fallback_hints(tool_name: str) -> str:
    """Get fallback tool suggestions when a tool fails."""
    fallbacks = {
        "read_file": (
            "if read_file failed, use list_directory to verify the path"
        ),
        "search_files": (
            "if search_files failed, try run_command with grep or rg"
        ),
        "list_directory": "if list_directory failed, try run_command with ls",
        "run_command": (
            "if run_command failed, check the command syntax or "
            "use another tool"
        ),
    }
    hint = fallbacks.get(tool_name, "")
    if hint:
        return f"\n**Fallback**: {hint}"
    return ""


class ToolChainLibrary:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir

    def get_relevant_chains(self, query: str = "") -> str:
        return get_relevant_chains(query)

    def get_fallback_hints(self, tool_name: str) -> str:
        return get_fallback_hints(tool_name)
