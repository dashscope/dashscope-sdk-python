from __future__ import annotations

import fnmatch
import os
import shutil

from dashscope.acli.tools.registry import PermissionLevel, tool
from dashscope.acli.utils.paths import SENSITIVE_NAMES, validate_path, validate_write_path

# Maximum directory depth for search_files traversal.
_MAX_SEARCH_DEPTH = 8

# Maximum file content size for write_file (50 MB).
_MAX_WRITE_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="read_file",
    description="读取文件内容。可指定起始行和读取行数。",
    permission=PermissionLevel.AUTO,
)
def read_file(path: str, offset: int | None = None, limit: int | None = None) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"错误: {e}"
    if not os.path.isfile(path):
        return f"错误: 文件不存在 - {path}"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    start = offset or 0
    end = start + limit if limit else len(lines)
    selected = lines[start:end]

    result_lines = []
    for i, line in enumerate(selected, start=start + 1):
        result_lines.append(f"{i:>4}\t{line.rstrip()}")
    return "\n".join(result_lines)


@tool(
    name="write_file",
    description="将内容写入文件。如果文件已存在则覆盖。返回结果会包含 unified diff。",
    permission=PermissionLevel.CONFIRM,
)
def write_file(path: str, content: str) -> str:
    import difflib

    try:
        path = validate_write_path(path)
    except ValueError as e:
        return f"错误: {e}"

    if len(content) > _MAX_WRITE_SIZE:
        return f"错误: 内容过大 ({len(content)} 字节)，上限为 {_MAX_WRITE_SIZE // (1024*1024)} MB"

    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # Capture before-state for diff (best-effort — binary or read errors
    # fall through to "new file" treatment).
    old_content = ""
    existed = os.path.isfile(path)
    if existed:
        try:
            with open(path, "r", encoding="utf-8") as f:
                old_content = f.read()
        except (OSError, UnicodeDecodeError):
            existed = False  # treat as new for diff purposes

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    if not existed:
        return f"已创建文件: {path} ({len(content)} 字符)"

    if old_content == content:
        return f"内容未变化: {path}"

    # Cap displayed diff so a 5000-line full rewrite doesn't drown the UI;
    # full content is in messages so LLM still sees everything it wrote.
    # Don't pass lineterm="" — default "\n" gives proper newlines on the
    # ---, +++, @@ control lines so the diff is readable when joined.
    diff_lines = list(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    MAX_DIFF_LINES = 80
    truncated = ""
    if len(diff_lines) > MAX_DIFF_LINES:
        total_added = sum(
            1 for l in diff_lines if l.startswith("+") and not l.startswith("+++")
        )
        diff_lines = diff_lines[:MAX_DIFF_LINES]
        truncated = (
            f"\n... (diff 截断，共 {total_added} 行新增，仅展示前 {MAX_DIFF_LINES} 行)"
        )
    diff_text = "".join(diff_lines)

    return (
        f"已写入文件: {path} ({len(content)} 字符)\n"
        f"--- diff ---\n"
        f"{diff_text}{truncated}"
    )


@tool(
    name="list_directory",
    description="列出目录内容，显示文件类型和大小。",
    permission=PermissionLevel.AUTO,
)
def list_directory(path: str) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"错误: {e}"
    if not os.path.isdir(path):
        return f"错误: 目录不存在 - {path}"

    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            entries.append(f"  [目录] {name}/")
        else:
            size = os.path.getsize(full)
            entries.append(f"  [文件] {name}  ({_format_size(size)})")

    header = f"目录: {path} ({len(entries)} 项)"
    return header + "\n" + "\n".join(entries)


@tool(
    name="create_directory",
    description="创建目录，支持创建多级目录。",
    permission=PermissionLevel.CONFIRM,
)
def create_directory(path: str) -> str:
    try:
        path = validate_write_path(path)
    except ValueError as e:
        return f"错误: {e}"
    os.makedirs(path, exist_ok=True)
    return f"已创建目录: {path}"


@tool(
    name="delete_file",
    description="删除指定文件。此操作不可撤销。",
    permission=PermissionLevel.DANGEROUS,
)
def delete_file(path: str) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"错误: {e}"
    if not os.path.isfile(path):
        return f"错误: 文件不存在 - {path}"
    os.remove(path)
    return f"已删除文件: {path}"


@tool(
    name="delete_directory",
    description="删除指定目录及其所有内容。此操作不可撤销。",
    permission=PermissionLevel.DANGEROUS,
)
def delete_directory(path: str) -> str:
    try:
        path = validate_path(path)
    except ValueError as e:
        return f"错误: {e}"
    if not os.path.isdir(path):
        return f"错误: 目录不存在 - {path}"
    # Prevent deleting cwd itself
    abs_path = os.path.abspath(path)
    cwd = os.path.abspath(".")
    if abs_path == cwd:
        return "错误: 不允许删除当前工作目录"
    shutil.rmtree(path)
    return f"已删除目录: {path}"


@tool(
    name="search_files",
    description="按文件名模式搜索文件。支持通配符如 *.py、test_*。",
    permission=PermissionLevel.AUTO,
)
def search_files(pattern: str, path: str | None = None) -> str:
    search_root = os.path.expanduser(path) if path else os.getcwd()
    try:
        search_root = validate_path(search_root)
    except ValueError as e:
        return f"错误: {e}"
    if not os.path.isdir(search_root):
        return f"错误: 搜索路径不存在 - {search_root}"

    matches = []
    root_depth = search_root.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(search_root):
        # Enforce depth limit
        current_depth = root.rstrip(os.sep).count(os.sep) - root_depth
        if current_depth >= _MAX_SEARCH_DEPTH:
            dirs[:] = []
        # Skip hidden and sensitive directories
        dirs[:] = [
            d for d in dirs if not d.startswith(".") and d not in SENSITIVE_NAMES
        ]
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if name in SENSITIVE_NAMES:
                    continue
                matches.append(os.path.join(root, name))
                if len(matches) >= 100:
                    break
        if len(matches) >= 100:
            break

    if not matches:
        return f"未找到匹配 '{pattern}' 的文件"
    result = f"找到 {len(matches)} 个匹配文件:\n"
    result += "\n".join(f"  {m}" for m in matches)
    return result


@tool(
    name="move_file",
    description="移动或重命名文件/目录。",
    permission=PermissionLevel.CONFIRM,
)
def move_file(src: str, dst: str) -> str:
    try:
        src = validate_path(src)
        dst = validate_write_path(dst)
    except ValueError as e:
        return f"错误: {e}"
    if not os.path.exists(src):
        return f"错误: 源路径不存在 - {src}"
    shutil.move(src, dst)
    return f"已移动: {src} → {dst}"


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
