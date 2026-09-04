# -*- coding: utf-8 -*-
# pylint: disable=too-many-return-statements,too-many-branches
from __future__ import annotations

import asyncio
import locale
import os
import re
import shlex
import shutil
import sys

from dashscope.acli.tools.registry import PermissionLevel, tool

MAX_OUTPUT_LENGTH = 10000
MAX_OUTPUT_BYTES = 5 * 1024 * 1024  # 5 MB hard limit before truncation
DEFAULT_TIMEOUT = 30
IS_WINDOWS = os.name == "nt"


def _resolve_output_encoding() -> str:
    """Encoding for decoding subprocess output.

    Resolved at import, not per call: the TUI replaces ``sys.stdout`` with
    textual's capture object, which has no ``.encoding``, so reading it later
    raises AttributeError and fails every command.
    """
    if not IS_WINDOWS:
        return "utf-8"
    return (
        getattr(sys.stdout, "encoding", None)
        or getattr(sys.__stdout__, "encoding", None)
        or locale.getpreferredencoding(False)
        or "utf-8"
    )


def _resolve_win_shell() -> tuple[str, ...]:
    """argv prefix for running a command on Windows.

    ``-NoProfile`` matters for latency: user profile scripts run on every
    spawn, and each run_command pays that cost.
    """
    exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    return (exe, "-NoProfile", "-NonInteractive", "-Command")


OUTPUT_ENCODING = _resolve_output_encoding()
WIN_SHELL = _resolve_win_shell()

BLOCKED_PATTERNS = [
    # POSIX
    "mkfs",
    "dd if=",
    "> /dev/sd",
    "chmod -R 777 /",
    "shred",
    "wipefs",
    # rm with this flag always targets `/` — no benign use exists
    "--no-preserve-root",
    # Windows
    "rd /s /q",
    "del /f /q /s",
    "> \\\\.\\PhysicalDrive",
]

# Patterns that need token-level precision (plain substring matching would
# false-positive on everyday commands):
# - root-wipe rm: `rm -rf /` or `rm -rf /*`, but NOT `rm -rf /tmp/build`
# - the Windows `format` command as an actual command token, but NOT
#   `clang-format`, `terraform`, or the word "information"
_RM_ROOT_RE = re.compile(r"\brm\s+-\w*[rf]\w*\s+/(?:\s*\*?\s*(?:$|[;&|]))")
_FORMAT_CMD_RE = re.compile(r"(?:^|[;&|]\s*)format(?:\s|$)")

# Home / cwd wipes: `rm -rf ~`, `rm -rf .`, but NOT `rm -rf ./build`.
_RM_HOME_RE = re.compile(
    r"\brm\s+(?:-\w+\s+)*-\w*[rf]\w*\s+(?:--\s+)?(?:~|\.\.?)/?"
    r"(?=\s|$|[;&|])",
)
# Fork bombs: a function that pipes itself into itself in the background,
# e.g. `:(){ :|:& };:` or `bomb(){ bomb|bomb& };bomb`.
_FORK_BOMB_RE = re.compile(
    r"(\S+)\s*\(\)\s*\{\s*\1\s*\|\s*\1\s*&\s*\}\s*;",
)
# Redirects writing to raw block devices (`> /dev/sda`, `>>/dev/nvme0n1`).
_DEV_WRITE_RE = re.compile(
    r">\s*/dev/(?:sd|hd|vd|xvd|nvme|mmcblk|disk)\w*",
)
# Recursive chown of whole system trees (`chown -R u:g /etc`, `/`, ...).
# `/home` & co only match bare: `chown -R u /home/lzs` is everyday work.
_CHOWN_SYSTEM_RE = re.compile(
    r"\bchown\s+(?:-\w+\s+)*-R\s+\S+\s+"
    r"(?:/(?:etc|usr|bin|sbin|lib64|lib|boot|dev|sys)(?:[/\s;&|]|$)"
    r"|/(?:home|var|opt|srv|root)?(?:[\s;&|]|$))",
)
# Power-state commands as actual command tokens (incl. `sudo reboot`).
_SYSTEM_STATE_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\s+)?"
    r"(?:shutdown|reboot|halt|poweroff)(?:\s|$|[;&|])",
)
_INIT_HALT_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\s+)?init\s+[06](?:\s|$|[;&|])",
)
_SYSTEMCTL_HALT_RE = re.compile(
    r"\bsystemctl\s+(?:-\S+\s+)*(?:poweroff|reboot|halt)(?:\s|$|[;&|])",
)
# Killing PID 1 drags the whole system down with it.
_KILL_PID1_RE = re.compile(r"\bkill\s+(?:-\S+\s+)*1(?:\s|$|[;&|])")
# Shell-history destruction (`history -c`, `history -cw`).
_HISTORY_CLEAR_RE = re.compile(r"\bhistory\s+-\w*c")
# Remote-exec pipes: `curl ... | sh`, `wget ... | bash`, with any flags
# between the downloader and the pipe.
_PIPE_TO_SHELL_RE = re.compile(
    r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|dash|ksh)\b",
)
# Partition-table editors as actual command tokens.
_PARTITION_CMD_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:sudo\s+)?(?:fdisk|parted)(?:\s|$)",
)
# macOS: whole-disk erase and Secure Boot bypass.
_DISKUTIL_ERASE_RE = re.compile(r"\bdiskutil\s+eraseDisk\b")
_CSRUTIL_DISABLE_RE = re.compile(r"\bcsrutil\s+disable\b")

# (label, regex) pairs checked by run_command after BLOCKED_PATTERNS;
# the label is quoted in the block error message.
_BLOCKED_REGEXES: list[tuple[str, re.Pattern[str]]] = [
    ("rm -rf ~", _RM_HOME_RE),
    ("fork bomb", _FORK_BOMB_RE),
    ("> /dev/<block device>", _DEV_WRITE_RE),
    ("chown -R on system paths", _CHOWN_SYSTEM_RE),
    ("shutdown/reboot/halt/poweroff", _SYSTEM_STATE_RE),
    ("init 0/init 6", _INIT_HALT_RE),
    ("systemctl poweroff/reboot/halt", _SYSTEMCTL_HALT_RE),
    ("kill PID 1", _KILL_PID1_RE),
    ("history -c", _HISTORY_CLEAR_RE),
    ("curl/wget piped into a shell", _PIPE_TO_SHELL_RE),
    ("fdisk/parted", _PARTITION_CMD_RE),
    ("diskutil eraseDisk", _DISKUTIL_ERASE_RE),
    ("csrutil disable", _CSRUTIL_DISABLE_RE),
]

# ----- Read-only command classifier ---------------------------------------
# Used by the executor to auto-approve obvious inspection commands so the
# user isn't asked to confirm every `grep`/`ls`/`git status`. Conservative
# by design: anything unrecognized stays in the CONFIRM lane.

_SAFE_BINS = frozenset(
    {
        # Listings + content inspection
        "ls",
        "ll",
        "la",
        "pwd",
        "cd",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "tree",
        # Search
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ag",
        "ack",
        "find",
        "fd",
        "locate",
        # Text processing (no in-place edits unless `sed -i` / `awk` writes —
        # caught by the redirect check below)
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "column",
        "paste",
        "comm",
        # Identity / environment probes
        "echo",
        "printf",
        "which",
        "type",
        "file",
        "stat",
        "basename",
        "dirname",
        "whoami",
        "id",
        "groups",
        "uname",
        "uptime",
        "hostname",
        "date",
        "history",
        "printenv",
        "whereis",
        "realpath",
        "readlink",
        # Process / disk probes
        "ps",
        "top",
        "htop",
        "df",
        "du",
        "free",
        "lsof",
        "netstat",
        "ss",
        # Network probes (GET only; redirects caught by _has_redirection,
        # write/exfil flags caught by _UNSAFE_FLAGS)
        "curl",
        "wget",
        "ping",
        "dig",
        "nslookup",
        "traceroute",
        # Structured data
        "jq",
        "yq",
        "xxd",
        "hexdump",
        "od",
        "diff",
        "cmp",
        "md5",
        "sha256sum",
        # Binary inspection
        "strings",
        "ldd",
        "otool",
        "nm",
        # No-ops
        "true",
        "false",
        ":",
        # Windows equivalents (read-only only; powershell/wmic/taskkill/net
        # are NOT here — they can execute code or mutate system state)
        "dir",
        "where",
        "systeminfo",
        "ipconfig",
        "tasklist",
        "set",
        "ver",
        "vol",
    },
)

# Binaries whose first positional sub-command determines safety. E.g.
# `git status` is read-only, `git push` is not. Anything not listed = unsafe.
_SUBCMD_SAFE: dict[str, frozenset[str]] = {
    "git": frozenset(
        {
            "status",
            "log",
            "diff",
            "show",
            "branch",
            "remote",
            "tag",
            "blame",
            "ls-files",
            "ls-tree",
            "rev-parse",
            "describe",
            "shortlog",
            "reflog",
            "grep",
            "cat-file",
            "rev-list",
            "name-rev",
            "for-each-ref",
        },
    ),
    "pip": frozenset({"list", "show", "freeze", "check", "-V", "--version"}),
    "pip3": frozenset({"list", "show", "freeze", "check", "-V", "--version"}),
    "npm": frozenset(
        {"list", "ls", "view", "outdated", "audit", "doctor", "config"},
    ),
    "yarn": frozenset({"list", "info", "audit", "why"}),
    "pnpm": frozenset({"list", "ls", "view", "outdated", "audit", "why"}),
    "docker": frozenset(
        {
            "ps",
            "images",
            "inspect",
            "logs",
            "stats",
            "version",
            "info",
            "history",
        },
    ),
    "kubectl": frozenset(
        {"get", "describe", "logs", "version", "explain", "api-resources"},
    ),
    "brew": frozenset({"list", "info", "search", "config", "doctor", "deps"}),
    "go": frozenset({"version", "env", "list", "vet"}),
    "cargo": frozenset({"version", "tree"}),
    "mvn": frozenset(
        {"-version", "help:effective-pom"},
    ),  # rare; mvn usually writes
    # Python interpreter: only version / help are safe (-c runs arbitrary code)
    "python": frozenset({"--version", "-V", "--help", "-h"}),
    "python3": frozenset({"--version", "-V", "--help", "-h"}),
    "node": frozenset({"--version", "-v", "--help", "-h"}),
    "ruby": frozenset({"--version", "-v", "--help", "-h"}),
    "java": frozenset({"-version", "--version", "-help"}),
}

# Any of these as a bare token anywhere in the line → unsafe, no matter
# what the head binary is. Catches pipelines like `grep foo file && rm bar`.
_UNSAFE_TOKENS = frozenset(
    {
        # POSIX
        "rm",
        "rmdir",
        "mv",
        "cp",
        "ln",
        "chmod",
        "chown",
        "chgrp",
        "sudo",
        "su",
        "doas",
        "kill",
        "killall",
        "pkill",
        "dd",
        "mkfs",
        "fdisk",
        "parted",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "systemctl",
        "service",
        "launchctl",
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "snap",
        "pacman",
        "make",
        "cmake",
        "ninja",
        "tee",  # writes to a file
        # Windows
        "reg",
        "sc",
        "schtasks",
        "diskpart",
        "bcdedit",
        "bootrec",
        "sfc",
        "dism",
    },
)

# Redirects / write-side operators that taint any otherwise-safe command.
# `2>&1`-style fd duplication is harmless, so strip it first; any remaining
# `<` or `>` (incl. `2>file`, `>>file`, `&>file`) writes/reads a file.
_FD_DUP_RE = re.compile(r"\d*>&\d+")


def _has_redirection(chunk: str) -> bool:
    """True if the chunk redirects to/from a file (2>&1 dupes excluded)."""
    s = _FD_DUP_RE.sub("", chunk)
    return ">" in s or "<" in s


# Dangerous flags within otherwise-safe binaries. Long flags also match the
# `--flag=value` form; short flags also match glued values (`-ofile`).
_UNSAFE_FLAGS: dict[str, tuple[str, ...]] = {
    # Network write / exfiltration
    "curl": (
        "-o",
        "-O",
        "-F",
        "-T",
        "-d",
        "--output",
        "--remote-name",
        "--upload-file",
        "--form",
        "--data",
        "--data-raw",
        "--data-binary",
        "--data-urlencode",
    ),
    "wget": (
        "-O",
        "--output-document",
        "--post-data",
        "--post-file",
    ),
    # Filesystem mutation / command execution
    "find": ("-delete", "-exec", "-execdir", "-ok", "-okdir"),
}


def _has_unsafe_flags(bin_name: str, args: list[str]) -> bool:
    flags = _UNSAFE_FLAGS.get(bin_name)
    if not flags:
        return False
    for token in args:
        for flag in flags:
            if token == flag or token.startswith(flag + "="):
                return True
            # short options can be glued to their value: -ofile
            if (
                not flag.startswith("--")
                and token.startswith(flag)
                and len(token) > len(flag)
            ):
                return True
    return False


# Dual-purpose git subcommands: bare/listing forms are read-only, but some
# flags or positional args mutate the repo.
_GIT_SUBCMD_DENY: dict[str, frozenset[str]] = {
    "branch": frozenset({"-d", "-D", "-m", "-M", "-c", "-C"}),
    "tag": frozenset({"-d", "-a", "-s", "-m", "-f", "-u"}),
    "remote": frozenset(
        {
            "add",
            "remove",
            "rm",
            "rename",
            "set-url",
            "set-head",
            "set-branches",
        },
    ),
}

# Read-only sub-subcommands of `git remote`.
_GIT_REMOTE_READONLY = frozenset({"show", "get-url"})


def _git_args_safe(sub: str, args: list[str]) -> bool:
    """Extra guard for dual-purpose git subcommands (branch/tag/remote)."""
    deny = _GIT_SUBCMD_DENY.get(sub)
    if deny is None:
        return True
    if any(t in deny for t in args):
        return False
    if sub == "remote":
        if not args:
            return True
        first, rest = args[0], args[1:]
        if first in _GIT_REMOTE_READONLY:
            return True  # show/get-url are read-only regardless of their args
        return first.startswith("-") and all(t.startswith("-") for t in rest)
    if sub == "tag" and ("-l" in args or "--list" in args):
        return True  # listing form: `git tag -l 'v*'`
    # `git branch foo` / `git tag v1` create refs — only flag-only forms
    # (listing) stay safe.
    return all(t.startswith("-") for t in args)


def is_safe_readonly(cmd: str) -> bool:
    """True if ``cmd`` is a pure inspection command that's safe to run
    without asking. Conservative: any unrecognized binary, any chained
    write op, or any unsafe token returns False so the confirm prompt
    stays in place."""
    s = cmd.strip()
    if not s:
        return False
    if "$(" in s or "`" in s:
        return False  # command substitution could hide anything
    # Each pipeline / chain segment must be individually safe. Split on
    # newlines too: `ls\npython3 -c ...` is two commands. Lone `&` is not
    # split on — it would chop `2>&1` apart, and the _UNSAFE_TOKENS /
    # _UNSAFE_FLAGS checks still catch `sleep & rm bar` as a single chunk.
    chunks = re.split(r"\s*(?:\|\||&&|;|\||\n)\s*", s)
    return all(_is_safe_single(c) for c in chunks if c.strip())


def _is_safe_single(chunk: str) -> bool:
    if _has_redirection(chunk):
        return False
    try:
        tokens = shlex.split(chunk, posix=not IS_WINDOWS)
    except ValueError:
        return False
    if not tokens:
        return False
    # Skip leading VAR=value assignments (FOO=1 BAR=2 grep ...)
    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", tokens[i]):
        i += 1
    if i >= len(tokens):
        return False
    if any(t in _UNSAFE_TOKENS for t in tokens[i:]):
        return False
    bin_name = os.path.basename(tokens[i].replace(".exe", ""))
    args = tokens[i + 1 :]
    if bin_name == "env":
        # `env` / `env VAR=1` only prints the environment; `env bash -c ...`
        # re-invokes an arbitrary command, so only assignments are allowed.
        return all(re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", t) for t in args)
    if bin_name in _SAFE_BINS:
        return not _has_unsafe_flags(bin_name, args)
    if bin_name in _SUBCMD_SAFE:
        sub = args[0] if args else ""
        if sub not in _SUBCMD_SAFE[bin_name]:
            return False
        if bin_name == "git":
            return _git_args_safe(sub, args[1:])
        return True
    return False


@tool(
    name="run_command",
    description=(
        "Execute a shell command and return its output. "
        "Dangerous commands are blocked."
    ),
    permission=PermissionLevel.CONFIRM,
)
async def run_command(command: str, timeout: int | None = None) -> str:
    if not command or not command.strip():
        return (
            "Error: empty command argument; re-invoke with the "
            "command to run"
        )
    for pattern in BLOCKED_PATTERNS:
        if pattern in command:
            return (
                f"Error: command blocked "
                f"(contains dangerous pattern: {pattern})"
            )
    if _RM_ROOT_RE.search(command):
        return "Error: command blocked (contains dangerous pattern: rm -rf /)"
    if _FORMAT_CMD_RE.search(command):
        return "Error: command blocked (contains dangerous pattern: format)"
    for name, regex in _BLOCKED_REGEXES:
        if regex.search(command):
            return (
                f"Error: command blocked "
                f"(contains dangerous pattern: {name})"
            )

    # Belt-and-suspenders on top of utils.validation.coerce_types: if the model
    # still managed to slip something un-castable through (e.g. "auto"),
    # fall back to the default instead of letting asyncio.wait_for raise
    # TypeError on `<= 0`.
    try:
        timeout = (
            int(timeout) if timeout not in (None, "", 0) else DEFAULT_TIMEOUT
        )
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT

    try:
        if IS_WINDOWS:
            # argv form, never a shell string: interpolating into
            # 'powershell -Command "..."' routes it through cmd.exe first and
            # any embedded double quote corrupts the command.
            proc = await asyncio.create_subprocess_exec(
                *WIN_SHELL,
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
            )
        else:
            # Optional OS sandbox (opt-in; degrades to normal execution
            # when disabled or when no backend is available).
            from dashscope.acli import sandbox

            sandbox_argv = None
            if sandbox.is_enabled():
                sandbox_argv = sandbox.build_argv(command, os.getcwd())
            if sandbox_argv is not None:
                proc = await asyncio.create_subprocess_exec(
                    *sandbox_argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.getcwd(),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=os.getcwd(),
                )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"Error: command timed out ({timeout}s)"
    except Exception as e:
        return f"Error: execution failed - {e}"

    # Enforce hard size limits to prevent OOM on huge outputs
    stdout_bytes = len(stdout) if stdout else 0
    stderr_bytes = len(stderr) if stderr else 0
    total_bytes = stdout_bytes + stderr_bytes

    if total_bytes > MAX_OUTPUT_BYTES:
        # Truncate raw bytes before decoding
        if stdout_bytes > MAX_OUTPUT_BYTES // 2:
            stdout = stdout[: MAX_OUTPUT_BYTES // 2]
        if stderr_bytes > MAX_OUTPUT_BYTES // 2:
            stderr = stderr[: MAX_OUTPUT_BYTES // 2]

    output = ""
    if stdout:
        output += stdout.decode(OUTPUT_ENCODING, errors="replace")
    if stderr:
        err = stderr.decode(OUTPUT_ENCODING, errors="replace")
        output += "\n[stderr]\n" + err

    if len(output) > MAX_OUTPUT_LENGTH:
        output = (
            output[:MAX_OUTPUT_LENGTH]
            + f"\n\n... (output truncated, {total_bytes} bytes total)"
        )

    exit_info = f"\n[exit code: {proc.returncode}]"
    return output.strip() + exit_info
