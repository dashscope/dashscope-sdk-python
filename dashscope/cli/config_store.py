# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read/write access to ``~/.dashscope/config.toml``.

The SDK already *reads* this file (``dashscope._config``) but nothing in
the repo ever wrote it -- ``save_api_key`` writes a separate
``~/.dashscope/api_key``. ``dashscope config set`` is that writer.

Two constraints shaped the implementation:

* **No new dependency.** ``requirements.txt`` carries a TOML *reader*
  (``tomllib``/``tomli``) but no writer. Pulling ``tomlkit`` in would add
  a runtime dependency to every dashscope install for one subcommand.
* **The file is the user's.** It may hold comments and keys this CLI has
  never heard of. A regenerate-from-dict writer would destroy both, so
  this is a line editor: it replaces one line in place or appends one,
  and leaves every other byte alone. Because a line editor can still be
  wrong about what a line means, every candidate edit is re-parsed and
  the key read back before it is allowed to land.

CLI-only defaults live in a ``[cli]`` table. The SDK's reader filters to
its own ``_FILE_KEYS`` allowlist, so it drops that table silently -- the
two readers share one file with no SDK change and no risk of the CLI's
keys leaking into connection config.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dashscope.cli.errors import usage_error

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]

CLI_SECTION = "cli"
API_KEY = "api_key"

# CLI-owned defaults, stored under [cli]. Deliberately disjoint from the
# SDK's _FILE_KEYS so neither reader picks up the other's keys.
CLI_KEYS = frozenset({"output", "default_model"})


def _sdk_keys() -> frozenset:
    """Keys the SDK itself reads from config.toml.

    Taken from the SDK rather than restated here, so adding a config key
    to ``dashscope._config`` automatically makes it settable.
    """
    from dashscope._config import (
        _FILE_KEYS,
    )  # pylint: disable=import-outside-toplevel

    return frozenset(_FILE_KEYS)


def config_path() -> Path:
    """The active config file (honours ``DASHSCOPE_CONFIG``)."""
    from dashscope._config import (  # pylint: disable=import-outside-toplevel
        default_config_path,
    )

    return default_config_path()


def load_raw(path: Optional[Path] = None) -> Dict[str, Any]:
    """Whole config file as a dict; missing file reads as empty."""
    target = path or config_path()
    if not target.is_file():
        return {}
    try:
        with target.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise usage_error(f"invalid TOML in {target}: {exc}") from exc
    except OSError as exc:
        raise usage_error(f"cannot read {target}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def load_cli_section(path: Optional[Path] = None) -> Dict[str, Any]:
    section = load_raw(path).get(CLI_SECTION)
    return section if isinstance(section, dict) else {}


def known_keys() -> Tuple[str, ...]:
    """Every key ``config set`` accepts, in a stable display order."""
    return tuple(sorted(_sdk_keys())) + tuple(sorted(CLI_KEYS))


def bare_key(key: str) -> str:
    """The key's own name inside its table (``cli.output`` → ``output``).

    Also the validation point: an unknown key or section raises here, so
    every command that names a config key gets the same error.
    """
    return _section_for(key)[1]


def _section_for(key: str) -> Tuple[str, str]:
    """Split a key into ``(section, bare_key)``.

    ``cli.output`` and ``output`` both address the [cli] table; SDK keys
    are always top level.
    """
    if "." in key:
        section, _, bare = key.partition(".")
        section = section.strip()
        bare = bare.strip()
        if not section or not bare:
            raise usage_error(f"invalid config key {key!r}")
        if section != CLI_SECTION:
            raise usage_error(
                f"unknown config section [{section}] in {key!r}",
            )
    else:
        bare = key.strip()
        section = CLI_SECTION if bare in CLI_KEYS else ""
    if section == "" and bare not in _sdk_keys():
        raise usage_error(
            f"unknown config key {key!r}; known keys: {', '.join(known_keys())}",
        )
    if section == CLI_SECTION and bare not in CLI_KEYS:
        raise usage_error(
            f"unknown [cli] key {bare!r}; known keys: "
            f"{', '.join(sorted(CLI_KEYS))}",
        )
    return section, bare


def parse_toml_value(raw: str) -> Any:
    """Coerce a command-line string into a typed TOML value.

    Parsing ``v = <raw>`` with the real TOML grammar gets ints, floats,
    bools, arrays and quoted strings exactly right for free. An unquoted
    bare word is not valid TOML, and that is the common case
    (``config set workspace ws-123``), so it falls back to a string.
    """
    text = raw.strip()
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def render_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        # JSON's string escaping is a subset of TOML basic strings, so
        # this round-trips quotes, backslashes and newlines correctly.
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _enum_choices() -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    """``key -> (label, allowed values)`` for the keys with a fixed set.

    Taken from the SDK and the renderer rather than restated here, so it
    cannot drift from what they actually accept.
    """
    from dashscope._config import (  # pylint: disable=import-outside-toplevel
        VALID_PROTOCOLS,
        VALID_REGIONS,
    )
    from dashscope.cli.output import (  # pylint: disable=import-outside-toplevel
        OUTPUT_FORMATS,
    )

    return {
        "protocol": ("protocol", tuple(VALID_PROTOCOLS)),
        "region": ("region", tuple(VALID_REGIONS)),
        "output": ("output format", tuple(OUTPUT_FORMATS)),
    }


# Seconds-valued keys: a float that has to be strictly positive.
_POSITIVE_SECONDS = ("timeout", "http_keepalive_expiry")
# Count-valued keys: a non-negative integer.
_NON_NEGATIVE_INTS = (
    "max_retries",
    "http_max_keepalive_connections",
)
# A pool with no connections cannot issue a request, and `Settings`
# rejects it -- writing 0 here would leave a config file that makes every
# later invocation fail at resolve time.
_POSITIVE_INTS = ("http_max_connections",)


def _validate_number(
    key: str,
    value: Any,
    *,
    integer: bool,
    positive: bool,
) -> Any:
    try:
        number: Any = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        kind = "an integer" if integer else "a number"
        raise usage_error(f"{key} must be {kind}, got {value!r}") from exc
    if positive:
        if number <= 0:
            raise usage_error(f"{key} must be positive, got {number}")
    elif number < 0:
        raise usage_error(f"{key} must be >= 0, got {number}")
    return number


def _validate(key: str, value: Any) -> Any:
    """Reject values the SDK would refuse at resolve time."""
    enums = _enum_choices()
    if key in enums:
        label, allowed = enums[key]
        if value not in allowed:
            raise usage_error(
                f"invalid {label} {value!r}, expected one of "
                f"{', '.join(allowed)}",
            )
        return value
    if key in _POSITIVE_SECONDS:
        return _validate_number(key, value, integer=False, positive=True)
    if key in _NON_NEGATIVE_INTS:
        return _validate_number(key, value, integer=True, positive=False)
    if key in _POSITIVE_INTS:
        return _validate_number(key, value, integer=True, positive=True)
    if not isinstance(value, str):
        raise usage_error(f"{key} must be a string, got {value!r}")
    if not value.strip():
        raise usage_error(f"{key} must not be empty")
    return value


_TRIPLE_QUOTES = ('"""', "'''")


def _scan(line: str, quote: Optional[str]) -> Tuple[str, Optional[str]]:
    """Strip *line*'s comment; return the rest and the still-open delimiter."""
    out: List[str] = []
    index = 0
    while index < len(line):
        character = line[index]
        if quote is None:
            if character == "#":
                break
            triple = line[index : index + 3]  # noqa: E203
            if triple in _TRIPLE_QUOTES:
                quote = triple
                out.append(triple)
                index += 3
                continue
            if character in "\"'":
                quote = character
            out.append(character)
            index += 1
        elif character == "\\":
            out.append(line[index : index + 2])  # noqa: E203
            index += 2
        elif line.startswith(quote, index):
            out.append(quote)
            index += len(quote)
            quote = None
        else:
            out.append(character)
            index += 1
    if quote not in _TRIPLE_QUOTES:
        # A one-line string cannot stay open across a newline; what looks
        # like one is a syntax error for the parser to report, not state
        # to carry into the next line.
        quote = None
    return "".join(out), quote


def _code_lines(lines: List[str]) -> Iterator[Tuple[int, str]]:
    """``(index, line)`` for the lines that carry TOML syntax.

    Comments come back stripped, so ``[cli] # mine`` still reads as a
    table header. Lines inside a multi-line string are dropped: their
    ``[table]`` and ``key = value`` shaped text is data, and editing it
    would corrupt the string.
    """
    quote: Optional[str] = None
    for index, line in enumerate(lines):
        continuation = quote in _TRIPLE_QUOTES
        text, quote = _scan(line, quote)
        if not continuation:
            yield index, text


def _header_name(code: str) -> Optional[str]:
    """The table name if *code* is a ``[table]`` header, else None."""
    stripped = code.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    return stripped[1:-1].strip().strip("\"'")


def _find_key_line(
    lines: List[str],
    section: str,
    key: str,
) -> Optional[int]:
    """Index of ``key =`` inside *section*, or None."""
    current = ""
    for index, code in _code_lines(lines):
        header = _header_name(code)
        if header is not None:
            current = header
            continue
        if current != section or "=" not in code:
            continue
        name = code.split("=", 1)[0].strip().strip("\"'")
        if name == key:
            return index
    return None


def _section_bounds(
    lines: List[str],
    section: str,
) -> Optional[Tuple[int, int]]:
    """``(header_index, insert_index)`` for *section*, or None if absent."""
    header: Optional[int] = None
    for index, code in _code_lines(lines):
        name = _header_name(code)
        if name is None:
            continue
        if header is None:
            if name == section:
                header = index
        elif name != section:
            return header, index
    if header is None:
        return None
    return header, len(lines)


def _drop_trailing_blanks(lines: List[str]) -> None:
    while lines and not lines[-1].strip():
        lines.pop()


def _append_toplevel_key(lines: List[str], rendered: str) -> None:
    """Insert ``key = value`` ahead of the first table header.

    A top-level key must precede every ``[table]``, or TOML would read it
    as belonging to the last table.
    """
    first_table = next(
        (
            index
            for index, code in _code_lines(lines)
            if _header_name(code) is not None
        ),
        None,
    )
    if first_table is None:
        _drop_trailing_blanks(lines)
        lines.append(rendered)
        return
    lines[first_table:first_table] = [rendered, ""]


def _insert_key(lines: List[str], section: str, rendered: str) -> None:
    """Append *rendered* to *section*, creating the section when absent."""
    bounds = _section_bounds(lines, section)
    if bounds is None:
        if not section:
            _append_toplevel_key(lines, rendered)
            return
        _drop_trailing_blanks(lines)
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        lines.append(rendered)
        return

    _, insert_at = bounds
    # Step back over trailing blanks so the key lands inside its section
    # rather than after the whitespace that closes it.
    while insert_at > 0 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, rendered)


def set_value(
    key: str,
    raw_value: str,
    *,
    path: Optional[Path] = None,
) -> Tuple[str, Any]:
    """Persist one key. Returns ``(canonical_key, typed_value)``.

    The edit is one line: an existing assignment is replaced in place,
    otherwise the key is appended to its section (creating ``[cli]`` when
    needed). Comments and unrelated keys are never touched.
    """
    section, bare = _section_for(key)
    value = _validate(bare, parse_toml_value(raw_value))
    target = path or config_path()

    original: Optional[str] = None
    if target.is_file():
        try:
            original = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise usage_error(f"cannot read {target}: {exc}") from exc
    lines = (original or "").splitlines()
    rendered = f"{bare} = {render_toml_value(value)}"

    existing = _find_key_line(lines, section, bare)
    if existing is not None:
        lines[existing] = rendered
    else:
        _insert_key(lines, section, rendered)

    content = _rendered(lines)
    data = _parsed_or_refuse(target, content)
    _check_landed(target, data, section, bare, value)
    _write(target, content)
    return bare, value


def _owner_only_mode(target: Path) -> Optional[int]:
    """The mode to give a rewritten *target*, or None to keep 0600.

    Keeps whatever owner bits the user chose but always drops group and
    other access: this file can hold an API key, so a rewrite must never
    widen it.
    """
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        return None
    return mode & 0o600


def _rendered(lines: List[str]) -> str:
    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    return content


def _parsed_or_refuse(target: Path, content: str) -> Dict[str, Any]:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        # Would leave the user with a file the SDK cannot parse. Bail out
        # with the original still on disk rather than corrupting it.
        raise usage_error(
            f"refusing to write {target}: the result would be invalid TOML "
            f"({exc}). If the file uses inline tables or dotted keys, edit "
            f"it by hand.",
        ) from exc
    return data if isinstance(data, dict) else {}


def _check_landed(
    target: Path,
    data: Dict[str, Any],
    section: str,
    key: str,
    value: Any,
) -> None:
    """Refuse to claim success for an edit the SDK would not read back.

    The edit is placed by a line scanner, not by a TOML parser, so
    syntax the scanner does not model -- a dotted key, an inline table --
    can leave the real value somewhere else. Parsing the result and
    reading the key back turns every such case into an error instead of
    a silent no-op.
    """
    table: Any = data.get(section) if section else data
    found = table.get(key) if isinstance(table, dict) else None
    if found == value:
        return
    where = f"[{section}] {key}" if section else key
    raise usage_error(
        f"refusing to write {target}: {where} = {value!r} did not take "
        f"effect; the file still resolves it to {found!r}. It probably "
        f"uses dotted keys or inline tables, so edit it by hand.",
        solution=(
            f"Open {target} and set {where} directly, then re-run "
            f"`dashscope config list` to confirm."
        ),
    )


def _write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = _owner_only_mode(target)
    # mkstemp is O_EXCL and 0600 whatever the umask, so the secret is
    # unreadable to anyone else from the first byte and a symlink planted
    # at a guessable name cannot redirect the write.
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=target.name + ".",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), mode)
        os.replace(temporary, target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise usage_error(f"cannot write {target}: {exc}") from exc


def restrict_permissions(path: Optional[Path] = None) -> None:
    """Chmod the config file to owner-only after a secret is written."""
    target = path or config_path()
    try:
        if target.is_file():
            os.chmod(target, 0o600)
    except OSError:
        pass


def get_value(key: str, *, path: Optional[Path] = None) -> Any:
    section, bare = _section_for(key)
    data = load_raw(path)
    if section:
        table = data.get(section)
        return table.get(bare) if isinstance(table, dict) else None
    return data.get(bare)


def mask(value: Any) -> str:
    """Render a secret so a screenshot or CI log cannot leak it whole."""
    text = str(value)
    if len(text) <= 6:
        return "***"
    return f"{text[:4]}{'*' * (len(text) - 6)}{text[-2:]}"


def _env_var_for(key: str) -> Optional[str]:
    """The environment variable the SDK consults for *key*.

    Derived from ``dashscope._config``'s own ``_<KEY>_ENV`` constants so
    this display cannot drift from what the SDK actually reads.
    """
    from dashscope import _config  # pylint: disable=import-outside-toplevel

    return getattr(_config, f"_{key.upper()}_ENV", None)


def effective_values(*, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Resolved view for ``config list``: value plus where it came from.

    Read-only replay of the SDK's precedence (explicit > env > file >
    default). Command-line ``-c`` overrides are not included: they only
    exist for the duration of one process, and this is a persisted-state
    report.
    """
    data = load_raw(path)
    rows: List[Dict[str, Any]] = []
    for key in sorted(_sdk_keys()):
        file_value = data.get(key)
        env_name = _env_var_for(key)
        env_value = os.environ.get(env_name) if env_name else None
        if env_value not in (None, ""):
            source, value = f"env {env_name}", env_value
        elif file_value is not None:
            source, value = "config file", file_value
        else:
            source, value = "default", None
        rows.append(
            {
                "key": key,
                "value": mask(value) if key == API_KEY and value else value,
                "source": source,
            },
        )
    cli_section = data.get(CLI_SECTION)
    cli_section = cli_section if isinstance(cli_section, dict) else {}
    for key in sorted(CLI_KEYS):
        value = cli_section.get(key)
        rows.append(
            {
                "key": f"{CLI_SECTION}.{key}",
                "value": mask(value) if key == API_KEY and value else value,
                "source": "config file" if value is not None else "default",
            },
        )
    return rows
