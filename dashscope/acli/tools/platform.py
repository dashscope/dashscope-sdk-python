# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

from dashscope.acli.config import Config
from dashscope.acli.platforms import get_cli_provider
from dashscope.acli.tools.registry import (
    PermissionLevel,
    ToolDefinition,
    registry,
)

# Track which tool names each capability registered, so /capability disable
# can unregister them mid-session (the prior "takes effect on restart"
# caveat is gone).
# Populated by register_one_capability via a registry-diff.
_capability_tools: dict[str, set[str]] = {}

# Canonical order of platform capability keys, used by register_platform_tools
# and surfaced to /capability commands. Keep in sync with cli.py's
# CAPABILITY_CATALOG ALL_CAPABILITY_KEYS.
_CAPABILITY_KEYS = [
    "bailian.mcp",
    "bailian.cli",
    "local.subagent",
    "local.delegate",
    "local.memory",
]


# Hints injected into the system prompt whenever the user has *explicitly*
# disabled a capability — i.e. enabled_capabilities is non-empty and the
# key is missing from it. This is the LLM's only signal that "this fallback
# is forbidden"; without it the model happily reaches for run_command + bl
# (or curl, or grep) after we've unregistered the real tools — bypassing
# the user's intent and, in bailian.cli's case, costing them real money.
#
# Each hint should be short, imperative, and name the channel(s) the model
# would otherwise reach for.
_DISABLED_CAPABILITY_HINTS: dict[str, str] = {
    "bailian.mcp": (
        "The user has disabled bailian.mcp. Do not call mcp_connect, "
        "do not call any mcp_<service>_* tools, and do not suggest "
        "using /skill <name> to trigger MCP-dependent skills."
    ),
    "bailian.cli": (
        "The user has disabled bailian.cli (the full Bailian CLI). "
        "**Never** invoke bl commands via run_command (including "
        "bl image generate / bl video / bl speech / bl knowledge / "
        "bl memory etc.) to replicate the same capability — that "
        "consumes the user's quota. When a request falls within bl's "
        "scope (image/video/speech generation, RAG retrieval, document "
        "parsing, TTS/ASR, etc.), tell the user the capability is off "
        "and suggest re-enabling it via /capability enable bailian.cli."
    ),
    "local.delegate": (
        "The user has disabled local.delegate. Do not call delegate or "
        "delegate_parallel; handle parallel work by calling other "
        "tools sequentially."
    ),
    "local.memory": (
        "The user has disabled local.memory. Do not call memory_search "
        "/ memory_store / memory_delete; ask the user directly when "
        "you need profile info."
    ),
}


def disabled_capabilities_hint(config: Config) -> str:
    """Build the system-prompt section that lists explicitly-disabled
    capabilities. None or empty = no disabled hints.
    """
    caps = config.enabled_capabilities
    if not caps:
        return ""
    disabled = [
        k
        for k in _CAPABILITY_KEYS
        if k not in caps and k in _DISABLED_CAPABILITY_HINTS
    ]
    if not disabled:
        return ""
    lines = [
        "\n\n## Disabled capabilities (user turned these off; "
        "strictly respect this — do not work around them)",
    ]
    for k in disabled:
        lines.append(f"- **{k}**: {_DISABLED_CAPABILITY_HINTS[k]}")
    return "\n".join(lines)


def all_capability_keys() -> list[str]:
    """Built-in capability keys plus anything declared in
    custom-extensions.toml. Used by register_platform_tools and surfaced
    to /capability."""
    from dashscope.acli.extensions import current as _ext_current

    return _CAPABILITY_KEYS + [c.key for c in _ext_current().capabilities]


def register_platform_tools(
    config: Config,
    connect_mcp_fn: Callable | None = None,
):
    caps = config.enabled_capabilities
    all_enabled = caps is None
    for cap_key in all_capability_keys():
        if config.privacy_mode and _is_cloud_capability(cap_key):
            # privacy mode: cloud capabilities never reach the tool surface
            continue
        if all_enabled or cap_key in (caps or []):
            register_one_capability(config, cap_key, connect_mcp_fn)


def register_one_capability(
    config: Config,
    cap_key: str,
    connect_mcp_fn: Callable | None = None,
) -> int:
    """Register a single capability's tools, tracking new names for later
    unregister. Returns the count of tools added (0 when creds are missing
    or the capability key is unknown — silent so the caller can treat it
    as best-effort)."""
    before = {t.name for t in registry.list_tools()}

    if cap_key == "bailian.mcp":
        if config.api_key and connect_mcp_fn:
            _register_mcp_tools(config, connect_mcp_fn)
    elif cap_key == "bailian.cli":
        if cli_client := get_cli_provider(config):
            _register_bailian_cli_tools(cli_client)
    elif cap_key == "local.subagent":
        from dashscope.acli.agents.subagent import (
            _has_parent,
            register_subagent_tool,
        )

        if _has_parent():
            register_subagent_tool()
        # else: parent agent not constructed yet; register_platform_tools
        # was called too early — cli.py wires _set_parent_agent + a final
        # re-register pass after Agent construction.
    elif cap_key == "local.delegate":
        from dashscope.acli.agents.delegate import (
            _has_parent,
            register_delegate_tools,
        )

        if _has_parent():
            register_delegate_tools()
    elif cap_key == "local.memory":
        _register_local_memory_tools(config)
    else:
        # Fall through to extension capabilities (custom-extensions.toml).
        # The cap_key is checked against load_extensions() output; if
        # found, its tools are wired via the HTTP-tool factory.
        if not _register_extension_capability(cap_key, config):
            return 0

    after = {t.name for t in registry.list_tools()}
    new_names = after - before
    _capability_tools.setdefault(cap_key, set()).update(new_names)
    return len(new_names)


def _register_extension_capability(cap_key: str, config=None) -> bool:
    """Find a custom-extensions.toml capability by key and register each
    of its tools. Returns True if a matching capability existed."""
    from dashscope.acli.extensions import (
        auth_env_name,
        build_http_tool,
        build_vision_tool,
        find_capability,
        loaded_key_targets,
        provider_key_for_env,
        tool_parameters_schema,
    )

    cap = find_capability(cap_key)
    if cap is None:
        return False
    # Reuse a key the user already stored for a built-in provider with the
    # same env var (DASHSCOPE_API_KEY ↔ tongyi) instead of failing at call
    # time or prompting twice for the same secret.
    if not cap.resolve_auth_key():
        env_name = cap.api_key_env or auth_env_name(cap.auth)
        cap.runtime_key = provider_key_for_env(
            config,
            env_name,
            loaded_key_targets(),
        )
    for tool in cap.tools:
        # tool name namespaced with cap key so two extensions can both
        # have a tool called "search" without colliding.
        full_name = f"{cap_key}.{tool.name}".replace(".", "_")
        permission = {
            "auto": PermissionLevel.AUTO,
            "confirm": PermissionLevel.CONFIRM,
            "dangerous": PermissionLevel.DANGEROUS,
        }.get(tool.permission, PermissionLevel.AUTO)
        # Vision tools (LLM calls with image content) and HTTP tools
        # (REST calls) share the same registration path; only the call
        # factory differs.
        if tool.type == "vision":
            call_fn = build_vision_tool(cap, tool)
        else:
            call_fn = build_http_tool(cap, tool)
        params = tool_parameters_schema(tool)
        if permission is PermissionLevel.AUTO:
            registry.register_mcp_tool(
                name=full_name,
                description=f"[{cap_key}] {tool.description}",
                parameters=params,
                call_fn=call_fn,
            )
        else:
            registry.register(
                ToolDefinition(
                    name=full_name,
                    description=f"[{cap_key}] {tool.description}",
                    permission=permission,
                    func=call_fn,
                    parameters=params,
                ),
            )
    return True


def _register_local_memory_tools(config: Config) -> None:
    """Wire the local.memory capability: register the LocalMemoryCapability
    implementation in the global CapabilityRegistry and expose agent tools
    backed by it. The capability shares the LocalMemoryClient store that
    serves /profile and /memory (same user_id → same file), so agent-stored
    facts and the user's profile stay in one place."""
    from dashscope.acli.capabilities import get_capability_registry
    from dashscope.acli.capabilities.local_memory import LocalMemoryCapability
    from dashscope.acli.platforms import get_memory_provider

    cap_registry = get_capability_registry()
    cap = cap_registry.get("local.memory")
    if cap is None:
        cap = LocalMemoryCapability(client=get_memory_provider(config))
        cap_registry.register(cap)

    async def memory_search(query: str, top_k: int = 5) -> str:
        results = await cap.search(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        lines = [f"- [{r['id']}] {r['content']}" for r in results]
        return "Relevant memories:\n" + "\n".join(lines)

    async def memory_store(content: str) -> str:
        entry_id = await cap.add(content, metadata={"source": "agent"})
        if not entry_id:
            return "Memory not saved (empty or duplicates existing)"
        return f"Remembered (id: {entry_id})"

    async def memory_delete(entry_id: str) -> str:
        ok = await cap.delete(entry_id)
        return "Deleted" if ok else f"Memory {entry_id} not found"

    registry.register_mcp_tool(
        name="memory_search",
        description=(
            "Search the user's long-term memory/profile (preferences, "
            "tech stack, project conventions) to enrich context"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "search keywords",
                },
                "top_k": {
                    "type": "integer",
                    "description": "max results to return, default 5",
                },
            },
            "required": ["query"],
        },
        call_fn=memory_search,
    )
    registry.register_mcp_tool(
        name="memory_store",
        description=(
            "Store important facts (user preferences, project "
            "conventions, environment info) into long-term memory"
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "content to remember",
                },
            },
            "required": ["content"],
        },
        call_fn=memory_store,
    )
    registry.register_mcp_tool(
        name="memory_delete",
        description="Delete a long-term memory entry by id",
        parameters={
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "memory entry id",
                },
            },
            "required": ["entry_id"],
        },
        call_fn=memory_delete,
    )


def unregister_capability_tools(cap_key: str) -> int:
    """Drop all tools previously registered under cap_key. Returns the
    count of tools removed.

    Note: for bailian.mcp this removes the mcp_connect dispatcher and any
    mcp_<service>_* tools that were registered through it. Active MCPClient
    instances are left running — they're held by cli._mcp_clients, which
    only the slash-command handler touches; here we just clear the visible
    tool surface. Re-enable + /mcp services would re-register cleanly."""
    names = _capability_tools.pop(cap_key, set())
    for name in names:
        registry.unregister(name)
    return len(names)


def capability_tool_names() -> set[str]:
    """All tool names currently registered under any capability. Consumed by
    ToolRegistry.to_schema_list: registration is the capability gate, so any
    tool present here must be exposed to the model (extension HTTP/vision
    tools and bailian_* wrappers would otherwise be invisible)."""
    names: set[str] = set()
    for tool_names in _capability_tools.values():
        names |= tool_names
    return names


def _is_cloud_capability(cap_key: str) -> bool:
    """Cloud = anything not local.*. Privacy mode must keep these out of the
    tool surface."""
    return not cap_key.startswith("local.")


def unregister_cloud_capability_tools() -> int:
    """Privacy mode on: drop every registered cloud capability tool (bailian.*,
    extension HTTP/vision tools). Returns total tools removed."""
    removed = 0
    for cap_key in [
        k for k in list(_capability_tools) if _is_cloud_capability(k)
    ]:
        removed += unregister_capability_tools(cap_key)
    return removed


def refresh_extension_capability_tools(config: Config) -> int:
    """Re-register all enabled extension capabilities so their tool closures
    rebind to the latest custom-extensions.toml definitions (endpoint, auth,
    token). Called by /capability reload and /dev hot reload — without it,
    edited configs keep running through stale closures until restart.
    Returns total tools re-registered."""
    from dashscope.acli.extensions import current as _ext_current

    ext_keys = {c.key for c in _ext_current().capabilities}
    if config.enabled_capabilities is None:
        targets = sorted(ext_keys)
    else:
        targets = sorted(ext_keys & set(config.enabled_capabilities))
    if config.privacy_mode:
        targets = [k for k in targets if not _is_cloud_capability(k)]
    total = 0
    for key in targets:
        unregister_capability_tools(key)
        total += register_one_capability(config, key)
    return total


# ===== MCP =====


def _register_mcp_tools(config: Config, connect_mcp_fn: Callable):
    async def mcp_connect(service: str) -> str:
        err = await connect_mcp_fn(service, config)
        if err:
            return f"Connection failed: {err}"
        return (
            f"Connected to MCP service: {service}. New tools are now "
            f"available; call them directly to complete the task."
        )

    registry.register_mcp_tool(
        name="mcp_connect",
        description=(
            "Connect to an MCP cloud service to gain more tools. "
            "Currently available: time (clock), code-interpreter "
            "(code execution), doc-analysis (document parsing). "
            "New tools are registered after connecting."
        ),
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "service name; available: time, "
                        "code-interpreter, doc-analysis"
                    ),
                },
            },
            "required": ["service"],
        },
        call_fn=mcp_connect,
    )


# ===== Bailian CLI (bl) =====

# Subcommand verbs that imply non-trivial side effects (cost money, create
# files, write to remote stores). These get CONFIRM permission; everything
# else (chat/list/status/get/retrieve/describe/recognize/search/show) stays
# AUTO so the LLM can poll freely.
_BL_WRITE_VERBS = {
    "generate",
    "edit",
    "upload",
    "add",
    "delete",
    "update",
    "create",
    "call",
    "synthesize",
    "download",
}


def _register_bailian_cli_tools(client):
    """Pull `bl config export-schema` and register every command as a tool.

    Tool name = the `name` field from the schema (e.g. `bailian_text_chat`).
    Args fed by the LLM are translated to `--kebab-case` CLI flags inside
    `client.invoke()`. Failures register zero tools rather than poison the
    rest of platform tool registration.

    Schemas are cached at ~/.acli/bl-schemas.json (24h TTL). The original
    sync subprocess took ~1.2s at every startup; cache hit is <5ms.
    Cache miss runs the subprocess synchronously to seed for next time.
    """
    schemas = _load_or_refresh_bl_schemas(client)
    if not schemas:
        return

    for schema in schemas:
        tool_name = schema.get("name", "")
        if not tool_name.startswith("bailian_"):
            continue
        command_path = tool_name[len("bailian_") :].split("_")
        description = "[bl] " + schema.get("description", "")
        params = schema.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        verb = command_path[-1]
        permission = (
            PermissionLevel.CONFIRM
            if verb in _BL_WRITE_VERBS
            else PermissionLevel.AUTO
        )

        async def _call(_p=command_path, _c=client, **kwargs):
            return await _c.invoke(_p, kwargs)

        if permission is PermissionLevel.CONFIRM:
            registry.register(
                ToolDefinition(
                    name=tool_name,
                    description=description,
                    permission=permission,
                    func=_call,
                    parameters=params,
                ),
            )
        else:
            registry.register_mcp_tool(
                name=tool_name,
                description=description,
                parameters=params,
                call_fn=_call,
            )


_BL_SCHEMA_CACHE_TTL = 24 * 60 * 60  # 24h


def _load_or_refresh_bl_schemas(client):
    """Return cached bl schemas; refresh from `bl config export-schema` on
    miss or staleness. Returns [] on hard failure so caller can short-circuit.
    """
    import json
    import time
    from pathlib import Path

    cache_path = Path.home() / ".acli" / "bl-schemas.json"
    now = time.time()

    # Try cache first
    if cache_path.exists():
        try:
            age = now - cache_path.stat().st_mtime
            if age < _BL_SCHEMA_CACHE_TTL:
                data = json.loads(cache_path.read_text())
                if isinstance(data, list):
                    return data
        except (OSError, json.JSONDecodeError):
            pass

    # Miss/stale: fetch fresh and write back
    try:
        schemas = client.export_schemas_sync()
    except Exception:
        # On failure, try serving a stale cache rather than empty registry
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(schemas, ensure_ascii=False))
        tmp.replace(cache_path)
    except OSError:
        # Cache write failure is non-fatal; we still have schemas in memory
        pass

    return schemas
