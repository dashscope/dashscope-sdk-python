# -*- coding: utf-8 -*-
# pylint: disable=protected-access
from __future__ import annotations

import inspect
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Union, get_type_hints


class PermissionLevel(Enum):
    AUTO = "auto"
    CONFIRM = "confirm"
    DANGEROUS = "dangerous"


@dataclass
class ToolDefinition:
    name: str
    description: str
    permission: PermissionLevel = PermissionLevel.AUTO
    func: Callable = field(default=None)
    parameters: dict = field(default_factory=dict)


_PYTHON_TYPE_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _build_parameters_schema(func: Callable) -> dict:
    # include_extras=False and pass globals to resolve stringified annotations
    try:
        hints = get_type_hints(func, globalns=getattr(func, "__globals__", {}))
    except Exception:
        hints = {}

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        type_hint = hints.get(name, str)

        origin = getattr(type_hint, "__origin__", None)
        if origin is type(None):
            continue

        # Handle Optional[X] / Union[X, None] / X | None (3.10+)
        is_optional = False
        actual_type = type_hint
        is_union = origin is Union
        if sys.version_info >= (3, 10):
            is_union = is_union or isinstance(type_hint, types.UnionType)
        if is_union:
            args = getattr(type_hint, "__args__", ())
            if type(None) in args:
                is_optional = True
                actual_type = next(a for a in args if a is not type(None))

        json_type = _PYTHON_TYPE_TO_JSON.get(actual_type, "string")
        prop: dict[str, Any] = {"type": json_type}
        properties[name] = prop

        if param.default is inspect.Parameter.empty and not is_optional:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


# Agent-orchestration tools gated by capabilities (local.delegate /
# local.subagent). They enter the registry only when the governing capability
# is enabled (tools/platform.register_one_capability) and leave it on
# /capability disable, so capability on/off is enforced at registration time;
# whenever they are present in the registry the model must be able to see
# them — otherwise delegation can never auto-start.
_CAPABILITY_AGENT_TOOLS = frozenset(
    {
        "delegate",
        "delegate_parallel",
        "subagent_invoke",
    },
)


Tool = ToolDefinition  # alias for backwards-compatibility


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool_def: ToolDefinition):
        """Register a tool. When used as a decorator with a ToolDefinition,
        returns a decorator."""
        if isinstance(tool_def, ToolDefinition):
            self._tools[tool_def.name] = tool_def

            def decorator(func: Callable) -> Callable:
                self._tools[tool_def.name].func = func
                func._tool_definition = tool_def
                return func

            return decorator
        # Called with a callable directly (legacy)
        self._tools[tool_def.name] = tool_def
        return None

    def register_mcp_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        call_fn: Callable,
    ):
        tool_def = ToolDefinition(
            name=name,
            description=description,
            permission=PermissionLevel.AUTO,
            func=call_fn,
            parameters=parameters,
        )
        self._tools[name] = tool_def

    def unregister(self, name: str):
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def to_schema_list(self, user_input: str = "") -> list[dict]:
        """Convert registered tools to JSON schema format.

        If user_input is provided, prioritize tools relevant to the input.
        Core tools (filesystem, shell) are always included.
        Specialized tools (platform, MCP) are included if keywords match."""
        result = []

        # Core tools always included
        core_tools = {
            "read_file",
            "write_file",
            "list_directory",
            "search_files",
            "create_directory",
            "delete_file",
            "delete_directory",
            "move_file",
            "run_command",
            "web_search",
        }

        # bailian_* CLI wrappers are numerous (~25 schemas each with full
        # parameter definitions); offering them only when the request mentions
        # the matching capability keeps every unrelated call much smaller.
        b_all = [
            "bailian_text_chat",
            "bailian_omni",
            "bailian_image_generate",
            "bailian_image_edit",
            "bailian_video_generate",
            "bailian_video_edit",
            "bailian_video_ref",
            "bailian_video_task_get",
            "bailian_video_download",
            "bailian_vision_describe",
            "bailian_app_call",
            "bailian_app_list",
            "bailian_memory_add",
            "bailian_memory_search",
            "bailian_memory_list",
            "bailian_memory_update",
            "bailian_memory_delete",
            "bailian_memory_profile_create",
            "bailian_memory_profile_get",
            "bailian_knowledge_retrieve",
            "bailian_search_web",
            "bailian_speech_synthesize",
            "bailian_speech_recognize",
            "bailian_file_upload",
            "bailian_model_list",
            "bailian_console_call",
            "bailian_usage_free",
        ]
        b_image = [
            "bailian_image_generate",
            "bailian_image_edit",
            "bailian_vision_describe",
            "bailian_omni",
        ]
        b_video = [
            "bailian_video_generate",
            "bailian_video_edit",
            "bailian_video_ref",
            "bailian_video_task_get",
            "bailian_video_download",
        ]
        b_speech = ["bailian_speech_synthesize", "bailian_speech_recognize"]
        b_memory = [
            "bailian_memory_add",
            "bailian_memory_search",
            "bailian_memory_list",
            "bailian_memory_update",
            "bailian_memory_delete",
            "bailian_memory_profile_create",
            "bailian_memory_profile_get",
        ]

        # Keyword → tool mapping for conditional inclusion.
        # Chinese keywords are kept for CJK input; English equivalents
        # are appended alongside them so English prompts match too.
        keyword_tools = {
            "memory": ["memory_search", "memory_store"] + b_memory,
            "档案": ["memory_search", "memory_store"],
            "profile": ["memory_search", "memory_store"],
            "偏好": ["memory_search", "memory_store"],
            "preference": ["memory_search", "memory_store"],
            "记忆": b_memory,
            "数据": [
                "data_upload",
                "data_files",
                "data_delete",
                "data_categories",
            ],
            "data": [
                "data_upload",
                "data_files",
                "data_delete",
                "data_categories",
            ],
            "prompt": [
                "prompt_list",
                "prompt_get",
                "prompt_create",
                "prompt_delete",
                "prompt_render",
            ],
            "模板": [
                "prompt_list",
                "prompt_get",
                "prompt_create",
                "prompt_delete",
                "prompt_render",
            ],
            "template": [
                "prompt_list",
                "prompt_get",
                "prompt_create",
                "prompt_delete",
                "prompt_render",
            ],
            "搜索": ["web_search", "context_search"],
            "search": ["web_search", "context_search"],
            "模型": ["switch_model", "switch_provider"],
            "model": ["switch_model", "switch_provider"],
            "能力": ["capability_enable", "capability_disable"],
            "capability": ["capability_enable", "capability_disable"],
            # Browser / web scraping tools
            "scrape": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "抓取": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "网页": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "webpage": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "截图": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "screenshot": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "browser": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "爬取": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            "crawl": [
                "scrape_web",
                "scrape_web_html",
                "scrape_web_screenshot",
            ],
            # Camera tools
            "拍照": ["camera_capture", "camera_record"],
            "photo": ["camera_capture", "camera_record"],
            "摄像头": ["camera_capture", "camera_record"],
            "录屏": ["camera_capture", "camera_record"],
            "录制": ["camera_capture", "camera_record"],
            "record": ["camera_capture", "camera_record"],
            "camera": ["camera_capture", "camera_record"],
            # Bailian platform tools (schemas are keyword-gated, see b_* above)
            "图片": b_image,
            "画图": b_image,
            "draw": b_image,
            "image": b_image,
            "vision": b_image,
            "多模态": ["bailian_omni"],
            "multimodal": ["bailian_omni"],
            "omni": ["bailian_omni"],
            "视频": b_video,
            "video": b_video,
            "语音": b_speech,
            "speech": b_speech,
            "tts": ["bailian_speech_synthesize"],
            "朗读": ["bailian_speech_synthesize"],
            "read aloud": ["bailian_speech_synthesize"],
            "知识库": ["bailian_knowledge_retrieve"],
            "knowledge": ["bailian_knowledge_retrieve"],
            "rag": ["bailian_knowledge_retrieve"],
            "百炼应用": ["bailian_app_call", "bailian_app_list"],
            "bailian app": ["bailian_app_call", "bailian_app_list"],
            "用量": ["bailian_usage_free"],
            "usage": ["bailian_usage_free"],
            "quota": ["bailian_usage_free"],
            "上传": ["bailian_file_upload"],
            "upload": ["bailian_file_upload"],
            # Catch-all: mentioning the platform unlocks the full set
            "百炼": b_all,
            "bailian": b_all,
        }

        # Determine which tools to include
        include_tools = set(core_tools)

        # Add session tools (always available)
        include_tools.update(
            [
                "switch_model",
                "switch_provider",
                "capability_enable",
                "capability_disable",
            ],
        )
        include_tools.update(["mcp_connect", "mcp_disconnect"])

        # Model-invocable skill templates (.acli/skills/*.md)
        include_tools.add("use_skill")

        # Add capability-gated agent tools (delegate/subagent_invoke) whenever
        # they are registered — registration itself is the capability gate.
        include_tools.update(_CAPABILITY_AGENT_TOOLS & set(self._tools))

        # Same gate applies to every tool registered under an enabled
        # capability: extension HTTP/vision tools ({cap}_{tool} names).
        # Without this they'd be registered but never offered to the model,
        # making /capability enable a no-op in practice.
        # Exception: bailian_* CLI wrappers are numerous (~25 full schemas);
        # they ride only when a keyword matches (see keyword_tools above),
        # with the platform name as the catch-all.
        # Lazy import: acli.tools.platform imports this module.
        from dashscope.acli.tools.platform import capability_tool_names

        include_tools.update(
            n
            for n in capability_tool_names() & set(self._tools)
            if not n.startswith("bailian_")
        )

        # Add keyword-matched tools if user input provided
        if user_input:
            if isinstance(user_input, list):
                input_lower = " ".join(
                    b.get("text", "")
                    for b in user_input
                    if isinstance(b, dict)
                ).lower()
            else:
                input_lower = user_input.lower()
            for keyword, tools in keyword_tools.items():
                if keyword.lower() in input_lower:
                    include_tools.update(tools)

        # Add all MCP tools (they're dynamically added and usually relevant)
        for name in self._tools:
            if name.startswith("mcp_"):
                include_tools.add(name)

        # Build schema list
        for name, t in self._tools.items():
            # Include if in the filtered set, or if it's an MCP tool
            if name in include_tools or name.startswith("mcp_"):
                result.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                )

        return result


# Global registry
registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    permission: PermissionLevel = PermissionLevel.AUTO,
):
    def decorator(func: Callable) -> Callable:
        params_schema = _build_parameters_schema(func)
        tool_def = ToolDefinition(
            name=name,
            description=description,
            permission=permission,
            func=func,
            parameters=params_schema,
        )
        registry.register(tool_def)
        func._tool_definition = tool_def
        return func

    return decorator
