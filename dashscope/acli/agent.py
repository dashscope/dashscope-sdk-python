# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements,protected-access
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator

from rich.console import Console

from dashscope.acli.compression import (
    HARD_THRESHOLD_RATIO,
    estimate_message_tokens,
    estimate_tokens,
    safety_compress_if_needed,
    shrink_old_tool_messages,
)
from dashscope.acli.config import WORKSPACE_DIR, context_window_for_model
from dashscope.acli.executor import EXEC_ERROR_PREFIX, Executor
from dashscope.acli.hooks import HookBus, HookContext, create_hook_bus
from dashscope.acli.memory.manager import MemoryManager
from dashscope.acli.platforms.base import MemoryProvider
from dashscope.acli.prompt_pipeline import PromptContext, default_pipeline
from dashscope.acli.providers.base import LLMProvider, LLMResponse, ToolCall
from dashscope.acli.skills import get_skill_manager, skills_summary_for_llm
from dashscope.acli.tools.registry import PermissionLevel, registry
from dashscope.acli.utils import (
    UserAbortedTurn,
    UserSupplement,
    normalize_for_model,
    sanitize,
    text_of,
    tool_result_for_display,
    tool_result_for_history,
)

console = Console()

# Tool results are plain strings; outcome classification relies on these
# prefixes. "Error" marks validation / unknown-tool failures,
# EXEC_ERROR_PREFIX marks executor exceptions, "Cancelled" marks user
# rejection/cancellation. The Chinese variants are kept so results from
# external tools / MCP servers that still return Chinese text are also
# classified correctly.
_TOOL_ERROR_PREFIXES = ("错误", "Error", EXEC_ERROR_PREFIX)
_TOOL_CANCEL_PREFIXES = ("操作已取消", "Cancelled")


def _classify_outcome(successes: int, failures: int) -> str:
    """Classify a turn's outcome from its tool execution counters.

    User cancellation is neutral (counted neither way), so a turn where
    everything was cancelled lands on "partial" rather than "success".
    """
    if failures == 0 and successes > 0:
        return "success"
    if failures > 0 and successes == 0:
        return "failure"
    return "partial"


SYSTEM_PROMPT = """You are acli, an intelligent command-line assistant.
The user states what they need and you execute it — never make the user
run commands themselves.

Core principle: **the user speaks, you act.** You have full tool access to
fulfill user requests; call tools directly instead of suggesting slash
commands.

Rules:
1. Only use tools that exist in the tool list; never guess or invent tool names
2. Only use file paths inside the current working directory, paths explicitly
   given by the user, or paths relative to the CWD. Never invent, guess, or
   reuse paths seen in training data (e.g. /Users/xxx/...)
3. For clear-intent requests (git commit, read file, search, edit code), call
   tools directly without stating a plan first. Only for irreversible
   operations (delete, overwrite, force push) or complex multi-step tasks,
   briefly explain and wait for user confirmation
4. If a task needs multiple steps, execute them one by one and report progress
5. When a tool call fails, do not retry the same call; report the error to
   the user with a suggestion
6. Tools prefixed [MCP:xxx] come from Bailian MCP services; call them directly
7. If the "user profile" contains info, personalize answers with it (e.g.
   recommend Python solutions for a Python user)
8. When a request matches an "available Skill template" below, prefer that
   flow — if the Skill's MCP is connected, call its tools directly; if not,
   connect first with mcp_connect, then call
9. When a question may involve personal preferences, tech stack, or work
   environment, proactively search the profile with memory_search for context
10. If no suitable tool can complete the task, answer from your knowledge;
    never call a nonexistent tool
11. To switch model/Provider → call switch_model or switch_provider directly
12. To use cloud services (time, code execution, doc parsing, etc.) → call
    mcp_connect directly
13. To enable/disable capabilities → call capability_enable /
    capability_disable directly
14. Never reply "please use the /xxx command" — if you have a tool that can
    do it, just do it
15. **Do not use run_command with python3 -c inline scripts to read/write
    files** — it is inefficient. Read with read_file, write with write_file;
    shell tools like sed/grep are fine to use
16. **If a question can be answered directly (no files, commands, or external
    info needed), answer from your knowledge without calling tools** — e.g.
    self-introduction, explaining concepts, translating text, small talk
17. **Multiple independent subtasks (e.g. "look at these 2 files", several
    unrelated tasks at once) → call delegate_parallel to fan out subagents**,
    or send multiple delegate calls in the same turn; a single independent
    but lengthy subtask (whole-file review, multi-file scan) → call
    subagent_invoke for isolated execution and take back only the conclusion.
    Do not grind through them serially yourself

Reply style:
- **Concise**. No filler like "let me see / test this / verify / let me help
  you / I'll analyze it"; just act or give the answer
- **One shot**. Read files with read_file (use offset/limit for large spans);
  never use python3 -c inline scripts for file I/O
- **Batch in parallel**. Issue multiple independent tool calls for the same
  purpose in one turn; do not call them one by one serially
- **No re-confirmation**. Do not re-read facts already fetched; if a tool
  fails once, report the error to the user instead of retrying a rephrased
  version of the same action
- Do not summarize what you just did unless asked — the user can see the
  diff / output
- User input may come from voice transcription (/v command); just understand
  the intent and do not comment on the voice/recording feature itself"""


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        executor: Executor,
        max_turns: int = 50,
        memory: MemoryProvider | None = None,
        user_name: str = "",
        provider_name: str = "",
        model_name: str = "",
        session_path: Path | None = None,
        disabled_caps_provider=None,
        directives_provider=None,
        system_prompt: str | None = None,
        allow_delegate: bool = True,
        allowed_tools: list[str] | None = None,
        hook_bus: HookBus | None = None,
        memory_manager: MemoryManager | None = None,
        json_mode: bool = False,
    ):
        self.provider = provider
        self.executor = executor
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.allow_delegate = allow_delegate
        self.allowed_tools = allowed_tools
        self.hook_bus = hook_bus or create_hook_bus()
        self.json_mode = json_mode
        # Load custom system prompt: workspace .acli/system-prompt.md first,
        # then global ~/.acli/system-prompt.md, then built-in default.
        # runners.py pre-populates system_prompt via _load_system_prompt();
        # this fallback ensures direct Agent construction (SDK, tests,
        # subagents) also respects the global file.
        if self.system_prompt is None:
            from dashscope.acli.cli.startup import (
                _compose_system_prompt,
                _load_system_prompt,
            )

            self.system_prompt = _compose_system_prompt(_load_system_prompt())
        # Discover project instructions from CWD (rules.jsonl,
        # .cursorrules, etc.)
        from dashscope.acli.prompt import discover_project_instructions

        project_instructions = discover_project_instructions()
        self._prompt_pipeline = default_pipeline(
            base_prompt=self.system_prompt or SYSTEM_PROMPT,
            project_instructions=project_instructions,
            skills_summary_fn=skills_summary_for_llm,
            active_prompts_fn=lambda text: get_skill_manager().active_prompts(
                text,
            ),
        )
        self.messages: list[dict] = []
        self.last_output: str = ""
        self.memory = memory
        self.provider_name = provider_name
        self.model_name = model_name
        self.user_name = user_name
        self.session_path = session_path
        # Callable[[], str] returning a system-prompt fragment describing
        # currently-disabled capabilities. Called fresh each turn so
        # /capability toggles take effect on the next user input without
        # re-instantiating the agent.
        self.disabled_caps_provider = disabled_caps_provider
        # Callable[[], list[str]] returning user-declared operational rules
        # to be injected unconditionally each turn (see _directives_section).
        self.directives_provider = directives_provider

        # Sub-agents reuse the parent's memory_manager; top-level agents
        # create their own instance.
        if memory_manager is not None:
            self.memory_manager = memory_manager
        else:
            workspace_dir = WORKSPACE_DIR
            self.memory_manager = MemoryManager(workspace_dir)

        # Resolved once: the compression token budget tracks the model's real
        # context window instead of a fixed 128k.
        self._context_window = context_window_for_model(model_name or "")

        # Convenience references for backward compatibility
        self.trace_logger = self.memory_manager.trace
        self.experience_tracker = self.memory_manager.persistent.experience

        # Track tools used in current turn for experience recording
        self._current_turn_tools: list[str] = []
        self._turn_tool_successes = 0
        self._turn_tool_failures = 0
        self._reflection_lesson_recorded = False
        # Per-turn counters for the TUI status line
        self.turn_tool_calls = 0
        self.turn_subagents = 0
        self.turn_mcp_calls = 0
        self.turn_skills = 0
        # Explicit /skill invocations to count on the next turn
        self._pending_skill_names: list[str] = []

    def note_skill_use(self, name: str) -> None:
        """Queue an explicit /skill invocation for turn/session stats."""
        self._pending_skill_names.append(name)

    def reset(self):
        """Clear in-memory messages without deleting the persisted history
        file."""
        self.messages = []

    def load_session(self) -> int:
        """Restore self.messages from session_path. Returns the number of
        messages loaded (0 if no file, file empty, or parse failed)."""
        if not self.session_path or not self.session_path.exists():
            return 0
        try:
            data = json.loads(self.session_path.read_text())
            if isinstance(data, list):
                self.messages = data
                return len(data)
        except (json.JSONDecodeError, OSError):
            pass
        return 0

    def save_session(self) -> None:
        """Persist self.messages atomically. No-op when session_path is
        unset."""
        if not self.session_path:
            return
        try:
            self.session_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.session_path.with_suffix(
                self.session_path.suffix + ".tmp",
            )
            tmp.write_text(
                json.dumps(
                    sanitize(self.messages),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            tmp.replace(self.session_path)
        except OSError:
            pass
        # Update session meta message count
        try:
            from dashscope.acli.session import get_session_manager

            topic = self.session_path.parent.name
            get_session_manager().update_message_count(
                len(self.messages),
                topic,
            )
        except Exception:
            pass

    def _connected_mcp_services(self) -> set[str]:
        """Infer connected MCP services from registry tool names
        (mcp_<service>_*)."""
        services: set[str] = set()
        for t in registry.list_tools():
            if t.name.startswith("mcp_"):
                rest = t.name[4:]
                if "_" in rest:
                    services.add(rest.split("_", 1)[0])
        return services

    def _filter_tools_schema(self, tools_schema: list[dict]) -> list[dict]:
        """Apply per-agent tool restrictions.

        - Child agents (allow_delegate=False) do not see delegate/subagent
          tools (no nested spawning).
        - If allowed_tools is set, only those tool names are exposed.
        """
        names = {t.get("name") for t in tools_schema}

        delegate_names = {"delegate", "delegate_parallel", "subagent_invoke"}
        if not self.allow_delegate:
            names -= delegate_names

        if self.allowed_tools is not None:
            allowed = set(self.allowed_tools)
            names = names & allowed

        return [t for t in tools_schema if t.get("name") in names]

    def _system_prompt_for_turn(self, user_input_text: str) -> str:
        """Assemble the full system prompt via the pluggable PromptPipeline.

        The pipeline owns the stable/ephemeral split and the section order;
        adding a section is now a matter of registering it on the pipeline
        rather than editing this method.
        """
        ctx = PromptContext(
            user_input=user_input_text,
            user_name=self.user_name,
            provider_name=self.provider_name,
            model_name=self.model_name,
            memory_manager=self.memory_manager,
            experience_tracker=self.experience_tracker,
            disabled_caps_provider=self.disabled_caps_provider,
            directives_provider=self.directives_provider,
            current_turn_tools=self._current_turn_tools,
            connected_mcp_services=self._connected_mcp_services,
        )
        return self._prompt_pipeline.render(ctx)

    def _reflection_section(self) -> str:
        """Inject reflection hints when repeated failures detected."""
        tracker = self.memory_manager.session.reflection
        if tracker.needs_reflection():
            # Also record lesson in experience memory — once per turn, since
            # this section is re-evaluated on every loop iteration.
            lesson = tracker.get_failure_lesson()
            if (
                lesson
                and self._current_turn_tools
                and not self._reflection_lesson_recorded
            ):
                self._reflection_lesson_recorded = True
                self.memory_manager.record_experience(
                    task_summary="repeated tool failures",
                    tools_used=self._current_turn_tools,
                    outcome="failure",
                    lesson=lesson,
                )
            return tracker.get_reflection_hint()
        return ""

    async def _recall_memory(self, user_input) -> str:
        """Search for relevant profile info and format as context.

        Recall is purely score-based (weighted keyword match + min_score
        threshold) — no keyword gate on the query, so stored profile facts
        surface whenever they are actually relevant.
        """
        if not self.memory:
            return ""
        query = text_of(user_input)
        if not query.strip():
            return ""

        try:
            nodes = await self.memory.search(query, top_k=3, min_score=0.3)
            if not nodes:
                return ""
            parts = [f"- {n.content}" for n in nodes]
            return "\n\nUser profile:\n" + "\n".join(parts)
        except Exception:
            return ""

    async def _store_memory(self, messages: list[dict[str, str]]):
        """Store conversation turn as memory (fire-and-forget)."""
        if not self.memory:
            return
        try:
            nodes = await self.memory.add(messages)
            if nodes:
                names = ", ".join(n.content[:20] for n in nodes)
                console.print(f"\n[dim](memory: +{len(nodes)} {names})[/dim]")
        except Exception as e:
            console.print(f"\n[dim red](memory error: {e})[/dim red]")

    async def run_stream(self, user_input) -> AsyncIterator[str]:
        self.messages.append({"role": "user", "content": user_input})

        from dashscope.acli.audit import get_audit_logger

        get_audit_logger().log_user_input(text_of(user_input))

        # Track partial assistant content so Ctrl-C doesn't lose it.
        _partial: list[str] = []
        try:
            async for chunk in self._run_stream_body(user_input):
                _partial.append(chunk)
                yield chunk
        except BaseException:
            # On cancellation / error, save whatever was received so the
            # user message and partial reply survive the next session load.
            partial_text = "".join(_partial).strip()
            if partial_text:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": partial_text + " [interrupted]",
                    },
                )
            self.save_session()
            raise
        finally:
            # Turn-scoped trust grants reset here so the next user prompt
            # starts fresh (Ctrl-C aborts also fall through this branch).
            self.executor.clear_session_trust()

    async def _run_stream_body(self, user_input) -> AsyncIterator[str]:
        user_input_text = text_of(user_input)

        # User-message hook
        await self.hook_bus.dispatch_async(
            HookContext(
                event="on_message",
                tool_name="",
                arguments={"input": user_input_text},
            ),
        )

        # Clear tool tracking for new turn
        self._current_turn_tools = []
        self.turn_tool_calls = 0
        self.turn_subagents = 0
        self.turn_mcp_calls = 0
        try:
            active = get_skill_manager().active_packages(user_input_text)
            names = [p.name for p in active] + self._pending_skill_names
            self._pending_skill_names = []
            self.turn_skills = len(names)
            self.executor.record_skills(names)
        except Exception:
            self.turn_skills = 0
        self._turn_tool_successes = 0
        self._turn_tool_failures = 0
        self._reflection_lesson_recorded = False

        # Reset reflection tracker for new turn
        self.memory_manager.session.reflection.reset()

        # Recall relevant memories
        memory_context = await self._recall_memory(user_input_text)
        system_prompt = self._system_prompt_for_turn(user_input_text)
        if memory_context:
            system_prompt += memory_context

        tools_schema = self._filter_tools_schema(
            registry.to_schema_list(user_input_text),
        )

        # Compress with the full request overhead in view: the system prompt
        # and tools schema ride on every API call, so they count against the
        # model's window too. Runs after both are built, before the request
        # list is assembled.
        overhead_tokens = estimate_tokens(system_prompt) + estimate_tokens(
            json.dumps(tools_schema, ensure_ascii=False),
        )
        await self._auto_compress(overhead_tokens)
        await self._ensure_fits_context(overhead_tokens)

        messages_with_system = [
            {"role": "system", "content": system_prompt},
            *self.messages,
        ]

        last_content = ""
        for _loop_i in range(self.max_turns):
            # Cheap in-place shrink of stale tool results (no LLM call):
            # keeps long tool loops from re-sending huge old outputs.
            if _loop_i > 0:
                shrink_old_tool_messages(self.messages)
            # Mid-turn guard: a long tool loop can overflow the window within
            # a single user turn. Estimate-only check; compress and rebuild
            # the request list only past the hard threshold.
            if _loop_i > 0 and (
                estimate_message_tokens(self.messages) + overhead_tokens
                >= int(self._context_window * HARD_THRESHOLD_RATIO)
            ):
                compressed = await safety_compress_if_needed(
                    self.messages,
                    self.provider.chat,
                    context_window=self._context_window,
                    extra_tokens=overhead_tokens,
                )
                if compressed is not None:
                    old_count = len(self.messages)
                    self.messages = compressed
                    messages_with_system = [
                        {"role": "system", "content": system_prompt},
                        *self.messages,
                    ]
                    console.print(
                        f"[dim]mid-turn safety compression: {old_count} "
                        f"messages → {len(self.messages)}[/dim]",
                    )

            # Inject reflection hint while repeated failures persist; assign
            # unconditionally so the hint disappears once failures stop.
            # The hint is appended, keeping the cache-friendly prefix stable.
            reflection_hint = self._reflection_section()
            messages_with_system[0]["content"] = (
                system_prompt + reflection_hint
            )

            full_content = ""
            full_reasoning = ""
            tool_calls: list[ToolCall] = []
            seen_tool_calls: set[tuple[str, str]] = set()
            last_chunk = None
            llm_start = time.monotonic()

            async for chunk in self.provider.chat_stream(
                normalize_for_model(messages_with_system, self.model_name),
                tools_schema,
                response_format={"type": "json_object"}
                if self.json_mode
                else None,
            ):
                if chunk.delta_content:
                    full_content += chunk.delta_content

                    yield chunk.delta_content

                if chunk.delta_reasoning_content:
                    full_reasoning += chunk.delta_reasoning_content

                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        key = (
                            tc.name,
                            json.dumps(
                                tc.arguments,
                                sort_keys=True,
                                ensure_ascii=False,
                            ),
                        )
                        if key in seen_tool_calls:
                            continue
                        seen_tool_calls.add(key)
                        tool_calls.append(tc)

                # Record token usage from the last chunk

                if chunk.usage:
                    self.executor.record_api_call(chunk.usage)
                    last_chunk = chunk

            # Log LLM call trace
            self.executor.record_prompt_composition(
                messages_with_system,
                tools_schema,
            )
            self.trace_logger.log_llm_call(
                messages=messages_with_system,
                tools=tools_schema,
                response={
                    "content": full_content,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in tool_calls
                    ],
                    "usage": (
                        last_chunk.usage
                        if last_chunk and last_chunk.usage
                        else None
                    ),
                },
                duration_ms=int((time.monotonic() - llm_start) * 1000),
                model=self.model_name,
            )

            # Filter out invalid tool calls
            tool_calls = [tc for tc in tool_calls if tc.name]

            self.turn_tool_calls += len(tool_calls)
            for tc in tool_calls:
                if tc.name in ("delegate", "subagent_invoke"):
                    self.turn_subagents += 1
                elif tc.name == "delegate_parallel":
                    tasks = tc.arguments.get("tasks")
                    self.turn_subagents += (
                        len(tasks) if isinstance(tasks, list) and tasks else 1
                    )
                if tc.name.startswith("mcp_"):
                    self.turn_mcp_calls += 1

            if not tool_calls:
                if full_content:
                    # Detect truncated tool intent: LLM described an action but
                    # the tool call never materialized (stream cut short).
                    _TOOL_HINTS = (
                        "write_file",
                        "read_file",
                        "run_command",
                        "search_files",
                        "list_directory",
                        "delete_file",
                        "edit_file",
                        "create_directory",
                    )
                    if any(h in full_content for h in _TOOL_HINTS):
                        warn = (
                            "\n[Response may be truncated: the model "
                            "described a tool action but produced no "
                            "complete tool call]\n"
                        )
                        full_content += warn
                        yield warn
                    final_msg: dict = {
                        "role": "assistant",
                        "content": full_content,
                    }
                    if full_reasoning:
                        final_msg["reasoning_content"] = full_reasoning
                    self.messages.append(final_msg)
                    last_content = full_content
                else:
                    yield (
                        "\n[Model returned no content; the model name "
                        "may not exist or the service is unresponsive]\n"
                    )
                break

            # Handle tool calls
            if full_content:
                last_content = full_content
            response = LLMResponse(
                content=full_content,
                tool_calls=tool_calls,
                reasoning_content=full_reasoning,
            )
            assistant_msg = self._build_assistant_message(response)
            self.messages.append(assistant_msg)
            messages_with_system.append(assistant_msg)

            # Execute tool calls in parallel when multiple tools are called.
            # If ANY tool may prompt for user confirmation (stdin), we MUST
            # fall back to sequential execution — concurrent stdin prompts
            # corrupt each other and produce garbled output.
            yield ""  # Stop thinking spinner before tool execution
            any_needs_confirm = any(
                self._needs_confirmation(tc) for tc in tool_calls
            )
            if len(tool_calls) > 1 and not any_needs_confirm:
                # Parallel execution
                results = await asyncio.gather(
                    *[self._execute_tool(tc) for tc in tool_calls],
                    return_exceptions=True,
                )
                supplement_info = None
                aborted: UserAbortedTurn | None = None
                for tool_call, result in zip(tool_calls, results):
                    if isinstance(result, UserAbortedTurn):
                        # Stop-the-turn signal (e.g. hook-forced confirm on an
                        # AUTO tool). Every call still gets a tool message so
                        # tool_call pairing stays intact, then we re-raise.
                        aborted = aborted or result
                        result = "Cancelled (user aborted this turn)"
                    elif isinstance(result, UserSupplement):
                        supplement_info = result.supplement
                        result = "Cancelled (user added info to replan)"
                    elif isinstance(result, Exception):
                        result = f"Error: {result}"

                    yield (
                        f"\n[{tool_call.name}] →\n"
                        f"{tool_result_for_display(tool_call.name, result)}\n"
                    )
                    tool_msg = {
                        "role": "tool",
                        "content": tool_result_for_history(result),
                        "name": tool_call.name,
                        "tool_use_id": tool_call.id,
                        "tool_call_id": tool_call.id,
                    }
                    self.messages.append(tool_msg)
                    messages_with_system.append(tool_msg)

                if aborted:
                    raise aborted
                if supplement_info:
                    supplement_msg = {
                        "role": "user",
                        "content": (
                            f"[User supplement]: {supplement_info}\n"
                            f"Replan with the new info and continue."
                        ),
                    }
                    self.messages.append(supplement_msg)
                    messages_with_system.append(supplement_msg)
                    continue
            else:
                # Sequential execution: either a single tool, or multiple tools
                # where at least one needs user confirmation (stdin prompts).
                try:
                    answered_ids: set[str] = set()
                    for tool_call in tool_calls:
                        result = await self._execute_tool(tool_call)
                        rendered = tool_result_for_display(
                            tool_call.name,
                            result,
                        )
                        yield f"\n[{tool_call.name}] →\n{rendered}\n"
                        tool_msg = {
                            "role": "tool",
                            "content": tool_result_for_history(result),
                            "name": tool_call.name,
                            "tool_use_id": tool_call.id,
                            "tool_call_id": tool_call.id,
                        }
                        self.messages.append(tool_msg)
                        messages_with_system.append(tool_msg)
                        answered_ids.add(tool_call.id)
                except UserSupplement as e:
                    self._close_pending_tool_calls(
                        tool_calls,
                        answered_ids,
                        messages_with_system,
                    )
                    supplement_text = (
                        f"[User supplement]: {e.supplement}\n"
                        f"Replan with the new info and continue."
                    )
                    yield f"\n{supplement_text}\n"
                    supplement_msg = {
                        "role": "user",
                        "content": supplement_text,
                    }
                    self.messages.append(supplement_msg)
                    messages_with_system.append(supplement_msg)
                    continue
                except UserAbortedTurn:
                    # Close out unanswered tool_calls before propagating —
                    # otherwise the saved/in-memory history keeps an assistant
                    # message whose tool_calls have no tool responses, and the
                    # next API request is rejected by the provider.
                    self._close_pending_tool_calls(
                        tool_calls,
                        answered_ids,
                        messages_with_system,
                        reason="Cancelled (user aborted this turn)",
                    )
                    raise
        else:
            yield "\n(max turn limit reached)"

        # Response hook
        await self.hook_bus.dispatch_async(
            HookContext(
                event="on_response",
                tool_name="",
                arguments={"input": user_input, "content": last_content},
            ),
        )

        # Persist session before memory write so a memory exception can't
        # cost us the conversation.
        self.save_session()

        # Store memory after conversation ends (whether normal or max_turns).
        # Strip images from the user side — memory backends expect text.
        if last_content:
            await self._store_memory(
                [
                    {"role": "user", "content": text_of(user_input)},
                    {"role": "assistant", "content": last_content},
                ],
            )

        # Store conversation history summary
        self._store_history()

        # Record experience for learning (only if tools were used)
        if self._current_turn_tools:
            task_summary = text_of(user_input)[:100]  # Truncate for storage
            outcome = _classify_outcome(
                self._turn_tool_successes,
                self._turn_tool_failures,
            )
            self.memory_manager.record_experience(
                task_summary=task_summary,
                tools_used=self._current_turn_tools,
                outcome=outcome,
            )

            # Record tool sequence for directives auto-learning
            if outcome == "success":
                from dashscope.acli.memory.directives_learning import (
                    _load_patterns,
                    analyze_patterns,
                    propose_directive,
                    record_tool_sequence,
                )

                record_tool_sequence(self._current_turn_tools)

                # Periodically analyze patterns and auto-propose directives
                try:
                    seq_count = len(_load_patterns().get("sequences", []))
                    if seq_count >= 5 and seq_count % 5 == 0:
                        proposals = analyze_patterns()
                        for p in proposals:
                            propose_directive(p["directive"], p["pattern"])
                except Exception:
                    pass  # directives learning must never break the agent

    def _store_history(self) -> None:
        """Persist a conversation history summary. Swallows errors so a
        storage failure never breaks the agent loop."""
        try:
            from dashscope.acli.platforms.local import history

            history.store_history(self.messages)
        except Exception as e:
            console.print(f"[dim red](history error: {e})[/dim red]")

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        tool_def = registry.get(tool_call.name)
        if not tool_def:
            return f"Error: unknown tool '{tool_call.name}'"

        # Track tool usage for experience recording (deduped, keep
        # first-use order)
        if tool_call.name not in self._current_turn_tools:
            self._current_turn_tools.append(tool_call.name)

        # Before-tool-call hooks
        before_ctx = HookContext(
            event="before_tool_call",
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
        )
        before_result = await self.hook_bus.dispatch_async(before_ctx)

        if before_result.blocked:
            msg = (
                before_result.warnings[0]
                if before_result.warnings
                else "blocked by hook"
            )
            from dashscope.acli.audit import get_audit_logger

            get_audit_logger().log_tool_call(
                tool_call.name,
                tool_call.arguments,
                decision="denied",
                reason=f"hook: {msg}",
            )
            return f"Blocked: {msg}"

        if before_result.confirm:
            from dashscope.acli.tools.registry import ToolDefinition

            synthetic = ToolDefinition(
                name=tool_call.name,
                description=tool_def.description,
                permission=PermissionLevel.CONFIRM,
                func=tool_def.func,
                parameters=tool_def.parameters,
            )
            if not await self.executor._async_check_permission(
                synthetic,
                tool_call.arguments,
            ):
                from dashscope.acli.audit import get_audit_logger

                get_audit_logger().log_tool_call(
                    tool_call.name,
                    tool_call.arguments,
                    decision="denied",
                    reason="hook confirm declined",
                )
                return "Cancelled"

        start_time = time.time()
        result = await self.executor.execute(tool_def, tool_call.arguments)
        duration_ms = int((time.time() - start_time) * 1000)
        success = not result.startswith(
            _TOOL_ERROR_PREFIXES,
        ) and not result.startswith(
            _TOOL_CANCEL_PREFIXES,
        )
        self.trace_logger.log_tool_execution(
            tool_call.name,
            tool_call.arguments,
            result,
            duration_ms,
            success,
        )

        # After-tool-call hooks
        after_ctx = HookContext(
            event="after_tool_call",
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            result=result,
            success=success,
        )
        after_result = await self.hook_bus.dispatch_async(after_ctx)

        # On-error hooks
        if not success:
            error_ctx = HookContext(
                event="on_error",
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=result,
                success=False,
            )
            error_result = await self.hook_bus.dispatch_async(error_ctx)
            after_result.warnings.extend(error_result.warnings)
            after_result.alerts.extend(error_result.alerts)
            after_result.outputs.extend(error_result.outputs)
            after_result.logs.extend(error_result.logs)

        # Emit non-silent hook outputs to the console
        for warning in after_result.warnings:
            console.print(f"[yellow]⚠️  {warning}[/yellow]")
        for alert in after_result.alerts:
            console.print(f"[red]🚨 {alert}[/red]")
        for log in after_result.logs:
            console.print(f"[dim]{log}[/dim]")

        # Record tool execution via memory manager (trace + reflection)
        self.memory_manager.log_trace(
            "tool_execution",
            {
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "result": result,
                "duration_ms": duration_ms,
                "success": success,
            },
        )
        # Reflection counts only real failures — user rejection/cancellation
        # is neutral and must not feed the consecutive-failure counter.
        if not result.startswith(_TOOL_CANCEL_PREFIXES):
            self.memory_manager.record_tool_execution(tool_call.name, success)
            if success:
                self._turn_tool_successes += 1
            else:
                self._turn_tool_failures += 1

        # Add fallback hints for common failures
        if not success and result.startswith(_TOOL_ERROR_PREFIXES):
            fallback_hint = (
                self.memory_manager.session.tool_chains.get_fallback_hints(
                    tool_call.name,
                )
            )
            if fallback_hint:
                result = result + fallback_hint

        return result

    async def _ensure_fits_context(self, extra_tokens: int = 0) -> None:
        """Gateway-style safety compression before an API call.

        If the accumulated conversation plus request overhead is near the
        model's hard threshold, compress older messages while keeping the most
        recent ones.  This prevents a single huge tool output from blowing
        the window.
        """
        compressed = await safety_compress_if_needed(
            self.messages,
            self.provider.chat,
            context_window=self._context_window,
            extra_tokens=extra_tokens,
        )
        if compressed is not None:
            old_count = len(self.messages)
            self.messages = compressed
            console.print(
                f"[dim]safety compression: {old_count} messages → "
                f"{len(self.messages)}[/dim]",
            )

    async def _auto_compress(self, extra_tokens: int = 0) -> None:
        """Auto-compress conversation when the token budget runs high.

        Token-aware triggering against the model's context window, with
        tail-message preservation and tool-output truncation.
        """
        from dashscope.acli.compression import compress_messages

        compressed = await compress_messages(
            self.messages,
            self.provider.chat,
            context_window=self._context_window,
            extra_tokens=extra_tokens,
        )
        if compressed is None:
            return

        old_count = len(self.messages)
        self.messages = compressed
        console.print(
            f"[dim]auto-compression: {old_count} messages → "
            f"{len(self.messages)}[/dim]",
        )

        # Log compression decision
        self.trace_logger.log_decision(
            "auto_compress",
            {
                "old_count": old_count,
                "new_count": len(self.messages),
                "context_window": self._context_window,
                "extra_tokens": extra_tokens,
            },
        )

    # Tool families that may need interactive confirmation (stdin prompts).
    # When multiple such tools appear in one batch, we must fall back to
    # sequential execution — concurrent stdin prompts corrupt each other.
    _NEEDS_CONFIRM_TOOLS = frozenset(
        {"write_file", "delete_file", "delete_directory", "move_file"},
    )

    def _needs_confirmation(self, tool_call: ToolCall) -> bool:
        """Check whether a tool call may trigger an interactive
        confirmation prompt.

        Conservative heuristic: known dangerous tool names OR any tool with
        a non-AUTO permission level in the registry.
        """
        if tool_call.name in self._NEEDS_CONFIRM_TOOLS:
            return True
        tool_def = registry.get(tool_call.name)
        if tool_def and tool_def.permission != PermissionLevel.AUTO:
            return True
        return False

    def _close_pending_tool_calls(
        self,
        tool_calls: list[ToolCall],
        answered_ids: set[str],
        messages_with_system: list[dict],
        reason: str = "Cancelled (user added info to replan)",
    ) -> None:
        """Append synthetic tool responses for tool_calls left unanswered.

        A UserSupplement aborts the remaining tool loop, which would leave
        the assistant message's tool_calls without matching tool responses —
        the next API request requires one tool response per tool_call.
        """
        for tc in tool_calls:
            if tc.id in answered_ids:
                continue
            tool_msg = {
                "role": "tool",
                "content": reason,
                "name": tc.name,
                "tool_use_id": tc.id,
                "tool_call_id": tc.id,
            }
            self.messages.append(tool_msg)
            messages_with_system.append(tool_msg)

    def _build_assistant_message(self, response: LLMResponse) -> dict:
        msg: dict = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        if getattr(response, "reasoning_content", ""):
            msg["reasoning_content"] = response.reasoning_content
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(
                            tc.arguments,
                            ensure_ascii=False,
                        ),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg
