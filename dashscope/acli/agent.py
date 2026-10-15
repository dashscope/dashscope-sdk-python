# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches,too-many-statements,protected-access
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncIterator

from rich.console import Console

from dashscope.acli.acceptance import AcceptanceGate
from dashscope.acli.compression import (
    HARD_THRESHOLD_RATIO,
    estimate_message_tokens,
    estimate_tokens,
    restore_note,
    safety_compress_if_needed,
    shrink_old_tool_messages,
)
from dashscope.acli.config import WORKSPACE_DIR, context_window_for_model
from dashscope.acli.executor import Executor
from dashscope.acli.hooks import HookBus, HookContext, create_hook_bus
from dashscope.acli.memory.manager import MemoryManager
from dashscope.acli.memory.reflection import (
    compression_hint,
    convergence_hint,
    is_readonly_tool_call,
    time_budget_hint,
)
from dashscope.acli.platforms.base import MemoryProvider
from dashscope.acli.prompt_pipeline import PromptContext, default_pipeline
from dashscope.acli.providers.base import LLMProvider, LLMResponse, ToolCall
from dashscope.acli.providers.hardening import (
    RetriesExhausted,
    is_retryable_error,
)
from dashscope.acli.skills import get_skill_manager, skills_summary_for_llm
from dashscope.acli.tools.registry import PermissionLevel, registry
from dashscope.acli.utils import (
    UserAbortedTurn,
    UserSupplement,
    normalize_for_model,
    sanitize,
    sanitize_text,
    text_of,
    tool_result_for_display,
    tool_result_for_history,
    truncate,
)

console = Console()


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


def _tool_message(tool_call: ToolCall, result: str) -> dict:
    """Build the history entry answering ``tool_call``.

    Redaction happens here, ahead of truncation. ``save_session`` already ran
    ``sanitize`` over the on-disk copy, so before this a key read out of the
    environment was scrubbed from the session file and still handed to the
    model in full — the persistence path was protecting the wrong reader.
    Scrubbing after truncation would not work either: ``truncate_head_tail``
    cuts on a character budget and can split a token in half, leaving a
    fragment that matches no pattern.

    The restore note must be attached here for the same reason: later
    shrinking skips a result that already carries an omission marker, so the
    largest results would end up with no way to be read back.
    """
    return {
        "role": "tool",
        "content": tool_result_for_history(
            sanitize_text(result),
            note=restore_note(tool_call.name, tool_call.arguments),
        ),
        "name": tool_call.name,
        "tool_use_id": tool_call.id,
        "tool_call_id": tool_call.id,
    }


def _budget_burned_slowly(error: BaseException) -> bool:
    """True when the provider stack already spent real wall clock on this turn.

    ``RetriesExhausted`` on its own is not the signal it looks like. Raised
    after a stalled endpoint it means ``max_retries x request_timeout`` went
    into attempts that each timed out (268s measured), and re-attempting here
    multiplies that — the 819s of silent wall clock the turn retry was
    removed for. Raised after a refused connection it means only the ~28s of
    backoff was spent, and re-attempting is cheap and is exactly what rides
    out a transient blip. Matching on the type alone gives up in both cases;
    the elapsed seconds the exception carries tell them apart.
    """
    return isinstance(error, RetriesExhausted) and error.exhausted_slowly


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
3. For clear-intent requests (git commit, read file, search), call tools
   directly without stating a plan first. For code changes, go straight to
   the edit — but you must verify it afterwards (rule 17). Only for
   irreversible operations (delete, overwrite, force push) or complex
   multi-step tasks, briefly explain and wait for user confirmation
4. When a tool call fails, do not retry the same call; report the error to
   the user with a suggestion
5. Tools prefixed [MCP:xxx] come from Bailian MCP services; call them directly
6. If the "user profile" contains info, personalize answers with it (e.g.
   recommend Python solutions for a Python user)
7. When a request matches an "available Skill template" below, prefer that
   flow — if the Skill's MCP is connected, call its tools directly; if not,
   connect first with mcp_connect, then call
8. When a question may involve personal preferences, tech stack, or work
   environment, proactively search the profile with memory_search for context
9. If no suitable tool can complete the task, answer from your knowledge;
   never call a nonexistent tool
10. To switch model/Provider → call switch_model or switch_provider directly
11. To use cloud services (time, code execution, doc parsing, etc.) → call
    mcp_connect directly
12. To enable/disable capabilities → call capability_enable /
    capability_disable directly
13. Never reply "please use the /xxx command" — if you have a tool that can
    do it, just do it
14. **Do not use run_command with python3 -c inline scripts to read/write
    files** — it is inefficient. Read with read_file, write with write_file;
    shell tools like sed/grep are fine to use
15. **If a question can be answered directly (no files, commands, or external
    info needed), answer from your knowledge without calling tools** — e.g.
    self-introduction, explaining concepts, translating text, small talk
16. **Multiple independent subtasks (e.g. "look at these 2 files", several
    unrelated tasks at once) → call delegate_parallel to fan out subagents**,
    or send multiple delegate calls in the same turn; a single independent
    but lengthy subtask (whole-file review, multi-file scan) → call
    subagent_invoke for isolated execution and take back only the conclusion.
    Do not grind through them serially yourself
17. **Verify code changes before reporting done.** Run the tests that cover
    what you touched; if nothing covers it, add a focused test for the new
    behaviour and run that. Report the command and its pass/fail result —
    never claim a change works without having executed it. When the output
    transforms or joins data (ETL, CSV/JSON, migrations), also assert every
    cross-reference resolves — schema-valid output with dangling IDs still
    fails. When the task names a ground-truth tool or command for a check,
    run the final verification with that exact tool and its stated
    parameters — a calibrated substitute proves nothing about the grade
18. **Snapshot a fragile artifact before inspecting it.** A tool meant to
    read can still mutate what it opens — SQLite deletes an invalid WAL
    when the connection closes, and destroyed evidence cannot be
    re-downloaded. Copy the file to scratch space first (cp to /tmp), or
    open it immutably (e.g. sqlite3 'file:...?immutable=1')
19. **Read images with the vision model, not by proxy.** When a task
    hinges on perceiving an image — text or digits in a picture, a chart,
    a rendered layout — call view_image on the file itself; reading it
    back through terminal-resolution ASCII art or hand-rolled pixel
    scripts throws away exactly the detail the answer depends on. If the
    artifact is data rather than an image (geometry, toolpaths, a
    waveform), render it to a PNG first and then look at it with
    view_image

Reply style:
- **Concise**. No filler like "let me see / let me help you / I'll analyze
  it"; just act or give the answer. Announcing that you are about to verify
  is filler; running the check and reporting its result is not
- **One shot**. Read files with read_file (use offset/limit for large spans);
  never use python3 -c inline scripts for file I/O
- **Batch in parallel**. Issue multiple independent tool calls for the same
  purpose in one turn; do not call them one by one serially
- **No re-confirmation**. Do not re-read facts already fetched; if a tool
  fails once, report the error to the user instead of retrying a rephrased
  version of the same action
- Do not recap the diff or restate what you just did unless asked — but do
  report verification results (command + pass/fail), which the user cannot
  see for themselves
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
        oneshot: bool = False,
    ):
        self.provider = provider
        self.executor = executor
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.allow_delegate = allow_delegate
        self.allowed_tools = allowed_tools
        self.hook_bus = hook_bus or create_hook_bus()
        self.json_mode = json_mode
        self.oneshot = oneshot
        # Hard stop for read-only stalls in oneshot (benchmark/script) mode.
        # Env override: ACLI_READONLY_HARD_CAP (0 disables).
        try:
            self.readonly_hard_cap = int(
                os.environ.get("ACLI_READONLY_HARD_CAP", "40"),
            )
        except ValueError:
            self.readonly_hard_cap = 40
        # Turn-level retry: re-run the current turn (only when nothing was
        # streamed yet, to avoid duplicate output) up to turn_retry_max times
        # with capped exponential backoff. Covers providers that carry no
        # retry layer of their own; a HardenedProvider that exhausted its own
        # budget slowly (stalled endpoint, not a refused connection) is not
        # re-paid here — see _budget_burned_slowly.
        # Env override: ACLI_TURN_RETRY_MAX (0 disables).
        try:
            self.turn_retry_max = int(
                os.environ.get("ACLI_TURN_RETRY_MAX", "2"),
            )
        except ValueError:
            self.turn_retry_max = 2
        self.turn_retry_delay = 5.0
        self.turn_retry_backoff_cap = 30.0
        # Empty-turn backstop (oneshot only): a turn that ends with no tool
        # calls and no content is a provider failure shape, not "done". An
        # autonomous run must not exit on it while budget remains — nudge
        # and continue, bounded so a persistently empty endpoint still
        # terminates. Interactive keeps the historical break: there the
        # user sees the notice and simply speaks again.
        self._empty_turns = 0
        self._empty_turn_max = 2
        # Budget-aware convergence nudge (oneshot only): as the turn budget is
        # consumed, tell the agent to switch approach if plateaued, lock in an
        # already-passing result, then finalize. Covers productive-but-stuck
        # iteration that ReflectionTracker (failures) and StagnationTracker
        # (read-only stalls) both miss. Env: ACLI_CONVERGE_SOFT/_HARD (ratios;
        # soft >= 1.0 disables).
        try:
            self.converge_soft_ratio = float(
                os.environ.get("ACLI_CONVERGE_SOFT", "0.6"),
            )
        except ValueError:
            self.converge_soft_ratio = 0.6
        try:
            self.converge_hard_ratio = float(
                os.environ.get("ACLI_CONVERGE_HARD", "0.85"),
            )
        except ValueError:
            self.converge_hard_ratio = 0.85
        # Wall-clock budget nudge (oneshot only): the turn-count convergence
        # hint never fires when a run is few-but-huge generations, yet the
        # harness kills the run on wall time all the same — and the hard-set
        # tasks that died so (dna-assembly, write-compressor) all died with
        # their deliverable never written. The bench adapter injects the
        # in-container `timeout` value as ACLI_TIME_BUDGET_SEC; unset/<=0
        # disables. Ratios: ACLI_TIME_SOFT / ACLI_TIME_HARD.
        try:
            self.time_budget_sec = float(
                os.environ.get("ACLI_TIME_BUDGET_SEC", "0"),
            )
        except ValueError:
            self.time_budget_sec = 0.0
        try:
            self.time_soft_ratio = float(
                os.environ.get("ACLI_TIME_SOFT", "0.5"),
            )
        except ValueError:
            self.time_soft_ratio = 0.5
        try:
            self.time_hard_ratio = float(
                os.environ.get("ACLI_TIME_HARD", "0.8"),
            )
        except ValueError:
            self.time_hard_ratio = 0.8
        # Monotonic anchor for the budget above, set on the first turn: a
        # wall clock would credit host-suspend time the process never had.
        self._run_started: float | None = None
        # Silence nudge: one reminder per turn after N tool calls have gone by
        # without any prose reaching the user. Mode-independent on purpose —
        # the shape this exists for is an interactive turn of *distinct*
        # run_command calls, which every existing backstop misses:
        # StagnationTracker's read-only classifier counts run_command as
        # mutating and so resets its streak, its repeat counters need
        # byte-identical calls, and _convergence_section is oneshot-only.
        # Env: ACLI_SILENT_TOOL_NUDGE (0 disables).
        try:
            self.silent_tool_nudge = int(
                os.environ.get("ACLI_SILENT_TOOL_NUDGE", "6"),
            )
        except ValueError:
            self.silent_tool_nudge = 6
        # Hard acceptance gate (oneshot only): run the project's test command
        # before accepting a final answer, and send the raw output back when
        # it fails. Built here, ahead of the policy text below, so the prompt
        # can promise the gate only when one is actually armed.
        # Env: ACLI_ACCEPTANCE (0 disables), ACLI_ACCEPTANCE_CMD (overrides
        # discovery), ACLI_ACCEPTANCE_RETRIES, ACLI_ACCEPTANCE_TIMEOUT.
        self._acceptance_gate = AcceptanceGate.from_env(self.oneshot)
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
        if self.oneshot:
            self.system_prompt = (self.system_prompt or SYSTEM_PROMPT) + (
                "\n\nTask execution policy (autonomous mode):\n"
                "- Work from a written plan, not from memory: early in the "
                "run, write a short plan file (goal, acceptance check, "
                "ordered steps); after each step, record in it what is done "
                "and what remains; before reporting completion, re-read it "
                "and verify each item. Sketch the deliverable itself first, "
                "then iterate on it. Never spend a whole turn only "
                "thinking — every turn must land something on disk: a file, "
                "a measurement, or an updated plan.\n"
                "- If the request has a measurable acceptance criterion "
                "(numeric threshold, size/time limit, test suite, "
                "similarity/accuracy target), iterate until it is met: "
                "implement -> measure against the criterion -> refine. Do "
                "not stop at the first version that merely runs; optimize "
                "toward the target — but once the criterion is met, lock it "
                "in and finish rather than risk regressing a passing result. "
                "Heed any budget-check hint: it tells you when to converge "
                "or switch to a different approach. A wall-clock hint means "
                "the run is killed when time runs out: get the required "
                "deliverable written early, then keep improving it.\n"
                "- Before reporting completion, turn every stated "
                "requirement into its own executable check and run them "
                "all — the measurable criterion, the format rules, and "
                "any constraint that mentions components the deliverable "
                "must contain. Where a check depends on domain facts, "
                "state the fact explicitly first (a wrong assumption is "
                "worse than a flagged unknown). Fix what fails instead "
                "of handing back a near miss.\n"
                "- When a check fails close to the target, treat it as "
                "actionable feedback: change the approach based on what "
                "the measurement shows, then re-measure.\n"
                "- Do not stake a large share of the remaining budget on "
                "one long unverified command: before an install, compile "
                "or download that could run for minutes, spend seconds "
                "probing that the path can succeed at all (package "
                "available, toolchain present); give long commands an "
                "explicit timeout so a stuck one fails fast and frees "
                "the rest of the budget."
            )
            if self._acceptance_gate.active:
                # An armed gate implies oneshot — from_env hands back an empty
                # command otherwise — so this belongs inside the block above.
                # Naming the command beats a generic promise: the model can
                # run it itself first instead of meeting the gate at the exit.
                check = self._acceptance_gate.command
                self.system_prompt += (
                    f"\n- Verification gate: `{check}` runs before your "
                    "final answer is accepted. If it fails, the answer is "
                    "withheld and the raw output comes back to you as the "
                    "next requirement. Make it pass by fixing the code — "
                    "never by editing, skipping, or weakening the check "
                    "itself."
                )
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
        self._reflection_hint_injected = False
        # Per-turn counters for the TUI status line
        self.turn_tool_calls = 0
        self.turn_subagents = 0
        self.turn_mcp_calls = 0
        self.turn_skills = 0
        # Consecutive tool calls that streamed no prose, and whether the
        # silence nudge has already been spent this turn.
        self._silent_tool_calls = 0
        self._silent_nudge_injected = False
        # Compression observability toward the model: the event is set by
        # either compression path and spent once by _compression_section;
        # the ratio is refreshed by the mid-turn guard's estimate.
        self._compression_event: tuple[int, int] | None = None
        self._context_ratio = 0.0
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
        """Restore self.messages. Returns the number of messages loaded
        (0 if nothing could be restored).

        The per-topic event stream is the source of truth: the latest
        ``messages/snapshot`` wins when present (crash recovery via
        append-only replay; torn trailing lines are skipped). The
        ``history.json`` file remains as the compat fallback for
        sessions written before snapshots existed.
        """
        if not self.session_path:
            return 0
        try:
            from dashscope.acli.session_events import (
                EVENTS_FILENAME,
                SessionEventLog,
                latest_snapshot_messages,
            )

            log = SessionEventLog(self.session_path.parent / EVENTS_FILENAME)
            resumed = latest_snapshot_messages(log.read_raw())
            if resumed:
                self.messages = resumed
                return len(resumed)
        except Exception:
            pass
        if not self.session_path.exists():
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
            scene_provider=self._scene_section,
            current_turn_tools=self._current_turn_tools,
            connected_mcp_services=self._connected_mcp_services,
        )
        return self._prompt_pipeline.render(ctx)

    def _scene_section(self) -> str:
        """Scene memory of the current session topic (best-effort).

        Subagents and SDK callers may run without a session manager;
        any failure simply yields no scene section.
        """
        try:
            from dashscope.acli.session import get_session_manager

            return get_session_manager().get_scene()
        except Exception:
            return ""

    def _reflection_section(self) -> str:
        """Inject an evidence-anchored hint on repeated tool failures.

        The lesson is deliberately *not* recorded here. This runs before the
        model has answered, so all it could capture was the tracker's own
        summary of a count. The diagnosis the model writes in response to the
        anchored hint is what makes a lesson recallable on a similar task
        later, so ``_record_reflection_lesson`` picks it up after the stream.
        """
        tracker = self.memory_manager.session.reflection
        if not tracker.needs_reflection():
            return ""
        hint = tracker.get_reflection_hint()
        if hint and self._current_turn_tools:
            self._reflection_hint_injected = True
        return hint

    def _record_reflection_lesson(self, diagnosis: str) -> None:
        """Persist the model's own failure diagnosis, once per turn.

        Called with the response to a turn that carried a reflection hint. An
        empty ``diagnosis`` — no content, or a stream cut short — falls back
        to the tracker's evidence summary, so a turn that hit the failure
        threshold still leaves something recallable behind.
        """
        if not self._reflection_hint_injected:
            return
        if self._reflection_lesson_recorded:
            return
        tracker = self.memory_manager.session.reflection
        lesson = tracker.get_failure_lesson(diagnosis=diagnosis)
        if not lesson:
            return
        self._reflection_lesson_recorded = True
        self._reflection_hint_injected = False
        self.memory_manager.record_experience(
            task_summary="repeated tool failures",
            tools_used=self._current_turn_tools,
            outcome="failure",
            lesson=lesson,
        )

    def _stagnation_section(self) -> str:
        """Inject a nudge on read-only stalls or repeated identical calls.

        ``get_stagnation_hint`` returns "" until one of those fires, so the
        tracker — not this method — decides whether the turn gets a section.
        """
        tracker = self.memory_manager.session.stagnation
        cap = self.readonly_hard_cap if self.oneshot else None
        return tracker.get_stagnation_hint(hard_cap=cap)

    def _convergence_section(self, loop_index: int) -> str:
        """Inject a budget-aware converge/switch nudge in oneshot runs.

        Fires on the fraction of the turn budget consumed, covering the
        productive-but-plateaued loop that failure/read-only trackers miss.
        """
        if not self.oneshot:
            return ""
        return convergence_hint(
            loop_index,
            self.max_turns,
            self.converge_soft_ratio,
            self.converge_hard_ratio,
        )

    def _time_budget_section(self) -> str:
        """Wall-clock budget nudge; see time_budget_hint for the rationale."""
        if not self.oneshot or self._run_started is None:
            return ""
        return time_budget_hint(
            time.monotonic() - self._run_started,
            self.time_budget_sec,
            self.time_soft_ratio,
            self.time_hard_ratio,
        )

    def _compression_section(self) -> str:
        """Tell the model about compression it would otherwise never see.

        The compression event is spent once — after a compression the ratio
        drops below the warn band anyway, so the state-based pre-hint alone
        cannot double-fire. The ratio is refreshed by the mid-turn guard, so
        the pre-hint can only fire from the second loop iteration on; the
        first iteration of a turn is covered by the event path instead
        (``_auto_compress`` runs just before the loop).
        """
        event = self._compression_event
        self._compression_event = None
        return compression_hint(event, self._context_ratio)

    # Seconds of wall-clock budget a single run_command may never consume:
    # the remainder minus a reserve for writing/finalizing the deliverable.
    # Run 2026-10-12__reqenum-0612 lost 720s of an 1800s budget to two
    # run_command timeouts (a 600s pip compile that could never succeed
    # offline plus a 120s follow-up) and died with the deliverable never
    # written; the hint alone can only ask, this enforces.
    _CMD_TIMEOUT_RESERVE_SEC = 180
    _CMD_TIMEOUT_MIN_SEC = 30

    def _command_timeout_ceiling(self) -> int | None:
        """Max seconds one run_command may take, or None when unbudgeted."""
        if (
            not self.oneshot
            or self.time_budget_sec <= 0
            or self._run_started is None
        ):
            return None
        remaining = self.time_budget_sec - (
            time.monotonic() - self._run_started
        )
        return max(
            self._CMD_TIMEOUT_MIN_SEC,
            int(remaining - self._CMD_TIMEOUT_RESERVE_SEC),
        )

    def _clamp_run_command_timeout(self, tool_call) -> str:
        """Clamp run_command's timeout to the budget ceiling.

        Mutates tool_call.arguments and returns a note for the tool
        result when it clamped, "" otherwise. The note must ride the
        result: the assistant's own tool_call entry in history keeps the
        model's original arguments, so without it the model cannot see
        why its 600s command died at 120s.
        """
        if tool_call.name != "run_command":
            return ""
        ceiling = self._command_timeout_ceiling()
        if ceiling is None or self._run_started is None:
            return ""
        from dashscope.acli.tools.shell import default_timeout

        raw = tool_call.arguments.get("timeout")
        try:
            current = (
                int(raw) if raw not in (None, "", 0) else default_timeout()
            )
        except (TypeError, ValueError):
            current = default_timeout()
        if current <= ceiling:
            return ""
        tool_call.arguments["timeout"] = ceiling
        remaining = self.time_budget_sec - (
            time.monotonic() - self._run_started
        )
        return (
            f"[timeout clamped from {current}s to {ceiling}s: "
            f"{int(remaining)}s of wall-clock budget remain, "
            f"{self._CMD_TIMEOUT_RESERVE_SEC}s reserved for finalizing. "
            "Break the work into commands that fit, or pick a path that "
            "does not need the long step at all.]\n"
        )

    def _silence_section(self) -> str:
        """Inject one reminder when tool calls stop producing prose.

        Spent at most once per turn. Repeating it every iteration teaches the
        model to skim past it, which is the failure mode prose instructions
        already have — the point of a structural backstop is that it fires on
        a count rather than on the model's willingness to re-read advice.

        Deliberately not gated on ``oneshot``: the user watching a spinner for
        five minutes with no text is the interactive case, and the oneshot
        runs already have the convergence and read-only-cap sections.
        """
        if self._silent_nudge_injected or not self.silent_tool_nudge:
            return ""
        if self._silent_tool_calls < self.silent_tool_nudge:
            return ""
        self._silent_nudge_injected = True
        return (
            "\n## ⏸ You have gone quiet\n"
            f"{self._silent_tool_calls} tool calls in a row and not one word "
            "of it reached the user.\n"
            "Before the next call, say in one or two sentences what you are "
            "doing and what you have found so far. If the calls were "
            "collecting information to answer a question you can already "
            "answer, answer it instead of calling again.\n"
        )

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
        # A failed/interrupted turn can leave its user message as the last
        # history entry; resubmitting the same text must not stack a second
        # copy on top of it — the orphan is reused as this turn's input.
        if not (
            self.messages
            and self.messages[-1].get("role") == "user"
            and self.messages[-1].get("content") == user_input
        ):
            self.messages.append({"role": "user", "content": user_input})

        from dashscope.acli.audit import get_audit_logger

        get_audit_logger().log_user_input(text_of(user_input))

        # Track partial assistant content so Ctrl-C doesn't lose it.
        _partial: list[str] = []
        try:
            async for chunk in self._run_stream_body(user_input):
                _partial.append(chunk)
                yield chunk
        except BaseException as e:
            # On cancellation / error, save whatever was received so the
            # user message and partial reply survive the next session load.
            # A partial that is pure markdown scaffolding ("**", "```")
            # carries no information and is dropped.
            partial_text = "".join(_partial).strip()
            if partial_text and any(ch.isalnum() for ch in partial_text):
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": partial_text + " [interrupted]",
                    },
                )
            elif not isinstance(
                e,
                (
                    UserAbortedTurn,
                    KeyboardInterrupt,
                    SystemExit,
                    GeneratorExit,
                    asyncio.CancelledError,
                ),
            ):
                # A provider/turn failure with zero output otherwise leaves
                # only the user message behind: the history then shows N
                # stacked user prompts with no trace of why each attempt
                # failed, and the model cannot learn the cause on resubmit.
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "[Turn failed before any response: "
                            f"{sanitize_text(str(e))[:500]}]"
                        ),
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
        if self._run_started is None:
            self._run_started = time.monotonic()
        self.turn_tool_calls = 0
        self.turn_subagents = 0
        self.turn_mcp_calls = 0
        self._silent_tool_calls = 0
        self._silent_nudge_injected = False
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
        # Cleared with the tracker below: a hint from the previous turn must
        # not license recording a lesson from this turn's first response.
        self._reflection_hint_injected = False

        # Reset reflection tracker for new turn
        self.memory_manager.session.reflection.reset()
        self.memory_manager.session.stagnation.reset()
        self._acceptance_gate.reset()

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
            # the request list only past the hard threshold. The estimate is
            # kept as the context-fill ratio for the compression hint below.
            if _loop_i > 0:
                used_tokens = (
                    estimate_message_tokens(self.messages) + overhead_tokens
                )
                self._context_ratio = (
                    used_tokens / self._context_window
                    if self._context_window > 0
                    else 0.0
                )
            if _loop_i > 0 and (
                used_tokens >= int(self._context_window * HARD_THRESHOLD_RATIO)
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
                    self._compression_event = (old_count, len(self.messages))
                    messages_with_system = [
                        {"role": "system", "content": system_prompt},
                        *self.messages,
                    ]
                    console.print(
                        f"[dim]mid-turn safety compression: {old_count} "
                        f"messages → {len(self.messages)}[/dim]",
                    )

            # Reflection / stagnation / convergence / silence hints are
            # volatile, so they ride at the tail of the request instead of
            # being folded into the system message. Rewriting
            # messages_with_system[0] on every iteration changed the whole
            # prompt prefix byte-for-byte and defeated provider-side prompt
            # caching, where cached input costs roughly a tenth of uncached.
            # The system message and the accumulated history below it stay
            # append-only and byte-stable.
            #
            # The hint has to be a user message, not a second system message:
            # the Anthropic converter keeps only the last system message, so a
            # second one would replace the real prompt outright, and the Tongyi
            # converter only reads messages[0] as system.
            #
            # All five sections are rebuilt every iteration because tracker
            # state changes as tools run; each returns "" when its condition
            # is not met, so an empty hint costs no message at all.
            hint_parts = {
                "reflection": self._reflection_section(),
                "stagnation": self._stagnation_section(),
                "convergence": self._convergence_section(_loop_i),
                "time_budget": self._time_budget_section(),
                "compression": self._compression_section(),
                "silence": self._silence_section(),
            }
            hint = "".join(hint_parts.values()).strip()
            # Which behaviour hints fired this call, and their sizes. Kept
            # out of the prompt and logged with the llm_call event so hint
            # efficacy is measurable from traces instead of guessed.
            hint_chars = {k: len(v) for k, v in hint_parts.items() if v}
            request_messages = (
                [*messages_with_system, {"role": "user", "content": hint}]
                if hint
                else messages_with_system
            )

            # Hard stop for read-only stalls in oneshot mode: finish with
            # whatever is on hand instead of inspecting forever.
            if (
                self.oneshot
                and self.readonly_hard_cap
                and self.memory_manager.session.stagnation.readonly_streak
                >= self.readonly_hard_cap
            ):
                stop_msg = (
                    f"\n[Stopped: {self.readonly_hard_cap} consecutive "
                    "read-only tool calls with no progress. Ending the run "
                    "to avoid an infinite verification loop.]\n"
                )
                yield stop_msg
                last_content = stop_msg
                break

            full_content = ""
            full_reasoning = ""
            tool_calls: list[ToolCall] = []
            seen_tool_calls: set[tuple[str, str]] = set()
            last_chunk = None
            finish_reason = ""
            llm_start = time.monotonic()

            # Turn-level retry around the streaming call, for providers that
            # carry no retry layer of their own. A HardenedProvider that spent
            # its budget on slow attempts raises RetriesExhausted and is NOT
            # re-attempted here: that multiplied max_retries x request_timeout
            # by turn_retry_max, which against a stalled endpoint measured
            # 819s of silent wall clock from a 60s configured timeout. A
            # budget spent failing fast is still re-attempted — see
            # _budget_burned_slowly. We only retry when nothing user-visible
            # was streamed yet; once content is yielded, a retry would
            # duplicate output, so that case degrades to a partial turn
            # instead. Reasoning chunks do not count as emitted: they are
            # never yielded to the user, so retrying a stream that carried
            # only thinking duplicates nothing. A stalled stream of pure
            # reasoning otherwise slipped past both retry layers and ended
            # the run as an empty turn (dna-insert, 2026-10-12__fix-071).
            turn_attempt = 0
            while True:
                full_content = ""
                full_reasoning = ""
                tool_calls = []
                seen_tool_calls = set()
                last_chunk = None
                finish_reason = ""
                emitted = False
                try:
                    async for chunk in self.provider.chat_stream(
                        normalize_for_model(
                            request_messages,
                            self.model_name,
                        ),
                        tools_schema,
                        response_format=(
                            {"type": "json_object"} if self.json_mode else None
                        ),
                    ):
                        if chunk.delta_content:
                            emitted = True
                            full_content += chunk.delta_content

                            yield chunk.delta_content

                        if chunk.delta_reasoning_content:
                            full_reasoning += chunk.delta_reasoning_content

                        if chunk.tool_calls:
                            emitted = True
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
                        if chunk.finish_reason:
                            finish_reason = chunk.finish_reason
                    break
                except Exception as e:
                    if (
                        not emitted
                        and is_retryable_error(e)
                        and turn_attempt < self.turn_retry_max
                        and not _budget_burned_slowly(e)
                    ):
                        turn_attempt += 1
                        await asyncio.sleep(
                            min(
                                self.turn_retry_delay
                                * (2 ** (turn_attempt - 1)),
                                self.turn_retry_backoff_cap,
                            ),
                        )
                        continue
                    if emitted and is_retryable_error(e):
                        # Mid-stream break. No layer retries this — the chunks
                        # already yielded would be duplicated — but it should
                        # not end the run either. Keep the partial response and
                        # let the loop move on: a connection reset 126s into a
                        # 1200s budget otherwise costs the other 1074s.
                        yield (
                            f"\n[Stream interrupted ({e}); continuing with "
                            "the partial response]\n"
                        )
                        break
                    raise

            # Log LLM call trace
            prompt_stats = self.executor.record_prompt_composition(
                request_messages,
                tools_schema,
            )
            # Char-based estimate logged next to the provider's usage in the
            # same event, so the two token accounts can be reconciled from
            # traces instead of drifting silently apart.
            prompt_stats["est_prompt_tokens"] = estimate_message_tokens(
                request_messages,
            ) + estimate_tokens(json.dumps(tools_schema, ensure_ascii=False))
            if hint_chars:
                prompt_stats["hint_chars"] = hint_chars
            # Size of the cacheable request prefix: every bucket except the
            # volatile trailing hint. A drop between consecutive llm_call
            # events means the prefix changed byte-for-byte and the
            # provider's prompt cache was defeated — this makes drift
            # measurable from traces instead of guessed (dsh's tracker has
            # the same lesson measured: 42% vs 99.8% cache hits).
            prompt_stats["prefix_chars"] = (
                prompt_stats["system_chars"]
                + prompt_stats["user_chars"]
                + prompt_stats["assistant_chars"]
                + prompt_stats["tool_result_chars"]
                + prompt_stats["tool_call_args_chars"]
                + prompt_stats["tools_schema_chars"]
                - len(hint)
            )
            self.trace_logger.log_llm_call(
                messages=request_messages,
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
                prompt_stats=prompt_stats,
            )

            # If this iteration carried a reflection hint, its answer is the
            # diagnosis worth remembering. Placed before both exits from the
            # iteration so neither path can skip it, and before the
            # truncation warning is appended below so the stored lesson is
            # the model's own text.
            self._record_reflection_lesson(full_content)

            # Filter out invalid tool calls
            tool_calls = [tc for tc in tool_calls if tc.name]

            self.turn_tool_calls += len(tool_calls)
            # Silence is per-iteration, not cumulative over the turn: an
            # iteration that streamed prose did talk to the user, so the run
            # of unheard calls starts over from the next one.
            if full_content.strip():
                self._silent_tool_calls = 0
            else:
                self._silent_tool_calls += len(tool_calls)
            if full_content.strip() or tool_calls:
                self._empty_turns = 0
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
                if finish_reason == "length":
                    # Output cap cut the generation mid-turn: this is not a
                    # final answer, so skip the gate and do not break. The
                    # partial goes into history and the loop continues with
                    # a shorter-chunks demand — which is also what makes
                    # every between-iteration mechanism (time budget,
                    # convergence, reflection) reachable again on the
                    # monolithic-generation failure shape (2 requests per
                    # 1800s in run 2026-10-12__time-069).
                    partial = full_content or (
                        "[response cut off by the output-length limit "
                        "during planning, before any visible output]"
                    )
                    cut_msg: dict = {"role": "assistant", "content": partial}
                    if full_reasoning:
                        cut_msg["reasoning_content"] = full_reasoning
                    self.messages.append(cut_msg)
                    messages_with_system.append(cut_msg)
                    nudge = (
                        "[Output-length limit reached — your response was "
                        "cut off. Continue working, but keep each response "
                        "short: put plans and intermediate results into "
                        "files with write_file in small pieces instead of "
                        "one long response; if you were composing the "
                        "deliverable, write it now in several smaller "
                        "calls.]"
                    )
                    yield f"\n{nudge}\n"
                    nudge_msg = {"role": "user", "content": nudge}
                    self.messages.append(nudge_msg)
                    messages_with_system.append(nudge_msg)
                    continue
                # Acceptance gate. This branch is the loop's only voluntary
                # exit, so it is the one place where "the model is done" can
                # be checked against something other than the model's own
                # opinion. Run before the answer is committed: a rejected
                # answer belongs in history as an attempt, not in last_content
                # as a result.
                verdict = await self._acceptance_gate.check()
                if verdict is not None and verdict.rejection:
                    self._withhold_answer(
                        full_content,
                        full_reasoning,
                        verdict.rejection,
                        messages_with_system,
                    )
                    yield (
                        "\n[Answer withheld: the verification check "
                        f"`{verdict.run.command}` did not pass; sending its "
                        "output back]\n"
                    )
                    continue
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
                    # The gate could not certify the answer but sending it back
                    # would not help either, so the caveat travels with it
                    # instead of the answer being silently downgraded.
                    if verdict is not None and verdict.note:
                        full_content += verdict.note
                        yield verdict.note
                    final_msg: dict = {
                        "role": "assistant",
                        "content": full_content,
                    }
                    if full_reasoning:
                        final_msg["reasoning_content"] = full_reasoning
                    self.messages.append(final_msg)
                    last_content = full_content
                else:
                    if (
                        self.oneshot
                        and self._empty_turns < self._empty_turn_max
                    ):
                        # A stall that slipped both retry layers (reasoning-
                        # only stream, then a clean-but-empty stop) must not
                        # pass for "done": with budget left, exiting here is
                        # how dna-insert died at 562/1800s with the
                        # deliverable never written (2026-10-12__fix-071).
                        self._empty_turns += 1
                        nudge = (
                            "[Empty response: the model returned no content "
                            "and no tool calls. The task is not finished — "
                            "keep working: re-read your plan file and take "
                            "the next step; if a deliverable file is "
                            "required, make sure it exists on disk.]"
                        )
                        yield f"\n{nudge}\n"
                        nudge_msg = {"role": "user", "content": nudge}
                        self.messages.append(nudge_msg)
                        messages_with_system.append(nudge_msg)
                        continue
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
                    tool_msg = _tool_message(tool_call, result)
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
                        tool_msg = _tool_message(tool_call, result)
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

        # Record the completed turn as events (session-as-event-log
        # direction) BEFORE persisting history.json: the event stream is
        # the source of truth, so it must never be older than the
        # fallback store if we crash between the two writes. Best-effort:
        # subagents/SDK callers may run without a session manager, and
        # event recording must never break the loop.
        try:
            from dashscope.acli.session import get_session_manager

            turn_topic = (
                self.session_path.parent.name if self.session_path else None
            )
            get_session_manager().record_turn_event(
                user_text=text_of(user_input),
                assistant_text=last_content,
                tools_used=self._current_turn_tools,
                outcome=(
                    _classify_outcome(
                        self._turn_tool_successes,
                        self._turn_tool_failures,
                    )
                    if self._current_turn_tools
                    else ""
                ),
                topic=turn_topic,
            )
            get_session_manager().record_messages_snapshot(
                self.messages,
                topic=turn_topic,
            )
        except Exception:
            pass

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
            # Flatten + ellipsis: raw prompts carry newlines, and a
            # mid-word cut breaks the "Past Lessons" list rendering.
            task_summary = truncate(" ".join(text_of(user_input).split()), 100)
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

        # Budget back-pressure on the one tool that can burn minutes of
        # wall clock in a single call; see _clamp_run_command_timeout.
        budget_note = self._clamp_run_command_timeout(tool_call)

        # Track tool usage for experience recording (deduped, keep
        # first-use order)
        if tool_call.name not in self._current_turn_tools:
            self._current_turn_tools.append(tool_call.name)

        # Futility refusal, checked before the hook so the user is never asked
        # to approve a call that will not run. Identical input cannot produce
        # a different output, so a repeat of an unchanged failure is refused
        # rather than nudged: the nudge was already being ignored.
        refusal = self.memory_manager.session.stagnation.refuse_if_futile(
            tool_call.name,
            tool_call.arguments,
        )
        if refusal:
            from dashscope.acli.audit import get_audit_logger

            get_audit_logger().log_tool_call(
                tool_call.name,
                tool_call.arguments,
                decision="denied",
                reason="futility: identical failure, no state change since",
            )
            return refusal

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

        # Acceptance baseline, taken from the last point where the call is
        # still guaranteed to run: after the futility refusal and the
        # permission hooks, before the executor. Read-only calls skip it, so a
        # question-answering turn never pays for a test suite, and the
        # baseline still comes from a workspace this turn has not touched.
        if not is_readonly_tool_call(tool_call.name, tool_call.arguments):
            await self._acceptance_gate.ensure_baseline()

        start_time = time.time()
        outcome = await self.executor.execute_detailed(
            tool_def,
            tool_call.arguments,
        )
        duration_ms = int((time.time() - start_time) * 1000)
        result = outcome.text
        if budget_note:
            result = budget_note + result
        success = outcome.ok
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
                "signal_kind": outcome.signal_kind,
                "exit_code": outcome.exit_code,
            },
        )
        # Reflection counts only real failures — user rejection/cancellation
        # is neutral and must not feed the consecutive-failure counter. The
        # outcome text goes along with the verdict: the hint quotes it back
        # verbatim, and outcome.text is the raw output as classified, before
        # any fallback hint was appended to `result`.
        if not outcome.cancelled:
            self.memory_manager.record_tool_execution(
                tool_call.name,
                success,
                evidence=outcome.text,
                signal_kind=outcome.signal_kind,
                exit_code=outcome.exit_code,
            )
            self.memory_manager.session.stagnation.record(
                is_readonly_tool_call(tool_call.name, tool_call.arguments),
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                ok=success,
                result=outcome.text,
            )
            if success:
                self._turn_tool_successes += 1
            else:
                self._turn_tool_failures += 1

        # Add fallback hints for common failures
        if not success and not outcome.cancelled:
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
        self._compression_event = (old_count, len(self.messages))
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

    def _withhold_answer(
        self,
        content: str,
        reasoning: str,
        rejection: str,
        messages_with_system: list[dict],
    ) -> None:
        """Put a gate-rejected answer and its rejection into history.

        The answer is kept rather than dropped: the model has to see what it
        claimed in order to retract the right part of it, and a rejection
        quoting output from an attempt the history never mentions reads as
        noise. Neither message becomes turn output — the turn is being sent
        back, not finished.
        """
        if content:
            withheld: dict = {"role": "assistant", "content": content}
            if reasoning:
                withheld["reasoning_content"] = reasoning
            self.messages.append(withheld)
            messages_with_system.append(withheld)
        # A user message, matching how the reflection/stagnation hints are
        # delivered: the provider converters keep only one system message, so
        # a second one would replace the real prompt.
        rejection_msg = {"role": "user", "content": rejection}
        self.messages.append(rejection_msg)
        messages_with_system.append(rejection_msg)

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
