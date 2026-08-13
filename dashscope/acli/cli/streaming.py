"""Streaming output and conversation management functions."""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from dashscope.acli.agent import Agent
from dashscope.acli.config import Config
from dashscope.acli.deliverable import collect_deliverables, surface_deliverables
from dashscope.acli.utils import AsyncSpinner, UserAbortedTurn, message_text_for_compress

console = Console()


def _render_tool_trail(line: str) -> None:
    """Render a `[tool_name] → result` trail line. For write_file (and any
    other diff-bearing tools), split out the unified diff section and
    render it with Syntax("diff") so additions/removals are color-coded.

    Trail format: `[write_file] → 已写入文件: path (N 字符)\\n--- diff ---\\n<diff>`.
    """
    if "[write_file]" in line and "--- diff ---" in line:
        head, _, diff_body = line.partition("--- diff ---")
        console.print(Text(head.rstrip(), style="dim cyan"))
        diff_body = diff_body.strip("\n")
        if diff_body:
            console.print(
                Syntax(diff_body, "diff", theme="ansi_dark", background_color="default")
            )
        return
    console.print(Text(line, style="dim cyan"))


async def _do_compress(agent: Agent) -> None:
    """Compress the current conversation into a short summary."""
    old_count = len(agent.messages)
    old_chars = sum(len(str(m.get("content", ""))) for m in agent.messages)
    console.print("[dim]正在压缩对话上下文...[/dim]")
    conversation_dump = ""
    for m in agent.messages:
        conversation_dump += f"[{m.get('role', '?')}]: {message_text_for_compress(m)}\n"
    compress_messages = [
        {
            "role": "system",
            "content": "把以下对话压缩为简要摘要，保留关键决策、代码变更、文件路径和用户偏好。直接输出摘要，不要解释。",
        },
        {"role": "user", "content": conversation_dump},
    ]
    try:
        resp = await agent.provider.chat(compress_messages, tools=[])
        summary = resp.content if hasattr(resp, "content") else str(resp)
        agent.messages = [
            {
                "role": "user",
                "content": (
                    "以下是之前对话的摘要（压缩生成，仅作背景参考；"
                    "其中提到的模型名、provider 等配置信息可能已过时，"
                    "当前配置以系统提示中的“当前模型”为准）：\n"
                    f"{summary}"
                ),
            },
            {
                "role": "assistant",
                "content": "好的，我已了解之前对话的上下文。请继续。",
            },
        ]
        agent.save_session()
        new_chars = sum(len(str(m.get("content", ""))) for m in agent.messages)
        console.print(
            f"[green]已压缩: {old_count} 条消息 → 2 条, {old_chars} 字符 → {new_chars} 字符[/green]"
        )
    except Exception as e:
        console.print(f"[red]压缩失败: {e}[/red]")


async def _do_summarize(
    agent: Agent, messages_to_summarize: list | None = None, silent: bool = False
) -> dict | None:
    """Summarize a task and record to experience tracker.

    Args:
        agent: The agent instance.
        messages_to_summarize: Messages to summarize. If None, use agent.messages.
        silent: If True, suppress console output (for auto-trigger).

    Returns:
        Parsed summary dict or None on failure.
    """
    import json as _json

    tracker = getattr(agent, "experience_tracker", None)
    if not tracker:
        if not silent:
            console.print("[yellow]经验追踪器未初始化[/yellow]")
        return None

    msgs = (
        messages_to_summarize if messages_to_summarize is not None else agent.messages
    )
    if len(msgs) < 2:
        if not silent:
            console.print("[yellow]对话内容不足，无法总结[/yellow]")
        return None

    if not silent:
        console.print("[dim]正在总结当前任务...[/dim]")

    conversation_dump = ""
    tools_used: list[str] = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        conversation_dump += f"[{role}]: {content}\n"
        if role == "assistant" and "tool_calls" in m:
            for tc in m["tool_calls"]:
                if "function" in tc:
                    name = tc["function"]["name"]
                    if name not in tools_used:
                        tools_used.append(name)

    summarize_messages = [
        {
            "role": "system",
            "content": (
                "你是一个任务总结助手。根据对话内容，用以下 JSON 格式输出总结（直接输出 JSON，不要解释）：\n"
                '{"task": "一句话描述完成的任务", "steps": ["关键步骤1", "关键步骤2"], '
                '"lesson": "从中学到的经验教训（如有）", "outcome": "success 或 partial"}'
            ),
        },
        {"role": "user", "content": conversation_dump},
    ]

    try:
        resp = await agent.provider.chat(summarize_messages, tools=[])
        raw = resp.content if hasattr(resp, "content") else str(resp)

        parsed = None
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(raw[start:end])
        except Exception:
            parsed = None

        task = (parsed or {}).get("task", "") or raw.split("\n", 1)[0][:100]
        lesson = (parsed or {}).get("lesson", "")
        steps = (parsed or {}).get("steps", [])
        outcome = (parsed or {}).get("outcome", "success")
        if outcome not in ("success", "partial", "failure"):
            outcome = "success"

        if lesson and steps:
            lesson = f"步骤: {'; '.join(steps)}\n教训: {lesson}"
        elif steps:
            lesson = f"步骤: {'; '.join(steps)}"

        tracker.record_experience(
            task_summary=task[:200],
            tools_used=tools_used,
            outcome=outcome,
            lesson=lesson,
        )

        if not silent:
            console.print(f"\n[bold green]✓ 任务总结已记录[/bold green]")
            console.print(f"  [bold]任务:[/bold] {task}")
            if steps:
                console.print(f"  [bold]步骤:[/bold]")
                for s in steps:
                    console.print(f"    - {s}")
            if lesson and not steps:
                console.print(f"  [bold]教训:[/bold] {lesson}")
            console.print()

        return parsed or {
            "task": task,
            "steps": steps,
            "lesson": lesson,
            "outcome": outcome,
        }
    except Exception as e:
        if not silent:
            console.print(f"[red]总结失败: {e}[/red]")
        return None


async def _stream_response(agent: Agent, config: Config, user_input):
    """Stream agent response with real-time output and tool call indicators."""
    buffer = ""
    first_chunk = True
    spinner_active = True
    spinner = AsyncSpinner("思考中...")
    await spinner.__aenter__()

    # Install SIGINT handler so Ctrl+C aborts the turn instead of killing the process.
    import signal

    loop = asyncio.get_running_loop()
    aborted = False

    def _sigint_handler():
        nonlocal aborted
        aborted = True

    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)
    except NotImplementedError:
        pass

    # Initialize streaming TTS feeder if enabled
    tts_feeder = None
    if config.tts_enabled:
        try:
            from dashscope.acli.ui.tts import StreamingTTSFeeder, is_available

            ok, _err = is_available()
            if ok:
                tts_feeder = StreamingTTSFeeder(
                    api_key=config.tongyi_api_key,
                    model=config.tts_model,
                    voice=config.tts_voice,
                    speech_rate=config.tts_speed,
                )
                tts_feeder.start()
        except Exception:
            tts_feeder = None

    try:
        async for chunk in agent.run_stream(user_input):
            if aborted:
                console.print("\n[yellow]已中断[/yellow]")
                break

            # Stop spinner on first chunk OR empty chunk (tool execution signal)
            if chunk == "" and spinner_active:
                await spinner.__aexit__(None, None, None)
                spinner_active = False
            if first_chunk:
                if spinner_active:
                    await spinner.__aexit__(None, None, None)
                    spinner_active = False
                first_chunk = False

            if chunk.startswith("\n[") and "] →" in chunk:
                if buffer:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    buffer = ""
                _render_tool_trail(chunk.strip())
            else:
                buffer += chunk
                sys.stdout.write(chunk)
                sys.stdout.flush()
                # Feed chunk to streaming TTS
                if tts_feeder:
                    tts_feeder.feed(chunk)
    except UserAbortedTurn:
        console.print("\n[dim]已中止本轮[/dim]")
    except Exception as e:
        console.print(f"\n[red]错误: {e}[/red]")
    finally:
        # Restore default SIGINT handler
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError):
            pass
        if spinner_active:
            await spinner.__aexit__(None, None, None)
        if buffer:
            sys.stdout.write("\n")
            sys.stdout.flush()
            agent.last_output = buffer
        # Stop the TTS feeder on ALL exit paths, otherwise its worker
        # thread leaks one per turn (empty-buffer turns never reached
        # finish()/cancel()).
        if tts_feeder:
            try:
                if buffer.strip():
                    # Flush remaining text to streaming TTS and wait
                    err = tts_feeder.finish()
                    if err:
                        console.print(f"[yellow]TTS: {err}[/yellow]")
                else:
                    tts_feeder.cancel()
            except Exception as e:
                console.print(f"[yellow]TTS 错误: {e}[/yellow]")
            finally:
                thread = getattr(tts_feeder, "_worker_thread", None)
                if thread is not None:
                    thread.join(timeout=5)

        # Surface any files produced by tool calls this turn.
        try:
            surface_deliverables(collect_deliverables(agent.messages))
        except Exception:
            pass
