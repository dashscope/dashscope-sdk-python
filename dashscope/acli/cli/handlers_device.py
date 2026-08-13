# -*- coding: utf-8 -*-
"""Device and multimedia command handlers (camera, voice, tts)."""
# pylint: disable=too-many-return-statements,too-many-branches
# pylint: disable=too-many-statements

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from dashscope.acli.cli.constants import ASR_MODELS
from dashscope.acli.config import Config

if TYPE_CHECKING:
    from dashscope.acli.agent import Agent

console = Console()


def _handle_camera_command(cmd: str) -> None:
    """/camera capture [file] | /camera record [duration] [file]"""
    from dashscope.acli.ui.camera import capture, is_available, record

    parts = cmd.split()
    if len(parts) < 2:
        ok, backend = is_available()
        status = (
            f"可用 ({backend})"
            if ok
            else "不可用 (安装: pip install acli[camera] 或 brew install imagesnap)"
        )
        console.print(f"[dim]摄像头状态: {status}[/dim]")
        console.print(
            "[dim]用法: /camera capture [file] | "
            "/camera record [duration] [file][/dim]",
        )
        return
    if parts[1] == "capture":
        filename = parts[2] if len(parts) >= 3 else "camera_capture.jpg"
        result = capture(filename)
    elif parts[1] == "record":
        duration = 5.0
        filename = "camera_record.mp4"
        args = parts[2:]
        if args:
            try:
                duration = float(args[0])
                filename = args[1] if len(args) >= 2 else filename
            except ValueError:
                filename = args[0]
        console.print(f"[dim]📹 录制中 ({duration}s)...[/dim]")
        result = record(filename, duration)
    else:
        console.print(
            "[dim]用法: /camera capture [file] | "
            "/camera record [duration] [file][/dim]",
        )
        return
    if result.startswith("错误"):
        console.print(f"[red]{result}[/red]")
    else:
        console.print(f"[green]{result}[/green]")


def _handle_voice_command(cmd: str, config: Config) -> bool | str:
    """Unified /voice command.

    /voice                — show status
    /voice on             — start voice input (replaces /v)
    /voice off            — cancel ongoing voice input (TUI only)
    /voice model <name>   — switch ASR model
    /voice silence <sec>  — seconds of silence before auto-stop
    /voice max <sec>      — maximum recording duration
    /voice threshold <rms>— RMS threshold for silence detection

    Returns "voice" to request voice input, True when handled normally.
    """
    parts = cmd.strip().split(maxsplit=2)
    sub = parts[1] if len(parts) > 1 else "status"

    if sub in ("on", "start", "record"):
        return "voice"

    if sub == "off":
        from dashscope.acli.ui.voice import try_cancel_voice_input

        if try_cancel_voice_input():
            console.print("[dim]已取消录音[/dim]")
        else:
            console.print("[dim]没有正在进行的录音[/dim]")
        return True

    if sub == "model":
        if len(parts) < 3:
            console.print(f"[dim]当前 ASR 模型: {config.asr_model}[/dim]")
            console.print(f"[dim]可选: {', '.join(ASR_MODELS)}[/dim]")
            return True
        name = parts[2]
        if name not in ASR_MODELS:
            console.print(f"[red]未知 ASR 模型: {name}[/red]")
            console.print(f"[dim]可选: {', '.join(ASR_MODELS)}[/dim]")
            return True
        config.asr_model = name
        config.save_workspace()
        config.save_global()
        console.print(f"[green]✓ ASR 模型已切换为: {name}[/green]")
        return True

    if sub == "silence":
        if len(parts) < 3:
            console.print(
                f"[dim]当前停顿阈值: {config.voice_silence_duration}s[/dim]",
            )
            console.print("[dim]用法: /voice silence <seconds>[/dim]")
            return True
        try:
            val = float(parts[2])
            if val <= 0:
                raise ValueError
            config.voice_silence_duration = val
            config.save_workspace()
            config.save_global()
            console.print(f"[green]✓ 停顿阈值已设置为: {val}s[/green]")
        except ValueError:
            console.print("[red]请输入大于 0 的数字[/red]")
        return True

    if sub == "max":
        if len(parts) < 3:
            console.print(f"[dim]当前最大录音时长: {config.voice_max_seconds}s[/dim]")
            console.print("[dim]用法: /voice max <seconds>[/dim]")
            return True
        try:
            val = int(parts[2])
            if val <= 0:
                raise ValueError
            config.voice_max_seconds = val
            config.save_workspace()
            config.save_global()
            console.print(f"[green]✓ 最大录音时长已设置为: {val}s[/green]")
        except ValueError:
            console.print("[red]请输入大于 0 的整数[/red]")
        return True

    if sub == "threshold":
        if len(parts) < 3:
            console.print(
                f"[dim]当前静音 RMS 阈值: {config.voice_silence_threshold}[/dim]",
            )
            console.print("[dim]用法: /voice threshold <rms>[/dim]")
            return True
        try:
            val = int(parts[2])
            if val < 0:
                raise ValueError
            config.voice_silence_threshold = val
            config.save_workspace()
            config.save_global()
            console.print(f"[green]✓ 静音 RMS 阈值已设置为: {val}[/green]")
        except ValueError:
            console.print("[red]请输入非负整数[/red]")
        return True

    if sub == "status":
        pass

    # Default: show status/help
    console.print("[bold]语音输入[/bold]")
    console.print(f"  ASR 模型: [cyan]{config.asr_model}[/cyan]")
    console.print(f"  停顿结束: [cyan]{config.voice_silence_duration}s[/cyan]")
    console.print(f"  最大时长: [cyan]{config.voice_max_seconds}s[/cyan]")
    console.print(f"  静音阈值: [cyan]{config.voice_silence_threshold}[/cyan]")
    console.print("\n[dim]用法:[/dim]")
    console.print("  /voice on              — 开始录音")
    console.print("  /voice off             — 取消录音")
    console.print("  /voice model <name>    — 切换 ASR 模型")
    console.print("  /voice silence <sec>   — 设置停顿结束秒数")
    console.print("  /voice max <sec>       — 设置最大录音秒数")
    console.print("  /voice threshold <rms> — 设置静音检测阈值")
    return True


def _handle_tts_command(
    cmd: str,
    config: Config,
    agent: "Agent" = None,
) -> None:
    """/tts — Text-to-Speech voice output control.

    /tts on/off           — enable/disable auto TTS on agent replies
    /tts status           — show current TTS config
    /tts model <name>     — switch TTS model
    /tts voice <name>     — switch TTS voice
    /tts speed <rate>     — set speech rate (0.5-2.0)
    /tts say <text>       — speak given text
    /tts last             — speak last agent reply
    """
    from dashscope.acli.ui.tts import (
        DEFAULT_VOICE,
        TTS_MODELS,
        TTS_VOICES,
        VOICE_DISPLAY,
        is_available,
        speak_text,
    )

    parts = cmd.split(maxsplit=2)
    sub = parts[1] if len(parts) >= 2 else "status"

    if sub == "on":
        ok, err = is_available()
        if not ok:
            console.print(f"[red]{err}[/red]")
            return
        config.tts_enabled = True
        config.save_workspace()
        config.save_global()
        console.print("[green]✓ 语音输出已开启[/green]")
        return

    if sub == "off":
        config.tts_enabled = False
        config.save_workspace()
        config.save_global()
        console.print("[dim]语音输出已关闭[/dim]")
        return

    if sub == "status":
        console.print("[bold]TTS 语音输出[/bold]")
        if config.tts_enabled:
            console.print("  状态: [green]开启[/green]")
        else:
            console.print("  状态: [dim]关闭[/dim]")
            console.print(
                "  [yellow]提示: 自动朗读默认关闭，需执行 /tts on 显式开启[/yellow]",
            )
        console.print(f"  模型: [cyan]{config.tts_model}[/cyan]")
        voice_name = VOICE_DISPLAY.get(config.tts_voice, config.tts_voice)
        console.print(f"  语音: [cyan]{config.tts_voice}[/cyan] ({voice_name})")
        console.print(f"  语速: [cyan]{config.tts_speed:.1f}[/cyan]")
        ok, err = is_available()
        if not ok:
            console.print(f"\n[yellow]提示: {err}[/yellow]")
        else:
            console.print(f"\n[dim]可用模型: {', '.join(TTS_MODELS)}[/dim]")
            voices = TTS_VOICES.get(config.tts_model, [])
            if voices:
                voice_list = ", ".join(
                    f"{v}({VOICE_DISPLAY.get(v, v)})" for v in voices[:5]
                )
                console.print(f"[dim]可用语音: {voice_list}...[/dim]")
        return

    if sub == "model":
        if len(parts) < 3:
            console.print(f"[dim]当前模型: {config.tts_model}[/dim]")
            console.print(f"[dim]可选模型: {', '.join(TTS_MODELS)}[/dim]")
            return
        model_name = parts[2]
        if model_name not in TTS_MODELS:
            console.print(f"[red]未知模型: {model_name}[/red]")
            console.print(f"[dim]可选模型: {', '.join(TTS_MODELS)}[/dim]")
            return
        config.tts_model = model_name
        config.tts_voice = DEFAULT_VOICE.get(model_name, config.tts_voice)
        config.save_workspace()
        config.save_global()
        console.print(f"[green]✓ TTS 模型已切换为: {model_name}[/green]")
        return

    if sub == "voice":
        if len(parts) < 3:
            # Interactive voice selection
            voices = TTS_VOICES.get(config.tts_model, [])
            if not voices:
                console.print(f"[dim]当前语音: {config.tts_voice}[/dim]")
                console.print("[dim]该模型无可选语音[/dim]")
                return
            console.print("[bold]选择 TTS 语音[/bold]")
            console.print(
                f"[dim]当前: {config.tts_voice} "
                f"({VOICE_DISPLAY.get(config.tts_voice, config.tts_voice)}"
                f")[/dim]\n",
            )
            for i, v in enumerate(voices, 1):
                display_name = VOICE_DISPLAY.get(v, v)
                marker = (
                    " [green]← 当前[/green]" if v == config.tts_voice else ""
                )
                console.print(
                    f"  [cyan][{i}][/cyan] {v} — {display_name}{marker}",
                )
            console.print("\n[dim]输入序号选择，q 取消[/dim]")
            try:
                choice = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("[dim]已取消[/dim]")
                return
            if choice.lower() == "q" or not choice:
                console.print("[dim]已取消[/dim]")
                return
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(voices):
                    config.tts_voice = voices[idx]
                    config.save_workspace()
                    config.save_global()
                    console.print(
                        f"[green]✓ TTS 语音已切换为: {voices[idx]} "
                        f"({VOICE_DISPLAY.get(voices[idx], voices[idx])}"
                        f")[/green]",
                    )
                else:
                    console.print("[red]无效的选择[/red]")
            except ValueError:
                console.print("[red]请输入数字[/red]")
            return

        voice_name = parts[2]
        voices = TTS_VOICES.get(config.tts_model, [])
        if voice_name not in voices:
            console.print(f"[red]未知语音: {voice_name}[/red]")
            if voices:
                voice_list = ", ".join(
                    f"{v}({VOICE_DISPLAY.get(v, v)})" for v in voices[:5]
                )
                console.print(f"[dim]可选语音: {voice_list}...[/dim]")
            return
        config.tts_voice = voice_name
        config.save_workspace()
        config.save_global()
        console.print(
            f"[green]✓ TTS 语音已切换为: {voice_name} "
            f"({VOICE_DISPLAY.get(voice_name, voice_name)})[/green]",
        )
        return

    if sub == "speed":
        if len(parts) < 3:
            console.print(f"[dim]当前语速: {config.tts_speed:.1f}[/dim]")
            console.print("[dim]可选范围: 0.5-2.0[/dim]")
            return
        try:
            speed = float(parts[2])
            if not 0.5 <= speed <= 2.0:
                console.print("[red]语速必须在 0.5-2.0 之间[/red]")
                return
            config.tts_speed = speed
            config.save_workspace()
            config.save_global()
            console.print(f"[green]✓ TTS 语速已设置为: {speed:.1f}[/green]")
        except ValueError:
            console.print("[red]请输入有效的数字[/red]")
        return

    if sub == "say":
        if len(parts) < 3:
            console.print("[dim]用法: /tts say <text>[/dim]")
            return
        text = parts[2]
        ok, err = is_available()
        if not ok:
            console.print(f"[red]{err}[/red]")
            return
        console.print(f"[dim]朗读: {text}[/dim]")
        speak_text(
            config.tongyi_api_key,
            text,
            config.tts_model,
            config.tts_voice,
            config.tts_speed,
        )
        return

    if sub == "last":
        if not agent or not agent.last_output:
            console.print("[dim]没有上一次回复可以朗读[/dim]")
            return
        ok, err = is_available()
        if not ok:
            console.print(f"[red]{err}[/red]")
            return
        console.print("[dim]朗读上一次回复...[/dim]")
        speak_text(
            config.tongyi_api_key,
            agent.last_output,
            config.tts_model,
            config.tts_voice,
            config.tts_speed,
        )
        return

    console.print(
        "[dim]用法:\n"
        "  /tts on/off           — 启用/禁用自动朗读\n"
        "  /tts status           — 显示当前配置\n"
        "  /tts model <name>     — 切换模型\n"
        "  /tts voice <name>     — 切换语音\n"
        "  /tts speed <0.5-2.0>  — 设置语速\n"
        "  /tts say <text>       — 朗读指定文本\n"
        "  /tts last             — 朗读上一次回复[/dim]",
    )
