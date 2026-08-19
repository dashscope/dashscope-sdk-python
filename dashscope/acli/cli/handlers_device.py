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
            f"available ({backend})"
            if ok
            else (
                "unavailable (install: pip install acli[camera] "
                "or brew install imagesnap)"
            )
        )
        console.print(f"[dim]Camera status: {status}[/dim]")
        console.print(
            "[dim]Usage: /camera capture [file] | "
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
        console.print(f"[dim]📹 Recording ({duration}s)...[/dim]")
        result = record(filename, duration)
    else:
        console.print(
            "[dim]Usage: /camera capture [file] | "
            "/camera record [duration] [file][/dim]",
        )
        return
    if result.startswith(("错误", "Error")):
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
            console.print("[dim]Recording cancelled[/dim]")
        else:
            console.print("[dim]No recording in progress[/dim]")
        return True

    if sub == "model":
        if len(parts) < 3:
            console.print(f"[dim]Current ASR model: {config.asr_model}[/dim]")
            console.print(f"[dim]Available: {', '.join(ASR_MODELS)}[/dim]")
            return True
        name = parts[2]
        if name not in ASR_MODELS:
            console.print(f"[red]Unknown ASR model: {name}[/red]")
            console.print(f"[dim]Available: {', '.join(ASR_MODELS)}[/dim]")
            return True
        config.asr_model = name
        config.save_workspace()
        config.save_global()
        console.print(f"[green]✓ ASR model switched to: {name}[/green]")
        return True

    if sub == "silence":
        if len(parts) < 3:
            console.print(
                f"[dim]Current silence threshold: "
                f"{config.voice_silence_duration}s[/dim]",
            )
            console.print("[dim]Usage: /voice silence <seconds>[/dim]")
            return True
        try:
            val = float(parts[2])
            if val <= 0:
                raise ValueError
            config.voice_silence_duration = val
            config.save_workspace()
            config.save_global()
            console.print(
                f"[green]✓ Silence threshold set to: {val}s[/green]",
            )
        except ValueError:
            console.print("[red]Please enter a number greater than 0[/red]")
        return True

    if sub == "max":
        if len(parts) < 3:
            console.print(
                f"[dim]Current max recording duration: "
                f"{config.voice_max_seconds}s[/dim]",
            )
            console.print("[dim]Usage: /voice max <seconds>[/dim]")
            return True
        try:
            val = int(parts[2])
            if val <= 0:
                raise ValueError
            config.voice_max_seconds = val
            config.save_workspace()
            config.save_global()
            console.print(
                f"[green]✓ Max recording duration set to: {val}s[/green]",
            )
        except ValueError:
            console.print("[red]Please enter an integer greater than 0[/red]")
        return True

    if sub == "threshold":
        if len(parts) < 3:
            console.print(
                f"[dim]Current silence RMS threshold: "
                f"{config.voice_silence_threshold}[/dim]",
            )
            console.print("[dim]Usage: /voice threshold <rms>[/dim]")
            return True
        try:
            val = int(parts[2])
            if val < 0:
                raise ValueError
            config.voice_silence_threshold = val
            config.save_workspace()
            config.save_global()
            console.print(
                f"[green]✓ Silence RMS threshold set to: {val}[/green]",
            )
        except ValueError:
            console.print("[red]Please enter a non-negative integer[/red]")
        return True

    if sub == "status":
        pass

    # Default: show status/help
    console.print("[bold]Voice Input[/bold]")
    console.print(f"  ASR model: [cyan]{config.asr_model}[/cyan]")
    console.print(
        f"  Silence stop: [cyan]" f"{config.voice_silence_duration}s[/cyan]",
    )
    console.print(
        f"  Max duration: [cyan]{config.voice_max_seconds}s[/cyan]",
    )
    console.print(
        f"  RMS threshold: [cyan]" f"{config.voice_silence_threshold}[/cyan]",
    )
    console.print("\n[dim]Usage:[/dim]")
    console.print("  /voice on              — start recording")
    console.print("  /voice off             — cancel recording")
    console.print("  /voice model <name>    — switch ASR model")
    console.print("  /voice silence <sec>   — set silence-stop seconds")
    console.print("  /voice max <sec>       — set max recording seconds")
    console.print("  /voice threshold <rms> — silence detection threshold")
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
        console.print("[green]✓ Voice output enabled[/green]")
        return

    if sub == "off":
        config.tts_enabled = False
        config.save_workspace()
        config.save_global()
        console.print("[dim]Voice output disabled[/dim]")
        return

    if sub == "status":
        console.print("[bold]TTS Voice Output[/bold]")
        if config.tts_enabled:
            console.print("  Status: [green]on[/green]")
        else:
            console.print("  Status: [dim]off[/dim]")
            console.print(
                "  [yellow]Hint: auto-speak is off by default; "
                "run /tts on to enable it[/yellow]",
            )
        console.print(f"  Model: [cyan]{config.tts_model}[/cyan]")
        voice_name = VOICE_DISPLAY.get(config.tts_voice, config.tts_voice)
        console.print(
            f"  Voice: [cyan]{config.tts_voice}[/cyan] ({voice_name})",
        )
        console.print(f"  Speed: [cyan]{config.tts_speed:.1f}[/cyan]")
        ok, err = is_available()
        if not ok:
            console.print(f"\n[yellow]Hint: {err}[/yellow]")
        else:
            console.print(
                f"\n[dim]Available models: " f"{', '.join(TTS_MODELS)}[/dim]",
            )
            voices = TTS_VOICES.get(config.tts_model, [])
            if voices:
                voice_list = ", ".join(
                    f"{v}({VOICE_DISPLAY.get(v, v)})" for v in voices[:5]
                )
                console.print(f"[dim]Available voices: {voice_list}...[/dim]")
        return

    if sub == "model":
        if len(parts) < 3:
            console.print(f"[dim]Current model: {config.tts_model}[/dim]")
            console.print(f"[dim]Available: {', '.join(TTS_MODELS)}[/dim]")
            return
        model_name = parts[2]
        if model_name not in TTS_MODELS:
            console.print(f"[red]Unknown model: {model_name}[/red]")
            console.print(f"[dim]Available: {', '.join(TTS_MODELS)}[/dim]")
            return
        config.tts_model = model_name
        config.tts_voice = DEFAULT_VOICE.get(model_name, config.tts_voice)
        config.save_workspace()
        config.save_global()
        console.print(
            f"[green]✓ TTS model switched to: {model_name}[/green]",
        )
        return

    if sub == "voice":
        if len(parts) < 3:
            # Interactive voice selection
            voices = TTS_VOICES.get(config.tts_model, [])
            if not voices:
                console.print(f"[dim]Current voice: {config.tts_voice}[/dim]")
                console.print("[dim]No voices available for this model[/dim]")
                return
            console.print("[bold]Select TTS Voice[/bold]")
            console.print(
                f"[dim]Current: {config.tts_voice} "
                f"({VOICE_DISPLAY.get(config.tts_voice, config.tts_voice)}"
                f")[/dim]\n",
            )
            for i, v in enumerate(voices, 1):
                display_name = VOICE_DISPLAY.get(v, v)
                marker = (
                    " [green]← current[/green]"
                    if v == config.tts_voice
                    else ""
                )
                console.print(
                    f"  [cyan][{i}][/cyan] {v} — {display_name}{marker}",
                )
            console.print("\n[dim]Enter a number to select, q to cancel[/dim]")
            try:
                choice = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("[dim]Cancelled[/dim]")
                return
            if choice.lower() == "q" or not choice:
                console.print("[dim]Cancelled[/dim]")
                return
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(voices):
                    config.tts_voice = voices[idx]
                    config.save_workspace()
                    config.save_global()
                    console.print(
                        f"[green]✓ TTS voice switched to: {voices[idx]} "
                        f"({VOICE_DISPLAY.get(voices[idx], voices[idx])}"
                        f")[/green]",
                    )
                else:
                    console.print("[red]Invalid selection[/red]")
            except ValueError:
                console.print("[red]Please enter a number[/red]")
            return

        voice_name = parts[2]
        voices = TTS_VOICES.get(config.tts_model, [])
        if voice_name not in voices:
            console.print(f"[red]Unknown voice: {voice_name}[/red]")
            if voices:
                voice_list = ", ".join(
                    f"{v}({VOICE_DISPLAY.get(v, v)})" for v in voices[:5]
                )
                console.print(f"[dim]Available voices: {voice_list}...[/dim]")
            return
        config.tts_voice = voice_name
        config.save_workspace()
        config.save_global()
        console.print(
            f"[green]✓ TTS voice switched to: {voice_name} "
            f"({VOICE_DISPLAY.get(voice_name, voice_name)})[/green]",
        )
        return

    if sub == "speed":
        if len(parts) < 3:
            console.print(f"[dim]Current speed: {config.tts_speed:.1f}[/dim]")
            console.print("[dim]Valid range: 0.5-2.0[/dim]")
            return
        try:
            speed = float(parts[2])
            if not 0.5 <= speed <= 2.0:
                console.print("[red]Speed must be between 0.5 and 2.0[/red]")
                return
            config.tts_speed = speed
            config.save_workspace()
            config.save_global()
            console.print(f"[green]✓ TTS speed set to: {speed:.1f}[/green]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")
        return

    if sub == "say":
        if len(parts) < 3:
            console.print("[dim]Usage: /tts say <text>[/dim]")
            return
        text = parts[2]
        ok, err = is_available()
        if not ok:
            console.print(f"[red]{err}[/red]")
            return
        console.print(f"[dim]Speaking: {text}[/dim]")
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
            console.print("[dim]No previous reply to speak[/dim]")
            return
        ok, err = is_available()
        if not ok:
            console.print(f"[red]{err}[/red]")
            return
        console.print("[dim]Speaking last reply...[/dim]")
        speak_text(
            config.tongyi_api_key,
            agent.last_output,
            config.tts_model,
            config.tts_voice,
            config.tts_speed,
        )
        return

    console.print(
        "[dim]Usage:\n"
        "  /tts on/off           — enable/disable auto-speak\n"
        "  /tts status           — show current config\n"
        "  /tts model <name>     — switch model\n"
        "  /tts voice <name>     — switch voice\n"
        "  /tts speed <0.5-2.0>  — set speech rate\n"
        "  /tts say <text>       — speak given text\n"
        "  /tts last             — speak last reply[/dim]",
    )
