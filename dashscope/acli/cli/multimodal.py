# -*- coding: utf-8 -*-
"""Multimodal content handling: @-references, images, audio."""

# pylint: disable=too-many-return-statements,too-many-statements

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

from dashscope.acli.cli.constants import (
    _AT_AUDIO_MAX_BYTES,
    _AT_FILE_MAX_CHARS,
    _AT_FILE_PATTERN,
    _AUDIO_MIME,
    _IMAGE_MIME,
    _MEDIA_SENTINEL_RE,
)


def _expand_at_references(text: str) -> tuple[str, list[str], list[dict]]:
    """Replace @<path> tokens with file contents.

    Returns ``(expanded_text, image_data_urls, audio_clips)``. Text files
    become fenced code blocks; image files become sentinels
    ``<<__ACLI_IMG_N__>>``; audio files become sentinels
    ``<<__ACLI_AUDIO_N__>>``. The caller splices the returned data into
    multimodal content blocks.
    """
    images: list[str] = []
    audio_clips: list[dict] = []

    _AT_DIR_MAX_CHARS = 50000
    _BINARY_EXTS = {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".ico",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".otf",
    }

    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".md": "markdown",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
    }

    def _expand_directory(dirpath: Path, raw: str) -> str:
        import subprocess as _sp

        try:
            result = _sp.run(
                [
                    "git",
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ],
                capture_output=True,
                text=True,
                cwd=str(dirpath),
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                files = [
                    dirpath / f
                    for f in result.stdout.strip().splitlines()
                    if f
                ]
            else:
                raise FileNotFoundError
        except (FileNotFoundError, _sp.TimeoutExpired):
            files = []
            for root, dirs, fnames in os.walk(str(dirpath)):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fn in fnames:
                    files.append(Path(root) / fn)

        total = 0
        blocks: list[str] = []
        file_count = 0
        for fp in sorted(files):
            if fp.suffix.lower() in _BINARY_EXTS:
                continue
            if not fp.is_file():
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if total + len(content) > _AT_DIR_MAX_CHARS:
                content = content[: _AT_DIR_MAX_CHARS - total]
                rel = (
                    fp.relative_to(dirpath)
                    if fp.is_relative_to(dirpath)
                    else fp
                )
                lang = lang_map.get(fp.suffix.lower(), "")
                blocks.append(
                    f"\n--- @{raw}{rel} ---\n```{lang}\n"
                    f"{content}\n... (truncated)\n```\n",
                )
                file_count += 1
                break
            rel = fp.relative_to(dirpath) if fp.is_relative_to(dirpath) else fp
            lang = lang_map.get(fp.suffix.lower(), "")
            blocks.append(
                f"\n--- @{raw}{rel} ---\n```{lang}\n{content}\n```\n",
            )
            total += len(content)
            file_count += 1

        if not blocks:
            return f"[@{raw} directory is empty or has no text files]"
        header = f"\n--- @{raw} ({file_count} files, {total} chars) ---\n"
        return header + "".join(blocks)

    def _replace(match: re.Match) -> str:
        raw = match.group(1)
        p = Path(raw).expanduser()
        if p.is_dir():
            return _expand_directory(p, raw)
        if not p.is_file():
            return match.group(0)  # leave @raw untouched

        suffix = p.suffix.lower()
        mime = _IMAGE_MIME.get(suffix)
        if mime:
            from dashscope.acli.utils.images import image_to_data_url

            try:
                url = image_to_data_url(p)
            except ValueError as e:
                return f"[@{raw} {e}]"
            except OSError as e:
                return f"[@{raw} read failed: {e}]"
            images.append(url)
            return f"<<__ACLI_IMG_{len(images) - 1}__>>"

        audio_info = _AUDIO_MIME.get(suffix)
        if audio_info:
            mime, fmt = audio_info
            try:
                data = p.read_bytes()
            except OSError as e:
                return f"[@{raw} read failed: {e}]"
            if len(data) > _AT_AUDIO_MAX_BYTES:
                return (
                    f"[@{raw} audio too large ({len(data)//1024} KB > "
                    f"{_AT_AUDIO_MAX_BYTES//1024} KB); ignored]"
                )
            audio_clips.append(
                {
                    "mime": mime,
                    "format": fmt,
                    "data": base64.b64encode(data).decode(),
                },
            )
            return f"<<__ACLI_AUDIO_{len(audio_clips) - 1}__>>"

        if suffix in _BINARY_EXTS:
            return f"[@{raw} binary file, skipped]"

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"[@{raw} read failed: {e}]"
        truncated = ""
        if len(content) > _AT_FILE_MAX_CHARS:
            content = content[:_AT_FILE_MAX_CHARS]
            truncated = (
                f"\n... (truncated; original > {_AT_FILE_MAX_CHARS} chars)"
            )
        lang = lang_map.get(suffix, "")
        return f"\n--- @{raw} ---\n```{lang}\n{content}{truncated}\n```\n"

    expanded = _AT_FILE_PATTERN.sub(_replace, text)
    return expanded, images, audio_clips


def _to_multimodal_content(
    text: str,
    images: list[str],
    audio_clips: list[dict] | None = None,
) -> str | list[dict]:
    """Convert (text-with-sentinels, image-data-urls, audio-clips) into either
    a plain string (no media) or an OpenAI-style content-block list."""
    audio_clips = audio_clips or []
    if not images and not audio_clips:
        return text
    blocks: list[dict] = []
    parts = _MEDIA_SENTINEL_RE.split(text)
    # split yields [text0, kind0, idx0, text1, kind1, idx1, ..., textN];
    # segments starting at index 0 with step 3 are text, then kind, then idx.
    for i, part in enumerate(parts):
        if i % 3 == 0:
            if part.strip():
                blocks.append({"type": "text", "text": part})
        elif i % 3 == 2:
            kind = parts[i - 1]
            try:
                idx = int(part)
            except ValueError:
                continue
            if kind == "IMG":
                try:
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": images[idx]},
                        },
                    )
                except IndexError:
                    continue
            elif kind == "AUDIO":
                try:
                    clip = audio_clips[idx]
                    blocks.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": clip["data"],
                                "format": clip["format"],
                            },
                        },
                    )
                except IndexError:
                    continue
    return blocks or text
