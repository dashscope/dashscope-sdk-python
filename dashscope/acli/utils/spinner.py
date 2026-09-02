# -*- coding: utf-8 -*-
"""Lightweight stderr spinners used by the CLI and executor."""

from __future__ import annotations

import asyncio
import sys
import threading
import time


class StderrSpinner:
    """Lightweight sync spinner — writes to stderr to avoid terminal
    conflicts."""

    FRAMES = ("·", "▪", "■", "█", "■", "▪", "·")
    COLORS = (
        "\033[36m",
        "\033[34m",
        "\033[35m",
        "\033[33m",
        "\033[35m",
        "\033[34m",
        "\033[36m",
    )
    CLEAR = "\r\033[2K"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def __init__(self, text: str, interval: float = 0.06):
        self._text = text
        self._interval = interval
        self._stop = False

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._thread.join(timeout=1)
        sys.stderr.write(self.CLEAR)
        sys.stderr.flush()

    def _run(self):
        i = 0
        while not self._stop:
            frame = self.FRAMES[i % len(self.FRAMES)]
            color = self.COLORS[(i // 3) % len(self.COLORS)]
            sys.stderr.write(
                f"\r{color}{frame}{self.RESET} "
                f"{self.DIM}{self._text}{self.RESET}",
            )
            sys.stderr.flush()
            i += 1
            time.sleep(self._interval)


class AsyncSpinner:
    """Lightweight async spinner — writes to stderr, safe outside
    patch_stdout()."""

    FRAMES = ("·", "▪", "■", "█", "■", "▪", "·")
    COLORS = (
        "\033[36m",
        "\033[34m",
        "\033[35m",
        "\033[33m",
        "\033[35m",
        "\033[34m",
        "\033[36m",
    )
    CLEAR = "\r\033[2K"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def __init__(self, text: str, interval: float = 0.06):
        self._text = text
        self._interval = interval
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        sys.stderr.write(self.CLEAR)
        sys.stderr.flush()

    async def _run(self):
        i = 0
        while True:
            frame = self.FRAMES[i % len(self.FRAMES)]
            color = self.COLORS[(i // 3) % len(self.COLORS)]
            sys.stderr.write(
                f"\r{color}{frame}{self.RESET} "
                f"{self.DIM}{self._text}{self.RESET}",
            )
            sys.stderr.flush()
            i += 1
            await asyncio.sleep(self._interval)
