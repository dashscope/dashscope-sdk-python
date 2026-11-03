# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Waiting on long-running jobs without reimplementing the polling.

``client.fine_tunes.wait`` already does exponential backoff, enforces the
timeout, and raises on FAILED/CANCELED. The CLI's only addition is a
spinner, threaded in through ``wait``'s ``_sleep`` seam so the backoff
schedule stays the SDK's business.

Two rules follow from that:

* Progress is rendered only on a real terminal in the human format.
  Under ``--output json`` the spinner would corrupt the machine-readable
  stream, and under a pipe it would write ANSI escapes into a log file.
* A failed or timed-out job raises out of ``wait`` and is left alone
  here. Catching it to print something friendlier is exactly how the 1.x
  tree ended up exiting 0 after an error.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Optional

from rich.status import Status

from dashscope.cli2.output import Renderer


def _show_progress(renderer: Renderer) -> bool:
    if not renderer.table_mode:
        return False
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def wait_job(
    client: Any,
    job: Any,
    *,
    timeout: Optional[float] = None,
    renderer: Optional[Renderer] = None,
    label: str = "fine-tune job",
) -> Any:
    """Block until *job* reaches a terminal status, then return it.

    *job* may be the object ``create`` returned or just its id. Raises
    whatever ``wait`` raises -- the caller must not swallow it.
    """
    waiter = client.fine_tunes
    kwargs: dict = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    if renderer is None or not _show_progress(renderer):
        return waiter.wait(job, **kwargs)

    job_id = getattr(job, "job_id", None) or job
    text = f"waiting for {label} {job_id}"
    started = time.monotonic()
    with Status(
        text,
        console=renderer.err_console,
        spinner="dots",
    ) as status:
        return waiter.wait(
            job,
            _sleep=_status_sleep(status, text, started),
            **kwargs,
        )


def _status_sleep(
    status: Status,
    text: str,
    started: float,
) -> Callable[[float], Any]:
    """A ``time.sleep`` replacement that refreshes the spinner each poll.

    Rebuilt from *text* rather than from the current status, so repeated
    polls do not stack suffixes on each other.
    """

    def sleep(seconds: float) -> None:
        elapsed = int(time.monotonic() - started)
        status.update(f"{text} ({elapsed}s elapsed)")
        time.sleep(seconds)

    return sleep
