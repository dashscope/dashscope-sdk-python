# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Waiting on long-running jobs without reimplementing the polling.

Every job-shaped SDK resource (``fine_tunes``, ``rl.jobs``,
``images.jobs``, ``videos.jobs``, ``embeddings.jobs``,
``audio.asr.jobs``) waits the same way: exponential backoff, an enforced
timeout, and a raise on FAILED/CANCELED. The CLI's only addition is a
spinner, threaded in through ``wait``'s ``_sleep`` seam so the backoff
schedule stays the SDK's business.

Two rules follow from that:

* Progress is rendered only on a real terminal in the human format.
  Under ``--output json`` the spinner would corrupt the machine-readable
  stream, and under a pipe it would write ANSI escapes into a log file.
* A failed or timed-out job raises out of ``wait`` and is left alone
  here. Catching it to print something friendlier is exactly how the 1.x
  tree ended up exiting 0 after an error.

A third applies to the commands that submit: ``--timeout`` is checked
with :func:`validate_wait_timeout` before the create call, so a bad
value costs an error message rather than a job nobody wanted. The
standalone ``wait`` commands check it before touching the client, so a
bad ``--timeout`` stays a usage error (exit 2) even with no API key
configured.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Optional

from rich.status import Status

from dashscope.cli.errors import usage_error
from dashscope.cli.output import Renderer


def _show_progress(renderer: Renderer) -> bool:
    if not renderer.table_mode:
        return False
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def validate_wait_timeout(timeout: Optional[float]) -> None:
    """Reject a ``--timeout`` the waiter could never honour.

    Submit sites call this *before* creating a job, not only through
    :func:`wait_job`: by the time a wait starts, the service has already
    been asked for something the user is paying for. ``nan`` is the case
    that matters most -- every comparison against it is false, so the
    SDK's ``remaining <= 0`` deadline check never fires and the poll loop
    spins forever instead of timing out.
    """
    if timeout is None:
        return
    if not math.isfinite(timeout):
        raise usage_error(
            f"--timeout must be a finite number of seconds, got {timeout!r}",
        )
    if timeout <= 0:
        raise usage_error(
            f"--timeout must be greater than 0 seconds, got {timeout!r}",
        )


def wait_job(
    client: Any,
    job: Any,
    *,
    timeout: Optional[float] = None,
    renderer: Optional[Renderer] = None,
    label: str = "fine-tune job",
    waiter: Optional[Any] = None,
) -> Any:
    """Block until *job* reaches a terminal status, then return it.

    *job* may be the object ``create`` returned or just its id. Raises
    whatever ``wait`` raises -- the caller must not swallow it.

    *waiter* is the resource whose ``wait`` does the polling; it defaults
    to ``client.fine_tunes``. Every job-shaped resource in the SDK
    (``rl.jobs``, ``images.jobs``, ``videos.jobs``, ``audio.asr.jobs``)
    exposes the same ``wait(job, timeout=, _sleep=)`` seam, so they pass
    themselves here rather than each growing its own spinner.
    """
    validate_wait_timeout(timeout)
    if waiter is None:
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
