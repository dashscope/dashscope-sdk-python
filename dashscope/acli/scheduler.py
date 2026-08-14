# -*- coding: utf-8 -*-
"""Cron-style task scheduler for periodic skill execution.

Supports three scheduling modes:
- interval: `every 5m` / `every 30s` / `every 2h`
- specific time: `at 14:30` (next occurrence) / `at 2026-06-12T09:00`
- cron expression: `cron "0 9 * * 1-5"`

Jobs persist across sessions in ~/.acli/cron_jobs.json.
"""
# pylint: disable=too-many-branches,too-many-return-statements
# pylint: disable=too-many-statements

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from rich.console import Console

from dashscope.acli.config import CONFIG_DIR

console = Console()

_PERSIST_PATH = CONFIG_DIR / "cron_jobs.json"

# ── Data models ──────────────────────────────────────────────


@dataclass
class SkillInvocation:
    name: str
    args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "args": self.args}

    @classmethod
    def from_dict(cls, d: dict) -> SkillInvocation:
        return cls(name=d["name"], args=d.get("args", []))


@dataclass
class ScheduleSpec:
    kind: str  # "interval" | "at" | "cron"
    raw: str
    interval_seconds: float | None = None
    at_datetime: str | None = None  # ISO format string for serialization
    cron_fields: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "raw": self.raw,
            "interval_seconds": self.interval_seconds,
            "at_datetime": self.at_datetime,
            "cron_fields": self.cron_fields,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScheduleSpec:
        return cls(
            kind=d["kind"],
            raw=d["raw"],
            interval_seconds=d.get("interval_seconds"),
            at_datetime=d.get("at_datetime"),
            cron_fields=d.get("cron_fields"),
        )


@dataclass
class CronJob:
    id: str
    schedule: ScheduleSpec
    skills: list[SkillInvocation]
    condition: str | None = None
    subagent: bool = True
    enabled: bool = True
    created_at: str = ""
    last_run: str | None = None
    last_result: str | None = None
    run_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "schedule": self.schedule.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
            "condition": self.condition,
            "subagent": self.subagent,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "last_result": self.last_result,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CronJob:
        return cls(
            id=d["id"],
            schedule=ScheduleSpec.from_dict(d["schedule"]),
            skills=[SkillInvocation.from_dict(s) for s in d["skills"]],
            condition=d.get("condition"),
            subagent=d.get("subagent", True),
            enabled=d.get("enabled", True),
            created_at=d.get("created_at", ""),
            last_run=d.get("last_run"),
            last_result=d.get("last_result"),
            run_count=d.get("run_count", 0),
        )


# ── Time parsing ─────────────────────────────────────────────

_INTERVAL_RE = re.compile(r"^(\d+)\s*([smh])$")


def parse_interval(s: str) -> float:
    m = _INTERVAL_RE.match(s.strip().lower())
    if not m:
        raise ValueError(f"cannot parse interval: {s} (formats: 30s, 5m, 2h)")
    val, unit = int(m.group(1)), m.group(2)
    return val * {"s": 1, "m": 60, "h": 3600}[unit]


def parse_at_time(s: str) -> datetime:
    s = s.strip()
    if "T" in s or (s.count("-") >= 2):
        return datetime.fromisoformat(s)
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"cannot parse time: {s} (formats: 14:30, 2026-06-12T09:00)",
        )
    h, m = int(parts[0]), int(parts[1])
    now = datetime.now()
    candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


# ── Cron expression parser ───────────────────────────────────


def _parse_cron_field(field_str: str, lo: int, hi: int) -> set[int]:
    values: set[int] = set()
    for part in field_str.split(","):
        part = part.strip()
        if part == "*":
            values.update(range(lo, hi + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            if step <= 0:
                raise ValueError(f"cron step must be > 0: {part}")
            values.update(range(lo, hi + 1, step))
        elif "-" in part:
            a, b = part.split("-", 1)
            values.update(range(int(a), int(b) + 1))
        else:
            values.add(int(part))
    return values


def _expand_dow(field_str: str) -> set[int]:
    """cron day-of-week (0=Sun..6=Sat, 7=Sun) → Python weekday
    (0=Mon..6=Sun)."""
    raw = _parse_cron_field(field_str, 0, 7)
    python_dows: set[int] = set()
    for d in raw:
        if d in (0, 7):
            python_dows.add(6)  # Sunday
        else:
            python_dows.add(d - 1)  # 1=Mon→0, 2=Tue→1, ...
    return python_dows


def next_cron_fire(fields: list[str], after: datetime) -> datetime | None:
    minutes = sorted(_parse_cron_field(fields[0], 0, 59))
    hours = sorted(_parse_cron_field(fields[1], 0, 23))
    doms = _parse_cron_field(fields[2], 1, 31)
    months = _parse_cron_field(fields[3], 1, 12)
    dows = _expand_dow(fields[4])

    earliest = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    day = earliest
    # Scan day-by-day (not minute-by-minute) so sparse crons don't block the
    # event loop. 9 years covers the longest Feb-29 gap (2096 → 2104).
    for _ in range(366 * 9):
        if day.month in months and day.day in doms and day.weekday() in dows:
            for hour in hours:
                for minute in minutes:
                    candidate = day.replace(hour=hour, minute=minute)
                    if candidate >= earliest:
                        return candidate
        day = (day + timedelta(days=1)).replace(hour=0, minute=0)
    return None


# ── Argument parsing ─────────────────────────────────────────


def parse_cron_add(
    args_str: str,
) -> tuple[ScheduleSpec, list[SkillInvocation], str | None, bool]:
    tokens = shlex.split(args_str, posix=os.name != "nt")

    schedule: ScheduleSpec | None = None
    skills: list[SkillInvocation] = []
    condition: str | None = None
    subagent: bool = True

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("every", "at", "cron"):
            if i + 1 >= len(tokens):
                raise ValueError(f"{tok} requires an argument")
            val = tokens[i + 1]
            if tok == "every":
                secs = parse_interval(val)
                schedule = ScheduleSpec(
                    kind="interval",
                    raw=f"every {val}",
                    interval_seconds=secs,
                )
            elif tok == "at":
                dt = parse_at_time(val)
                schedule = ScheduleSpec(
                    kind="at",
                    raw=f"at {val}",
                    at_datetime=dt.isoformat(),
                )
            elif tok == "cron":
                parts = val.strip().split()
                if len(parts) != 5:
                    raise ValueError(f"cron expression needs 5 fields: {val}")
                schedule = ScheduleSpec(
                    kind="cron",
                    raw=f"cron {val}",
                    cron_fields=parts,
                )
            i += 2
        elif tok == "condition":
            if i + 1 >= len(tokens):
                raise ValueError("condition requires an argument")
            condition = tokens[i + 1]
            i += 2
        elif tok == "no-subagent":
            subagent = False
            i += 1
        elif tok == "skill":
            i += 1
            if i >= len(tokens):
                raise ValueError("skill requires a skill name")
            name = tokens[i]
            i += 1
            skill_args: list[str] = []
            while i < len(tokens) and tokens[i] not in (
                "skill",
                "every",
                "at",
                "cron",
                "condition",
                "no-subagent",
            ):
                skill_args.append(tokens[i])
                i += 1
            skills.append(SkillInvocation(name, skill_args))
        else:
            raise ValueError(f"unknown argument: {tok}")

    if not schedule:
        raise ValueError("a schedule is required: every, at, or cron")
    if not skills:
        raise ValueError("at least one skill is required")

    return schedule, skills, condition, subagent


# ── Scheduler ────────────────────────────────────────────────


class Scheduler:
    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        self.jobs: dict[str, CronJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._pending_prompts: list[str] = []

    async def load_and_start(self) -> None:
        if _PERSIST_PATH.exists():
            try:
                data = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
                for jid, jd in data.items():
                    self.jobs[jid] = CronJob.from_dict(jd)
            except (json.JSONDecodeError, KeyError):
                pass

        for jid, job in self.jobs.items():
            if job.enabled:
                self._start_job_task(jid)

    async def shutdown(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    def add_job(
        self,
        schedule: ScheduleSpec,
        skills: list[SkillInvocation],
        condition: str | None,
        subagent: bool,
    ) -> CronJob:
        job = CronJob(
            id=secrets.token_hex(3),
            schedule=schedule,
            skills=skills,
            condition=condition,
            subagent=subagent,
            enabled=True,
            created_at=datetime.now().isoformat(),
        )
        self.jobs[job.id] = job
        self._persist()
        self._start_job_task(job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        if job_id not in self.jobs:
            return False
        task = self._tasks.pop(job_id, None)
        if task:
            task.cancel()
        del self.jobs[job_id]
        self._persist()
        return True

    def pause_job(self, job_id: str) -> bool:
        if job_id not in self.jobs:
            return False
        self.jobs[job_id].enabled = False
        task = self._tasks.pop(job_id, None)
        if task:
            task.cancel()
        self._persist()
        return True

    def resume_job(self, job_id: str) -> bool:
        if job_id not in self.jobs:
            return False
        self.jobs[job_id].enabled = True
        self._start_job_task(job_id)
        self._persist()
        return True

    def get_pending_prompts(self) -> list[str]:
        prompts = self._pending_prompts[:]
        self._pending_prompts.clear()
        return prompts

    def print_jobs(self) -> None:
        if not self.jobs:
            console.print("[dim]No scheduled jobs[/dim]")
            console.print(
                "[dim]Usage: /cron add every 5m skill "
                "weather Hangzhou[/dim]",
            )
            return

        console.print("[bold]📋 Scheduled jobs[/bold]")
        for job in self.jobs.values():
            status = (
                "[green]running[/green]"
                if job.enabled
                else "[yellow]paused[/yellow]"
            )
            skill_names = ", ".join(s.name for s in job.skills)
            console.print(f"  [{job.id}] {status}  {job.schedule.raw}")
            console.print(f"    skills: {skill_names}")
            if job.condition:
                console.print(f"    condition: {job.condition}")
            if job.last_run:
                console.print(f"    last run: {job.last_run}")
            if job.last_result:
                result_preview = job.last_result[:80].replace("\n", " ")
                console.print(f"    result: {result_preview}...")
            console.print(f"    runs: {job.run_count}")
            console.print()

    # ── Internal ──────────────────────────────────────────────

    def _start_job_task(self, job_id: str) -> None:
        if job_id in self._tasks:
            self._tasks[job_id].cancel()
        task = asyncio.create_task(self._job_loop(job_id))
        task.add_done_callback(lambda t: self._on_job_done(job_id, t))
        self._tasks[job_id] = task

    def _on_job_done(self, job_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(
                f"\n[CRON] job {job_id} exited unexpectedly: "
                f"{type(exc).__name__}: {exc}",
            )
        elif job_id in self.jobs and self.jobs[job_id].enabled:
            print(f"\n[CRON] job {job_id} ended (schedule may be one-shot)")

    async def _job_loop(self, job_id: str) -> None:
        job = self.jobs[job_id]
        try:
            while True:
                delay = self._seconds_until_next_fire(job)
                if delay is None:
                    job.enabled = False
                    self._persist()
                    break
                await asyncio.sleep(delay)
                if not job.enabled:
                    break
                try:
                    # Timeout: 5 minutes per execution to prevent hanging
                    await asyncio.wait_for(
                        self._execute_job(job),
                        timeout=300.0,
                    )
                except asyncio.TimeoutError:
                    job.last_run = datetime.now().isoformat()
                    job.last_result = "execution timed out (>300s)"
                    job.run_count += 1
                    self._persist()
                    print(
                        f"\n[CRON {job.id}] timeout: job ran over "
                        f"300s and was terminated",
                    )
                    print(
                        f"  schedule: {job.schedule.raw} "
                        f"· run #{job.run_count}",
                    )
                except Exception as e:
                    # Log error but continue the loop
                    job.last_run = datetime.now().isoformat()
                    job.last_result = (
                        f"execution error: {type(e).__name__}: {e}"
                    )
                    job.run_count += 1
                    self._persist()
                    print(
                        f"\n[CRON {job.id}] failed: "
                        f"{type(e).__name__}: {e}",
                    )
                    print(
                        f"  schedule: {job.schedule.raw} "
                        f"· run #{job.run_count}",
                    )
                if job.schedule.kind == "at":
                    job.enabled = False
                    self._persist()
                    break
        except asyncio.CancelledError:
            pass

    def _seconds_until_next_fire(self, job: CronJob) -> float | None:
        now = datetime.now()
        kind = job.schedule.kind

        if kind == "interval":
            return job.schedule.interval_seconds or 60.0

        if kind == "at":
            if not job.schedule.at_datetime:
                return None
            target = datetime.fromisoformat(job.schedule.at_datetime)
            # Tz-aware targets (e.g. "2026-06-12T09:00+08:00") can't be
            # compared with naive datetime.now(); convert to naive local time.
            if target.tzinfo is not None:
                target = target.astimezone().replace(tzinfo=None)
            delta = (target - now).total_seconds()
            return max(0.0, delta)

        if kind == "cron":
            if not job.schedule.cron_fields:
                return None
            fire = next_cron_fire(job.schedule.cron_fields, now)
            if fire is None:
                return None
            return max(0.0, (fire - now).total_seconds())

        return None

    async def _execute_job(self, job: CronJob) -> None:
        print(f"\n[CRON {job.id}] starting...")

        # Privacy mode: skip every job. Jobs may embed cloud-capable tools
        # (MCP, subagent) in their prompt, and there is no reliable way to
        # prove a job is local-only from the outside — so all of them pause.
        if getattr(self.config, "privacy_mode", False):
            job.last_result = (
                "skipped: privacy mode on, cron jobs do not run "
                "cloud capabilities"
            )
            job.last_run = datetime.now().isoformat()
            job.run_count += 1
            self._persist()
            print(f"[CRON {job.id}] skipped (privacy mode)")
            return

        # Condition check
        if job.condition:
            try:
                proc = await asyncio.create_subprocess_shell(
                    job.condition,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                exit_code = await proc.wait()
                if exit_code != 0:
                    job.last_result = f"condition not met (exit {exit_code})"
                    job.last_run = datetime.now().isoformat()
                    job.run_count += 1
                    self._persist()
                    return
            except Exception as e:
                job.last_result = f"condition check error: {e}"
                job.last_run = datetime.now().isoformat()
                self._persist()
                return

        # Auto-connect MCP services
        from dashscope.acli.skills.base import BUILTIN_SKILLS, render_skill

        for inv in job.skills:
            skill = BUILTIN_SKILLS.get(inv.name)
            if skill and skill.mcp_service:
                from dashscope.acli.cli import _connect_mcp, _mcp_clients

                if skill.mcp_service not in _mcp_clients:
                    print(
                        f"[CRON {job.id}] connecting MCP: "
                        f"{skill.mcp_service}...",
                    )
                    try:
                        err = await asyncio.wait_for(
                            _connect_mcp(skill.mcp_service, self.config),
                            timeout=30.0,
                        )
                        if err:
                            print(
                                f"[CRON {job.id}] MCP connect "
                                f"failed: {err}",
                            )
                    except asyncio.TimeoutError:
                        print(
                            f"[CRON {job.id}] MCP connect "
                            f"timeout: {skill.mcp_service}",
                        )
                    except Exception as e:
                        print(f"[CRON {job.id}] MCP connect error: {e}")

        # Execute skills
        results = []
        for inv in job.skills:
            skill = BUILTIN_SKILLS.get(inv.name)
            if not skill:
                results.append(f"unknown skill: {inv.name}")
                continue

            # Check MCP dependency
            if skill.mcp_service:
                from dashscope.acli.cli import _mcp_clients

                if skill.mcp_service not in _mcp_clients:
                    results.append(
                        f"skipped {inv.name}: requires MCP service "
                        f"{skill.mcp_service} (connect failed)",
                    )
                    continue

            rendered = render_skill(skill, inv.args)
            if not rendered:
                arg_hint = " ".join(f"<{a}>" for a in skill.arguments)
                results.append(
                    f"missing args: /skill {inv.name} {arg_hint}",
                )
                continue

            if job.subagent:
                from dashscope.acli.agents.subagent import _subagent_invoke

                result = await _subagent_invoke(
                    prompt=rendered,
                    system_prompt=(
                        f"You are the execution agent for a scheduled "
                        f"job. Current job: {inv.name}. Output the "
                        f"result directly, no extra explanation."
                    ),
                    max_turns=15,
                )
            else:
                self._pending_prompts.append(rendered)
                result = "(queued to the main conversation)"

            results.append(result)

        job.last_run = datetime.now().isoformat()
        job.last_result = "\n---\n".join(results)[:500]
        job.run_count += 1
        self._persist()

        # Display results
        skill_names = ", ".join(inv.name for inv in job.skills)
        content = "\n\n".join(results)
        print(f"\n⏰ [CRON {job.id}] {skill_names}")
        print(f"  schedule: {job.schedule.raw} · run #{job.run_count}")
        print("─" * 60)
        print(content)
        print("─" * 60)

    def _persist(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {jid: job.to_dict() for jid, job in self.jobs.items()}
        tmp = _PERSIST_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(_PERSIST_PATH)
