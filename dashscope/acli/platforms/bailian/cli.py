# -*- coding: utf-8 -*-
"""Wrapper around the `bl` CLI (Aliyun Model Studio).

`bl` is a Node-based CLI bundled separately (`npm i -g @bailian/cli`
or similar).
We treat it as a *platform capability*: when `bl` is on PATH, we pull its full
command catalog via `bl config export-schema` and register each subcommand as
a native function-call tool. The LLM then sees `bailian_text_chat`,
`bailian_image_generate`, etc. with proper JSON-Schema parameters.

Why subprocess instead of a Python SDK: `bl` already implements the auth,
streaming, retries, multimodal upload, and (importantly) the schema export,
so we avoid duplicating ~thousands of lines of provider code.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass


class BailianCLIError(Exception):
    pass


@dataclass
class CLIResult:
    code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.code == 0


def _camel_to_kebab(name: str) -> str:
    # Split before an uppercase that starts a new word (prev is
    # lowercase/digit)
    # or that ends an acronym run (next is lowercase): APIKey → api-key.
    import re

    return re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
        "-",
        name,
    ).lower()


class BailianCLIClient:
    """Subprocess wrapper around the `bl` binary."""

    def __init__(
        self,
        binary: str = "bl",
        api_key: str = "",
        region: str = "",
        base_url: str = "",
        timeout: int = 180,
    ):
        self.binary = binary
        self.api_key = api_key
        self.region = region
        self.base_url = base_url
        self.timeout = timeout

    # ===== Discovery =====

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def export_schemas_sync(self) -> list[dict]:
        """Synchronous variant — used at startup tool-registration time when
        we're already inside an event loop and can't spin up another."""
        try:
            cp = subprocess.run(
                [self.binary, "config", "export-schema"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            raise BailianCLIError(
                f"bl config export-schema failed: {e}",
            ) from e
        if cp.returncode != 0:
            raise BailianCLIError(
                f"bl exit={cp.returncode}: {cp.stderr or cp.stdout}",
            )
        try:
            return json.loads(cp.stdout)
        except json.JSONDecodeError as e:
            raise BailianCLIError(f"bl schema parse failed: {e}") from e

    # ===== Invocation =====

    async def run(
        self,
        args: list[str],
        stdin: str = "",
        inject_globals: bool = True,
    ) -> CLIResult:
        """Run `bl <args>` and capture stdout/stderr."""
        cmd = [self.binary, *args]
        env = None
        if inject_globals:
            if self.api_key:
                # env, not argv: `--api-key <key>` is visible in `ps` to
                # every local user.
                import os

                env = {**os.environ, "DASHSCOPE_API_KEY": self.api_key}
            if self.region:
                cmd.extend(["--region", self.region])
            if self.base_url:
                cmd.extend(["--base-url", self.base_url])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(stdin.encode() if stdin else None),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            return CLIResult(
                code=124,
                stdout="",
                stderr=f"timeout ({self.timeout}s)",
            )
        return CLIResult(
            code=proc.returncode or 0,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
        )

    async def invoke(self, command_path: list[str], params: dict) -> str:
        """Translate `{kw: val}` into `--kebab-case <value>` flags and
        call `bl`.

        - bool=True    →  --flag
        - bool=False   →  omitted
        - list         →  repeat --flag <item>
        - str/number   →  --flag <value>
        - None         →  omitted
        """
        args = list(command_path)
        for key, value in params.items():
            if value is None:
                continue
            flag = "--" + _camel_to_kebab(key)
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, list):
                for item in value:
                    args.extend([flag, str(item)])
            else:
                args.extend([flag, str(value)])

        result = await self.run(args)
        if not result.ok:
            tail = (result.stderr or result.stdout).strip().splitlines()[-20:]
            return f"[bl exit={result.code}]\n" + "\n".join(tail)
        return result.stdout
