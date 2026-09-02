# -*- coding: utf-8 -*-
"""Skill package loader.

A skill package is a directory under ``.acli/skills/<name>/`` containing:

  skill.toml   — metadata (name, version, description, triggers, always_active)
  prompt.md    — system prompt supplement
  tools.py     — optional Python module exposing ``register(registry)``
  hooks.toml   — optional hooks definition

Loaded packages contribute:
  - namespaced tools (``<skill>_<tool>``)
  - prompt supplements when active
  - hooks registered on the global hook bus
"""
# pylint: disable=redefined-outer-name,too-many-branches

from __future__ import annotations

import fnmatch
import importlib.util
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dashscope.acli.config import CONFIG_DIR
from dashscope.acli.hooks import Hook, HookBus
from dashscope.acli.tools.registry import ToolRegistry, registry
from dashscope.acli.utils import load_toml


@dataclass
class SkillPackage:
    """A loaded skill package."""

    name: str
    version: str
    description: str
    author: str
    path: Path
    triggers: list[str] = field(default_factory=list)
    always_active: bool = False
    prompt: str = ""
    tools_registered: list[str] = field(default_factory=list)
    hooks_registered: list[Hook] = field(default_factory=list)

    def is_active(self, user_input: str) -> bool:
        if self.always_active:
            return True
        if not user_input:
            return False
        if isinstance(user_input, list):
            user_input = " ".join(
                b.get("text", "") for b in user_input if isinstance(b, dict)
            )
        for pattern in self.triggers:
            if fnmatch.fnmatch(user_input.lower(), pattern.lower()):
                return True
        return False


def _load_toml(path: Path) -> dict[str, Any] | None:
    return load_toml(path)


def _namespace_tool_name(skill_name: str, tool_name: str) -> str:
    if tool_name.startswith(f"{skill_name}_"):
        return tool_name
    return f"{skill_name}_{tool_name}"


def _register_package_tools(pkg: SkillPackage, registry: ToolRegistry) -> None:
    """Import tools.py and namespace any tools it registers."""
    tools_py = pkg.path / "tools.py"
    if not tools_py.exists():
        return
    try:
        # Snapshot before importing so module-level registrations are caught.
        before = {t.name: t for t in registry.list_tools()}

        # Module names must be valid Python identifiers (no hyphens/dots).
        safe_name = pkg.name.replace("-", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(
            f"acli_skill_tools_{safe_name}",
            tools_py,
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        # Keep the module alive so registered tool functions remain reachable.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)

            # Also call register(registry) if the module exposes it.
            register_fn = getattr(module, "register", None)
            if register_fn is not None:
                register_fn(registry)
        except Exception:
            # Roll back: a half-loaded package must not leave orphaned tools
            # in the registry or a broken module in sys.modules.
            for t in registry.list_tools():
                if t.name not in before:
                    registry.unregister(t.name)
            for name, original in before.items():
                if registry.get(name) is not original:
                    registry.unregister(name)
                    registry.register(original)
            sys.modules.pop(spec.name, None)
            return

        # Packages must not shadow pre-existing tools: restore any tool the
        # package re-registered under an existing name.
        for name, original in before.items():
            if registry.get(name) is not original:
                registry.unregister(name)
                registry.register(original)
                warnings.warn(
                    f"skill package '{pkg.name}' tried to override "
                    f"registered tool '{name}', ignored",
                    stacklevel=2,
                )

        after = {t.name for t in registry.list_tools()}

        # Rename newly added tools to include skill namespace.
        for name in after - set(before):
            tool = registry.get(name)
            if tool is None:
                continue
            new_name = _namespace_tool_name(pkg.name, name)
            if new_name != name:
                if registry.get(new_name) is not None:
                    warnings.warn(
                        f"skill package '{pkg.name}' tool '{new_name}' "
                        "collides with a registered tool, ignored",
                        stacklevel=2,
                    )
                    registry.unregister(name)
                    continue
                registry.unregister(name)
                tool.name = new_name
                registry.register(tool)
            pkg.tools_registered.append(new_name)
    except Exception:
        # Tool registration failures are non-fatal; the rest of the package
        # (prompts, hooks) can still work.
        pass


def _register_package_hooks(pkg: SkillPackage, hook_bus: HookBus) -> None:
    """Load hooks.toml and register each rule on the bus."""
    hooks_toml = pkg.path / "hooks.toml"
    data = _load_toml(hooks_toml)
    if not data:
        return
    hooks_section = data.get("hooks", {})
    if not isinstance(hooks_section, dict):
        return
    for event_name, entries in hooks_section.items():
        if not isinstance(entries, list):
            continue
        for spec in entries:
            if isinstance(spec, dict):
                hook = Hook(event_name, spec)
                hook_bus.register(hook)
                pkg.hooks_registered.append(hook)


def load_skill_package(path: Path) -> SkillPackage | None:
    """Load a single skill package directory."""
    if not path.is_dir():
        return None
    meta = _load_toml(path / "skill.toml")
    if not meta:
        return None
    name = meta.get("name", path.name)
    prompt_path = path / "prompt.md"
    prompt = ""
    if prompt_path.exists():
        try:
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    triggers = meta.get("triggers", [])
    if isinstance(triggers, str):
        triggers = [triggers]

    return SkillPackage(
        name=name,
        version=str(meta.get("version", "0.0.1")),
        description=str(meta.get("description", "")),
        author=str(meta.get("author", "")),
        path=path,
        triggers=triggers,
        always_active=bool(meta.get("always_active", False)),
        prompt=prompt,
    )


def discover_skill_packages(
    extra_dirs: list[Path] | None = None,
) -> list[SkillPackage]:
    """Discover skill packages.

    Priority: extra_dirs (project) > CONFIG_DIR/skills (global).
    Within a directory, each sub-directory is a candidate package.
    """
    dirs: list[Path] = []
    if extra_dirs:
        dirs.extend(extra_dirs)
    dirs.append(CONFIG_DIR / "skills")

    seen: set[str] = set()
    packages: list[SkillPackage] = []
    for base_dir in dirs:
        if not base_dir.is_dir():
            continue
        for candidate in sorted(base_dir.iterdir()):
            pkg = load_skill_package(candidate)
            if pkg is None or pkg.name in seen:
                continue
            seen.add(pkg.name)
            packages.append(pkg)
    return packages


def register_skill_package(
    pkg: SkillPackage,
    registry: ToolRegistry,
    hook_bus: HookBus,
) -> None:
    """Register a package's tools and hooks."""
    _register_package_tools(pkg, registry)
    _register_package_hooks(pkg, hook_bus)


def unregister_skill_package(
    pkg: SkillPackage,
    hook_bus: HookBus | None = None,
) -> None:
    """Remove a package's tools from the registry and its hooks from
    the bus."""
    for name in pkg.tools_registered:
        registry.unregister(name)
    pkg.tools_registered.clear()
    if hook_bus is not None:
        for hook in pkg.hooks_registered:
            hook_bus.unregister(hook)
        pkg.hooks_registered.clear()
