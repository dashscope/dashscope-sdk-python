# -*- coding: utf-8 -*-
"""Skill package manager — discovery, activation, and slash commands."""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dashscope.acli.hooks import HookBus
from dashscope.acli.skills import store
from dashscope.acli.skills.package import (
    SkillPackage,
    discover_skill_packages,
    register_skill_package,
    unregister_skill_package,
)
from dashscope.acli.tools.registry import registry


@dataclass
class SkillManager:
    """Manages loaded skill packages and their activation state."""

    packages: list[SkillPackage] = field(default_factory=list)
    _force_enabled: set[str] = field(default_factory=set)
    _force_disabled: set[str] = field(default_factory=set)
    _hook_bus: HookBus | None = None
    _global: bool = False
    _registry_url: str = ""

    def load(self, hook_bus: HookBus | None = None) -> None:
        """Discover and register packages from workspace and global dirs."""
        if hook_bus is not None:
            self._hook_bus = hook_bus
        bus = self._hook_bus
        # Clear previously registered tools and hooks
        for pkg in self.packages:
            unregister_skill_package(pkg, bus)
        self.packages = discover_skill_packages(
            extra_dirs=[store.get_skills_dir(self._global)],
        )
        for pkg in self.packages:
            if bus is not None:
                register_skill_package(pkg, registry, bus)
            else:
                register_skill_package(pkg, registry, HookBus())

    def reload(self) -> None:
        """Reload packages from disk."""
        self._force_enabled.clear()
        self._force_disabled.clear()
        self.load(self._hook_bus)

    def list(self) -> list[dict[str, Any]]:
        """Return package info for display."""
        return [
            {
                "name": pkg.name,
                "version": pkg.version,
                "description": pkg.description,
                "always_active": pkg.always_active,
                "triggers": pkg.triggers,
                "tools": list(pkg.tools_registered),
            }
            for pkg in self.packages
        ]

    def install(self, source: str) -> str:
        """Install a skill package and reload."""
        name = store.install(
            source,
            global_=self._global,
            registry_url=self._registry_url,
        )
        self.reload()
        return name

    def link(self, source_dir: str) -> str:
        """Symlink a local skill package and reload."""
        name = store.link(source_dir, global_=self._global)
        self.reload()
        return name

    def uninstall(self, name: str) -> bool:
        """Remove an installed skill package and reload."""
        result = store.uninstall(name, global_=self._global)
        if result:
            self._force_enabled.discard(name)
            self._force_disabled.discard(name)
            self.reload()
        return result

    def update(self, name: str) -> str:
        """Update a single installed skill package and reload."""
        store.update(
            name,
            global_=self._global,
            registry_url=self._registry_url,
        )
        self.reload()
        return name

    def update_all(self) -> builtins.list[tuple[str, str | None]]:
        """Update all installed skill packages and reload."""
        results = store.update_all(
            global_=self._global,
            registry_url=self._registry_url,
        )
        self.reload()
        return results

    def search(self, query: str) -> builtins.list[dict[str, Any]]:
        """Search installed packages and the registry."""
        return store.search(
            query,
            global_=self._global,
            registry_url=self._registry_url,
        )

    def publish(self, source_dir: str) -> Path:
        """Validate and package a skill directory."""
        return store.publish(source_dir)

    def active_packages(self, user_input: str) -> builtins.list[SkillPackage]:
        """Return packages that should be active for the given user input."""
        active: list[SkillPackage] = []
        for pkg in self.packages:
            if pkg.name in self._force_disabled:
                continue
            if pkg.name in self._force_enabled or pkg.is_active(user_input):
                active.append(pkg)
        return active

    def active_prompts(self, user_input: str) -> str:
        """Concatenate prompts from active packages."""
        prompts = [
            pkg.prompt
            for pkg in self.active_packages(user_input)
            if pkg.prompt
        ]
        if not prompts:
            return ""
        return "\n\n".join(prompts)

    def enable(self, name: str) -> bool:
        pkg = self._find(name)
        if pkg is None:
            return False
        self._force_enabled.add(name)
        self._force_disabled.discard(name)
        return True

    def disable(self, name: str) -> bool:
        pkg = self._find(name)
        if pkg is None:
            return False
        self._force_disabled.add(name)
        self._force_enabled.discard(name)
        return True

    def _find(self, name: str) -> SkillPackage | None:
        for pkg in self.packages:
            if pkg.name == name:
                return pkg
        return None


# Global manager instance
_skill_manager = SkillManager()


def get_skill_manager() -> SkillManager:
    return _skill_manager
