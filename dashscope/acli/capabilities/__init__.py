"""Capability Interface Abstraction: Decouple from Bailian.

Abstracts cloud capabilities into pluggable interfaces.
Enables:
- Local implementations (for privacy mode, offline usage)
- Multiple providers (not just Bailian)
- Easy capability swapping

Architecture:
- CapabilityInterface: base class with common methods
- MemoryCapability: user memory/profile operations
- CapabilityRegistry: manages active capability implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CapabilityInterface(ABC):
    """Base interface for all capabilities."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Capability identifier (e.g., 'bailian.mcp', 'bailian.cli')."""
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name (e.g., 'bailian', 'local', 'ollama')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this capability is currently available."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the capability (connect, authenticate, etc.)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass


class MemoryCapability(CapabilityInterface):
    """Interface for user memory/profile operations."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search user memory for relevant information."""
        pass

    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Add a memory entry. Returns entry ID."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def list(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent memory entries."""
        pass


class CapabilityRegistry:
    """Registry for managing capability implementations."""

    def __init__(self):
        self._capabilities: dict[str, CapabilityInterface] = {}

    def register(self, capability: CapabilityInterface) -> None:
        """Register a capability implementation."""
        self._capabilities[capability.name] = capability

    def unregister(self, name: str) -> None:
        """Unregister a capability."""
        self._capabilities.pop(name, None)

    def get(self, name: str) -> CapabilityInterface | None:
        """Get a registered capability by name."""
        return self._capabilities.get(name)


# Global registry instance
_registry = CapabilityRegistry()


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry."""
    return _registry


def get_capability(name: str) -> CapabilityInterface | None:
    """Get a capability from the global registry."""
    return _registry.get(name)
