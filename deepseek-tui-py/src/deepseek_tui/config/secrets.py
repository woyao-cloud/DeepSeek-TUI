"""Secrets management — port of `deepseek-secrets` crate.

Provides a pluggable keyring abstraction with an in-memory fallback.
"""

from __future__ import annotations

import abc
import os
from typing import Optional


class KeyringStore(abc.ABC):
    """Abstract credential store."""

    @abc.abstractmethod
    def get(self, service: str) -> Optional[str]:
        """Retrieve a credential."""

    @abc.abstractmethod
    def set(self, service: str, value: str) -> None:
        """Store a credential."""

    @abc.abstractmethod
    def delete(self, service: str) -> None:
        """Remove a credential."""


class InMemoryKeyringStore(KeyringStore):
    """Ephemeral key-value store (for testing / fallback)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, service: str) -> Optional[str]:
        return self._store.get(service)

    def set(self, service: str, value: str) -> None:
        self._store[service] = value

    def delete(self, service: str) -> None:
        self._store.pop(service, None)


class Secrets:
    """Secrets façade with automatic platform detection."""

    def __init__(self, store: KeyringStore) -> None:
        self._store = store

    @classmethod
    def auto_detect(cls) -> Secrets:
        # For now, always use in-memory store. OS keyring integration
        # (keyring lib) can be added as an opt-in.
        return cls(InMemoryKeyringStore())

    def resolve(self, service: str) -> Optional[str]:
        """Resolve a credential: env var > keyring."""
        env_value = env_for(service)
        if env_value is not None:
            return env_value
        return self._store.get(service)

    def get(self, service: str) -> Optional[str]:
        return self._store.get(service)

    def set(self, service: str, value: str) -> None:
        self._store.set(service, value)

    def delete(self, service: str) -> None:
        self._store.delete(service)

    @property
    def backend_name(self) -> str:
        return type(self._store).__name__


def env_for(service: str) -> Optional[str]:
    """Look up a provider API key from environment variables."""
    # Map provider names to env var names
    var_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "nvidia-nim": "NVIDIA_NIM_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "novita": "NOVITA_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "sglang": "SGLANG_API_KEY",
    }
    env_var = var_map.get(service)
    if env_var is None:
        return None
    value = os.environ.get(env_var)
    if value and value.strip():
        return value.strip()
    return None
