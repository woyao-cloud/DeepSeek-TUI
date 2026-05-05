"""Model registry — port of `deepseek-agent` ModelRegistry.

Resolves model IDs to provider endpoints with alias support and fallback chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from deepseek_tui.config import ProviderKind


@dataclass
class ModelInfo:
    id: str
    provider: ProviderKind
    aliases: list[str] = field(default_factory=list)
    supports_tools: bool = True
    supports_reasoning: bool = True


@dataclass
class ModelResolution:
    requested: Optional[str]
    resolved: ModelInfo
    used_fallback: bool
    fallback_chain: list[str] = field(default_factory=list)


def _normalize(value: str) -> str:
    return value.strip().lower()


_BUILTIN_MODELS = [
    ModelInfo("deepseek-v4-pro", ProviderKind.DEEPSEEK, [], True, True),
    ModelInfo("deepseek-v4-flash", ProviderKind.DEEPSEEK, [
        "deepseek-chat", "deepseek-reasoner", "deepseek-r1",
        "deepseek-v3", "deepseek-v3.2",
    ], True, True),
    ModelInfo("deepseek-ai/deepseek-v4-pro", ProviderKind.NVIDIA_NIM, [
        "deepseek-v4-pro", "nvidia-deepseek-v4-pro", "nim-deepseek-v4-pro",
    ], True, True),
    ModelInfo("deepseek-ai/deepseek-v4-flash", ProviderKind.NVIDIA_NIM, [
        "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner",
        "nvidia-deepseek-v4-flash", "nim-deepseek-v4-flash",
    ], True, True),
    ModelInfo("gpt-4.1", ProviderKind.OPENAI, ["gpt4.1", "gpt-4o"], True, True),
    ModelInfo("gpt-4.1-mini", ProviderKind.OPENAI, ["gpt-4o-mini"], True, False),
    ModelInfo("deepseek/deepseek-v4-pro", ProviderKind.OPENROUTER, [
        "deepseek-v4-pro", "openrouter-deepseek-v4-pro",
    ], True, True),
    ModelInfo("deepseek/deepseek-v4-flash", ProviderKind.OPENROUTER, [
        "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner",
        "openrouter-deepseek-v4-flash",
    ], True, True),
    ModelInfo("deepseek/deepseek-v4-pro", ProviderKind.NOVITA, [
        "deepseek-v4-pro", "novita-deepseek-v4-pro",
    ], True, True),
    ModelInfo("deepseek/deepseek-v4-flash", ProviderKind.NOVITA, [
        "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner",
        "novita-deepseek-v4-flash",
    ], True, True),
    ModelInfo("accounts/fireworks/models/deepseek-v4-pro", ProviderKind.FIREWORKS, [
        "deepseek-v4-pro", "fireworks-deepseek-v4-pro",
    ], True, True),
    ModelInfo("deepseek-ai/DeepSeek-V4-Pro", ProviderKind.SGLANG, [
        "deepseek-v4-pro", "sglang-deepseek-v4-pro",
    ], True, True),
    ModelInfo("deepseek-ai/DeepSeek-V4-Flash", ProviderKind.SGLANG, [
        "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner",
        "sglang-deepseek-v4-flash",
    ], True, True),
]


class ModelRegistry:
    """Registry for resolving model IDs to provider endpoints."""

    def __init__(self, models: Optional[list[ModelInfo]] = None) -> None:
        models = models or _BUILTIN_MODELS
        self._models = list(models)
        self._alias_map: dict[str, int] = {}
        for idx, model in enumerate(self._models):
            key = _normalize(model.id)
            if key not in self._alias_map:
                self._alias_map[key] = idx
            for alias in model.aliases:
                alias_key = _normalize(alias)
                if alias_key not in self._alias_map:
                    self._alias_map[alias_key] = idx

    def list(self) -> list[ModelInfo]:
        return list(self._models)

    def resolve(
        self,
        requested: Optional[str] = None,
        provider_hint: Optional[ProviderKind] = None,
    ) -> ModelResolution:
        chain: list[str] = []

        if requested is not None:
            chain.append(f"requested:{requested}")
            norm = _normalize(requested)

            # Check provider-specific match first
            if provider_hint is not None:
                for model in self._models:
                    if model.provider == provider_hint and self._model_matches(model, norm):
                        return ModelResolution(requested, model, False, chain)

            # Check alias map
            if norm in self._alias_map:
                idx = self._alias_map[norm]
                if idx < len(self._models):
                    return ModelResolution(requested, self._models[idx], False, chain)

        provider = provider_hint or ProviderKind.DEEPSEEK
        chain.append(f"provider_default:{provider.value}")

        for model in self._models:
            if model.provider == provider:
                return ModelResolution(requested, model, True, chain)

        # Ultimate fallback
        chain.append("global_default:deepseek-v4-pro")
        fallback = ModelInfo("deepseek-v4-pro", ProviderKind.DEEPSEEK, [], True, True)
        return ModelResolution(requested, fallback, True, chain)

    @staticmethod
    def _model_matches(model: ModelInfo, norm: str) -> bool:
        if _normalize(model.id) == norm:
            return True
        for alias in model.aliases:
            if _normalize(alias) == norm:
                return True
        return False
