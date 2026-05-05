"""Tests for the agent (model registry) module."""

import pytest

from deepseek_tui.agent import ModelRegistry, ModelInfo
from deepseek_tui.config import ProviderKind


class TestModelRegistry:
    def test_deepseek_v4_pro_default(self):
        registry = ModelRegistry()
        resolved = registry.resolve()
        assert resolved.resolved.provider == ProviderKind.DEEPSEEK
        assert resolved.resolved.id == "deepseek-v4-pro"
        assert resolved.used_fallback

    def test_deepseek_v4_pro_by_name(self):
        registry = ModelRegistry()
        resolved = registry.resolve("deepseek-v4-pro")
        assert resolved.resolved.provider == ProviderKind.DEEPSEEK
        assert resolved.resolved.id == "deepseek-v4-pro"
        assert not resolved.used_fallback

    def test_deepseek_v4_flash_aliases(self):
        registry = ModelRegistry()
        for alias in ("deepseek-chat", "deepseek-reasoner", "deepseek-r1", "deepseek-v3"):
            resolved = registry.resolve(alias)
            assert resolved.resolved.id == "deepseek-v4-flash", f"{alias} -> {resolved.resolved.id}"

    def test_nvidia_nim_hint(self):
        registry = ModelRegistry()
        resolved = registry.resolve("deepseek-v4-pro", provider_hint=ProviderKind.NVIDIA_NIM)
        assert resolved.resolved.provider == ProviderKind.NVIDIA_NIM
        assert resolved.resolved.id == "deepseek-ai/deepseek-v4-pro"

    def test_nvidia_nim_default(self):
        registry = ModelRegistry()
        resolved = registry.resolve(None, provider_hint=ProviderKind.NVIDIA_NIM)
        assert resolved.resolved.provider == ProviderKind.NVIDIA_NIM
        assert resolved.resolved.id == "deepseek-ai/deepseek-v4-pro"
        assert resolved.used_fallback

    def test_openrouter_default(self):
        registry = ModelRegistry()
        resolved = registry.resolve(None, provider_hint=ProviderKind.OPENROUTER)
        assert resolved.resolved.provider == ProviderKind.OPENROUTER
        assert resolved.resolved.id == "deepseek/deepseek-v4-pro"

    def test_novita_default(self):
        registry = ModelRegistry()
        resolved = registry.resolve(None, provider_hint=ProviderKind.NOVITA)
        assert resolved.resolved.provider == ProviderKind.NOVITA
        assert resolved.resolved.id == "deepseek/deepseek-v4-pro"

    def test_fireworks_default(self):
        registry = ModelRegistry()
        resolved = registry.resolve(None, provider_hint=ProviderKind.FIREWORKS)
        assert resolved.resolved.provider == ProviderKind.FIREWORKS
        assert resolved.resolved.id == "accounts/fireworks/models/deepseek-v4-pro"

    def test_sglang_default(self):
        registry = ModelRegistry()
        resolved = registry.resolve(None, provider_hint=ProviderKind.SGLANG)
        assert resolved.resolved.provider == ProviderKind.SGLANG
        assert resolved.resolved.id == "deepseek-ai/DeepSeek-V4-Pro"

    def test_fallback_chain_populated(self):
        registry = ModelRegistry()
        resolved = registry.resolve("nonexistent-model")
        assert resolved.used_fallback
        assert len(resolved.fallback_chain) > 0

    def test_list_models(self):
        registry = ModelRegistry()
        models = registry.list()
        assert len(models) >= 12  # 12 model infos in the builtin list
        assert all(isinstance(m, ModelInfo) for m in models)
