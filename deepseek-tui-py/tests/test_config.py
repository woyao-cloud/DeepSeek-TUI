"""Tests for the config module."""

import os
import tempfile
from pathlib import Path

import pytest

from deepseek_tui.config import (
    ConfigStore,
    ConfigToml,
    ProviderKind,
    ProviderConfig,
    CliRuntimeOverrides,
)


class TestProviderKind:
    def test_parse_deepseek(self):
        assert ProviderKind.parse("deepseek") == ProviderKind.DEEPSEEK
        assert ProviderKind.parse("deep-seek") == ProviderKind.DEEPSEEK
        assert ProviderKind.parse("DEEPSEEK") == ProviderKind.DEEPSEEK

    def test_parse_nvidia(self):
        assert ProviderKind.parse("nvidia-nim") == ProviderKind.NVIDIA_NIM
        assert ProviderKind.parse("nvidia") == ProviderKind.NVIDIA_NIM
        assert ProviderKind.parse("nim") == ProviderKind.NVIDIA_NIM

    def test_parse_openai(self):
        assert ProviderKind.parse("openai") == ProviderKind.OPENAI
        assert ProviderKind.parse("open-ai") == ProviderKind.OPENAI

    def test_parse_openrouter(self):
        assert ProviderKind.parse("openrouter") == ProviderKind.OPENROUTER
        assert ProviderKind.parse("open_router") == ProviderKind.OPENROUTER

    def test_parse_novita(self):
        assert ProviderKind.parse("novita") == ProviderKind.NOVITA

    def test_parse_fireworks(self):
        assert ProviderKind.parse("fireworks") == ProviderKind.FIREWORKS
        assert ProviderKind.parse("fireworks-ai") == ProviderKind.FIREWORKS

    def test_parse_sglang(self):
        assert ProviderKind.parse("sglang") == ProviderKind.SGLANG
        assert ProviderKind.parse("sg-lang") == ProviderKind.SGLANG

    def test_parse_unknown(self):
        assert ProviderKind.parse("unknown") is None


class TestConfigToml:
    def test_defaults(self):
        config = ConfigToml()
        assert config.provider == ProviderKind.DEEPSEEK
        assert config.api_key is None

    def test_resolve_runtime_deepseek(self):
        config = ConfigToml(
            api_key="sk-test",
            default_text_model="deepseek-v4-pro",
        )
        resolved = config.resolve_runtime_options(CliRuntimeOverrides())
        assert resolved.provider == ProviderKind.DEEPSEEK
        assert resolved.api_key == "sk-test"
        assert resolved.model == "deepseek-v4-pro"
        assert resolved.base_url == "https://api.deepseek.com"

    def test_resolve_cli_overrides(self):
        config = ConfigToml()
        cli = CliRuntimeOverrides(
            api_key="cli-key",
            model="deepseek-v4-flash",
            base_url="https://custom.example.com",
        )
        resolved = config.resolve_runtime_options(cli)
        assert resolved.api_key == "cli-key"
        assert resolved.model == "deepseek-v4-flash"
        assert resolved.base_url == "https://custom.example.com"

    def test_resolve_nvidia_provider(self):
        config = ConfigToml(provider=ProviderKind.NVIDIA_NIM)
        resolved = config.resolve_runtime_options(CliRuntimeOverrides())
        assert resolved.provider == ProviderKind.NVIDIA_NIM
        assert resolved.model == "deepseek-ai/deepseek-v4-pro"
        assert resolved.base_url == "https://integrate.api.nvidia.com/v1"

    def test_resolve_nvidia_model_aliasing(self):
        config = ConfigToml(provider=ProviderKind.NVIDIA_NIM)
        cli = CliRuntimeOverrides(model="deepseek-v4-flash")
        resolved = config.resolve_runtime_options(cli)
        assert resolved.model == "deepseek-ai/deepseek-v4-flash"

    def test_resolve_openrouter(self):
        config = ConfigToml(provider=ProviderKind.OPENROUTER)
        resolved = config.resolve_runtime_options(CliRuntimeOverrides())
        assert resolved.provider == ProviderKind.OPENROUTER
        assert resolved.base_url == "https://openrouter.ai/api/v1"
        assert resolved.model == "deepseek/deepseek-v4-pro"

    def test_merge_project_overrides(self):
        base = ConfigToml(api_key="global-key", base_url="https://api.deepseek.com")
        project = ConfigToml(
            base_url="https://project.deepseek.com",
            model="deepseek-v4-flash",
        )
        base.merge_project_overrides(project)
        assert base.api_key == "global-key"
        assert base.base_url == "https://project.deepseek.com"
        assert base.model == "deepseek-v4-flash"

    def test_get_value(self):
        config = ConfigToml(
            api_key="sk-secret-key-12345",
            provider=ProviderKind.OPENAI,
            model="gpt-4.1",
        )
        # Check redaction
        api_val = config.get_value("api_key")
        assert api_val is not None
        assert "***" in api_val
        assert "sk-s" in api_val

        provider_val = config.get_value("provider")
        assert provider_val == "openai"

        model_val = config.get_value("model")
        assert model_val == "gpt-4.1"

    def test_set_value_provider(self):
        config = ConfigToml()
        config.set_value("provider", "openai")
        assert config.provider == ProviderKind.OPENAI

    def test_set_value_api_key(self):
        config = ConfigToml()
        config.set_value("api_key", "sk-new")
        assert config.api_key == "sk-new"


class TestConfigStore:
    def test_load_and_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            store = ConfigStore(path)
            store.config.api_key = "sk-roundtrip"
            store.config.provider = ProviderKind.OPENAI
            store.config.providers[ProviderKind.OPENAI] = ProviderConfig(model="gpt-4.1")
            store.save()

            # Re-load
            store2 = ConfigStore.load(path)
            assert store2.config.api_key == "sk-roundtrip"
            assert store2.config.provider == ProviderKind.OPENAI
            assert store2.config.providers[ProviderKind.OPENAI].model == "gpt-4.1"


class TestEnvOverrides:
    def test_env_provider(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_PROVIDER", "openai")
        config = ConfigToml()
        resolved = config.resolve_runtime_options(CliRuntimeOverrides())
        assert resolved.provider == ProviderKind.OPENAI

    def test_env_api_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        config = ConfigToml()
        resolved = config.resolve_runtime_options(CliRuntimeOverrides())
        assert resolved.api_key == "sk-env"

    def test_cli_beats_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        config = ConfigToml()
        cli = CliRuntimeOverrides(api_key="sk-cli")
        resolved = config.resolve_runtime_options(cli)
        assert resolved.api_key == "sk-cli"
