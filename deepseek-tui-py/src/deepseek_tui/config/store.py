"""ConfigStore — TOML config file read/write (port of `deepseek-config` ConfigStore)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .model import ConfigToml, ProviderKind, ProviderConfig

# Try to import tomllib (Python 3.11+) or tomli
try:
    import tomllib as _toml_loads
except ImportError:
    import tomli as _toml_loads  # type: ignore[no-redef]

import tomli_w as _toml_dumps


CONFIG_FILE_NAME = "config.toml"


class ConfigStore:
    """Persistent config store backed by a TOML file."""

    def __init__(self, path: Path, config: ConfigToml | None = None) -> None:
        self._path = path
        self.config = config or ConfigToml()

    @classmethod
    def load(cls, path: Optional[Path] = None) -> ConfigStore:
        resolved = resolve_config_path(path)
        if not resolved.exists():
            return cls(resolved)
        raw = resolved.read_text(encoding="utf-8")
        parsed = _parse_toml(raw)
        return cls(resolved, parsed)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = _serialize_toml(self.config)
        self._path.write_text(raw, encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path


def resolve_config_path(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit
    env_path = os.environ.get("DEEPSEEK_CONFIG_PATH")
    if env_path and env_path.strip():
        return Path(env_path.strip())
    return default_config_path()


def default_config_path() -> Path:
    home = Path.home()
    return home / ".deepseek" / CONFIG_FILE_NAME


# ── TOML Parsing / Serialization ─────────────────────────────────────

def _parse_toml(raw: str) -> ConfigToml:
    data = _toml_loads.loads(raw)
    config = ConfigToml()

    # Root scalar fields
    config.api_key = data.get("api_key")
    config.base_url = data.get("base_url")
    config.default_text_model = data.get("default_text_model")
    config.model = data.get("model")
    config.auth_mode = data.get("auth_mode")
    config.chatgpt_access_token = data.get("chatgpt_access_token")
    config.device_code_session = data.get("device_code_session")
    config.output_mode = data.get("output_mode")
    config.log_level = data.get("log_level")
    config.telemetry = data.get("telemetry")
    config.approval_policy = data.get("approval_policy")
    config.sandbox_mode = data.get("sandbox_mode")

    if "provider" in data:
        parsed = ProviderKind.parse(str(data["provider"]))
        if parsed:
            config.provider = parsed

    # Provider sub-tables
    providers_raw = data.get("providers", {})
    if isinstance(providers_raw, dict):
        for key, prov_data in providers_raw.items():
            if isinstance(prov_data, dict):
                provider = ProviderKind.parse(key)
                if provider:
                    cfg = ProviderConfig(
                        api_key=prov_data.get("api_key"),
                        base_url=prov_data.get("base_url"),
                        model=prov_data.get("model"),
                    )
                    config.providers[provider] = cfg

    # Collect extras (unknown top-level keys)
    known = {
        "api_key", "base_url", "default_text_model", "provider", "model",
        "auth_mode", "chatgpt_access_token", "device_code_session",
        "output_mode", "log_level", "telemetry", "approval_policy",
        "sandbox_mode", "providers", "network", "skills", "snapshots", "lsp",
    }
    for k, v in data.items():
        if k not in known:
            config.extras[k] = v

    return config


def _serialize_toml(config: ConfigToml) -> str:
    data = {}

    # Root scalars (only non-None values)
    for key in ("api_key", "base_url", "default_text_model", "model",
                 "auth_mode", "chatgpt_access_token", "device_code_session",
                 "output_mode", "log_level", "approval_policy", "sandbox_mode"):
        val = getattr(config, key, None)
        if val is not None:
            data[key] = val

    if config.telemetry is not None:
        data["telemetry"] = config.telemetry
    if config.provider != ProviderKind.DEEPSEEK:
        data["provider"] = config.provider.value

    # Provider sub-tables
    if config.providers:
        providers_out = {}
        for provider, cfg in config.providers.items():
            entry = {}
            for k in ("api_key", "base_url", "model"):
                v = getattr(cfg, k, None)
                if v is not None:
                    entry[k] = v
            if entry:
                providers_out[provider.value] = entry
        if providers_out:
            data["providers"] = providers_out

    # Extras
    data.update(config.extras)

    return _toml_dumps.dumps(data)
