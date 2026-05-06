"""Configuration data models — port of `deepseek-config` ConfigToml, ProviderKind, etc."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProviderKind(Enum):
    DEEPSEEK = "deepseek"
    NVIDIA_NIM = "nvidia-nim"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    NOVITA = "novita"
    FIREWORKS = "fireworks"
    SGLANG = "sglang"

    @classmethod
    def parse(cls, value: str) -> Optional[ProviderKind]:
        normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
        # Alias map
        aliases = {
            "deep-seek": cls.DEEPSEEK,
            "nvidia": cls.NVIDIA_NIM,
            "nim": cls.NVIDIA_NIM,
            "nvidia-nim": cls.NVIDIA_NIM,
            "open-ai": cls.OPENAI,
            "open-router": cls.OPENROUTER,
            "fireworks-ai": cls.FIREWORKS,
            "sg-lang": cls.SGLANG,
        }
        try:
            return cls(normalized)
        except ValueError:
            return aliases.get(normalized)

    def provider_slot(self) -> str:
        return self.value


class RunMode(str, Enum):
    PLAN = "plan"      # Read-only mode
    AGENT = "agent"    # Interactive approval mode (default)
    YOLO = "yolo"      # Auto-approve mode


# ── Provider Config ──────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


# ── Nested Config Tables ─────────────────────────────────────────────

@dataclass
class NetworkPolicyToml:
    default: str = "prompt"
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    audit: bool = True


@dataclass
class SkillsToml:
    registry_url: Optional[str] = None
    max_install_size_bytes: Optional[int] = None


@dataclass
class SnapshotsToml:
    enabled: bool = True
    max_age_days: int = 7


@dataclass
class LspConfigToml:
    enabled: Optional[bool] = None
    poll_after_edit_ms: Optional[int] = None
    max_diagnostics_per_file: Optional[int] = None
    include_warnings: Optional[bool] = None
    servers: Optional[dict[str, list[str]]] = None


# ── Main Config ──────────────────────────────────────────────────────

_DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
_DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_NVIDIA_NIM_MODEL = "deepseek-ai/deepseek-v4-pro"
_DEFAULT_NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_OPENAI_MODEL = "gpt-4.1"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-pro"
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_NOVITA_MODEL = "deepseek/deepseek-v4-pro"
_DEFAULT_NOVITA_BASE_URL = "https://api.novita.ai/v1"
_DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
_DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_DEFAULT_SGLANG_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_DEFAULT_SGLANG_BASE_URL = "http://localhost:30000/v1"


def _default_providers() -> dict[ProviderKind, ProviderConfig]:
    return {}


@dataclass
class ConfigToml:
    """On-disk TOML schema mirroring config.example.toml."""
    # Root-level fields
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_text_model: Optional[str] = None
    provider: ProviderKind = ProviderKind.DEEPSEEK
    model: Optional[str] = None
    auth_mode: Optional[str] = None
    chatgpt_access_token: Optional[str] = None
    device_code_session: Optional[str] = None
    output_mode: Optional[str] = None
    log_level: Optional[str] = None
    telemetry: Optional[bool] = None
    approval_policy: Optional[str] = None
    sandbox_mode: Optional[str] = None
    default_mode: RunMode = RunMode.AGENT
    # Sub-tables
    providers: dict[ProviderKind, ProviderConfig] = field(default_factory=_default_providers)
    network: Optional[NetworkPolicyToml] = None
    skills: Optional[SkillsToml] = None
    snapshots: Optional[SnapshotsToml] = None
    lsp: Optional[LspConfigToml] = None
    # Extras (unknown keys)
    extras: dict[str, object] = field(default_factory=dict)

    def merge_project_overrides(self, project: ConfigToml) -> None:
        """Merge project-level overrides from $WORKSPACE/.deepseek/config.toml."""
        if project.api_key is not None:
            self.api_key = project.api_key
            self.provider = project.provider
        if project.base_url is not None:
            self.base_url = project.base_url
        if project.default_text_model is not None:
            self.default_text_model = project.default_text_model
        if project.model is not None:
            self.model = project.model
        if project.auth_mode is not None:
            self.auth_mode = project.auth_mode
        if project.output_mode is not None:
            self.output_mode = project.output_mode
        if project.telemetry is not None:
            self.telemetry = project.telemetry
        if project.approval_policy is not None:
            self.approval_policy = project.approval_policy
        if project.sandbox_mode is not None:
            self.sandbox_mode = project.sandbox_mode
        if project.provider != ProviderKind.DEEPSEEK or project.api_key is not None:
            self.provider = project.provider
        # Merge provider sub-tables
        for prov, cfg in project.providers.items():
            if cfg.api_key is not None:
                self.providers.setdefault(prov, ProviderConfig()).api_key = cfg.api_key
            if cfg.base_url is not None:
                self.providers.setdefault(prov, ProviderConfig()).base_url = cfg.base_url
            if cfg.model is not None:
                self.providers.setdefault(prov, ProviderConfig()).model = cfg.model
        if project.network is not None:
            self.network = project.network
        if project.skills is not None:
            self.skills = project.skills
        if project.snapshots is not None:
            self.snapshots = project.snapshots
        if project.lsp is not None:
            self.lsp = project.lsp
        self.extras.update(project.extras)

    def resolve_runtime_options(self, cli: CliRuntimeOverrides) -> ResolvedRuntimeOptions:
        """Resolve runtime options with CLI flags > config file > env vars."""
        env = EnvRuntimeOverrides.load()
        provider = cli.provider or env.provider or self.provider

        provider_cfg = self.providers.get(provider, ProviderConfig())
        root_deepseek_api_key = self.api_key if provider == ProviderKind.DEEPSEEK else None
        root_deepseek_base_url = self.base_url if provider == ProviderKind.DEEPSEEK else None
        root_deepseek_model = self.default_text_model if provider == ProviderKind.DEEPSEEK else None

        # API key: CLI > config file > env
        from_file = provider_cfg.api_key or root_deepseek_api_key
        from_env = _api_key_env_for(provider)
        api_key = cli.api_key or from_file or from_env

        # Base URL: CLI > env > provider config > root > default
        base_url = (
            cli.base_url
            or env.base_url_for(provider)
            or provider_cfg.base_url
            or root_deepseek_base_url
            or _default_base_url(provider)
        )

        # Model: CLI > env > provider config > root > self.model > default
        model = (
            cli.model
            or env.model
            or provider_cfg.model
            or root_deepseek_model
            or self.model
            or _default_model(provider)
        )

        return ResolvedRuntimeOptions(
            provider=provider,
            model=_normalize_model_for_provider(provider, model),
            api_key=api_key,
            base_url=base_url,
            auth_mode=cli.auth_mode or env.auth_mode or self.auth_mode,
            output_mode=cli.output_mode or env.output_mode or self.output_mode,
            log_level=cli.log_level or env.log_level or self.log_level,
            telemetry=cli.telemetry or env.telemetry or self.telemetry or False,
            approval_policy=cli.approval_policy or env.approval_policy or self.approval_policy,
            sandbox_mode=cli.sandbox_mode or env.sandbox_mode or self.sandbox_mode,
        )

    def get_value(self, key: str) -> Optional[str]:
        """Get a config value by dotted key path."""
        mapping = {
            "provider": self.provider.value,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "default_text_model": self.default_text_model,
            "model": self.model,
            "auth.mode": self.auth_mode,
            "auth.chatgpt_access_token": self.chatgpt_access_token,
            "auth.device_code_session": self.device_code_session,
            "output_mode": self.output_mode,
            "log_level": self.log_level,
            "approval_policy": self.approval_policy,
            "sandbox_mode": self.sandbox_mode,
            "default_mode": self.default_mode.value,
        }
        # Provider-specific keys
        for prov in ProviderKind:
            prefix = f"providers.{prov.value}"
            cfg = self.providers.get(prov)
            if cfg is None:
                continue
            mapping[f"{prefix}.api_key"] = cfg.api_key
            mapping[f"{prefix}.base_url"] = cfg.base_url
            mapping[f"{prefix}.model"] = cfg.model

        # Check extras
        if key not in mapping:
            return str(self.extras.get(key, "")) if key in self.extras else None
        value = mapping[key]
        if value is None:
            return None
        if key == "telemetry":
            return str(value).lower()
        if "api_key" in key or "token" in key or "session" in key:
            return _redact_secret(str(value))
        return str(value)

    def list_values(self) -> dict[str, str]:
        """List all config values with redacted secrets."""
        result: dict[str, str] = {}
        result["provider"] = self.provider.value
        if self.api_key is not None:
            result["api_key"] = _redact_secret(self.api_key)
        if self.base_url is not None:
            result["base_url"] = self.base_url
        if self.default_text_model is not None:
            result["default_text_model"] = self.default_text_model
        if self.model is not None:
            result["model"] = self.model
        if self.auth_mode is not None:
            result["auth.mode"] = self.auth_mode
        if self.chatgpt_access_token is not None:
            result["auth.chatgpt_access_token"] = _redact_secret(self.chatgpt_access_token)
        if self.device_code_session is not None:
            result["auth.device_code_session"] = _redact_secret(self.device_code_session)
        if self.output_mode is not None:
            result["output_mode"] = self.output_mode
        if self.log_level is not None:
            result["log_level"] = self.log_level
        if self.telemetry is not None:
            result["telemetry"] = str(self.telemetry).lower()
        if self.approval_policy is not None:
            result["approval_policy"] = self.approval_policy
        if self.sandbox_mode is not None:
            result["sandbox_mode"] = self.sandbox_mode
        result["default_mode"] = self.default_mode.value
        # Provider-specific
        for prov in ProviderKind:
            prefix = f"providers.{prov.value}"
            cfg = self.providers.get(prov)
            if cfg is None:
                continue
            if cfg.api_key is not None:
                result[f"{prefix}.api_key"] = _redact_secret(cfg.api_key)
            if cfg.base_url is not None:
                result[f"{prefix}.base_url"] = cfg.base_url
            if cfg.model is not None:
                result[f"{prefix}.model"] = cfg.model
        for k, v in self.extras.items():
            result[k] = str(v)
        return result

    def set_value(self, key: str, value: str) -> None:
        """Set a config value by dotted key path."""
        if key == "provider":
            parsed = ProviderKind.parse(value)
            if parsed is None:
                raise ValueError(f"unknown provider '{value}'")
            self.provider = parsed
        elif key == "api_key":
            self.api_key = value
        elif key == "base_url":
            self.base_url = value
        elif key == "default_text_model":
            self.default_text_model = value
        elif key == "model":
            self.model = value
        elif key == "auth.mode":
            self.auth_mode = value
        elif key == "auth.chatgpt_access_token":
            self.chatgpt_access_token = value
        elif key == "auth.device_code_session":
            self.device_code_session = value
        elif key == "output_mode":
            self.output_mode = value
        elif key == "log_level":
            self.log_level = value
        elif key == "telemetry":
            self.telemetry = value.lower() in ("1", "true", "yes", "on", "enabled")
        elif key == "approval_policy":
            self.approval_policy = value
        elif key == "sandbox_mode":
            self.sandbox_mode = value
        elif key == "default_mode":
            try:
                self.default_mode = RunMode(value.lower())
            except ValueError:
                valid_modes = [m.value for m in RunMode]
                raise ValueError(f"invalid mode '{value}'. valid: {valid_modes}")
        elif key.startswith("providers."):
            parts = key.split(".")
            if len(parts) == 3:
                p = ProviderKind.parse(parts[1])
                if p is None:
                    raise ValueError(f"unknown provider '{parts[1]}'")
                cfg = self.providers.setdefault(p, ProviderConfig())
                if parts[2] == "api_key":
                    cfg.api_key = value
                    if p == ProviderKind.DEEPSEEK:
                        self.api_key = value
                elif parts[2] == "base_url":
                    cfg.base_url = value
                    if p == ProviderKind.DEEPSEEK:
                        self.base_url = value
                elif parts[2] == "model":
                    cfg.model = value
                    if p == ProviderKind.DEEPSEEK:
                        self.default_text_model = value
        else:
            self.extras[key] = value


def _default_base_url(provider: ProviderKind) -> str:
    return {
        ProviderKind.DEEPSEEK: _DEFAULT_DEEPSEEK_BASE_URL,
        ProviderKind.NVIDIA_NIM: _DEFAULT_NVIDIA_NIM_BASE_URL,
        ProviderKind.OPENAI: _DEFAULT_OPENAI_BASE_URL,
        ProviderKind.OPENROUTER: _DEFAULT_OPENROUTER_BASE_URL,
        ProviderKind.NOVITA: _DEFAULT_NOVITA_BASE_URL,
        ProviderKind.FIREWORKS: _DEFAULT_FIREWORKS_BASE_URL,
        ProviderKind.SGLANG: _DEFAULT_SGLANG_BASE_URL,
    }[provider]


def _default_model(provider: ProviderKind) -> str:
    return {
        ProviderKind.DEEPSEEK: _DEFAULT_DEEPSEEK_MODEL,
        ProviderKind.NVIDIA_NIM: _DEFAULT_NVIDIA_NIM_MODEL,
        ProviderKind.OPENAI: _DEFAULT_OPENAI_MODEL,
        ProviderKind.OPENROUTER: _DEFAULT_OPENROUTER_MODEL,
        ProviderKind.NOVITA: _DEFAULT_NOVITA_MODEL,
        ProviderKind.FIREWORKS: _DEFAULT_FIREWORKS_MODEL,
        ProviderKind.SGLANG: _DEFAULT_SGLANG_MODEL,
    }[provider]


_PROVIDER_MODEL_ALIASES: dict[ProviderKind, dict[str, str]] = {
    ProviderKind.NVIDIA_NIM: {
        "deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
        "deepseek-v4pro": "deepseek-ai/deepseek-v4-pro",
        "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
        "deepseek-v4flash": "deepseek-ai/deepseek-v4-flash",
        "deepseek-chat": "deepseek-ai/deepseek-v4-flash",
        "deepseek-reasoner": "deepseek-ai/deepseek-v4-flash",
        "deepseek-r1": "deepseek-ai/deepseek-v4-flash",
        "deepseek-v3": "deepseek-ai/deepseek-v4-flash",
        "deepseek-v3.2": "deepseek-ai/deepseek-v4-flash",
    },
    ProviderKind.OPENROUTER: {
        "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
        "deepseek-v4pro": "deepseek/deepseek-v4-pro",
        "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
        "deepseek-v4flash": "deepseek/deepseek-v4-flash",
        "deepseek-chat": "deepseek/deepseek-v4-flash",
        "deepseek-reasoner": "deepseek/deepseek-v4-flash",
        "deepseek-r1": "deepseek/deepseek-v4-flash",
        "deepseek-v3": "deepseek/deepseek-v4-flash",
        "deepseek-v3.2": "deepseek/deepseek-v4-flash",
    },
    ProviderKind.NOVITA: {
        "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
        "deepseek-v4pro": "deepseek/deepseek-v4-pro",
        "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
        "deepseek-v4flash": "deepseek/deepseek-v4-flash",
        "deepseek-chat": "deepseek/deepseek-v4-flash",
        "deepseek-reasoner": "deepseek/deepseek-v4-flash",
        "deepseek-r1": "deepseek/deepseek-v4-flash",
        "deepseek-v3": "deepseek/deepseek-v4-flash",
        "deepseek-v3.2": "deepseek/deepseek-v4-flash",
    },
    ProviderKind.FIREWORKS: {
        "deepseek-v4-pro": "accounts/fireworks/models/deepseek-v4-pro",
        "deepseek-v4pro": "accounts/fireworks/models/deepseek-v4-pro",
    },
    ProviderKind.SGLANG: {
        "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-v4pro": "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-v4flash": "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-chat": "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-reasoner": "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-r1": "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-v3": "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-v3.2": "deepseek-ai/DeepSeek-V4-Flash",
    },
}


def _normalize_model_for_provider(provider: ProviderKind, model: str) -> str:
    lower = model.strip().lower()
    aliases = _PROVIDER_MODEL_ALIASES.get(provider, {})
    return aliases.get(lower, model)


def _redact_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "********"
    return f"{secret[:4]}***{secret[-4:]}"


# ── Runtime Overrides ────────────────────────────────────────────────

@dataclass
class CliRuntimeOverrides:
    provider: Optional[ProviderKind] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    auth_mode: Optional[str] = None
    output_mode: Optional[str] = None
    log_level: Optional[str] = None
    telemetry: Optional[bool] = None
    approval_policy: Optional[str] = None
    sandbox_mode: Optional[str] = None


@dataclass
class ResolvedRuntimeOptions:
    provider: ProviderKind
    model: str
    api_key: Optional[str]
    base_url: str
    auth_mode: Optional[str]
    output_mode: Optional[str]
    log_level: Optional[str]
    telemetry: bool
    approval_policy: Optional[str]
    sandbox_mode: Optional[str]


# ── Environment Variable Overrides ───────────────────────────────────

@dataclass
class EnvRuntimeOverrides:
    provider: Optional[ProviderKind] = None
    model: Optional[str] = None
    output_mode: Optional[str] = None
    auth_mode: Optional[str] = None
    log_level: Optional[str] = None
    telemetry: Optional[bool] = None
    approval_policy: Optional[str] = None
    sandbox_mode: Optional[str] = None
    # Per-provider base URLs
    deepseek_base_url: Optional[str] = None
    nvidia_base_url: Optional[str] = None
    openai_base_url: Optional[str] = None
    openrouter_base_url: Optional[str] = None
    novita_base_url: Optional[str] = None
    fireworks_base_url: Optional[str] = None
    sglang_base_url: Optional[str] = None

    @classmethod
    def load(cls) -> EnvRuntimeOverrides:
        return cls(
            provider=_env_provider(),
            model=os.environ.get("DEEPSEEK_MODEL"),
            output_mode=os.environ.get("DEEPSEEK_OUTPUT_MODE"),
            auth_mode=os.environ.get("DEEPSEEK_AUTH_MODE"),
            log_level=os.environ.get("DEEPSEEK_LOG_LEVEL"),
            telemetry=_parse_bool_env("DEEPSEEK_TELEMETRY"),
            approval_policy=os.environ.get("DEEPSEEK_APPROVAL_POLICY"),
            sandbox_mode=os.environ.get("DEEPSEEK_SANDBOX_MODE"),
            deepseek_base_url=_non_empty_env("DEEPSEEK_BASE_URL"),
            nvidia_base_url=(
                _non_empty_env("NVIDIA_NIM_BASE_URL")
                or _non_empty_env("NIM_BASE_URL")
                or _non_empty_env("NVIDIA_BASE_URL")
            ),
            openai_base_url=_non_empty_env("OPENAI_BASE_URL"),
            openrouter_base_url=_non_empty_env("OPENROUTER_BASE_URL"),
            novita_base_url=_non_empty_env("NOVITA_BASE_URL"),
            fireworks_base_url=_non_empty_env("FIREWORKS_BASE_URL"),
            sglang_base_url=_non_empty_env("SGLANG_BASE_URL"),
        )

    def base_url_for(self, provider: ProviderKind) -> Optional[str]:
        return {
            ProviderKind.DEEPSEEK: self.deepseek_base_url,
            ProviderKind.NVIDIA_NIM: self.nvidia_base_url,
            ProviderKind.OPENAI: self.openai_base_url,
            ProviderKind.OPENROUTER: self.openrouter_base_url,
            ProviderKind.NOVITA: self.novita_base_url,
            ProviderKind.FIREWORKS: self.fireworks_base_url,
            ProviderKind.SGLANG: self.sglang_base_url,
        }.get(provider)


def _api_key_env_for(provider: ProviderKind) -> Optional[str]:
    """Look up API key from environment variables for a specific provider."""
    var_map = {
        ProviderKind.DEEPSEEK: "DEEPSEEK_API_KEY",
        ProviderKind.NVIDIA_NIM: "NVIDIA_NIM_API_KEY",
        ProviderKind.OPENAI: "OPENAI_API_KEY",
        ProviderKind.OPENROUTER: "OPENROUTER_API_KEY",
        ProviderKind.NOVITA: "NOVITA_API_KEY",
        ProviderKind.FIREWORKS: "FIREWORKS_API_KEY",
        ProviderKind.SGLANG: "SGLANG_API_KEY",
    }
    env_var = var_map.get(provider)
    if env_var is None:
        return None
    return _non_empty_env(env_var)


def _env_provider() -> Optional[ProviderKind]:
    raw = os.environ.get("DEEPSEEK_PROVIDER")
    if raw is None:
        return None
    return ProviderKind.parse(raw)


def _non_empty_env(key: str) -> Optional[str]:
    value = os.environ.get(key)
    if value and value.strip():
        return value.strip()
    return None


def _parse_bool_env(key: str) -> Optional[bool]:
    raw = os.environ.get(key)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on", "enabled")
