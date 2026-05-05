"""Model registry — resolve model IDs to provider endpoints.

Port of the `deepseek-agent` crate.
"""

from .registry import ModelInfo, ModelRegistry, ModelResolution

__all__ = ["ModelInfo", "ModelRegistry", "ModelResolution"]
