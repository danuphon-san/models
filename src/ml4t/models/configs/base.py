"""Base config types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaseModelConfig:
    """Common configuration for ML4T models."""

    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
