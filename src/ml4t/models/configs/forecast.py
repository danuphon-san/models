"""Config dataclasses for factor forecasters."""

from __future__ import annotations

from dataclasses import dataclass

from ml4t.models.configs.base import BaseModelConfig


@dataclass(frozen=True, slots=True)
class ExpandingMeanForecasterConfig(BaseModelConfig):
    """Config for the historical-mean factor-premium baseline."""

    model_name: str = "expanding_mean"


@dataclass(frozen=True, slots=True)
class AR1ForecasterConfig(BaseModelConfig):
    """Config for per-factor AR(1) forecasts."""

    model_name: str = "ar1"


@dataclass(frozen=True, slots=True)
class EWMABaseForecasterConfig(BaseModelConfig):
    """Config for EWMA factor-premium forecasts."""

    model_name: str = "ewma"
    half_life: float = 12.0
