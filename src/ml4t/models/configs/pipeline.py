"""Pipeline-level config types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MapperConfig:
    """Config for asset-return or weight mappers."""

    model_name: str = "beta_lambda"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Declarative description of a latent-factor forecast pipeline."""

    latent_factor_model: str
    factor_forecaster: str
    asset_mapper: str = "beta_lambda"
