"""Supervised autoencoder scaffold."""

from __future__ import annotations

from ml4t.models.api import PanelBatch
from ml4t.models.configs import SAEConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import FitSummary, LatentFactorState


class SAEModel(BaseLatentFactorModel[SAEConfig]):
    """Placeholder for the SAE structural extractor."""

    def __init__(self, config: SAEConfig) -> None:
        super().__init__(config)

    def fit(self, batch: PanelBatch) -> FitSummary:
        raise NotImplementedError("SAE port not implemented yet")

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        raise NotImplementedError("SAE port not implemented yet")
