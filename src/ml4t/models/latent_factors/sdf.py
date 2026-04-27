"""SDF estimator scaffold."""

from __future__ import annotations

from ml4t.models.api import PanelBatch
from ml4t.models.configs import SDFConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import FitSummary, LatentFactorState


class SDFModel(BaseLatentFactorModel[SDFConfig]):
    """Placeholder for the SDF structural extractor."""

    def __init__(self, config: SDFConfig) -> None:
        super().__init__(config)

    def fit(self, batch: PanelBatch) -> FitSummary:
        raise NotImplementedError("SDF port not implemented yet")

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        raise NotImplementedError("SDF port not implemented yet")
