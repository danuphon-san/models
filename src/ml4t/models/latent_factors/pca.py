"""Persistent-panel PCA baseline."""

from __future__ import annotations

import numpy as np

from ml4t.models.api import PanelBatch
from ml4t.models.configs import PCAConfig
from ml4t.models.latent_factors.base import BaseLatentFactorModel
from ml4t.models.types import FitSummary, LatentFactorState, PersistentPanelBatch


class PCAModel(BaseLatentFactorModel[PCAConfig]):
    """Persistent-panel PCA structural extractor."""

    def __init__(self, config: PCAConfig) -> None:
        super().__init__(config)
        self._asset_mean: np.ndarray | None = None
        self._loadings: np.ndarray | None = None
        self._train_factor_returns: np.ndarray | None = None
        self._asset_ids: tuple[str, ...] = ()

    def fit(self, batch: PanelBatch) -> FitSummary:
        persistent = _require_persistent_panel(batch)
        if persistent.returns is None:
            raise ValueError("PCA requires returns in the training batch")

        returns = np.asarray(persistent.returns, dtype=np.float64)
        asset_mean = np.nanmean(returns, axis=0)
        centered = returns - asset_mean[None, :]
        centered = np.where(np.isfinite(centered), centered, 0.0)

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        loadings = vt[: self.config.n_factors].T
        factor_returns = centered @ loadings

        self._asset_mean = asset_mean
        self._loadings = loadings
        self._train_factor_returns = factor_returns
        self._asset_ids = persistent.asset_ids
        self._mark_fitted()

        explained_variance = np.var(factor_returns, axis=0, ddof=0).sum()
        total_variance = np.var(centered, axis=0, ddof=0).sum()
        return FitSummary(
            converged=True,
            train_metrics={
                "explained_variance_ratio": float(explained_variance / total_variance)
                if total_variance > 0
                else 0.0,
            },
            notes=("Static loadings extracted from demeaned return panel.",),
        )

    def extract(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorState:
        del checkpoint
        persistent = _require_persistent_panel(batch)
        if not self.is_fitted or self._loadings is None:
            raise RuntimeError("PCA model must be fitted before extract()")

        n_periods = persistent.n_periods
        asset_betas = np.broadcast_to(
            self._loadings[None, :, :],
            (n_periods, self._loadings.shape[0], self._loadings.shape[1]),
        ).copy()
        factor_returns = None
        if persistent.returns is not None and self._asset_mean is not None:
            centered = np.asarray(persistent.returns, dtype=np.float64) - self._asset_mean[None, :]
            centered = np.where(np.isfinite(centered), centered, 0.0)
            factor_returns = centered @ self._loadings

        return LatentFactorState(
            asset_betas=asset_betas,
            factor_returns=factor_returns,
            checkpoint_epoch=None,
            timestamps=persistent.timestamps,
            asset_ids=persistent.asset_ids or self._asset_ids,
            metadata={
                "model_name": self.config.model_name,
                "persistent_entities": True,
            },
        )


def _require_persistent_panel(batch: PanelBatch) -> PersistentPanelBatch:
    if not isinstance(batch, PersistentPanelBatch):
        raise TypeError("PCA requires PersistentPanelBatch input")
    return batch
