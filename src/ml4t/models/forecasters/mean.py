"""Historical-mean factor-premium forecaster."""

from __future__ import annotations

import numpy as np

from ml4t.models.configs import ExpandingMeanForecasterConfig
from ml4t.models.forecasters.base import BaseFactorForecaster
from ml4t.models.types import FactorForecastResult, FitSummary, LatentFactorState


class ExpandingMeanFactorForecaster(BaseFactorForecaster[ExpandingMeanForecasterConfig]):
    """Forecast factor premia with the training-sample mean."""

    def __init__(self, config: ExpandingMeanForecasterConfig | None = None) -> None:
        super().__init__(config or ExpandingMeanForecasterConfig())
        self._mean_factor_premium: np.ndarray | None = None

    def fit(self, state: LatentFactorState) -> FitSummary:
        if state.factor_returns is None:
            raise ValueError("ExpandingMeanFactorForecaster requires training factor_returns")

        self._mean_factor_premium = np.nanmean(state.factor_returns, axis=0)
        self._mark_fitted()
        return FitSummary(
            converged=True,
            train_metrics={
                "n_train_periods": float(state.n_periods),
                "mean_abs_factor_premium": float(np.mean(np.abs(self._mean_factor_premium))),
            },
            notes=("Forecast uses the historical mean of fitted factor returns.",),
        )

    def predict(self, state: LatentFactorState) -> FactorForecastResult:
        if not self.is_fitted or self._mean_factor_premium is None:
            raise RuntimeError("Forecaster must be fitted before predict()")

        factor_premia = np.broadcast_to(
            self._mean_factor_premium[None, :],
            (state.n_periods, self._mean_factor_premium.shape[0]),
        ).copy()
        return FactorForecastResult(
            factor_premia=factor_premia,
            timestamps=state.timestamps,
            metadata={"model_name": self.config.model_name},
        )
