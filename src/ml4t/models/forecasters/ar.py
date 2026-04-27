"""Per-factor autoregressive forecaster."""

from __future__ import annotations

import numpy as np

from ml4t.models.configs import AR1ForecasterConfig
from ml4t.models.forecasters.base import BaseFactorForecaster
from ml4t.models.types import FactorForecastResult, FitSummary, LatentFactorState


class AR1FactorForecaster(BaseFactorForecaster[AR1ForecasterConfig]):
    """Forecast factor premia with independent AR(1) models."""

    def __init__(self, config: AR1ForecasterConfig | None = None) -> None:
        super().__init__(config or AR1ForecasterConfig())
        self._intercepts: np.ndarray | None = None
        self._slopes: np.ndarray | None = None
        self._last_values: np.ndarray | None = None
        self._fallback_mean: np.ndarray | None = None

    def fit(self, state: LatentFactorState) -> FitSummary:
        if state.factor_returns is None:
            raise ValueError("AR1FactorForecaster requires training factor_returns")

        factors = np.asarray(state.factor_returns, dtype=np.float64)
        n_periods, n_factors = factors.shape
        self._fallback_mean = np.nanmean(factors, axis=0)
        self._last_values = factors[-1].copy()
        self._intercepts = np.zeros(n_factors, dtype=np.float64)
        self._slopes = np.zeros(n_factors, dtype=np.float64)

        if n_periods < 2:
            self._mark_fitted()
            return FitSummary(
                converged=True,
                train_metrics={"n_train_periods": float(n_periods)},
                notes=("Insufficient history for AR(1); using mean fallback.",),
            )

        x = factors[:-1]
        y = factors[1:]
        for factor_idx in range(n_factors):
            x_k = x[:, factor_idx]
            y_k = y[:, factor_idx]
            valid = np.isfinite(x_k) & np.isfinite(y_k)
            if valid.sum() < 2:
                self._intercepts[factor_idx] = self._fallback_mean[factor_idx]
                self._slopes[factor_idx] = 0.0
                continue
            design = np.column_stack([np.ones(valid.sum(), dtype=np.float64), x_k[valid]])
            coeffs, *_ = np.linalg.lstsq(design, y_k[valid], rcond=None)
            self._intercepts[factor_idx], self._slopes[factor_idx] = coeffs

        self._mark_fitted()
        return FitSummary(
            converged=True,
            train_metrics={"n_train_periods": float(n_periods)},
            notes=("Independent AR(1) fitted per factor.",),
        )

    def predict(self, state: LatentFactorState) -> FactorForecastResult:
        if not self.is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict()")
        assert self._intercepts is not None
        assert self._slopes is not None
        assert self._last_values is not None
        assert self._fallback_mean is not None

        n_periods = state.n_periods
        forecasts = np.zeros((n_periods, self._intercepts.shape[0]), dtype=np.float64)
        previous = self._last_values.copy()
        for step in range(n_periods):
            next_values = self._intercepts + self._slopes * previous
            next_values = np.where(np.isfinite(next_values), next_values, self._fallback_mean)
            forecasts[step] = next_values
            previous = next_values

        return FactorForecastResult(
            factor_premia=forecasts,
            timestamps=state.timestamps,
            metadata={"model_name": self.config.model_name},
        )
