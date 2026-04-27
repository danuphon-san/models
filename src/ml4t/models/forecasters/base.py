"""Base classes for factor forecasters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ml4t.models.configs import BaseModelConfig
from ml4t.models.types import FactorForecastResult, FitSummary, LatentFactorState


class BaseFactorForecaster[ConfigT: BaseModelConfig](ABC):
    """Abstract base for factor-premium forecasters."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, state: LatentFactorState) -> FitSummary:
        """Fit on extracted training-state factor returns."""

    @abstractmethod
    def predict(self, state: LatentFactorState) -> FactorForecastResult:
        """Forecast factor premia for the target batch."""

    def _mark_fitted(self) -> None:
        self._is_fitted = True
