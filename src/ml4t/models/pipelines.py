"""Composable pipelines for finance-native model workflows."""

from __future__ import annotations

from dataclasses import dataclass

from ml4t.models.api import AssetMapper, FactorForecaster, LatentFactorModel, PanelBatch
from ml4t.models.types import FitSummary, LatentFactorPrediction


@dataclass(slots=True)
class PipelineFitResult:
    """Fit summaries for each stage of a latent-factor pipeline."""

    structural_fit: FitSummary
    factor_forecast_fit: FitSummary


class LatentFactorForecastPipeline:
    """Compose structural extraction, factor forecasting, and asset mapping."""

    def __init__(
        self,
        model: LatentFactorModel,
        forecaster: FactorForecaster,
        mapper: AssetMapper,
    ) -> None:
        self.model = model
        self.forecaster = forecaster
        self.mapper = mapper

    def fit(self, batch: PanelBatch) -> PipelineFitResult:
        structural_fit = self.model.fit(batch)
        train_state = self.model.extract(batch)
        factor_forecast_fit = self.forecaster.fit(train_state)
        return PipelineFitResult(
            structural_fit=structural_fit,
            factor_forecast_fit=factor_forecast_fit,
        )

    def predict(
        self,
        batch: PanelBatch,
        *,
        checkpoint: int | None = None,
    ) -> LatentFactorPrediction:
        state = self.model.extract(batch, checkpoint=checkpoint)
        factor_forecast = self.forecaster.predict(state)
        asset_forecast = self.mapper.predict(state, factor_forecast)
        return LatentFactorPrediction(
            state=state,
            factor_forecast=factor_forecast,
            asset_forecast=asset_forecast,
        )
