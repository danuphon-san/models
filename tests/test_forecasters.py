from __future__ import annotations

import numpy as np

from ml4t.models import (
    AR1FactorForecaster,
    AR1ForecasterConfig,
    EWMABaseFactorForecaster,
    EWMABaseForecasterConfig,
    LatentFactorState,
)


def test_ar1_forecaster_produces_recursive_factor_path() -> None:
    state = LatentFactorState(
        asset_betas=np.ones((4, 3, 1), dtype=np.float64),
        factor_returns=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        timestamps=("2024-01", "2024-02", "2024-03", "2024-04"),
    )
    future = LatentFactorState(
        asset_betas=np.ones((3, 2, 1), dtype=np.float64),
        timestamps=("2024-05", "2024-06", "2024-07"),
    )

    forecaster = AR1FactorForecaster(AR1ForecasterConfig())
    forecaster.fit(state)
    forecast = forecaster.predict(future)

    assert forecast.factor_premia.shape == (3, 1)
    assert forecast.factor_premia[0, 0] < forecast.factor_premia[1, 0]
    assert forecast.factor_premia[1, 0] <= forecast.factor_premia[2, 0]


def test_ewma_forecaster_broadcasts_last_level() -> None:
    state = LatentFactorState(
        asset_betas=np.ones((5, 2, 2), dtype=np.float64),
        factor_returns=np.array(
            [
                [0.1, -0.2],
                [0.2, -0.1],
                [0.3, 0.0],
                [0.4, 0.1],
                [0.5, 0.2],
            ],
            dtype=np.float64,
        ),
    )
    future = LatentFactorState(asset_betas=np.ones((2, 2, 2), dtype=np.float64))

    forecaster = EWMABaseFactorForecaster(EWMABaseForecasterConfig(half_life=2.0))
    forecaster.fit(state)
    forecast = forecaster.predict(future)

    assert forecast.factor_premia.shape == (2, 2)
    assert np.allclose(forecast.factor_premia[0], forecast.factor_premia[1])
