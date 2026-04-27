"""Factor-premium forecasters."""

from ml4t.models.forecasters.ar import AR1FactorForecaster
from ml4t.models.forecasters.base import BaseFactorForecaster
from ml4t.models.forecasters.ewma import EWMABaseFactorForecaster
from ml4t.models.forecasters.mean import ExpandingMeanFactorForecaster

__all__ = [
    "AR1FactorForecaster",
    "BaseFactorForecaster",
    "EWMABaseFactorForecaster",
    "ExpandingMeanFactorForecaster",
]
