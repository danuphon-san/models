"""Config dataclasses for end-to-end portfolio learners."""

from __future__ import annotations

from dataclasses import dataclass

from ml4t.models.configs.base import BaseModelConfig


@dataclass(frozen=True, slots=True)
class PortfolioConfig(BaseModelConfig):
    """Base config for portfolio-learning models."""

    model_name: str = "portfolio_model"
    turnover_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class LSTMPortfolioConfig(PortfolioConfig):
    """Starter config for a sequence-based portfolio learner."""

    model_name: str = "lstm_portfolio"
    hidden_size: int = 64
    n_layers: int = 1
