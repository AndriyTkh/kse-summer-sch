"""Model B — core. 4 direct LightGBM classifiers (30m/1h/3h/6h).

Direct multi-horizon: one model per horizon, target shifted to its distance
(recursive compounds error — rejected, see STRUCTURE §8).

Target for horizon H at row t = 1 if an alert occurs in window (t, t+H].
Class imbalance (issue #5): PR-AUC metric + scale_pos_weight, NOT accuracy.
"""

from __future__ import annotations

import pandas as pd

from . import config


def make_target(grid: pd.DataFrame, horizon: str) -> pd.Series:
    """Future-window label for `horizon`: alert in (t, t+H]? Forward shift only."""
    raise NotImplementedError


def train_horizon(X: pd.DataFrame, y: pd.Series, horizon: str):
    """Fit one LightGBM classifier for `horizon`. Recency-weighted (issue #6).

    Returns the fitted booster. scale_pos_weight set from train positive rate.
    """
    raise NotImplementedError


def train_all(X: pd.DataFrame, grid: pd.DataFrame) -> dict:
    """Train all 4 horizon models. Returns {horizon: model}."""
    raise NotImplementedError


def predict_all(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon alert probabilities. Returns [oblast, ts, h30m, h1h, h3h, h6h]."""
    raise NotImplementedError
