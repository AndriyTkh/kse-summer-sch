"""Model B — core. 4 direct LightGBM classifiers (30m/1h/3h/6h).

Direct multi-horizon: one model per horizon, target shifted to its distance
(recursive compounds error — rejected, see STRUCTURE §8).

Target for horizon H at row t = 1 if an alert is active in window (t, t+H].
Horizons are ceil'd to whole grid hours; on the HOURLY grid 30m and 1h both
collapse to "next 1 hour" (documented limit — sub-hourly needs a finer grid).

Class imbalance (issue #5): PR-AUC metric + scale_pos_weight, NOT accuracy.
Non-stationarity (issue #6): exponential recency weighting on training rows.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from . import config

# Recency half-life for sample weights (issue #6): older rows count exponentially less.
_RECENCY_HALFLIFE_DAYS = 180

_LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=50,
    n_jobs=-1,
    verbosity=-1,
)


def horizon_steps(horizon: str) -> int:
    """Whole grid-hour steps for a horizon (30m -> 1, 1h -> 1, 3h -> 3, 6h -> 6)."""
    return max(1, math.ceil(config.HORIZON_HOURS[horizon]))


def make_target(grid: pd.DataFrame, horizon: str) -> pd.Series:
    """Future-window label for `horizon`: 1 if alert active in (t, t+H]. Forward shift only.

    Tail rows (last H hours per oblast, no full future window) come back NaN — drop them
    before training. Uses only FUTURE cells, so it is a label, never a feature leak.
    """
    k = horizon_steps(horizon)
    g = grid.groupby(level="oblast", sort=False)["alert"]
    # max over shift(-1..-k) = "any alert in the next k hours".
    fut = pd.concat([g.shift(-i) for i in range(1, k + 1)], axis=1).max(axis=1)
    return fut.rename(f"target_{horizon}")


def feature_columns(X: pd.DataFrame) -> list[str]:
    """Model inputs = every column except the raw label and any target columns."""
    return [c for c in X.columns if c != "alert" and not c.startswith("target_")]


def _recency_weights(index: pd.MultiIndex) -> np.ndarray:
    """Exponential decay by row age relative to the latest timestamp in `index`."""
    ts = index.get_level_values("ts_utc")
    age_days = (ts.max() - ts).total_seconds().to_numpy() / 86400.0
    return np.power(0.5, age_days / _RECENCY_HALFLIFE_DAYS)


def train_horizon(X: pd.DataFrame, y: pd.Series, horizon: str) -> LGBMClassifier:
    """Fit one LightGBM classifier for `horizon`. Recency-weighted (issue #6).

    scale_pos_weight set from the training positive rate (issue #5). Rows with NaN
    target (future window runs off the end) are dropped here.
    """
    feats = feature_columns(X)
    mask = y.notna()
    Xt, yt = X.loc[mask, feats], y.loc[mask].astype(int)

    pos = int(yt.sum())
    neg = int(len(yt) - pos)
    # Need both classes for a meaningful re-balance; a saturated slice (all-pos or
    # all-neg, e.g. a degenerate walk-forward fold) -> spw 1.0 (LightGBM rejects <=0).
    spw = (neg / pos) if pos and neg else 1.0

    model = LGBMClassifier(scale_pos_weight=spw, **_LGBM_PARAMS)
    model.fit(Xt, yt, sample_weight=_recency_weights(Xt.index))
    return model


def train_all(X: pd.DataFrame, grid: pd.DataFrame, horizons=None) -> dict:
    """Train all horizon models on training-fold rows. Returns {horizon: model}.

    `X` is the feature matrix (training fold), `grid` carries the raw `alert` column
    aligned to X's index (targets are derived from it per horizon).
    """
    horizons = horizons or config.HORIZONS
    models = {}
    for h in horizons:
        y = make_target(grid, h).reindex(X.index)
        models[h] = train_horizon(X, y, h)
    return models


def predict_all(models: dict, X: pd.DataFrame, calibrators: dict | None = None) -> pd.DataFrame:
    """Per-horizon alert probabilities. Returns a frame indexed like X, one col per horizon.

    If `calibrators` is given ({horizon: fitted IsotonicRegression}, see
    evaluate.fit_isotonic), each horizon's raw prob is mapped through its calibrator
    (issue #10). Monotone, so ranking/PR-AUC is unchanged; only the prob scale shifts.
    """
    feats = feature_columns(X)
    out = pd.DataFrame(index=X.index)
    for h, model in models.items():
        p = model.predict_proba(X[feats])[:, 1]
        if calibrators and h in calibrators:
            p = calibrators[h].predict(p)
        out[h] = p
    return out
