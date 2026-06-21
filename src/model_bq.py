"""Model Bq — quantile LightGBM for prediction intervals (Phase 3).

B answers "will an alert fire?" with a point probability. Bq answers "how much of
the next H hours will be under alert, and how sure are we?" — it regresses the
continuous alert-FRACTION over (t, t+H] at several quantiles, yielding an
uncertainty band (the free bands STRUCTURE §8 listed as missing).

Why a fraction target (not the binary label): a probability has no spread to put a
quantile on. The fraction in [0, 1] is a genuine continuous intensity — "expect 40%
of the next 6h under alert (10–70% band)" — which is what shelter/resource planning
actually needs, and quantile regression gives honest bands on it.

  - 30m/1h collapse to k=1 grid step, so their fraction is just the next-hour binary
    {0,1}; the band degenerates there (honest — no sub-hourly info on this grid).
  - 3h/6h are genuinely continuous and where the interval earns its keep.

Leak guard (issue #1): the target reads ONLY future cells (t, t+H]; features are the
same leak-safe matrix B uses. Recency weighting + feature selection reuse model_b.

Quantile crossing: independent per-quantile fits can cross (q90 < q10 on a row); we
sort each row's quantiles ascending after predicting, then clip to [0, 1] (a fraction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from . import config, model_b

# Bq reuses B's tree config; only the objective/alpha differ per quantile.
_BQ_PARAMS = dict(model_b._LGBM_PARAMS)


def q_label(alpha: float) -> str:
    """Quantile -> compact column suffix (0.1 -> 'q10', 0.05 -> 'q05')."""
    return f"q{int(round(alpha * 100)):02d}"


def make_fraction_target(grid: pd.DataFrame, horizon: str) -> pd.Series:
    """Continuous future-window target: alert FRACTION over (t, t+H], in [0, 1].

    Mirrors model_b.make_target but averages the next-k cells instead of max-ing them.
    k = horizon_steps(horizon). Tail rows whose entire future window runs off the grid
    come back NaN (drop before fit); partial windows average the available cells (same
    skip-NaN convention as make_target — consistent, accepted).
    """
    k = model_b.horizon_steps(horizon)
    g = grid.groupby(level="oblast", sort=False)["alert"]
    fut = pd.concat([g.shift(-i) for i in range(1, k + 1)], axis=1).mean(axis=1)
    return fut.rename(f"frac_{horizon}")


def train_quantile_horizon(
    X: pd.DataFrame,
    y: pd.Series,
    quantiles=config.QUANTILES,
    params: dict | None = None,
) -> dict:
    """Fit one LGBMRegressor per quantile for a horizon. Returns {alpha: model}.

    Recency-weighted (issue #6) on training rows; NaN-target rows dropped here.
    `params` overrides tree config (tests pass tiny n_estimators for speed).
    """
    feats = model_b.feature_columns(X)
    mask = y.notna()
    Xt, yt = X.loc[mask, feats], y.loc[mask].astype(float)
    w = model_b._recency_weights(Xt.index)
    base = {**_BQ_PARAMS, **(params or {})}

    models = {}
    for a in quantiles:
        m = LGBMRegressor(objective="quantile", alpha=a, **base)
        m.fit(Xt, yt, sample_weight=w)
        models[a] = m
    return models


def train_all_quantiles(
    X: pd.DataFrame,
    grid: pd.DataFrame,
    horizons=None,
    quantiles=config.QUANTILES,
    params: dict | None = None,
) -> dict:
    """Train quantile models for every horizon. Returns {horizon: {alpha: model}}.

    `grid` supplies the raw `alert` column the fraction target is derived from
    (X may be the same frame — build_feature_matrix keeps `alert`).
    """
    horizons = horizons or config.HORIZONS
    out = {}
    for h in horizons:
        y = make_fraction_target(grid, h).reindex(X.index)
        out[h] = train_quantile_horizon(X, y, quantiles, params)
    return out


def predict_quantiles(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon quantile predictions, monotone-sorted + clipped to [0, 1].

    `models` = {horizon: {alpha: model}} from train_all_quantiles. Columns are
    '{horizon}_{qNN}' (e.g. '6h_q10', '6h_q50', '6h_q90'). Within a horizon the
    per-row quantiles are sorted ascending to remove quantile crossing.
    """
    feats = model_b.feature_columns(X)
    Xf = X[feats]
    out = pd.DataFrame(index=X.index)
    for h, qmods in models.items():
        alphas = sorted(qmods)
        block = np.column_stack([qmods[a].predict(Xf) for a in alphas])
        block = np.sort(block, axis=1)          # kill quantile crossing
        block = np.clip(block, 0.0, 1.0)        # it's a fraction
        for j, a in enumerate(alphas):
            out[f"{h}_{q_label(a)}"] = block[:, j]
    return out


def interval_columns(horizon: str, low=config.PI_LOW, high=config.PI_HIGH) -> tuple[str, str, str]:
    """Column names for a horizon's (low, median, high) band in predict_quantiles output."""
    return f"{horizon}_{q_label(low)}", f"{horizon}_q50", f"{horizon}_{q_label(high)}"
