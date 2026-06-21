"""Model ONSET — alt approach (the "when", parallel to Model B's "whether").

WHY this exists. Model B asks "is an alert ACTIVE somewhere in (t, t+H]". On the
HOURLY grid the 6h base rate is ~0.51 and the target is trivially monotone in H, so
B wins by persistence/autocorrelation (alert_roll_168h alone = 53% of gain) rather than
by forecasting a strike. High PR-AUC, hollow product (see the 2026-06-21 phase-4
write-down). This module retargets the SAME pipeline to alert ONSET:

    target_H[t] = 1 iff a NEW alert STARTS in (t, t+H], evaluated ONLY from a quiet
                  state (no alert active at t); rows with an alert active at t are
                  excluded (NaN) — the "does a new one begin" question is ill-posed
                  while one is already running.

A "start" is a rising edge: alert==1 at hour s with alert==0 at s-1. Multi-horizon
(30m/1h/3h/6h) reads as a timing profile. Numbers will look WORSE than B but mean more,
and this is where threat LEADING indicators (revived via config.ONSET_THREAT_*) should
finally pay off.

Everything else is reused from model_b (feature columns, recency weights, scale_pos_weight
re-balance, isotonic-aware predict): only the TARGET changes. Path "A" of the onset
reframe; "B" (time-to-next-onset survival) is a separate later step.
"""

from __future__ import annotations

import pandas as pd

from . import config, model_b


def make_onset_target(grid: pd.DataFrame, horizon: str) -> pd.Series:
    """Onset-in-window label for `horizon`: 1 if a NEW alert starts in (t, t+H].

    Quiet-state only: rows where an alert is already active at t come back NaN and are
    dropped before training/eval (same NaN-drop path model_b.train_horizon uses for the
    no-future-window tail). Uses only FUTURE cells for the label and only the current/past
    `alert` for the quiet mask — no leak.
    """
    k = model_b.horizon_steps(horizon)
    a = grid["alert"]
    g = a.groupby(level="oblast", sort=False)
    prev = g.shift(1)
    # Rising edge: alert on now, off the previous hour. First row per oblast has prev NaN
    # -> not counted as an onset (can't confirm it's new). float so NaN can flow through.
    onset = ((a == 1) & (prev == 0)).astype("float64")

    og = onset.groupby(level="oblast", sort=False)
    # max over shift(-1..-k) = "any onset in the next k hours". Mirrors model_b.make_target;
    # only the final row per oblast (whole window off the end) is all-NaN -> NaN.
    fut = pd.concat([og.shift(-i) for i in range(1, k + 1)], axis=1).max(axis=1)
    # Evaluate from a quiet state only: drop rows with an alert active at t.
    fut = fut.where(a == 0)
    return fut.rename(f"onset_{horizon}")


def train_all(X: pd.DataFrame, grid: pd.DataFrame, horizons=None) -> dict:
    """Train one LightGBM onset classifier per horizon. Returns {horizon: model}.

    Mirrors model_b.train_all but swaps in the onset target. Reuses model_b.train_horizon
    (recency weights + scale_pos_weight + NaN-target drop), so the quiet-state mask and the
    tail naturally fall out of training.
    """
    horizons = horizons or config.HORIZONS
    models = {}
    for h in horizons:
        y = make_onset_target(grid, h).reindex(X.index)
        models[h] = model_b.train_horizon(X, y, h)
    return models


# Prediction is identical to the whether-model (probabilities, optional isotonic).
predict_all = model_b.predict_all
feature_columns = model_b.feature_columns
