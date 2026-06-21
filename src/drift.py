"""Drift detection (Phase 3) — the operational answer to non-stationarity (issue #6).

War tempo is non-stationary: 2022 cruise-missile waves ≠ 2025 Shahed swarms. A model
frozen on early data silently rots. This module quantifies that shift so the
auto-retrain loop (retrain.py) can fire BEFORE labels confirm the damage.

Signal: Population Stability Index (PSI) per feature, between a REFERENCE window (what
the live model trained on) and a CURRENT block. PSI bins the reference distribution,
then measures how much the current distribution's mass moved between those bins:

    PSI = Σ (cur_pct - ref_pct) * ln(cur_pct / ref_pct)

Rule of thumb (config): < 0.10 stable, 0.10–0.25 moderate, > 0.25 significant.

PSI is a leading (covariate) signal — feature shape moves before performance craters.
Performance-drop (a lagging signal) is handled directly in retrain.py off the pinball
trajectory; here we own the data-distribution side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_EPS = 1e-6


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples. 0 = identical.

    Bin edges are quantiles of `expected` (equal-frequency), so a well-spread reference
    gives stable bins; outer edges open to ±inf so the current block can't fall outside.
    Constant/degenerate features (one unique edge) -> 0.0 (no distribution to shift).
    Empty bins are floored to `_EPS` to keep the log finite.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:                      # constant feature: nothing to shift
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_pct = np.histogram(expected, edges)[0] / expected.size
    a_pct = np.histogram(actual, edges)[0] / actual.size
    e_pct = np.clip(e_pct, _EPS, None)
    a_pct = np.clip(a_pct, _EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame, features=None) -> pd.Series:
    """Per-feature PSI (reference vs current), sorted most-drifted first.

    `features` defaults to the numeric columns common to both frames (raw `alert` and
    any `target_`/`frac_` columns excluded — drift is about INPUTS, not the label).
    """
    if features is None:
        common = [c for c in reference.columns if c in current.columns]
        features = [
            c for c in common
            if c != "alert"
            and not c.startswith(("target_", "frac_"))
            and np.issubdtype(reference[c].dropna().to_numpy().dtype, np.number)
        ]
    scores = {f: psi(reference[f].to_numpy(), current[f].to_numpy()) for f in features}
    return pd.Series(scores, dtype=float).sort_values(ascending=False)


def classify_psi(value: float) -> str:
    """PSI -> band label using config thresholds: stable | moderate | significant."""
    if value >= config.DRIFT_PSI_ALERT:
        return "significant"
    if value >= config.DRIFT_PSI_WARN:
        return "moderate"
    return "stable"


def drift_score(reference: pd.DataFrame, current: pd.DataFrame, features=None) -> dict:
    """Aggregate drift verdict for one reference→current comparison.

    Returns {'max', 'mean', 'n_significant', 'top', 'band'} where `max` is the worst
    single-feature PSI (the trigger the retrain loop watches), `top` is that feature,
    and `n_significant` counts features past the ALERT threshold.
    """
    s = feature_psi(reference, current, features)
    if s.empty:
        return {"max": 0.0, "mean": 0.0, "n_significant": 0, "top": None, "band": "stable"}
    mx = float(s.iloc[0])
    return {
        "max": mx,
        "mean": float(s.mean()),
        "n_significant": int((s >= config.DRIFT_PSI_ALERT).sum()),
        "top": str(s.index[0]),
        "band": classify_psi(mx),
    }
