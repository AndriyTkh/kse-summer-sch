"""Evaluation — temporal split, PR-AUC, calibration, heatmap.

Rare-positive data: accuracy is misleading (issue #5). Report PR-AUC + calibration.
Split is TEMPORAL only — train early, test last TEST_WEEKS (issue #6). Never random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from . import config


def temporal_split(
    df: pd.DataFrame,
    test_weeks: int = config.TEST_WEEKS,
    purge_hours: float = config.PURGE_HOURS,
):
    """Split by time: train = early, test = last `test_weeks`. Returns (train, test).

    Cut is a single global timestamp (max ts - test_weeks). NEVER random (issue #6):
    rows at or after the cut are test, everything earlier is train. Works on any frame
    carrying a `ts_utc` index level.

    Purged split (leak guard): targets span t -> t+H, so train rows in the last
    `purge_hours` before the cut would carry labels reaching into test. Drop that
    gap band so no train label peeks across the cut. Pass purge_hours=0 to disable.
    """
    ts = df.index.get_level_values("ts_utc")
    cut = ts.max() - pd.Timedelta(weeks=test_weeks)
    train = df[ts < cut - pd.Timedelta(hours=purge_hours)]
    test = df[ts >= cut]
    return train, test


def pr_auc(y_true, y_score) -> float:
    """Average precision (area under PR curve). Primary metric (issue #5)."""
    return float(average_precision_score(y_true, y_score))


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Mean |confidence - accuracy| over equal-width probability bins (issue #10)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.mean()) * abs(y_score[m].mean() - y_true[m].mean())
    return float(ece)


def fit_isotonic(y_true, y_score) -> IsotonicRegression:
    """Fit isotonic regression mapping raw score -> calibrated probability (issue #10).

    LightGBM with scale_pos_weight inflates probabilities (rebalances the prior), so
    raw probs are miscalibrated even when ranking is good. Isotonic is monotone, so it
    leaves PR-AUC (a ranking metric) unchanged while correcting the probability scale.

    MUST be fit out-of-fold: pass a calibration slice the model never trained on
    (the CALIB_WEEKS fold), never the test fold or in-sample train rows.
    `out_of_bounds="clip"` keeps test scores outside the calib range in [0, 1].
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(y_score, dtype=float), np.asarray(y_true, dtype=float))
    return iso


def calibration_plot(y_true, y_score, n_bins: int = 10, ax=None):
    """Reliability curve + ECE/Brier. Flags miscalibration (apply isotonic if bad, #10).

    Returns (ax, {'ece':..., 'brier':...}). Lazy matplotlib import so the metric path
    (and headless test runs) needn't touch a display backend.
    """
    import matplotlib.pyplot as plt

    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability")
    ax.legend(loc="upper left")
    metrics = {
        "ece": expected_calibration_error(y_true, y_score, n_bins),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    return ax, metrics


def compare_b_vs_a(b_pred: pd.DataFrame, a_pred: pd.DataFrame, y_true: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side B vs A PR-AUC per horizon. Deliverable: B wins short horizon.

    `b_pred`/`y_true` are horizon-keyed frames aligned on the test index; `a_pred` is
    the Prophet baseline broadcast to the same index/horizons. Missing A -> NaN row.
    Returns a table indexed by horizon with columns [pr_auc_b, pr_auc_a, lift].
    """
    rows = {}
    for h in b_pred.columns:
        y = y_true[h] if isinstance(y_true, pd.DataFrame) else y_true
        m = y.notna()
        b = pr_auc(y[m].astype(int), b_pred.loc[m, h])
        a = (
            pr_auc(y[m].astype(int), a_pred.loc[m, h])
            if a_pred is not None and h in a_pred.columns
            else float("nan")
        )
        rows[h] = {"pr_auc_b": b, "pr_auc_a": a, "lift": b / a if a else float("nan")}
    return pd.DataFrame(rows).T


def oblast_horizon_heatmap(probs: pd.DataFrame, ax=None):
    """Oblast x horizon mean-probability heatmap — the headline visual output.

    `probs` is the predict_all frame (MultiIndex oblast/ts, one col per horizon).
    Averages predicted probability per oblast over the test window. Returns (ax, table).
    """
    import matplotlib.pyplot as plt

    table = probs.groupby(level="oblast").mean()
    table = table[[h for h in config.HORIZONS if h in table.columns]]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.3 * len(table) + 1))
    im = ax.imshow(table.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("horizon")
    ax.set_title("Mean alert probability by oblast x horizon")
    ax.figure.colorbar(im, ax=ax, label="P(alert)")
    return ax, table
