"""Evaluation — temporal split, PR-AUC, calibration, heatmap.

Rare-positive data: accuracy is misleading (issue #5). Report PR-AUC + calibration.
Split is TEMPORAL only — train early, test last TEST_WEEKS (issue #6). Never random.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score

from . import config


def temporal_split(df: pd.DataFrame, test_weeks: int = config.TEST_WEEKS):
    """Split by time: train = early, test = last `test_weeks`. Returns (train, test).

    Cut is a single global timestamp (max ts - test_weeks). NEVER random (issue #6):
    rows at or after the cut are test, everything earlier is train. Works on any frame
    carrying a `ts_utc` index level.
    """
    ts = df.index.get_level_values("ts_utc")
    cut = ts.max() - pd.Timedelta(weeks=test_weeks)
    train = df[ts < cut]
    test = df[ts >= cut]
    return train, test


def pr_auc(y_true, y_score) -> float:
    """Average precision (area under PR curve). Primary metric (issue #5)."""
    return float(average_precision_score(y_true, y_score))


def calibration_plot(y_true, y_score, ax=None):
    """Reliability curve. Apply isotonic/Platt if miscalibrated (issue #10)."""
    raise NotImplementedError


def compare_b_vs_a(b_pred: pd.DataFrame, a_pred: pd.DataFrame, y_true) -> pd.DataFrame:
    """Side-by-side B vs A at each horizon. Deliverable: B wins short horizon."""
    raise NotImplementedError


def oblast_horizon_heatmap(probs: pd.DataFrame, ax=None):
    """Oblast x horizon probability heatmap — the headline visual output."""
    raise NotImplementedError
