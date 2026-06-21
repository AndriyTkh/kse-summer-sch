"""walk-forward CV: rolling-origin folds (non-overlapping, purged) + B eval/summary."""

import numpy as np
import pandas as pd

from src import config, evaluate, features, index


def _periodic_grid(weeks=14, oblasts=("kyivska", "lvivska")):
    """Hourly grid whose alert fires every 8th hour -> the calendar `hour` feature makes
    the next-hour target perfectly learnable (PR-AUC high by construction). Period 8
    (not 4) keeps even the 6h target from saturating to all-positive.
    """
    end = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=weeks) - pd.Timedelta(hours=1)
    g = index.build_master_index(start="2024-01-01", end=end, oblasts=list(oblasts))
    hour = g.index.get_level_values("ts_utc").hour
    g["alert"] = (hour % 8 == 0).astype("int8")
    return features.build_feature_matrix(g, {})   # lags + calendar only (no waves/daily)


def test_walk_forward_splits_are_nonoverlapping_and_purged():
    fm = _periodic_grid()
    folds = list(evaluate.walk_forward_splits(fm, n_folds=3, test_weeks=2))
    assert [k for k, _, _ in folds] == [0, 1, 2]          # newest -> oldest

    prev_lo = None
    for k, train, test in folds:
        tr = train.index.get_level_values("ts_utc")
        te = test.index.get_level_values("ts_utc")
        # train strictly before test, with the purge gap honoured
        assert te.min() - tr.max() >= pd.Timedelta(hours=config.PURGE_HOURS)
        # each test window ~2 weeks
        assert pd.Timedelta(days=13) <= (te.max() - te.min()) <= pd.Timedelta(weeks=2)
        # non-overlap: this fold's test ends at/below the previous (newer) fold's start
        if prev_lo is not None:
            assert te.max() <= prev_lo
        prev_lo = te.min()


def test_walk_forward_skips_empty_train_folds():
    # Ask for more folds than history supports: oldest folds have no train side -> dropped.
    fm = _periodic_grid(weeks=8)
    folds = list(evaluate.walk_forward_splits(fm, n_folds=10, test_weeks=2))
    assert len(folds) < 10
    assert all(not train.empty and not test.empty for _, train, test in folds)


def test_walk_forward_eval_and_summary():
    fm = _periodic_grid()
    folds = evaluate.walk_forward_eval(fm, n_folds=3, test_weeks=2, horizons=["1h", "6h"])
    assert set(folds.columns) == {"fold", "horizon", "n_test", "base", "pr_auc", "lift"}
    assert folds["fold"].nunique() == 3
    # average_precision_score can return a hair over 1.0 (float rounding) -> tolerance
    assert ((folds["pr_auc"] >= 0) & (folds["pr_auc"] <= 1 + 1e-9)).all()
    # learnable signal -> strong discrimination on the 1h next-hour target
    assert folds.loc[folds["horizon"] == "1h", "pr_auc"].mean() > 0.9

    summary = evaluate.walk_forward_summary(folds)
    assert list(summary.index) == ["1h", "6h"]          # canonical horizon order
    assert (summary["n_folds"] == 3).all()
    assert {"pr_auc_mean", "pr_auc_std", "pr_auc_min", "pr_auc_max"} <= set(summary.columns)
    assert (summary["pr_auc_min"] <= summary["pr_auc_mean"]).all()
    assert (summary["pr_auc_mean"] <= summary["pr_auc_max"]).all()


def test_walk_forward_summary_empty_is_empty():
    assert evaluate.walk_forward_summary(pd.DataFrame()).empty
