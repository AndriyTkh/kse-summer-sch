"""operational_eval: simulated ragged-edge degradation + backtest-vs-live gap."""

import numpy as np
import pandas as pd

from src import config, features, index, operational_eval


def _grid_with_threat_cols(weeks=10, oblasts=("kyivska", "lvivska")):
    """Hourly grid with every-8h alerts + synthetic threat/tempo columns so the
    degradation path has something to mask.
    """
    end = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=weeks) - pd.Timedelta(hours=1)
    g = index.build_master_index(start="2024-01-01", end=end, oblasts=list(oblasts))
    hour = g.index.get_level_values("ts_utc").hour
    g["alert"] = (hour % 8 == 0).astype("int8")
    fm = features.build_feature_matrix(g, {})
    # Inject synthetic threat/tempo columns so degrade_features has targets.
    rng = np.random.default_rng(42)
    fm["thr_ballistic_launched_3h"] = rng.random(len(fm))
    fm["thr_drone-strike_waves_6h"] = rng.random(len(fm))
    fm["tempo_prev_launched"] = rng.random(len(fm))
    return fm


def test_degrade_features_zeros_threat_tempo_in_tail():
    fm = _grid_with_threat_cols()
    degraded = operational_eval.degrade_features(fm, lag_hours=6)
    ts = degraded.index.get_level_values("ts_utc")
    cutoff = ts.max() - pd.Timedelta(hours=6)
    tail = degraded[ts >= cutoff]
    head = degraded[ts < cutoff]
    # Threat/tempo columns are zeroed in the tail
    for col in ["thr_ballistic_launched_3h", "thr_drone-strike_waves_6h", "tempo_prev_launched"]:
        assert (tail[col] == 0).all(), f"{col} should be zeroed in tail"
        assert not (head[col] == 0).all(), f"{col} should NOT be all-zero in head"
    # Non-degradable columns are untouched
    assert (degraded["hour"] == fm["hour"]).all()
    assert (degraded["alert"] == fm["alert"]).all()


def test_degrade_features_zero_lag_is_noop():
    fm = _grid_with_threat_cols()
    degraded = operational_eval.degrade_features(fm, lag_hours=0)
    pd.testing.assert_frame_equal(degraded, fm)


def test_degrade_features_no_threat_cols_is_noop():
    end = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=4) - pd.Timedelta(hours=1)
    g = index.build_master_index(start="2024-01-01", end=end, oblasts=["kyivska"])
    g["alert"] = 0
    fm = features.build_feature_matrix(g, {})
    degraded = operational_eval.degrade_features(fm, lag_hours=6)
    pd.testing.assert_frame_equal(degraded, fm)


def test_operational_eval_shape_and_gap_direction():
    fm = _grid_with_threat_cols()
    results = operational_eval.operational_eval(
        fm, lag_hours_list=[3, 12], test_weeks=2, horizons=["1h"]
    )
    assert set(results.columns) == {
        "lag_hours", "horizon", "pr_auc_full", "pr_auc_degraded", "gap", "gap_pct"
    }
    assert len(results) == 2   # 2 lags x 1 horizon
    assert set(results["lag_hours"]) == {3, 12}
    # Full features should be >= degraded (degradation can't help, though gap may be ~0
    # when threat cols have no signal in synthetic data).
    assert (results["gap"] >= -1e-9).all()


def test_operational_summary_pivot():
    fm = _grid_with_threat_cols()
    results = operational_eval.operational_eval(
        fm, lag_hours_list=[3, 6], test_weeks=2, horizons=["1h", "6h"]
    )
    summary = operational_eval.operational_summary(results)
    assert list(summary.index) == ["1h", "6h"]
    assert set(summary.columns) == {3, 6}


def test_operational_summary_empty():
    assert operational_eval.operational_summary(pd.DataFrame()).empty
