"""retrain: walk-forward log shape, policy behaviour, drift-triggered adaptation."""

import numpy as np
import pandas as pd

from src import config, index, retrain

_FAST = {"n_estimators": 15, "num_leaves": 7, "min_child_samples": 5}
_KW = dict(horizon="6h", quantiles=(0.1, 0.5, 0.9), params=_FAST,
           window_days=60, block_days=20)


def _drifting_fm():
    """1 oblast, 210 days hourly. `sig` distribution jumps at the midpoint -> drift."""
    g = index.build_master_index(start="2024-01-01", end="2024-07-29 00:00",
                                 oblasts=["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g["alert"] = ts.hour.isin([1, 2, 3]).astype("int8")
    half = ts.min() + (ts.max() - ts.min()) / 2
    # feature shifts hard in the second half -> large reference-vs-block PSI
    g["sig"] = ts.hour.astype(float) + np.where(ts >= half, 50.0, 0.0)
    return g


def test_log_shape_and_temporal_order():
    log = retrain.walk_forward_retrain(_drifting_fm(), policy="never", **_KW)
    assert {"block_start", "block_end", "pinball", "coverage", "width",
            "psi", "retrained", "n_train"} <= set(log.columns)
    assert log["block_start"].is_monotonic_increasing
    assert (log["n_train"] > 0).all()                 # every block had a fitted model
    assert ((log["coverage"] >= 0) & (log["coverage"] <= 1)).all()


def test_never_policy_does_not_retrain():
    log = retrain.walk_forward_retrain(_drifting_fm(), policy="never", **_KW)
    assert not log["retrained"].any()
    assert log["n_train"].nunique() == 1              # frozen: one training set throughout


def test_periodic_policy_retrains_on_cadence():
    log = retrain.walk_forward_retrain(_drifting_fm(), policy="periodic",
                                       period_blocks=2, **_KW)
    # every 2nd block flips the retrain flag
    flagged = np.where(log["retrained"].to_numpy())[0]
    assert flagged.size >= 1
    assert all((i + 1) % 2 == 0 for i in flagged)


def test_drift_policy_fires_on_shift():
    log = retrain.walk_forward_retrain(_drifting_fm(), policy="drift", **_KW)
    assert log["retrained"].any()                     # the midpoint jump must trigger
    # a retrain block must coincide with a PSI past the alert threshold
    fired = log[log["retrained"]]
    assert (fired["psi"] >= config.DRIFT_PSI_ALERT).all()


def test_short_history_raises():
    g = index.build_master_index(start="2024-01-01", end="2024-01-20",
                                 oblasts=["kyivska"])
    g["alert"] = 0
    g["sig"] = 1.0
    try:
        retrain.walk_forward_retrain(g, policy="never", **_KW)
        assert False, "expected ValueError for too-short history"
    except ValueError:
        pass


def test_compare_policies_returns_all():
    out = retrain.compare_policies(_drifting_fm(),
                                   policies=("never", "drift"), **_KW)
    assert set(out) == {"never", "drift"}
    assert all(isinstance(v, pd.DataFrame) for v in out.values())
