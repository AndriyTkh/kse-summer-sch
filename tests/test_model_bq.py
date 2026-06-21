"""model_bq: fraction target (future-only), quantile monotonicity, train/predict smoke."""

import numpy as np
import pandas as pd

from src import config, index, model_bq

_FAST = {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5}


def _grid(start, end, oblasts):
    g = index.build_master_index(start=start, end=end, oblasts=oblasts)
    g["alert"] = 0
    return g


def test_fraction_target_is_window_mean():
    g = _grid("2024-05-01 00:00", "2024-05-01 08:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    # alerts at 02:00 and 03:00
    g.loc[ts.hour.isin([2, 3]), "alert"] = 1
    y = model_bq.make_fraction_target(g, "3h").xs("kyivska", level="oblast")
    # from t=00: window (01,02,03] hits 02 and 03 -> 2/3
    assert y.loc["2024-05-01 00:00+00:00"] == 2 / 3
    # from t=01: window (02,03,04] hits 02,03 -> 2/3
    assert y.loc["2024-05-01 01:00+00:00"] == 2 / 3
    # from t=03: window (04,05,06] hits none -> 0
    assert y.loc["2024-05-01 03:00+00:00"] == 0.0
    # tail row with no future window at all -> NaN
    assert pd.isna(y.loc["2024-05-01 08:00+00:00"])


def test_fraction_target_in_unit_range():
    g = _grid("2024-05-01 00:00", "2024-05-03 00:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([1, 2, 3, 4]), "alert"] = 1
    y = model_bq.make_fraction_target(g, "6h").dropna()
    assert ((y >= 0) & (y <= 1)).all()


def test_predict_quantiles_monotone_and_clipped():
    g = _grid("2024-05-01 00:00", "2024-05-06 23:00", ["kyivska", "lvivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([2, 3, 4]), "alert"] = 1
    X = g.copy()
    X["sig"] = (ts.hour + 1).astype(float)

    models = model_bq.train_all_quantiles(X, g, horizons=["6h"], params=_FAST)
    preds = model_bq.predict_quantiles(models, X)

    lo, med, hi = model_bq.interval_columns("6h")
    assert {lo, med, hi} <= set(preds.columns)
    vals = preds[[lo, med, hi]].to_numpy()
    assert (vals >= 0).all() and (vals <= 1).all()          # clipped to fraction range
    assert np.all(np.diff(vals, axis=1) >= -1e-9)           # q10 <= q50 <= q90 per row


def test_train_all_quantiles_structure():
    g = _grid("2024-05-01 00:00", "2024-05-04 23:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([2, 3]), "alert"] = 1
    X = g.copy()
    X["sig"] = (ts.hour + 1).astype(float)

    models = model_bq.train_all_quantiles(X, g, horizons=["1h", "6h"],
                                          quantiles=(0.1, 0.5, 0.9), params=_FAST)
    assert set(models) == {"1h", "6h"}
    assert set(models["6h"]) == {0.1, 0.5, 0.9}


def test_q_label_formatting():
    assert model_bq.q_label(0.1) == "q10"
    assert model_bq.q_label(0.5) == "q50"
    assert model_bq.q_label(0.05) == "q05"
