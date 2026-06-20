"""model_b: target windowing (future-only), horizon steps, train/predict smoke."""

import numpy as np
import pandas as pd

from src import config, index, model_b


def _grid(start, end, oblasts):
    g = index.build_master_index(start=start, end=end, oblasts=oblasts)
    g["alert"] = 0
    return g


def test_horizon_steps_hourly_grid():
    # 30m and 1h both collapse to 1 step on the hourly grid (documented limit).
    assert model_b.horizon_steps("30m") == 1
    assert model_b.horizon_steps("1h") == 1
    assert model_b.horizon_steps("3h") == 3
    assert model_b.horizon_steps("6h") == 6


def test_make_target_future_window_only():
    g = _grid("2024-05-01 00:00", "2024-05-01 06:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour == 4, "alert"] = 1                  # single alert at 04:00
    y = model_b.make_target(g, "3h").xs("kyivska", level="oblast")
    # window (t, t+3h] hits 04:00 from t = 01:00, 02:00, 03:00 only
    assert y.loc["2024-05-01 01:00+00:00"] == 1
    assert y.loc["2024-05-01 03:00+00:00"] == 1
    assert y.loc["2024-05-01 00:00+00:00"] == 0       # 04:00 is 4h ahead, outside
    assert y.loc["2024-05-01 04:00+00:00"] == 0       # never counts the current/past hour
    assert pd.isna(y.loc["2024-05-01 06:00+00:00"])   # tail: no future window


def test_feature_columns_excludes_label_and_targets():
    X = pd.DataFrame({"alert": [0], "target_1h": [1], "hour": [3], "thr_x_3h": [0]})
    assert model_b.feature_columns(X) == ["hour", "thr_x_3h"]


def test_train_predict_smoke():
    # Two oblasts, a clean signal feature so LightGBM has something to split on.
    g = _grid("2024-05-01 00:00", "2024-05-05 23:00", ["kyivska", "lvivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([2, 3, 4]), "alert"] = 1
    X = g.copy()
    X["hour"] = ts.hour
    X["sig"] = (ts.hour + 1).astype(float)

    models = model_b.train_all(X, g, horizons=["1h", "6h"])
    assert set(models) == {"1h", "6h"}
    probs = model_b.predict_all(models, X)
    assert list(probs.columns) == ["1h", "6h"]
    assert probs.index.equals(X.index)
    assert ((probs.values >= 0) & (probs.values <= 1)).all()


def test_predict_all_calibrators_applied():
    # A calibrator maps raw prob -> calibrated; predict_all must route through it.
    from src import evaluate

    g = _grid("2024-05-01 00:00", "2024-05-05 23:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([2, 3, 4]), "alert"] = 1
    X = g.copy()
    X["sig"] = (ts.hour + 1).astype(float)

    models = model_b.train_all(X, g, horizons=["1h"])
    raw = model_b.predict_all(models, X)
    iso = evaluate.fit_isotonic(g["alert"].to_numpy(), raw["1h"].to_numpy())
    cal = model_b.predict_all(models, X, calibrators={"1h": iso})

    assert ((cal.values >= 0) & (cal.values <= 1)).all()
    # isotonic is monotone -> sorting by raw prob leaves calibrated non-decreasing
    order = np.argsort(raw["1h"].to_numpy())
    assert np.all(np.diff(cal["1h"].to_numpy()[order]) >= -1e-9)
