"""model_a: daily resampling + Prophet baseline broadcast to the hourly grid.

Prophet fit is slow and needs cmdstan; gate the live-fit test behind import so the
fast suite still runs without it. to_daily / broadcast logic is tested fit-free.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, index, model_a


def _grid(start, end, oblasts):
    g = index.build_master_index(start=start, end=end, oblasts=oblasts)
    g["alert"] = 0
    return g


def test_to_daily_rate_and_tz_naive():
    g = _grid("2024-05-01 00:00", "2024-05-02 23:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[(ts.normalize() == "2024-05-01") & (ts.hour < 12), "alert"] = 1  # 12/24 on day 1
    daily = model_a.to_daily(g)
    assert list(daily.columns) == ["oblast", "ds", "y"]
    assert daily["ds"].dt.tz is None                       # Prophet needs tz-naive
    d1 = daily.loc[daily["ds"] == "2024-05-01", "y"].iloc[0]
    assert d1 == pytest.approx(0.5)                         # 12 of 24 hours active
    assert (daily["y"].between(0, 1)).all()


@pytest.mark.skipif(
    pytest.importorskip("prophet", reason="prophet/cmdstan not installed") is None,
    reason="prophet not installed",
)
def test_baseline_for_grid_shape_and_range():
    # Small two-oblast grid; B-style train/test slabs by hand.
    g = _grid("2024-01-01", "2024-03-10 23:00", ["kyivska", "lvivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.weekday < 2, "alert"] = 1                     # weekly signal for Prophet
    train = g[ts < pd.Timestamp("2024-03-01", tz="UTC")]
    test = g[ts >= pd.Timestamp("2024-03-01", tz="UTC")]

    a_pred = model_a.baseline_for_grid(train, test, horizons=["1h", "6h"])
    assert a_pred.index.equals(test.index)
    assert list(a_pred.columns) == ["1h", "6h"]
    assert ((a_pred.values >= 0) & (a_pred.values <= 1)).all()
    # A is horizon-flat: every horizon column identical
    assert (a_pred["1h"].to_numpy() == a_pred["6h"].to_numpy()).all()
