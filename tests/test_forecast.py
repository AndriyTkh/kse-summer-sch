"""forecast_now: operational nowcast at the grid edge (one row per oblast)."""

import pandas as pd

from src import config, features, forecast, index, model_b


def _periodic_grid(weeks=8, oblasts=("kyivska", "lvivska", "odeska")):
    end = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=weeks) - pd.Timedelta(hours=1)
    g = index.build_master_index(start="2024-01-01", end=end, oblasts=list(oblasts))
    hour = g.index.get_level_values("ts_utc").hour
    g["alert"] = (hour % 8 == 0).astype("int8")
    return features.build_feature_matrix(g, {})


def test_latest_rows_is_origin_per_oblast():
    fm = _periodic_grid()
    edge = forecast.latest_rows(fm)
    assert len(edge) == fm.index.get_level_values("oblast").nunique()
    # each picked row is its oblast's max timestamp (rectangular grid -> the global max)
    gmax = fm.index.get_level_values("ts_utc").max()
    assert (edge.index.get_level_values("ts_utc") == gmax).all()
    # ...and its target is unknown (future window runs off the grid)
    y = model_b.make_target(fm, "1h").reindex(fm.index)
    assert y.reindex(edge.index).isna().all()


def test_forecast_now_shape_and_range():
    fm = _periodic_grid()
    table = forecast.forecast_now(fm)
    # one row per oblast, indexed by oblast
    assert table.index.name == "oblast"
    assert len(table) == fm.index.get_level_values("oblast").nunique()
    assert list(table.columns) == ["origin_utc"] + config.HORIZONS
    for h in config.HORIZONS:
        assert ((table[h] >= 0) & (table[h] <= 1)).all()
    # ranked by the longest horizon, descending
    assert table[config.HORIZONS[-1]].is_monotonic_decreasing


def test_forecast_now_applies_calibrators():
    fm = _periodic_grid()
    # a degenerate calibrator that maps everything to 0 must flow through to the output
    from sklearn.isotonic import IsotonicRegression
    zero = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=0.0)
    zero.fit([0.0, 1.0], [0.0, 0.0])
    cals = {h: zero for h in config.HORIZONS}
    table = forecast.forecast_now(fm, calibrators=cals)
    for h in config.HORIZONS:
        assert (table[h] == 0).all()


def test_format_forecast_carries_caveat_and_calibration_flag():
    fm = _periodic_grid()
    table = forecast.forecast_now(fm)
    raw = forecast.format_forecast(table, calibrated=False)
    cal = forecast.format_forecast(table, calibrated=True)
    assert "ragged right edge" in raw
    assert "RAW model scores" in raw
    assert "CALIBRATED" in cal
