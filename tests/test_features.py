"""features: leak-safety (issue #1) + builder correctness on synthetic grids."""

import pandas as pd

from src import config, features, index


def _grid(start, end, oblasts):
    g = index.build_master_index(start=start, end=end, oblasts=oblasts)
    g["alert"] = 0
    return g


def test_calendar_known_values():
    g = _grid("2024-05-04 23:00", "2024-05-05 02:00", ["kyivska"])  # Sat 23:00 -> Sun 02:00
    out = features.add_calendar_features(g)
    row = out.xs("kyivska", level="oblast")
    assert row.loc["2024-05-04 23:00+00:00", "hour"] == 23
    assert row.loc["2024-05-04 23:00+00:00", "is_night"] == 1     # 23:00 is night
    assert row.loc["2024-05-04 23:00+00:00", "is_weekend"] == 1   # Saturday
    assert row.loc["2024-05-05 02:00+00:00", "is_night"] == 1     # 02:00 is night
    # day-of-war from 2022-02-24
    assert row.loc["2024-05-05 02:00+00:00", "day_of_war"] == (
        pd.Timestamp("2024-05-05", tz="UTC") - pd.Timestamp(config.WAR_START, tz="UTC")
    ).days


def test_lag_uses_only_past():
    g = _grid("2024-05-01 00:00", "2024-05-01 05:00", ["kyivska"])
    # alert on at 01:00 and 02:00 only
    s = g.copy()
    ts = s.index.get_level_values("ts_utc")
    s.loc[(ts.hour == 1) | (ts.hour == 2), "alert"] = 1
    out = features.add_lag_features(s)
    r = out.xs("kyivska", level="oblast")
    # lag_1h at 02:00 sees 01:00 (==1); at 01:00 sees 00:00 (==0); never the current hour
    assert r.loc["2024-05-01 02:00+00:00", "alert_lag_1h"] == 1
    assert r.loc["2024-05-01 01:00+00:00", "alert_lag_1h"] == 0
    assert pd.isna(r.loc["2024-05-01 00:00+00:00", "alert_lag_1h"])  # no prior history


def _waves(rows):
    return pd.DataFrame(rows, columns=["time_start_utc", "channels", "oblasts", "launched"])


def _enable_threat(monkeypatch):
    """Threat is DEPRECATED in prod (config.THREAT_CHANNELS=()); re-enable the full set
    so the builder's leak-safety stays under test for the planned onset-model revival."""
    monkeypatch.setattr(config, "THREAT_CHANNELS", ("ballistic", "drone-strike", "air-cruise"))
    monkeypatch.setattr(config, "THREAT_VALUES", ("launched", "waves"))
    monkeypatch.setattr(config, "THREAT_WINDOWS", (3, 6, 24))


def test_threat_no_same_hour_leak(monkeypatch):
    """A wave launched in hour t must NOT count in thr_*_3h at t, only from t+1."""
    _enable_threat(monkeypatch)
    g = _grid("2024-05-01 00:00", "2024-05-01 04:00", ["kyivska"])
    waves = _waves([
        (pd.Timestamp("2024-05-01 02:30", tz="UTC"), {"ballistic"}, ["kyivska"], 5.0),
    ])
    out = features.add_threat_features(g, waves)
    r = out.xs("kyivska", level="oblast")
    assert r.loc["2024-05-01 02:00+00:00", "thr_ballistic_launched_3h"] == 0  # before wave
    assert r.loc["2024-05-01 03:00+00:00", "thr_ballistic_launched_3h"] == 5  # hour after
    assert r.loc["2024-05-01 03:00+00:00", "thr_ballistic_waves_3h"] == 1


def test_threat_national_broadcast(monkeypatch):
    """A wave with empty oblasts list reaches every oblast."""
    _enable_threat(monkeypatch)
    g = _grid("2024-05-01 00:00", "2024-05-01 03:00", ["kyivska", "lvivska"])
    waves = _waves([
        (pd.Timestamp("2024-05-01 00:30", tz="UTC"), {"drone-strike"}, [], 10.0),
    ])
    out = features.add_threat_features(g, waves)
    for ob in ("kyivska", "lvivska"):
        r = out.xs(ob, level="oblast")
        assert r.loc["2024-05-01 01:00+00:00", "thr_drone-strike_launched_3h"] == 10


def test_ucdp_prior_lagged_and_per_oblast():
    """UCDP prior at year Y reads only years < Y (leak-safe) and separates oblasts."""
    import numpy as np
    g = _grid("2024-05-01 00:00", "2024-05-01 02:00", ["donetska", "lvivska"])
    ucdp = pd.DataFrame({
        "oblast": ["donetska", "donetska", "donetska", "lvivska"],
        "year":   [2022,        2023,        2024,        2022],
        "deaths": [100,         50,          9999,        1],     # 2024 must NOT leak
        "events": [10,          5,           99,          1],
    })
    out = features.add_ucdp_features(g, ucdp)
    don = out.xs("donetska", level="oblast")["ucdp_deaths_prior"].iloc[0]
    lvi = out.xs("lvivska", level="oblast")["ucdp_deaths_prior"].iloc[0]
    assert abs(don - np.log1p(150)) < 1e-9      # 2022+2023 only, 2024 excluded
    assert abs(lvi - np.log1p(1)) < 1e-9
    assert don > lvi                            # frontline >> western prior


def test_ucdp_none_yields_zero_columns():
    g = _grid("2024-05-01 00:00", "2024-05-01 02:00", ["kyivska"])
    out = features.add_ucdp_features(g, None)
    assert (out["ucdp_deaths_prior"] == 0).all()
    assert (out["ucdp_events_prior"] == 0).all()


def test_tempo_uses_previous_day():
    g = _grid("2024-05-02 00:00", "2024-05-02 02:00", ["kyivska"])
    daily = pd.DataFrame({
        "date_utc": pd.to_datetime(["2024-05-01", "2024-05-02"]).tz_localize("UTC"),
        "launched": [100, 999], "destroyed": [50, 0], "n_waves": [2, 9],
    })
    out = features.add_tempo_features(g, daily)
    r = out.xs("kyivska", level="oblast")
    # cells on 2024-05-02 see the 05-01 tempo (100), never the same-day 999
    assert (r["tempo_prev_launched"] == 100).all()


def test_build_requires_alert_column():
    g = index.build_master_index(start="2024-05-01", end="2024-05-01 02:00", oblasts=["kyivska"])
    try:
        features.build_feature_matrix(g, {})
        assert False, "expected ValueError for missing alert column"
    except ValueError:
        pass
