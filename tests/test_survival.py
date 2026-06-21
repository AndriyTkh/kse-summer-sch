"""survival: build_survival_dataset, KM baseline, Cox model, temporal split."""

import numpy as np
import pandas as pd
import pytest

from src import config, features, index, survival

lifelines = pytest.importorskip("lifelines")


def _alert_events(n=20, oblasts=("kyivska", "lvivska")):
    """Synthetic alert events spread across a 10-week window."""
    rng = np.random.default_rng(99)
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i in range(n):
        ob = oblasts[i % len(oblasts)]
        start = base + pd.Timedelta(hours=rng.integers(0, 10 * 7 * 24))
        dur = rng.exponential(scale=3.0)  # ~3h mean
        end = start + pd.Timedelta(hours=dur)
        rows.append({"oblast": ob, "start_utc": start, "end_utc": end})
    return pd.DataFrame(rows)


def _feature_matrix(oblasts=("kyivska", "lvivska"), weeks=12):
    end = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=weeks) - pd.Timedelta(hours=1)
    g = index.build_master_index(start="2024-01-01", end=end, oblasts=list(oblasts))
    hour = g.index.get_level_values("ts_utc").hour
    g["alert"] = (hour % 8 == 0).astype("int8")
    return features.build_feature_matrix(g, {})


def test_build_survival_dataset_shape_and_columns():
    alerts = _alert_events()
    fm = _feature_matrix()
    surv = survival.build_survival_dataset(alerts, fm)
    assert len(surv) > 0
    assert "duration_hours" in surv.columns
    assert "observed" in surv.columns
    assert (surv["duration_hours"] > 0).all()
    assert surv["observed"].isin([0, 1]).all()
    # should have some feature columns from fm
    feature_cols = [c for c in surv.columns if c.startswith("alert_lag") or c == "hour"]
    assert len(feature_cols) > 0


def test_build_survival_dataset_censoring():
    """An alert that ends after the data cutoff should be censored."""
    alerts = pd.DataFrame([{
        "oblast": "kyivska",
        "start_utc": pd.Timestamp("2024-01-15 10:00", tz="UTC"),
        "end_utc": pd.Timestamp("2024-12-31 00:00", tz="UTC"),  # way past data
    }])
    fm = _feature_matrix(weeks=4)
    surv = survival.build_survival_dataset(alerts, fm)
    assert len(surv) == 1
    assert surv.iloc[0]["observed"] == 0


def test_build_survival_dataset_uncensored():
    """An alert that ends within the data window should be uncensored."""
    alerts = pd.DataFrame([{
        "oblast": "kyivska",
        "start_utc": pd.Timestamp("2024-01-10 10:00", tz="UTC"),
        "end_utc": pd.Timestamp("2024-01-10 12:00", tz="UTC"),
    }])
    fm = _feature_matrix(weeks=4)
    surv = survival.build_survival_dataset(alerts, fm)
    assert len(surv) == 1
    assert surv.iloc[0]["observed"] == 1
    assert abs(surv.iloc[0]["duration_hours"] - 2.0) < 0.01


def test_km_baseline_median():
    alerts = _alert_events(n=50)
    fm = _feature_matrix()
    surv = survival.build_survival_dataset(alerts, fm)
    kmf = survival.km_baseline(surv)
    assert hasattr(kmf, "median_survival_time_")
    assert kmf.median_survival_time_ > 0


def test_cox_model_fits_and_has_concordance():
    alerts = _alert_events(n=80)
    fm = _feature_matrix()
    surv = survival.build_survival_dataset(alerts, fm)
    cph = survival.cox_model(surv)
    assert 0 < cph.concordance_index_ <= 1
    assert len(cph.summary) > 0


def test_predict_duration_returns_series():
    alerts = _alert_events(n=80)
    fm = _feature_matrix()
    surv = survival.build_survival_dataset(alerts, fm)
    train, test = survival.temporal_split_events(surv, test_weeks=2)
    if len(train) < 5 or train["observed"].sum() == 0:
        pytest.skip("not enough training events")
    cph = survival.cox_model(train)
    pred = survival.predict_duration(cph, test)
    assert isinstance(pred, pd.Series)
    assert len(pred) == len(test)
    assert pred.name == "predicted_median_hours"


def test_temporal_split_events_is_temporal():
    alerts = _alert_events(n=40)
    fm = _feature_matrix()
    surv = survival.build_survival_dataset(alerts, fm)
    train, test = survival.temporal_split_events(surv, test_weeks=2)
    if not train.empty and not test.empty:
        assert train["start_utc"].max() < test["start_utc"].min()
    assert len(train) + len(test) == len(surv)
