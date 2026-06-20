"""evaluate: temporal split (never random) + PR-AUC + calibration/heatmap/compare."""

import matplotlib
matplotlib.use("Agg")          # headless: no display backend in CI

import numpy as np
import pandas as pd

from src import config, evaluate, index


def test_temporal_split_cut_is_time_based():
    g = index.build_master_index(
        start="2024-01-01", end="2024-03-25 23:00", oblasts=["kyivska", "lvivska"]
    )
    train, test = evaluate.temporal_split(g, test_weeks=4)
    tr = train.index.get_level_values("ts_utc")
    te = test.index.get_level_values("ts_utc")
    assert tr.max() < te.min()                         # no temporal overlap
    span = te.max() - te.min()
    assert pd.Timedelta(days=27) <= span <= pd.Timedelta(weeks=4)   # ~last 4 weeks
    assert len(train) + len(test) == len(g)


def test_pr_auc_perfect_and_baseline():
    y = [0, 0, 1, 1]
    assert evaluate.pr_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0   # perfect ranking
    # all-equal scores -> average precision collapses to the positive rate
    assert evaluate.pr_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_ece_perfect_calibration_is_zero():
    # scores equal to outcomes -> zero calibration error
    y = np.array([0, 0, 1, 1])
    assert evaluate.expected_calibration_error(y, y.astype(float)) == 0.0


def test_calibration_plot_returns_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.random(500)
    ax, m = evaluate.calibration_plot(y, s, n_bins=5)
    assert {"ece", "brier"} <= set(m)
    assert 0 <= m["ece"] <= 1 and 0 <= m["brier"] <= 1


def test_fit_isotonic_corrects_scale_keeps_ranking():
    # Inflated scores (e.g. scale_pos_weight) with good ranking: isotonic should pull
    # ECE down while leaving PR-AUC (ranking) intact.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    # raw score correlated with y but biased high -> miscalibrated
    s = np.clip(0.5 + 0.3 * y + rng.normal(0, 0.15, 2000), 0, 1)
    iso = evaluate.fit_isotonic(y, s)
    cal = iso.predict(s)
    assert (cal >= 0).all() and (cal <= 1).all()
    assert evaluate.expected_calibration_error(y, cal) <= evaluate.expected_calibration_error(y, s)
    # isotonic is monotone non-decreasing in the raw score (flats may create ties)
    order = np.argsort(s)
    assert np.all(np.diff(cal[order]) >= -1e-9)
    # ranking preserved up to ties -> PR-AUC essentially unchanged
    assert abs(evaluate.pr_auc(y, cal) - evaluate.pr_auc(y, s)) < 0.02


def test_fit_isotonic_clips_out_of_range_scores():
    # scores outside the fitted range must stay in [0, 1] (out_of_bounds="clip")
    iso = evaluate.fit_isotonic([0, 0, 1, 1], [0.2, 0.3, 0.6, 0.7])
    out = iso.predict([-0.5, 0.0, 1.0, 2.0])
    assert (out >= 0).all() and (out <= 1).all()


def test_compare_b_vs_a_table():
    idx = pd.MultiIndex.from_product([["kyivska"], pd.date_range("2024-05-01", periods=8, freq="h", tz="UTC")], names=["oblast", "ts_utc"])
    y = pd.DataFrame({"1h": [0, 0, 0, 0, 1, 1, 1, 1]}, index=idx)
    b = pd.DataFrame({"1h": [.1, .2, .3, .4, .6, .7, .8, .9]}, index=idx)  # perfect ranking
    a = pd.DataFrame({"1h": [.5] * 8}, index=idx)                          # no signal
    out = evaluate.compare_b_vs_a(b, a, y)
    assert out.loc["1h", "pr_auc_b"] == 1.0
    assert out.loc["1h", "pr_auc_b"] > out.loc["1h", "pr_auc_a"]


def test_heatmap_table_shape():
    idx = pd.MultiIndex.from_product(
        [["kyivska", "lvivska"], pd.date_range("2024-05-01", periods=4, freq="h", tz="UTC")],
        names=["oblast", "ts_utc"],
    )
    probs = pd.DataFrame({h: np.linspace(0, 1, 8) for h in config.HORIZONS}, index=idx)
    ax, table = evaluate.oblast_horizon_heatmap(probs)
    assert list(table.index) == ["kyivska", "lvivska"]
    assert list(table.columns) == config.HORIZONS
