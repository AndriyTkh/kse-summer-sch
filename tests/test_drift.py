"""drift: PSI is ~0 for same dist, large under shift, robust to constants/labels."""

import numpy as np
import pandas as pd

from src import config, drift


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    assert drift.psi(x, y) < 0.1                 # same dist -> stable band


def test_psi_large_under_mean_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)                 # shifted 3 sigma
    assert drift.psi(ref, cur) >= config.DRIFT_PSI_ALERT


def test_psi_constant_feature_is_zero():
    assert drift.psi(np.ones(100), np.ones(100)) == 0.0
    assert drift.psi(np.ones(100), np.zeros(100)) == 0.0   # no reference spread to bin


def test_psi_handles_nan():
    ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    cur = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    assert drift.psi(ref, cur) >= 0.0            # finite, no crash


def test_feature_psi_excludes_label_and_targets():
    rng = np.random.default_rng(2)
    idx = pd.RangeIndex(2000)
    ref = pd.DataFrame({
        "feat": rng.normal(0, 1, 2000),
        "alert": rng.integers(0, 2, 2000),
        "target_1h": rng.integers(0, 2, 2000),
        "frac_6h": rng.random(2000),
    }, index=idx)
    cur = ref.copy()
    s = drift.feature_psi(ref, cur)
    assert list(s.index) == ["feat"]             # only the real input feature scored


def test_drift_score_bands():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 4000), "b": rng.normal(0, 1, 4000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 4000), "b": rng.normal(4, 1, 4000)})
    d = drift.drift_score(ref, cur)
    assert d["top"] == "b"                        # b is the shifted feature
    assert d["band"] == "significant"
    assert d["n_significant"] >= 1


def test_classify_psi_thresholds():
    assert drift.classify_psi(0.0) == "stable"
    assert drift.classify_psi(0.15) == "moderate"
    assert drift.classify_psi(0.5) == "significant"
