"""Operational eval — quantify the backtest-vs-live gap from the ragged right edge.

The core insight (STRUCTURE §6): backtest PR-AUC overstates live performance not because
of leakage (the timestamp guard is correct) but because of DATA AVAILABILITY. Live, the
launch/tempo sources publish with lag — the most recent hours of threat/tempo features are
missing or stale at forecast time. The historical CSV has them complete, so the backtest
sees a world that doesn't exist live.

This module simulates that gap WITHOUT real vintage snapshots: for each test-fold origin
it masks the last `lag_hours` of threat/tempo features (zeroing them as if the source
hadn't refreshed yet), rebuilds Model B predictions, and compares PR-AUC to the
non-degraded backtest. The delta is the **estimated operational gap** — a lower bound on
how much live performance drops vs the backtest headline.

When real vintage snapshots become available (Phase-3 data pipeline), replace the
simulated masking with actual stale data and re-run the same scoring path.

Config:
  OPERATIONAL_LAG_HOURS — list of lag scenarios to sweep (default [3, 6, 12, 24]).
  Each lag simulates a source that last refreshed `lag_hours` before the test window.
"""

from __future__ import annotations

import pandas as pd

from . import config

OPERATIONAL_LAG_HOURS: list[int] = [3, 6, 12, 24]

_THREAT_PREFIX = "thr_"
_TEMPO_PREFIX = "tempo_"


def _is_degradable(col: str) -> bool:
    return col.startswith(_THREAT_PREFIX) or col.startswith(_TEMPO_PREFIX)


def degrade_features(fm: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """Zero out threat/tempo features in the last `lag_hours` of the feature matrix.

    Simulates a source that hasn't refreshed: the grid rows whose ts_utc falls within
    `lag_hours` of the maximum timestamp get their threat/tempo columns set to 0 (the
    same value `features.py` fills for missing data). Alert-lag and calendar features are
    untouched — the alert stream is assumed real-time (sirens are immediate).
    """
    if lag_hours <= 0:
        return fm
    ts = fm.index.get_level_values("ts_utc")
    cutoff = ts.max() - pd.Timedelta(hours=lag_hours)
    mask = ts >= cutoff
    cols = [c for c in fm.columns if _is_degradable(c)]
    if not cols or not mask.any():
        return fm
    out = fm.copy()
    out.loc[mask, cols] = 0.0
    return out


def operational_eval(
    fm: pd.DataFrame,
    *,
    lag_hours_list: list[int] | None = None,
    test_weeks: int = config.TEST_WEEKS,
    horizons=None,
    progress: bool = False,
) -> pd.DataFrame:
    """Compare backtest PR-AUC (full features) vs degraded PR-AUC across lag scenarios.

    For each lag in `lag_hours_list`:
      1. Split fm into train / test (temporal, purged).
      2. Train Model B on the FULL train set (the model is trained with complete data —
         the degradation only affects the test-side features, simulating what the model
         would see live at inference time).
      3. Score test with FULL features (backtest) and DEGRADED features (operational).
      4. Record PR-AUC for both, plus the gap.

    Returns a tidy long frame [lag_hours, horizon, pr_auc_full, pr_auc_degraded, gap].
    """
    from . import evaluate, model_b

    lag_hours_list = lag_hours_list or OPERATIONAL_LAG_HOURS
    horizons = horizons or config.HORIZONS

    train, test = evaluate.temporal_split(fm, test_weeks=test_weeks)
    models = model_b.train_all(train, train, horizons=horizons)
    probs_full = model_b.predict_all(models, test)

    rows = []
    for lag in lag_hours_list:
        test_degraded = degrade_features(test, lag)
        probs_deg = model_b.predict_all(models, test_degraded)

        for h in horizons:
            y = model_b.make_target(test, h).reindex(test.index)
            m = y.notna()
            yt = y[m].astype(int)
            if yt.nunique() < 2:
                continue
            ap_full = evaluate.pr_auc(yt, probs_full.loc[m, h])
            ap_deg = evaluate.pr_auc(yt, probs_deg.loc[m, h])
            gap = ap_full - ap_deg
            rows.append({
                "lag_hours": lag,
                "horizon": h,
                "pr_auc_full": ap_full,
                "pr_auc_degraded": ap_deg,
                "gap": gap,
                "gap_pct": 100 * gap / ap_full if ap_full else float("nan"),
            })
            if progress:
                print(f"  lag {lag:>3}h  {h:>4}  full={ap_full:.3f}  "
                      f"degraded={ap_deg:.3f}  gap={gap:+.4f} ({100*gap/ap_full:+.1f}%)")

    return pd.DataFrame(rows)


def operational_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot operational eval results: one row per horizon, columns per lag scenario.

    Produces a compact table showing PR-AUC gap (pct) at each lag, making it easy to
    read "at 6h source lag, the 1h-horizon model loses X% PR-AUC".
    """
    if results.empty:
        return pd.DataFrame()
    piv = results.pivot_table(
        index="horizon", columns="lag_hours", values="gap_pct", aggfunc="first"
    )
    order = [h for h in config.HORIZONS if h in piv.index]
    return piv.loc[order]
