"""Operational eval (Phase 2): backtest-vs-live gap from the ragged right edge.

Quantifies how much PR-AUC drops when launch/tempo features are stale (source publish
lag). Sweeps multiple lag scenarios (3h, 6h, 12h, 24h) and prints the gap per horizon.
This is the headline number that says "our backtest claims X but live we'd get Y".

Run: PYTHONUTF8=1 python scripts/runs/run_operational_eval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Partial moved under scripts/runs/ — put repo root on sys.path so `from src import`
# resolves when run standalone (the combined base run.py imports src directly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src import config, features, index, loaders, operational_eval, viz_export


def main() -> None:
    t0 = time.time()
    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()

    grid = index.build_master_index()  # starts at config.GRID_START (2023-07)
    grid = index.expand_alerts_to_grid(grid, alerts)
    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily})
    print(f"features {fm.shape[1]} cols  {len(fm):,} rows  [{time.time()-t0:.0f}s]")

    print(f"\noperational eval: lags {operational_eval.OPERATIONAL_LAG_HOURS}h  "
          f"test {config.TEST_WEEKS}wk")
    results = operational_eval.operational_eval(fm, progress=True)
    summary = operational_eval.operational_summary(results)

    print(f"\nPR-AUC gap (%) by horizon x source lag  [{time.time()-t0:.0f}s]")
    print("(positive = backtest overstatement vs live)")
    print(summary.round(2).to_string())

    # viz: operational.json — "operational (backtest-vs-live gap)" evaluation method
    by_h = {}
    for h in config.HORIZONS:
        sub = results[results["horizon"] == h]
        if sub.empty:
            continue
        by_h[h] = {
            "pr_auc_full": viz_export.serializable(float(sub["pr_auc_full"].iloc[0])),
            "by_lag": {
                int(r["lag_hours"]): {
                    "pr_auc_degraded": viz_export.serializable(float(r["pr_auc_degraded"])),
                    "gap_pct": viz_export.serializable(float(r["gap_pct"])),
                }
                for _, r in sub.iterrows()
            },
        }
    viz_export.write_viz_json("operational.json", {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "test_weeks": int(config.TEST_WEEKS),
        "lag_hours": list(operational_eval.OPERATIONAL_LAG_HOURS),
        "by_horizon": by_h,
    })
    print(f"\ndone  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
