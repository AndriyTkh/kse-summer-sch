"""Operational eval (Phase 2): backtest-vs-live gap from the ragged right edge.

Quantifies how much PR-AUC drops when launch/tempo features are stale (source publish
lag). Sweeps multiple lag scenarios (3h, 6h, 12h, 24h) and prints the gap per horizon.
This is the headline number that says "our backtest claims X but live we'd get Y".

Run: PYTHONUTF8=1 python run_operational_eval.py
"""

from __future__ import annotations

import time

from src import config, features, index, loaders, operational_eval


def main() -> None:
    t0 = time.time()
    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()

    start = alerts["start_utc"].min().floor("h").tz_convert("UTC").tz_localize(None)
    grid = index.build_master_index(start=start)
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
    print(f"\ndone  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
