"""Walk-forward backtest (Phase 2): rolling-origin CV for Model B.

Upgrades the single holdout (run_mvp) to a variance estimate + drift across war
regimes. Slides the test window back WALK_FORWARD_FOLDS times, scores each fold, and
prints per-fold PR-AUC plus the mean ± spread per horizon. Leak guard unchanged: each
fold purges PURGE_HOURS before its cut.

Run: PYTHONUTF8=1 python run_walkforward.py
"""

from __future__ import annotations

import time

import pandas as pd

from src import config, evaluate, features, index, loaders, viz_export


def main() -> None:
    t0 = time.time()
    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()

    grid = index.build_master_index()  # starts at config.GRID_START (2023-07)
    grid = index.expand_alerts_to_grid(grid, alerts)
    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily})
    print(f"features {fm.shape[1]} cols  {len(fm):,} rows  [{time.time()-t0:.0f}s]")

    print(f"\nwalk-forward: {config.WALK_FORWARD_FOLDS} folds x "
          f"{config.TEST_WEEKS}wk test, purge {config.PURGE_HOURS}h  (newest=fold 0)")
    folds = evaluate.walk_forward_eval(fm, progress=True)
    summary = evaluate.walk_forward_summary(folds)

    print(f"\nPR-AUC mean ± std across folds  [{time.time()-t0:.0f}s]")
    print(f"{'horizon':<8}{'folds':>6}{'mean':>8}{'std':>7}{'min':>7}{'max':>7}"
          f"{'base':>7}{'lift':>7}")
    for h, r in summary.iterrows():
        print(f"{h:<8}{int(r['n_folds']):>6}{r['pr_auc_mean']:>8.3f}{r['pr_auc_std']:>7.3f}"
              f"{r['pr_auc_min']:>7.3f}{r['pr_auc_max']:>7.3f}"
              f"{r['base_mean']:>7.3f}{r['lift_mean']:>7.2f}")

    # viz: walkforward.json — "walk-forward CV" evaluation method (stats page)
    viz_export.write_viz_json("walkforward.json", {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "n_folds": int(config.WALK_FORWARD_FOLDS),
        "test_weeks": int(config.TEST_WEEKS),
        "purge_hours": int(config.PURGE_HOURS),
        "by_horizon": {
            h: {k: viz_export.serializable(v) for k, v in summary.loc[h].items()}
            for h in summary.index
        },
    })
    print(f"\ndone  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
