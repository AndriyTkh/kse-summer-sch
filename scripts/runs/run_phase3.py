"""Phase 3 run: quantile prediction intervals (Bq).

Model Bq — quantile LightGBM over the alert-FRACTION target: per oblast×horizon
uncertainty bands, scored by pinball loss + interval coverage/width. Exports
intervals.json for the viz dashboard.

NOTE: the former second deliverable — drift-triggered auto-retrain (walk-forward) — was
DROPPED from this run on 2026-06-21 (too time-consuming for ≤0.005 PR-AUC; recency
weighting already handles drift). `retrain.py` / `drift.py` stay in-tree as a run-once
study only. See PLAN.md and the phase-4 write-down.

Leak-safe by construction (features < t; temporal split). Bands currently UNDER-cover
(~0.64 vs 0.80 nominal) — conformal fix queued (PLAN.md).
Run: PYTHONUTF8=1 python scripts/runs/run_phase3.py   (needs data/ — gitignored).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Partial moved under scripts/runs/ — put repo root on sys.path so `from src import`
# resolves when run standalone (the combined base run.py imports src directly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src import config, evaluate, features, index, loaders, model_bq, viz_export


def main() -> None:
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()
    ucdp = loaders.load_ucdp()

    grid = index.build_master_index()  # starts at config.GRID_START (2023-07)
    grid = index.expand_alerts_to_grid(grid, alerts)
    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily, "ucdp": ucdp})
    print(f"grid {len(fm):,} rows  features {fm.shape[1]}  [{time.time()-t0:.0f}s]")

    # --- 1. Quantile intervals on the temporal test fold ------------------
    rest, test = evaluate.temporal_split(fm)
    models = model_bq.train_all_quantiles(rest, rest)
    preds = model_bq.predict_quantiles(models, test)
    print(f"\nModel Bq quantile intervals (test fold)  [{time.time()-t0:.0f}s]")
    print(f"{'horizon':<8}{'pinball':>9}{'cover':>8}{'width':>8}")
    by_horizon = {}
    for h in config.HORIZONS:
        y = model_bq.make_fraction_target(test, h).reindex(test.index)
        m = y.notna()
        yt = y[m].to_numpy()
        pin = sum(evaluate.pinball_loss(yt, preds.loc[m, f"{h}_{model_bq.q_label(a)}"], a)
                  for a in config.QUANTILES) / len(config.QUANTILES)
        lo, _, hi = model_bq.interval_columns(h)
        cov = evaluate.interval_coverage(yt, preds.loc[m, lo], preds.loc[m, hi])
        wid = evaluate.interval_width(preds.loc[m, lo], preds.loc[m, hi])
        print(f"{h:<8}{pin:>9.4f}{cov:>8.2f}{wid:>8.3f}")
        by_horizon[h] = {
            "pinball": pin, "coverage": cov, "width": wid,
            "base_fraction": float(yt.mean()),
        }

    # --- intervals.json: aggregate band metrics + latest interval per oblast ---
    last_ts = preds.index.get_level_values("ts_utc").max()
    latest = preds.loc[preds.index.get_level_values("ts_utc") == last_ts]
    interval_preds = {}
    for oblast in config.OBLAST_CODES:
        try:
            row = latest.xs(oblast, level="oblast")
            interval_preds[oblast] = {
                h: {
                    "q10": float(row[f"{h}_q10"].iloc[0]),
                    "q50": float(row[f"{h}_q50"].iloc[0]),
                    "q90": float(row[f"{h}_q90"].iloc[0]),
                }
                for h in config.HORIZONS
            }
        except KeyError:
            interval_preds[oblast] = {h: None for h in config.HORIZONS}

    viz_export.write_viz_json("intervals.json", {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "forecast_base_utc": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
        "test_weeks": config.TEST_WEEKS,
        "horizons": config.HORIZONS,
        "nominal_coverage": config.PI_HIGH - config.PI_LOW,
        "by_horizon": by_horizon,
        "predictions": interval_preds,
    })

    # --- 2. Auto-retrain walk-forward: DROPPED (2026-06-21) ----------------
    # The frozen-vs-periodic-vs-drift walk-forward (retrain.compare_policies) was the
    # dominant compute cost and a single-fit-vs-walk-forward experiment showed it buys
    # <=0.005 PR-AUC at the operating point — recency weighting already absorbs the slow
    # regime drift. Dropped from the production run as too time-consuming for its payoff;
    # retained only as a run-once drift STUDY (see PLAN.md / phase-4 write-down). With it
    # gone there is no drift.json / drift_retrain.png; the viz drift panel hides (optional).

    print(f"\nartifacts -> {config.ARTIFACTS_DIR}  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
