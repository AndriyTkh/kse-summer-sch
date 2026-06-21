"""Phase 3 run: quantile prediction intervals (Bq) + drift-triggered auto-retrain.

Two deliverables on top of the MVP:
  1. Model Bq — quantile LightGBM over the alert-FRACTION target: per oblast×horizon
     uncertainty bands, scored by pinball loss + interval coverage/width.
  2. Auto-retrain — walk the model forward block-by-block over 2022→2025 and adapt on
     drift; compare the frozen floor vs periodic vs drift-triggered retraining.

Leak-safe by construction (features < t, purged trailing train windows, temporal walk).
Run: PYTHONUTF8=1 python run_phase3.py   (needs data/ — gitignored).
Writes interval + drift-retrain PNGs to artifacts/.
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, evaluate, features, index, loaders, model_bq, retrain


def main() -> None:
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()

    start = alerts["start_utc"].min().floor("h").tz_convert("UTC").tz_localize(None)
    grid = index.build_master_index(start=start)
    grid = index.expand_alerts_to_grid(grid, alerts)
    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily})
    print(f"grid {len(fm):,} rows  features {fm.shape[1]}  [{time.time()-t0:.0f}s]")

    # --- 1. Quantile intervals on the temporal test fold ------------------
    rest, test = evaluate.temporal_split(fm)
    models = model_bq.train_all_quantiles(rest, rest)
    preds = model_bq.predict_quantiles(models, test)
    print(f"\nModel Bq quantile intervals (test fold)  [{time.time()-t0:.0f}s]")
    print(f"{'horizon':<8}{'pinball':>9}{'cover':>8}{'width':>8}")
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

    # --- 2. Auto-retrain: frozen vs periodic vs drift ---------------------
    logs = retrain.compare_policies(fm, horizon="6h")
    print(f"\nAuto-retrain walk-forward, 6h  [{time.time()-t0:.0f}s]")
    print(f"{'policy':<10}{'mean pinball':>14}{'retrains':>10}{'mean cover':>12}")
    for p, log in logs.items():
        print(f"{p:<10}{log['pinball'].mean():>14.4f}"
              f"{int(log['retrained'].sum()):>10}{log['coverage'].mean():>12.2f}")

    ax = plt.subplots(figsize=(10, 4))[1]
    for p, log in logs.items():
        ax.plot(log["block_start"], log["pinball"], "o-", label=p, markersize=3)
    drift_log = logs["drift"]
    for bs in drift_log.loc[drift_log["retrained"], "block_start"]:
        ax.axvline(bs, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("block start"); ax.set_ylabel("pinball loss (6h)")
    ax.set_title("Auto-retrain: frozen vs periodic vs drift (dotted = drift retrain)")
    ax.legend()
    ax.figure.savefig(config.ARTIFACTS_DIR / "drift_retrain.png", bbox_inches="tight", dpi=120)
    plt.close(ax.figure)

    print(f"\nartifacts -> {config.ARTIFACTS_DIR}  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
