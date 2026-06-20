"""End-to-end MVP run: real grid -> features -> Model B -> metrics + artifacts.

Headline output for Phase-1 B (LightGBM forecasting). Leak-safe by construction
(features < t, temporal split). Prophet baseline (A) is a separate, deferred part.

Run: PYTHONUTF8=1 .venv/Scripts/python run_mvp.py
Writes calibration + heatmap PNGs to artifacts/ (gitignored).
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, evaluate, features, index, loaders, model_b


def main() -> None:
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()

    # Start at first recorded alert (pre-data hours would be false zeros).
    # build_master_index re-localizes to UTC, so hand it a naive wall-clock.
    start = alerts["start_utc"].min().floor("h").tz_convert("UTC").tz_localize(None)
    grid = index.build_master_index(start=start)
    grid = index.expand_alerts_to_grid(grid, alerts)
    print(f"grid {len(grid):,} rows  ({grid.index.get_level_values('oblast').nunique()} oblasts)"
          f"  positives {int(grid.alert.sum()):,} ({grid.alert.mean():.3f})  [{time.time()-t0:.0f}s]")

    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily})
    train, test = evaluate.temporal_split(fm)
    print(f"features {fm.shape[1]} cols  train {len(train):,}  test {len(test):,}  [{time.time()-t0:.0f}s]")

    models = model_b.train_all(train, train)
    probs = model_b.predict_all(models, test)
    print(f"trained {len(models)} horizons  [{time.time()-t0:.0f}s]\n")

    # Per-horizon metrics on the temporal test fold.
    print(f"{'horizon':<8}{'base':>7}{'PR-AUC':>8}{'lift':>7}{'ECE':>7}{'Brier':>8}")
    for h in config.HORIZONS:
        y = model_b.make_target(test, h).reindex(test.index)
        m = y.notna()
        yt, ys = y[m].astype(int), probs.loc[m, h]
        base = yt.mean()
        ap = evaluate.pr_auc(yt, ys)
        ece = evaluate.expected_calibration_error(yt, ys)
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(yt, ys)
        print(f"{h:<8}{base:>7.3f}{ap:>8.3f}{ap/base:>7.2f}{ece:>7.3f}{brier:>8.3f}")

    # Artifacts: 1h reliability curve + oblast x horizon heatmap.
    y1 = model_b.make_target(test, "1h").reindex(test.index)
    m1 = y1.notna()
    ax, _ = evaluate.calibration_plot(y1[m1].astype(int), probs.loc[m1, "1h"])
    ax.figure.savefig(config.ARTIFACTS_DIR / "calibration_1h.png", bbox_inches="tight", dpi=120)
    plt.close(ax.figure)

    ax2, table = evaluate.oblast_horizon_heatmap(probs)
    ax2.figure.savefig(config.ARTIFACTS_DIR / "heatmap.png", bbox_inches="tight", dpi=120)
    plt.close(ax2.figure)

    print(f"\nartifacts -> {config.ARTIFACTS_DIR}  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
