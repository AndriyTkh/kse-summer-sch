"""End-to-end MVP run: real grid -> features -> Model B -> metrics + artifacts.

Headline output for Phase-1 B (LightGBM forecasting). Leak-safe by construction
(features < t, temporal split). Prophet baseline (A) is a separate, deferred part.

Run: PYTHONUTF8=1 .venv/Scripts/python scripts/runs/run_mvp.py
Writes calibration + heatmap PNGs to artifacts/ (gitignored).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Partial moved under scripts/runs/ — put repo root on sys.path so `from src import`
# resolves when run standalone (the combined base run.py imports src directly).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, evaluate, features, index, loaders, model_a, model_b


def main() -> None:
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()

    # Start at first recorded alert (pre-data hours would be false zeros).
    # build_master_index re-localizes to UTC, so hand it a naive wall-clock.
    grid = index.build_master_index()  # starts at config.GRID_START (2023-07)
    grid = index.expand_alerts_to_grid(grid, alerts)
    print(f"grid {len(grid):,} rows  ({grid.index.get_level_values('oblast').nunique()} oblasts)"
          f"  positives {int(grid.alert.sum()):,} ({grid.alert.mean():.3f})  [{time.time()-t0:.0f}s]")

    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily})
    # Three-way temporal split (issue #10): train_fit < calib < test, never random.
    # calib is held out to fit isotonic out-of-fold so reported calibration is honest.
    rest, test = evaluate.temporal_split(fm)
    train_fit, calib = evaluate.temporal_split(rest, test_weeks=config.CALIB_WEEKS)
    print(f"features {fm.shape[1]} cols  train_fit {len(train_fit):,}  "
          f"calib {len(calib):,}  test {len(test):,}  [{time.time()-t0:.0f}s]")

    models = model_b.train_all(train_fit, train_fit)

    # Fit one isotonic calibrator per horizon on the held-out calib fold.
    calib_probs = model_b.predict_all(models, calib)
    calibrators = {}
    for h in config.HORIZONS:
        yc = model_b.make_target(calib, h).reindex(calib.index)
        mc = yc.notna()
        calibrators[h] = evaluate.fit_isotonic(yc[mc].astype(int), calib_probs.loc[mc, h])

    raw = model_b.predict_all(models, test)
    probs = model_b.predict_all(models, test, calibrators)
    print(f"trained {len(models)} horizons + isotonic  [{time.time()-t0:.0f}s]\n")

    # Per-horizon metrics on the temporal test fold. ECE/Brier shown raw -> calibrated:
    # PR-AUC is unchanged by isotonic (monotone), ECE/Brier should drop.
    from sklearn.metrics import brier_score_loss
    print(f"{'horizon':<8}{'base':>7}{'PR-AUC':>8}{'lift':>7}"
          f"{'ECE0':>7}{'ECE':>7}{'Brier0':>8}{'Brier':>8}")
    for h in config.HORIZONS:
        y = model_b.make_target(test, h).reindex(test.index)
        m = y.notna()
        yt, ys, yr = y[m].astype(int), probs.loc[m, h], raw.loc[m, h]
        base = yt.mean()
        ap = evaluate.pr_auc(yt, ys)
        ece0 = evaluate.expected_calibration_error(yt, yr)
        ece = evaluate.expected_calibration_error(yt, ys)
        b0 = brier_score_loss(yt, yr)
        brier = brier_score_loss(yt, ys)
        print(f"{h:<8}{base:>7.3f}{ap:>8.3f}{ap/base:>7.2f}"
              f"{ece0:>7.3f}{ece:>7.3f}{b0:>8.3f}{brier:>8.3f}")

    # Artifacts: 1h reliability curve (calibrated) + oblast x horizon heatmap.
    y1 = model_b.make_target(test, "1h").reindex(test.index)
    m1 = y1.notna()
    ax, _ = evaluate.calibration_plot(y1[m1].astype(int), probs.loc[m1, "1h"])
    ax.figure.savefig(config.ARTIFACTS_DIR / "calibration_1h.png", bbox_inches="tight", dpi=120)
    plt.close(ax.figure)

    ax2, table = evaluate.oblast_horizon_heatmap(probs)
    ax2.figure.savefig(config.ARTIFACTS_DIR / "heatmap.png", bbox_inches="tight", dpi=120)
    plt.close(ax2.figure)

    # Model A (Prophet daily baseline) + B-vs-A: B must win short horizon (DoD).
    a_pred = model_a.baseline_for_grid(train_fit, test)
    y_true = pd.DataFrame(
        {h: model_b.make_target(test, h).reindex(test.index) for h in config.HORIZONS}
    )
    cmp = evaluate.compare_b_vs_a(probs, a_pred, y_true)
    print(f"\nB vs A (Prophet) PR-AUC  [{time.time()-t0:.0f}s]")
    print(f"{'horizon':<8}{'B':>8}{'A':>8}{'lift':>7}")
    for h in config.HORIZONS:
        r = cmp.loc[h]
        print(f"{h:<8}{r['pr_auc_b']:>8.3f}{r['pr_auc_a']:>8.3f}{r['lift']:>7.2f}")

    print(f"\nartifacts -> {config.ARTIFACTS_DIR}  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
