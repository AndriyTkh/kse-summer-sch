"""End-to-end ONSET run (alt approach) — parallel to run_mvp.py.

Same grid -> features -> LightGBM -> leak-safe temporal eval pipeline, but the target is
alert ONSET ("a NEW alert starts in (t, t+H], from a quiet state"), not the whether-state
B predicts. Threat features are REVIVED here (config.ONSET_THREAT_*) because onset is where
launch leading-indicators should matter. Kept deliberately parallel to run_mvp.py so the two
can be wired side-by-side as a comparison later.

Run: PYTHONUTF8=1 .venv/Scripts/python scripts/runs/run_onset.py
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
from sklearn.metrics import brier_score_loss

from src import config, evaluate, features, index, loaders, model_onset


def main() -> None:
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()
    ucdp = loaders.load_ucdp()

    grid = index.build_master_index()  # starts at config.GRID_START (2023-07)
    grid = index.expand_alerts_to_grid(grid, alerts)
    print(f"grid {len(grid):,} rows  ({grid.index.get_level_values('oblast').nunique()} oblasts)"
          f"  positives {int(grid.alert.sum()):,} ({grid.alert.mean():.3f})  [{time.time()-t0:.0f}s]")

    # Revive threat for onset (the lean Phase-4 allowlist) without touching the
    # whether-model's empty config.THREAT_CHANNELS.
    threat = dict(
        channels=config.ONSET_THREAT_CHANNELS,
        values=config.ONSET_THREAT_VALUES,
        windows=config.ONSET_THREAT_WINDOWS,
    )
    fm = features.build_feature_matrix(
        grid, {"waves": waves, "daily": daily, "ucdp": ucdp, "threat": threat}
    )
    rest, test = evaluate.temporal_split(fm)
    train_fit, calib = evaluate.temporal_split(rest, test_weeks=config.CALIB_WEEKS)
    print(f"features {fm.shape[1]} cols  train_fit {len(train_fit):,}  "
          f"calib {len(calib):,}  test {len(test):,}  [{time.time()-t0:.0f}s]")

    models = model_onset.train_all(train_fit, train_fit)

    # Isotonic per horizon on the held-out calib fold (out-of-fold => honest calibration).
    calib_probs = model_onset.predict_all(models, calib)
    calibrators = {}
    for h in config.HORIZONS:
        yc = model_onset.make_onset_target(calib, h).reindex(calib.index)
        mc = yc.notna()
        calibrators[h] = evaluate.fit_isotonic(yc[mc].astype(int), calib_probs.loc[mc, h])

    raw = model_onset.predict_all(models, test)
    probs = model_onset.predict_all(models, test, calibrators)
    print(f"trained {len(models)} horizons + isotonic  [{time.time()-t0:.0f}s]\n")

    # Per-horizon metrics on the QUIET-state test rows (onset target masks active rows).
    # Onset base rate is far below B's; PR-AUC will be lower but lift is the honest signal.
    print(f"{'horizon':<8}{'base':>7}{'n':>9}{'PR-AUC':>8}{'lift':>7}"
          f"{'ECE0':>7}{'ECE':>7}{'Brier0':>8}{'Brier':>8}")
    for h in config.HORIZONS:
        y = model_onset.make_onset_target(test, h).reindex(test.index)
        m = y.notna()
        yt, ys, yr = y[m].astype(int), probs.loc[m, h], raw.loc[m, h]
        base = yt.mean()
        ap = evaluate.pr_auc(yt, ys)
        ece0 = evaluate.expected_calibration_error(yt, yr)
        ece = evaluate.expected_calibration_error(yt, ys)
        b0 = brier_score_loss(yt, yr)
        brier = brier_score_loss(yt, ys)
        lift = ap / base if base else float("nan")
        print(f"{h:<8}{base:>7.3f}{int(m.sum()):>9,}{ap:>8.3f}{lift:>7.2f}"
              f"{ece0:>7.3f}{ece:>7.3f}{b0:>8.3f}{brier:>8.3f}")

    # Threat-revival gain probe: did launch leading-indicators finally pay off on onset?
    # Sum LightGBM gain importance over the thr_* feature group per horizon.
    print(f"\nthreat-group gain share (revived for onset)  [{time.time()-t0:.0f}s]")
    print(f"{'horizon':<8}{'thr_gain%':>10}")
    feats = model_onset.feature_columns(train_fit)
    thr_idx = [i for i, c in enumerate(feats) if c.startswith("thr_")]
    for h in config.HORIZONS:
        gains = models[h].booster_.feature_importance(importance_type="gain")
        total = gains.sum()
        share = gains[thr_idx].sum() / total if total else 0.0
        print(f"{h:<8}{100*share:>9.1f}%")

    # Artifact: 1h reliability curve (calibrated) on quiet-state rows.
    y1 = model_onset.make_onset_target(test, "1h").reindex(test.index)
    m1 = y1.notna()
    ax, _ = evaluate.calibration_plot(y1[m1].astype(int), probs.loc[m1, "1h"])
    ax.figure.savefig(config.ARTIFACTS_DIR / "onset_calibration_1h.png",
                      bbox_inches="tight", dpi=120)
    plt.close(ax.figure)

    print(f"\nartifacts -> {config.ARTIFACTS_DIR}  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
