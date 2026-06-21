"""Export pre-computed predictions + metrics as JSON for the viz dashboard.

Runs the full pipeline (load → grid → features → B + A → metrics) and writes
predictions.json + metrics.json into viz/public/ for the React frontend.
"""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src import config, evaluate, features, index, loaders, model_a, model_b, model_onset


def _serializable(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _export_onset(grid, waves, daily, ucdp, out_dir, t0) -> None:
    """Train the ONSET model (alt approach) and write onset.json for the viz timing mode.

    Same leak-safe pipeline as B but: target = a NEW alert STARTS in (t, t+H] from a quiet
    state, and threat features are REVIVED (config.ONSET_THREAT_*). The per-horizon onset
    probabilities form a CDF (P(onset by T+n)) the frontend turns into a time-to-alert color
    + a distribution chart. Shape mirrors predictions.json so the map can switch sources.
    """
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

    models = model_onset.train_all(train_fit, train_fit)

    calib_probs = model_onset.predict_all(models, calib)
    calibrators = {}
    for h in config.HORIZONS:
        yc = model_onset.make_onset_target(calib, h).reindex(calib.index)
        mc = yc.notna()
        calibrators[h] = evaluate.fit_isotonic(yc[mc].astype(int), calib_probs.loc[mc, h])

    probs = model_onset.predict_all(models, test, calibrators)
    y_true = pd.DataFrame(
        {h: model_onset.make_onset_target(test, h).reindex(test.index) for h in config.HORIZONS}
    )
    print(f"Onset model trained, computing onset export...  [{time.time()-t0:.0f}s]")

    # Latest onset prediction per oblast per horizon = the onset CDF at the grid edge.
    last_ts = probs.index.get_level_values("ts_utc").max()
    latest = probs.loc[probs.index.get_level_values("ts_utc") == last_ts]
    predictions = {}
    for oblast in config.OBLAST_CODES:
        try:
            row = latest.xs(oblast, level="oblast")
            predictions[oblast] = {h: _serializable(row[h].iloc[0]) for h in config.HORIZONS}
        except KeyError:
            predictions[oblast] = {h: None for h in config.HORIZONS}

    # Aggregate + per-oblast onset skill (quiet-state rows only; honest lift over base).
    aggregate = {}
    for h in config.HORIZONS:
        y = y_true[h]
        m = y.notna()
        yt = y[m].astype(int)
        ys = probs.loc[m, h]
        base = float(yt.mean())
        pr = evaluate.pr_auc(yt, ys)
        aggregate[h] = {
            "base_rate": _serializable(base),
            "pr_auc": _serializable(pr),
            "lift": _serializable(pr / base if base else None),
            "n_samples": int(m.sum()),
        }

    per_oblast = {}
    for oblast in config.OBLAST_CODES:
        ob_mask = probs.index.get_level_values("oblast") == oblast
        ob_probs = probs[ob_mask]
        ob_y = y_true[ob_mask]
        ob_data = {}
        for h in config.HORIZONS:
            m = ob_y[h].notna()
            if m.sum() == 0:
                continue
            yt = ob_y.loc[m, h].astype(int)
            ys = ob_probs.loc[m, h]
            ob_data[h] = {
                "pr_auc": _serializable(evaluate.pr_auc(yt, ys)),
                "base_rate": _serializable(float(yt.mean())),
                "mean_pred": _serializable(float(ys.mean())),
                "n_samples": int(m.sum()),
            }
        per_oblast[oblast] = ob_data

    onset_out = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "forecast_base_utc": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
        "horizons": config.HORIZONS,
        "horizon_hours": config.HORIZON_HOURS,
        "test_weeks": config.TEST_WEEKS,
        "predictions": predictions,
        "aggregate": aggregate,
        "per_oblast": per_oblast,
    }
    (out_dir / "onset.json").write_text(json.dumps(onset_out, indent=2, default=_serializable))
    print(f"Exported onset.json -> {out_dir}  [{time.time()-t0:.0f}s]")


def main() -> None:
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_dir = config.ROOT / "viz" / "public"
    out_dir.mkdir(parents=True, exist_ok=True)

    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()
    ucdp = loaders.load_ucdp()

    grid = index.build_master_index()  # starts at config.GRID_START (2023-07)
    grid = index.expand_alerts_to_grid(grid, alerts)

    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily, "ucdp": ucdp})
    rest, test = evaluate.temporal_split(fm)
    train_fit, calib = evaluate.temporal_split(rest, test_weeks=config.CALIB_WEEKS)

    models = model_b.train_all(train_fit, train_fit)

    calib_probs = model_b.predict_all(models, calib)
    calibrators = {}
    for h in config.HORIZONS:
        yc = model_b.make_target(calib, h).reindex(calib.index)
        mc = yc.notna()
        calibrators[h] = evaluate.fit_isotonic(yc[mc].astype(int), calib_probs.loc[mc, h])

    probs = model_b.predict_all(models, test, calibrators)
    a_pred = model_a.baseline_for_grid(train_fit, test)
    y_true = pd.DataFrame(
        {h: model_b.make_target(test, h).reindex(test.index) for h in config.HORIZONS}
    )

    print(f"Models trained, computing exports...  [{time.time()-t0:.0f}s]")

    # --- predictions.json: latest prediction per oblast per horizon ---
    last_ts = probs.index.get_level_values("ts_utc").max()
    latest = probs.loc[probs.index.get_level_values("ts_utc") == last_ts]

    predictions = {}
    for oblast in config.OBLAST_CODES:
        try:
            row = latest.xs(oblast, level="oblast")
            predictions[oblast] = {h: _serializable(row[h].iloc[0]) for h in config.HORIZONS}
        except KeyError:
            predictions[oblast] = {h: None for h in config.HORIZONS}

    pred_out = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "forecast_base_utc": last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
        "horizons": config.HORIZONS,
        "predictions": predictions,
    }
    (out_dir / "predictions.json").write_text(json.dumps(pred_out, indent=2, default=_serializable))

    # --- metrics.json: aggregate + per-oblast accuracy ---
    from sklearn.metrics import brier_score_loss

    aggregate = {}
    for h in config.HORIZONS:
        y = y_true[h]
        m = y.notna()
        yt = y[m].astype(int)
        ys_b = probs.loc[m, h]
        ys_a = a_pred.loc[m, h]
        base_rate = float(yt.mean())
        pr_b = evaluate.pr_auc(yt, ys_b)
        pr_a = evaluate.pr_auc(yt, ys_a)
        ece_b = evaluate.expected_calibration_error(yt, ys_b)
        ece_a = evaluate.expected_calibration_error(yt, ys_a)
        brier_b = float(brier_score_loss(yt, ys_b))
        brier_a = float(brier_score_loss(yt, ys_a))
        aggregate[h] = {
            "base_rate": _serializable(base_rate),
            "pr_auc_b": _serializable(pr_b),
            "pr_auc_a": _serializable(pr_a),
            "lift": _serializable(pr_b / pr_a if pr_a else None),
            "ece_b": _serializable(ece_b),
            "ece_a": _serializable(ece_a),
            "brier_b": _serializable(brier_b),
            "brier_a": _serializable(brier_a),
        }

    per_oblast = {}
    for oblast in config.OBLAST_CODES:
        try:
            ob_mask = probs.index.get_level_values("oblast") == oblast
            ob_probs = probs[ob_mask]
            ob_y = y_true[ob_mask]
            ob_data = {}
            for h in config.HORIZONS:
                m = ob_y[h].notna()
                if m.sum() == 0:
                    continue
                yt = ob_y.loc[m, h].astype(int)
                ys = ob_probs.loc[m, h]
                ob_data[h] = {
                    "pr_auc_b": _serializable(evaluate.pr_auc(yt, ys)),
                    "base_rate": _serializable(float(yt.mean())),
                    "mean_pred": _serializable(float(ys.mean())),
                    "n_samples": int(m.sum()),
                }
            per_oblast[oblast] = ob_data
        except Exception:
            per_oblast[oblast] = {}

    # Calibration curve data (1h horizon for the chart)
    cal_data = {}
    for h in config.HORIZONS:
        y = y_true[h]
        m = y.notna()
        yt = y[m].astype(int).values
        ys = probs.loc[m, h].values
        from sklearn.calibration import calibration_curve
        try:
            frac_pos, mean_pred = calibration_curve(yt, ys, n_bins=10, strategy="quantile")
            cal_data[h] = {
                "frac_pos": [_serializable(x) for x in frac_pos],
                "mean_pred": [_serializable(x) for x in mean_pred],
            }
        except Exception:
            cal_data[h] = {"frac_pos": [], "mean_pred": []}

    metrics_out = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "test_weeks": config.TEST_WEEKS,
        "aggregate": aggregate,
        "per_oblast": per_oblast,
        "calibration_curves": cal_data,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=_serializable))

    print(f"Exported predictions.json + metrics.json -> {out_dir}  [{time.time()-t0:.0f}s]")

    # Alt approach: onset / timing mode for the viz (parallel to B).
    _export_onset(grid, waves, daily, ucdp, out_dir, t0)


if __name__ == "__main__":
    main()
