"""Operational nowcast (Phase 2): emit the next-6h per-oblast alert probabilities.

The product output, not a backtest. Trains Model B on ALL available data, fits isotonic
out-of-fold on the last CALIB_WEEKS so the emitted probabilities are honest, then
predicts the latest grid hour per oblast. Prints a ranked table + the ragged-edge
caveat (origin-hour launch/tempo features may be partial live — see src/forecast.py).

Run: PYTHONUTF8=1 python run_forecast.py
"""

from __future__ import annotations

import time

import pandas as pd

from src import config, evaluate, features, forecast, index, loaders, model_b, viz_export


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

    # Honest probabilities: fit isotonic on a held-out recent slice (out-of-fold), then
    # forecast_now retrains B on ALL rows for the freshest possible emitted model.
    train_fit, calib = evaluate.temporal_split(fm, test_weeks=config.CALIB_WEEKS)
    cal_models = model_b.train_all(train_fit, train_fit)
    cal_probs = model_b.predict_all(cal_models, calib)
    calibrators = {}
    for h in config.HORIZONS:
        yc = model_b.make_target(calib, h).reindex(calib.index)
        mc = yc.notna()
        calibrators[h] = evaluate.fit_isotonic(yc[mc].astype(int), cal_probs.loc[mc, h])

    table = forecast.forecast_now(fm, calibrators=calibrators)
    print(f"\n{forecast.format_forecast(table, calibrated=True)}")

    # viz: nowcast.json — the "nowcast" prediction option (map source switcher)
    origin = table["origin_utc"].max()
    preds = {}
    for oblast in config.OBLAST_CODES:
        if oblast in table.index:
            row = table.loc[oblast]
            preds[oblast] = {h: viz_export.serializable(row[h]) for h in config.HORIZONS}
        else:
            preds[oblast] = {h: None for h in config.HORIZONS}
    viz_export.write_viz_json("nowcast.json", {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "origin_utc": origin.isoformat() if hasattr(origin, "isoformat") else str(origin),
        "horizons": config.HORIZONS,
        "calibrated": True,
        "predictions": preds,
    })
    print(f"\ndone  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
