"""Operational nowcast — train Model B on ALL data, emit the next-H per-oblast probs.

This is the product output, NOT a backtest: for every oblast it answers "what is the
probability of an air-raid alert in the next 30m / 1h / 3h / 6h, starting from the
latest grid hour?". Trains on every row whose target window is fully observed and
predicts on the single most recent row per oblast (the forecast origin, whose target
is still unknown).

HONESTY CAVEAT — ragged right edge (Phase-2 operational eval):
  The newest rows are complete in a historical CSV but PARTIAL live: launch / tempo
  sources publish with lag, so threat/tempo features at the origin can be understated.
  This is a DATA-AVAILABILITY gap, not leakage (the timestamp guard is still correct).
  Quantifying it is the Phase-2 vintage eval (STRUCTURE §6); until then, treat the
  origin-hour threat channels as a lower bound, not ground truth.

Calibration: pass `calibrators` (per-horizon isotonic from evaluate.fit_isotonic, fit
out-of-fold in the eval pipeline) to emit calibrated probabilities; without them the
raw Model B scores are returned (well-ranked but overconfident at short horizons).
"""

from __future__ import annotations

import pandas as pd

from . import config


def latest_rows(fm: pd.DataFrame) -> pd.DataFrame:
    """Most recent row per oblast = the forecast origin (its target is unknown -> NaN).

    Assumes the standard (oblast, ts_utc) MultiIndex; sorts so `tail(1)` per oblast is
    the max-timestamp row.
    """
    return fm.sort_index().groupby(level="oblast", sort=False).tail(1)


def forecast_now(fm: pd.DataFrame, horizons=None, calibrators: dict | None = None) -> pd.DataFrame:
    """Train Model B on all labelled rows, emit next-H probabilities at the edge.

    `fm` is the full feature matrix up to "now" (index.build_master_index defaults its
    end to the current hour). Model B's `train_horizon` drops rows whose future window
    runs off the end, so the origin rows are auto-excluded from training and only used
    for prediction. Returns a frame indexed by `oblast` with columns
    `[origin_utc, 30m, 1h, 3h, 6h]`; probabilities are calibrated iff `calibrators` given.
    """
    from . import model_b   # lazy: keep lightgbm off the import path for caveat-only use

    horizons = horizons or config.HORIZONS
    models = model_b.train_all(fm, fm, horizons=horizons)   # NaN-target origin rows dropped
    edge = latest_rows(fm)
    probs = model_b.predict_all(models, edge, calibrators)

    out = probs.copy()
    out.insert(0, "origin_utc", edge.index.get_level_values("ts_utc"))
    out.index = edge.index.get_level_values("oblast")
    out.index.name = "oblast"
    return out.sort_values(horizons[-1] if horizons else config.HORIZONS[-1], ascending=False)


def format_forecast(table: pd.DataFrame, calibrated: bool) -> str:
    """Render the forecast_now table as a ranked text block with the ragged-edge caveat.

    `calibrated` flips the probability-trust line so the caller never over-claims raw
    scores as honest frequencies.
    """
    horizons = [h for h in config.HORIZONS if h in table.columns]
    origin = pd.Timestamp(table["origin_utc"].max())
    lines = [
        f"Air-raid alert nowcast — origin {origin:%Y-%m-%d %H:%M} UTC",
        "P(alert within horizon), per oblast (ranked by 6h):",
        "",
        f"{'oblast':<18}" + "".join(f"{h:>7}" for h in horizons),
    ]
    for ob, r in table.iterrows():
        lines.append(f"{ob:<18}" + "".join(f"{r[h]:>7.3f}" for h in horizons))
    lines += [
        "",
        ("probabilities: ISOTONIC-CALIBRATED (trust as frequencies)" if calibrated
         else "probabilities: RAW model scores (well-ranked, overconfident short-horizon)"),
        "caveat: ragged right edge — origin-hour launch/tempo features may be partial "
        "live (data-availability, not leakage). Phase-2 vintage eval quantifies the gap.",
    ]
    return "\n".join(lines)
