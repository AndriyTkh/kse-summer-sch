"""Auto-retrain walk-forward harness (Phase 3) — adaptive model under drift.

The non-stationarity verdict (issue #6): a model trained once and frozen decays as the
war regime shifts. This module operates Model Bq the way you would in production —
forward through time, block by block — and ADAPTS when drift fires, so we can measure
the gap between a frozen model and a self-retraining one on the real 2022→2025 record.

Online protocol per block (no leakage, score-then-adapt):
  1. SCORE the current model on the block (pinball / coverage / width).
  2. MEASURE drift: feature PSI between the model's training window and this block.
  3. ADAPT: if the policy says so, retrain on a trailing window ending at the block
     start (purged by PURGE_HOURS so no future label bleeds back) — the new model
     governs the NEXT block onward. The block just scored is never trained on.

Policies:
  drift     — retrain when block PSI.max exceeds `psi_threshold` (config.DRIFT_PSI_ALERT)
  periodic  — retrain every `period_blocks` blocks (calendar cadence, drift-blind)
  never     — frozen baseline (train once, ride it out) — the floor adaptive must beat

Leak guard: every train window is data STRICTLY before the block it will predict, minus
the purge band; blocks are walked in time order. Targets read only (t, t+H] (model_bq).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, drift, evaluate, model_b, model_bq


def _blocks(ts_index: pd.DatetimeIndex, start, end, block_days: int):
    """Yield [block_start, block_end) edges from `start` to `end` in block_days steps."""
    step = pd.Timedelta(days=block_days)
    bs = start
    while bs < end:
        yield bs, min(bs + step, end)
        bs += step


def _train_window(fm: pd.DataFrame, end_ts, window_days: int, purge_hours: float):
    """Trailing leak-safe slice: rows in [end_ts - window_days, end_ts - purge_hours).

    The purge drops rows whose (t, t+H] target would peek past the cut (label bleed),
    exactly as evaluate.temporal_split does at the global cut.
    """
    ts = fm.index.get_level_values("ts_utc")
    lo = end_ts - pd.Timedelta(days=window_days)
    hi = end_ts - pd.Timedelta(hours=purge_hours)
    return fm[(ts >= lo) & (ts < hi)]


def _score_block(models: dict, block: pd.DataFrame, horizon: str, quantiles) -> dict:
    """Pinball (avg over quantiles), interval coverage + width for one block."""
    y = model_bq.make_fraction_target(block, horizon).reindex(block.index)
    m = y.notna()
    if not m.any():
        return {"pinball": np.nan, "coverage": np.nan, "width": np.nan, "n": 0}

    preds = model_bq.predict_quantiles(models, block.loc[m])
    yt = y.loc[m].to_numpy()
    pin = np.mean([
        evaluate.pinball_loss(yt, preds[f"{horizon}_{model_bq.q_label(a)}"], a)
        for a in quantiles
    ])
    lo_col, _, hi_col = model_bq.interval_columns(horizon)
    return {
        "pinball": float(pin),
        "coverage": evaluate.interval_coverage(yt, preds[lo_col], preds[hi_col]),
        "width": evaluate.interval_width(preds[lo_col], preds[hi_col]),
        "n": int(m.sum()),
    }


def walk_forward_retrain(
    fm: pd.DataFrame,
    *,
    horizon: str = "6h",
    quantiles=config.QUANTILES,
    policy: str = "drift",
    block_days: int = config.RETRAIN_BLOCK_DAYS,
    window_days: int = config.RETRAIN_WINDOW_DAYS,
    period_blocks: int = config.RETRAIN_PERIOD_BLOCKS,
    psi_threshold: float = config.DRIFT_PSI_ALERT,
    purge_hours: float = config.PURGE_HOURS,
    monitor_features=None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Walk `fm` forward, scoring then adapting per `policy`. Returns a per-block log.

    `fm` is the full feature matrix (with raw `alert`), MultiIndex (oblast, ts_utc).
    The first `window_days` seed the initial fit; evaluation blocks start after it.

    Log columns: block_start, block_end, n, pinball, coverage, width, psi, psi_top,
    retrained, n_train. One row per forward block; `psi` is reference-vs-block max PSI.
    """
    ts = fm.index.get_level_values("ts_utc")
    t0, tN = ts.min(), ts.max()
    seed_end = t0 + pd.Timedelta(days=window_days)
    if seed_end >= tN:
        raise ValueError(
            f"history {(tN - t0).days}d too short for window_days={window_days}; "
            "shrink the window or feed more data"
        )

    ref = _train_window(fm, seed_end, window_days, purge_hours)
    models = model_bq.train_all_quantiles(ref, ref, horizons=[horizon],
                                          quantiles=quantiles, params=params)
    feats = monitor_features or model_b.feature_columns(fm)

    rows = []
    for i, (bs, be) in enumerate(_blocks(ts, seed_end, tN, block_days)):
        block = fm[(ts >= bs) & (ts < be)]
        if block.empty:
            continue

        score = _score_block(models, block, horizon, quantiles)
        dr = drift.drift_score(ref, block, features=feats)

        if policy == "drift":
            retrain = dr["max"] >= psi_threshold
        elif policy == "periodic":
            retrain = (i + 1) % period_blocks == 0
        elif policy == "never":
            retrain = False
        else:
            raise ValueError(f"unknown policy {policy!r}")

        n_train = len(ref)
        if retrain:
            new_ref = _train_window(fm, bs, window_days, purge_hours)
            if not new_ref.empty:
                models = model_bq.train_all_quantiles(
                    new_ref, new_ref, horizons=[horizon],
                    quantiles=quantiles, params=params)
                ref = new_ref
                n_train = len(new_ref)
            else:
                retrain = False

        rows.append({
            "block_start": bs, "block_end": be, "n": score["n"],
            "pinball": score["pinball"], "coverage": score["coverage"],
            "width": score["width"], "psi": dr["max"], "psi_top": dr["top"],
            "retrained": bool(retrain), "n_train": n_train,
        })

    return pd.DataFrame(rows)


def compare_policies(fm: pd.DataFrame, policies=("never", "periodic", "drift"), **kw) -> dict:
    """Run several policies on the same walk-forward; return {policy: log}.

    The deliverable comparison: drift/periodic adaptive vs the `never` frozen floor.
    Mean pinball across blocks summarizes each (lower = better under drift).
    """
    return {p: walk_forward_retrain(fm, policy=p, **kw) for p in policies}
