# Phase-2 Write-down — Walk-forward CV + Operational nowcast

**Branch:** `claude/phase2` (off `claude/next-mvp-phase-ovxke1` post-MVP).
**Scope this session:** first two Phase-2 tracks — walk-forward backtest + `forecast_now()`.
The MVP (B + A + threat-features + leak-safe holdout + heatmap) was already DONE; this is
the robustness + product-output layer on top of it.

---

## What shipped

### 1. Walk-forward CV (rolling-origin) — `src/evaluate.py`
Upgrades the single 8-week holdout (one verdict, one regime, no spread) to a variance
estimate + drift across war regimes — the Phase-2 evaluation item in STRUCTURE §6.

- `walk_forward_splits(df, n_folds, test_weeks, purge_hours, train_weeks)` — yields
  `(fold, train, test)` newest→oldest. Fold 0 = the canonical last-`test_weeks` holdout;
  each later fold steps the **non-overlapping** test window back another `test_weeks`.
  Train is EXPANDING by default (all earlier history, matches model_b recency weighting);
  `train_weeks` switches to a fixed-width SLIDING window. **Same purge guard** as
  `temporal_split` (drop `PURGE_HOURS` before each cut so t→t+H labels can't bleed across).
  Empty-train / empty-test folds skipped.
- `walk_forward_eval(...)` — runs Model B per fold, records PR-AUC / base / lift per
  horizon. **Calibration omitted on purpose**: PR-AUC is a ranking metric and isotonic is
  monotone, so the calibrator can't move the number whose variance we're estimating.
  Skips a fold/horizon with no positives (PR-AUC undefined).
- `walk_forward_summary(...)` — mean / std / min / max PR-AUC per horizon, canonical order.
- `config.WALK_FORWARD_FOLDS = 4`. Entrypoint: `run_walkforward.py`.

### 2. Operational nowcast — `src/forecast.py`
The product output (not a backtest): train B on ALL labelled rows, predict the latest
grid hour per oblast → next-30m/1h/3h/6h probabilities.

- `latest_rows(fm)` — most-recent row per oblast = forecast origin (target still NaN, so
  it's auto-excluded from training by `model_b.train_horizon`).
- `forecast_now(fm, horizons, calibrators)` — ranked frame `[origin_utc, 30m,1h,3h,6h]`,
  calibrated iff `calibrators` (out-of-fold isotonic) passed.
- `format_forecast(table, calibrated)` — ranked text + the **ragged-edge caveat**.
- Entrypoint `run_forecast.py` fits isotonic on the last `CALIB_WEEKS` (out-of-fold) then
  retrains on ALL rows for the freshest emitted model → honest probabilities.

**Honesty caveat carried in code + output:** the newest rows are complete in the
historical CSV but PARTIAL live (sources publish with lag → ragged right edge). This is a
DATA-AVAILABILITY gap, not leakage. Quantifying it = the Phase-2 vintage eval (still TODO).

---

## Fix made along the way
`model_b.train_horizon`: `scale_pos_weight` guarded against a saturated training slice
(all-positive or all-negative → `neg/pos` was 0, which LightGBM rejects with
`scale_pos_weight > 0`). Now falls back to 1.0. Surfaced by a degenerate walk-forward fold;
a real fold in a quiet/intense regime could hit the same edge.

---

## Tests (73 passing, 6 Prophet skips)
- `test_walk_forward.py` — non-overlapping + purged folds, newest→oldest, empty-train
  folds dropped, eval/summary shape + ordering, learnable-signal sanity (PR-AUC > 0.9 on a
  deterministic periodic grid).
- `test_forecast.py` — one row per oblast, origin = max ts, columns/range, calibrators flow
  through (degenerate zero-calibrator), caveat + calibration flag in the rendered text.

Synthetic grids use an every-8h alert so the `hour` calendar feature makes the next-hour
target perfectly learnable without saturating the 6h target.

---

## Not run here
Real-data entrypoints (`run_walkforward.py`, `run_forecast.py`) need `data/` (gitignored,
not in this container) + Prophet for the A path. Logic is unit-covered on synthetic grids;
both scripts parse and their pure-Python paths were smoke-tested.

---

## Phase-2 remaining (STRUCTURE §6, in honesty-payoff order)
1. **Operational / vintage eval** — snapshot sources at forecast time, validate `forecast_now`
   vs the later refresh → report backtest-vs-live gap (the real live capability). `forecast_now`
   is the predictor it will score.
2. **Duration model** — survival (lifelines), time-to-all-clear; reuses Phase-1 covariates.
