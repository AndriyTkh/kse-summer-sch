# PLAN.md — Current Stage: Phase 3 (uncertainty + auto-retrain)

> Active to-do. The *why* lives in [STRUCTURE.md](STRUCTURE.md); this is the *what now*.
> Phase 1 MVP is **done** (summary below). Phase 3 slice in flight: **quantile prediction
> intervals + drift-triggered auto-retrain** — the two "Accepted drawbacks" STRUCTURE §8
> flagged (no uncertainty bands; non-stationarity unmanaged). Other Phase-3 roadmap items
> (TG real-time scrape, nowcast tier, C/TCN/TFT, Hawkes, spatial) stay deferred.

---

## Phase 1 MVP — DONE ✅

- [x] 4 direct LightGBM models (30m/1h/3h/6h) per-oblast alert probability — `model_b.py`.
- [x] Prophet daily baseline + B-vs-A comparison — `model_a.py`, `evaluate.compare_b_vs_a`.
- [x] Threat-type features from the launch dataset — `threat_map.py`, `features.add_threat_features`.
- [x] Leak-safe temporal eval, PR-AUC + isotonic calibration — `evaluate.py` (3-way split).
- [x] Oblast × horizon heatmap + end-to-end `run_mvp.py`. (65 tests green at handoff.)

---

## Phase 2 — DONE ✅

- [x] Walk-forward backtest (rolling-origin CV) replacing single holdout — `evaluate.py`, `run_walkforward.py`.
- [x] Operational nowcast at the grid edge (`forecast_now`) — `forecast.py`, `run_forecast.py`.
- [x] Simulated-vintage operational eval (backtest-vs-live PR-AUC gap) — `operational_eval.py`, `run_operational_eval.py`.
- [x] Duration model: Kaplan-Meier + Cox PH, censoring-aware — `survival.py`, `run_survival.py`.
- [x] Viz dashboard (React + MapLibre) wired to P1/P2 JSON exports — `viz/`, `viz_export.py`.

---

## Phase 3 definition of done

- [ ] **Model Bq** — quantile LightGBM on the alert-FRACTION target → per oblast×horizon
      prediction intervals (q10/q50/q90), no quantile crossing, clipped to [0, 1].
- [ ] **Quantile metrics** — pinball loss + interval coverage + width, leak-safe temporal eval.
- [ ] **Drift detection** — per-feature PSI (reference window vs live block), banded by
      config thresholds (stable / moderate / significant).
- [ ] **Auto-retrain** — walk-forward harness; score-then-adapt; drift / periodic / never
      policies; frozen-vs-adaptive comparison shows adaptive wins under drift.
- [ ] **`run_phase3.py`** entrypoint + artifacts (interval table, drift-retrain trajectory PNG).

---

## Build order

1. **`model_bq.py`** — `make_fraction_target` (mean over (t, t+H], mirrors B's max), one
   `LGBMRegressor(objective="quantile")` per (horizon, alpha); reuse B's recency weights +
   feature columns; sort+clip quantiles on predict.
2. **`evaluate.py`** — add `pinball_loss`, `interval_coverage`, `interval_width`.
3. **`drift.py`** — `psi`, `feature_psi`, `classify_psi`, `drift_score` (max PSI = trigger).
4. **`retrain.py`** — `walk_forward_retrain` (purged trailing windows, online score→adapt),
   `compare_policies`.
5. **`run_phase3.py`** — quantile report on the test fold + 3-policy retrain comparison plot.
6. Tests for each (model_bq, drift, retrain, evaluate additions).

---

## Issue resolutions (locked, Phase 3 additions)

| # | Issue | Resolution |
|---|---|---|
| 12 | Quantiles on a probability | A point prob has no spread. Regress the continuous **alert-fraction** over (t, t+H] instead — a real intensity in [0, 1] that bands meaningfully. 30m/1h (k=1) degenerate to the binary next hour (honest — no sub-hourly info). |
| 13 | Quantile crossing | Independent per-α fits can cross. **Sort each row's quantiles ascending + clip [0, 1]** on predict. Monotone, so it never widens a valid band. |
| 14 | Drift signal | **PSI** per feature, reference-window vs live-block; max-PSI is the retrain trigger. Leading (covariate) signal — fires before labels confirm rot. Performance-drop trajectory (pinball) is the lagging cross-check in the log. |
| 15 | Retrain leakage | Each retrain fits a **purged trailing window** (PURGE_HOURS dropped) ending at the block start; blocks walked in time order; targets read only (t, t+H]. Score-then-adapt: the scored block is never trained on. |
| 16 | Phase-3 scope | Only **quantile + drift/retrain** this slice. TG scrape / nowcast / C / Hawkes / spatial stay roadmap (STRUCTURE §6). |

---

## Next action
Wire the interval + drift-retrain artifacts into a short writeup (B vs Bq sharpness; frozen
vs adaptive pinball under the 2022→2025 regime shift). Then revisit roadmap: TG nowcast tier.
