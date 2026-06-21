# PLAN.md — Current Stage: Phase 4 → pivot to ONSET (when, not whether)

> Active to-do. The *why* lives in [STRUCTURE.md](STRUCTURE.md); this is the *what now*.
> Phases 1–3 done. Phase-4 (2026-06-21) was a measure-everything pass: pruned features,
> clamped the grid to 2023-07, excluded dead regions, wired UCDP, dropped walk-forward.
> **Outcome: the "whether an alert fires" target is hollow → next session reframes to alert
> ONSET (when).** See `pr-writedown/2026-06-21_phase4-recency-pruning-onset-pivot.md`.
>
> **Submission scope cut:** alt models **C/D/E deferred** (GRU/TCN/TFT, Hawkes,
> IsolationForest); **walk-forward (eval + retrain) dropped** — run-once study only.

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

## Phase 3 — DONE ✅ (code merged; viz wiring + writeup carried into Phase 4)

- [x] **Model Bq** — quantile LightGBM on the alert-FRACTION target → per oblast×horizon
      prediction intervals (q10/q50/q90), no quantile crossing, clipped to [0, 1]. `model_bq.py`.
- [x] **Quantile metrics** — pinball loss + interval coverage + width, leak-safe temporal eval. `evaluate.py`.
- [x] **Drift detection** — per-feature PSI (reference window vs live block), banded by
      config thresholds (stable / moderate / significant). `drift.py`.
- [x] **Auto-retrain** — walk-forward harness; score-then-adapt; drift / periodic / never
      policies; frozen-vs-adaptive comparison. `retrain.py`.
- [x] **`run_phase3.py`** entrypoint + artifacts (interval table, drift-retrain trajectory PNG).

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

## Phase 4 — DONE this session (2026-06-21)

Measure-everything pass. Full retrospective: `pr-writedown/2026-06-21_phase4-recency-pruning-onset-pivot.md`.

- [x] **Wire Phase 3 into viz** — intervals.json + React Intervals panel. (drift.json/panel built
      then made inactive when walk-forward was dropped — see below.)
- [x] **UCDP regional prior (2a / #17)** — `load_ucdp` + `add_ucdp_features` wired, leak-safe
      per-oblast cumulative prior. Verdict: +0.0045 PR-AUC @6h, ~0 @1h. Kept (the location signal).
- [x] **Dead-region exclusion** — `config.MODEL_OBLASTS` drops crimea/sevastopol/luhanska
      (0–0.2% base rate, no siren coverage). Frontline kept; per-oblast lift reporting chosen.
- [x] **Grid clamp `GRID_START = 2023-07`** (B + Bq) — learning curve showed only the 2022
      ground-war regime is droppable; 6h at peak, 1h flat, −31% rows.
- [x] **Threat features DEPRECATED** — only +1% PR-AUC on the *persistence* target; disabled
      (`THREAT_CHANNELS=()`), lean 6-col allowlist kept for onset revival.
- [x] **Walk-forward DROPPED** (eval + retrain) — single-fit experiment showed ≤0.005 PR-AUC
      payoff; recency weighting suffices. Removed from `run_phase3`; `retrain.py`/`drift.py` stay
      as a run-once study only.

---

## NEXT SESSION — reframe target: *whether* → *when* (onset)

**Why:** "will an alert fire in next H" is trivially monotone in H (6h base rate 0.51) and
wins by persistence/autocorrelation, not strike forecasting. Useless as a product. Switch to
predicting alert **ONSET**. Path chosen: **A then B.**

1. **(A) Onset-in-window.** Retarget B: `make_target` → "a NEW alert STARTS in (t, t+H],
   evaluated only from a quiet state (no alert active at t)." Reuses grid/features/LightGBM/eval.
   Multi-horizon (30m/1h/3h/6h) = a timing profile. Expect lower-but-honest PR-AUC + real lift.
2. **Revive threat features** for A — they're leading indicators of *new* strikes (widen
   `config.THREAT_CHANNELS/VALUES/WINDOWS` back to the 6-col allowlist). Re-probe their gain on
   the onset target; this is where the project thesis should finally pay off.
3. **(B) Time-to-next-onset.** Retarget `survival.py` (currently alert *duration*) to
   hours-until-next-onset (censored). True "when": "next strike ~40min [10–90]". Do after A
   confirms onset is predictable.

**Order:** A (onset target + threat revival + re-probe) → B (survival timing).

---

## Open issues / loose ends (carry forward)

| # | Issue | Plan |
|---|---|---|
| 17 | ~~UCDP no-op stub~~ | DONE — wired, kept (6h signal). |
| 18 | Operational eval is simulated | Real publish-lag vintage snapshots (deferred, low priority). |
| 19 | Survival capped (national-only tempo) | Folds into onset reframe B (retarget survival) + TG per-oblast counts. |
| 20 | **Bq bands under-cover** (0.64 vs 0.80 nominal) | Conformal calibration on q10/q90 before shipping bands as a confidence claim. |
| 21 | **Per-oblast lift reporting** not built | Add PR-AUC-vs-base-rate per oblast to `evaluate` + viz (decided this session). |
| 22 | **Gain-share-by-group viz panel** | User flagged "very useful" — feature-group gain importance in dashboard. |
| 23 | **Present Bq as hours, not pinball** | End-user UI = expected alert-hours + plain range; pinball/coverage stay in methodology panel. |
| 24 | Walk-forward write-up | Document as run-once study ("drift slow, recency suffices"); drift viz panel now dead. |
| 25 | Clamp single-holdout caveat | Optional multi-fold confirm of 2023-07 (2024-01 was a non-monotone dip). |
