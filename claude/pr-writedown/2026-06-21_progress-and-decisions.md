# PR Write-down — Progress & Decisions

**Timestamp:** 2026-06-21T05:09:55Z (Day 2 start)
**Scope:** Phase-1 MVP — B (LightGBM forecasting) + threat-features + A (Prophet baseline).

---

## Current progress

**Implemented & wired on real data (72 tests pass):**

| Module | State |
|---|---|
| `loaders` | Vadimkin alerts, piterfm massive-attacks, missile_daily — done |
| `index` | hourly UTC oblast×hour grid + leak-guard join; `expand_alerts_to_grid` vectorized (22×, full grid ~35s) |
| `threat_map` | `model` → threat-type table (decoy tagging dead #8; rework #9) |
| `features` | lags, calendar, region, threat channels — leak-safe |
| `model_b` | 4 direct LightGBM (30m/1h/3h/6h), one target-shift per horizon |
| `model_a` | Prophet daily baseline (one Prophet/oblast, weekly+yearly seas) |
| `evaluate` | temporal split (now purged), PR-AUC, ECE/Brier, isotonic, heatmap |
| `run_mvp.py` | end-to-end: load → grid → features → 3-way split → 4 LGBM → metrics + artifacts |

**Splits:** `train_fit < calib < test`. `TEST_WEEKS=8`, `CALIB_WEEKS=4`, `PURGE_HOURS=6`.

---

## Latest results (post-purge full-grid B)

```
horizon    base  PR-AUC   lift   ECE0    ECE  Brier0   Brier
30m       0.315   0.899   2.86  0.092  0.016   0.106   0.083
1h        0.315   0.899   2.86  0.092  0.016   0.106   0.083
3h        0.390   0.908   2.33  0.062  0.024   0.107   0.099
6h        0.457   0.916   2.00  0.057  0.038   0.111   0.110
```

- PR-AUC ~0.90+ all horizons, floor (base) 0.315→0.457. Strong discrimination.
- lift falls with H because base rises faster than PR-AUC → short-horizon = biggest
  relative win (the DoD story; B beats A most at short horizon).
- isotonic fixes calibration: ECE 30m 0.084→0.013; 6h already near-honest (0.049→0.040).
- Purge added this session; numbers held vs pre-purge (0.904–0.923) → leak-safe, real.

**B vs A (same run):** B beats A every horizon, gap biggest short (30m/1h 0.899 vs 0.857;
6h converges 0.916 vs 0.901). DoD met.

---

## Decisions this session

### 1. Purge gap added (DONE)
- `config.PURGE_HOURS = max(HORIZON_HOURS)` = 6.
- `evaluate.temporal_split(..., purge_hours=)` drops train rows in the last 6h before
  each cut. Targets span t→t+H, so without it train labels peek across the cut into the
  held-out fold (label bleed). Applies to BOTH splits in `run_mvp` via default param →
  zero-bleed end to end. `purge_hours=0` disables (test asserts full partition).
- Distinction: **8 weeks = test fold SIZE; 6h = purge gap.** Different knobs.

### 2. Single holdout → walk-forward = Phase 2 (LOGGED)
- Current eval = one temporal holdout: one 8wk window, no variance estimate, one war
  regime. Honest + leak-safe but not robustness-proof.
- Upgrade = walk-forward / rolling-origin CV: slide the cut, score many folds →
  mean ± spread + drift across regimes. → STRUCTURE.md §6 + Accepted drawbacks.

### 3. Operational / live eval — SEPARATE metric = Phase 2 (LOGGED)
- Product value = forecast the REAL next 6h. Must be scored separate from backtest.
- Backtest overstates live perf — **not leakage** (timestamp guard is correct) but
  **data availability**: recent rows complete in the historical CSV are missing/partial
  live because sources publish with lag (ragged right edge / nowcast problem).
- **Honest test = real data VINTAGE** (user insight): snapshot the sources at forecast
  time (e.g. before the overnight refresh) into a separate dataset, predict next 6h,
  validate vs the later-updated source. The stale snapshot carries the true ragged edge
  → no synthetic lag-masking needed.
- Report **backtest-vs-operational gap** as a headline = the real live capability.
- → STRUCTURE.md §6 (PHASE 2 — operational eval).

---

## Open questions

- **`forecast_now()` placement:** simple "train on all data, emit next-6h per-oblast
  probabilities" entrypoint + incomplete-data caveat — MVP or Phase 2? (undecided)

## Left for MVP done

1. ~~**Writeup** — B vs A, calibration, accepted limits, roadmap pointer.~~ DONE (below).

---

# Writeup — Phase-1 MVP

## What this is

A short-horizon forecaster for air-raid alerts in Ukraine. For every oblast and every
hour it estimates the probability that an alert will fire within the next 30 minutes,
1 hour, 3 hours, and 6 hours. The unit of prediction is one (oblast, hour); the target is
the event "an alert occurs in the window t → t+H". Four horizons = four direct models.

The dataset is rare-positive and non-stationary (war intensity drifts), so the whole
pipeline is built around two non-negotiables: a temporal-only, leak-safe evaluation, and
metrics that survive class imbalance (PR-AUC + calibration, never accuracy).

## Method in one paragraph

Hourly UTC oblast×hour grid from war start. Features at row t use only data with
timestamp < t (leak guard); targets are future windows. Three-way temporal split,
`train_fit < calib < test` (test = last 8 weeks, calib = 4 weeks before it), with a 6-hour
**purge gap** dropped from the train side of each cut so no training label — which spans up
to t+6h — can peek across the boundary. Model B = four direct LightGBM regressors, one per
horizon. Probabilities are calibrated with isotonic regression fit out-of-fold on the calib
slice. Model A = a Prophet daily baseline (one model per oblast, weekly + yearly
seasonality) broadcast to the hourly test index.

## B vs A — the headline comparison

Model B (LightGBM) beats the Prophet baseline at every horizon, and by the largest margin
at short horizons — exactly where early warning has the most value and where a
daily-seasonal baseline has the least to say.

Same-run, all four horizons:

```
horizon    base  PR-AUC(B)  PR-AUC(A)  B/A lift
30m       0.315    0.899      0.857      1.05
1h        0.315    0.899      0.857      1.05
3h        0.390    0.908      0.881      1.03
6h        0.457    0.916      0.901      1.02
```

Reading it: `base` is the positive rate (the floor a random ranker scores). PR-AUC near
0.90+ against a 0.315 floor is strong discrimination. `lift` = PR-AUC / base *falls* with
horizon even though PR-AUC *rises* — because the base rate climbs faster than the score: a
wider window almost always contains an alert, so there is less headroom to beat. Hence the
model's relative value is greatest at 30m/1h. A converges toward B at 6h, its daily-seasonal
home turf, but never overtakes it. **DoD met: B beats A, gap biggest short-horizon.**

(30m and 1h are identical: the hourly grid cannot resolve sub-hour timing, so both collapse
to the same one-step target. Splitting them needs a finer grid — roadmap.)

## Calibration

Ranking well is not the same as quoting honest probabilities. Expected Calibration Error
(ECE) measures the gap between predicted probability and observed frequency; isotonic
regression rewrites the probabilities to close that gap without changing their order (so
PR-AUC is untouched).

```
horizon   ECE raw -> calibrated    Brier raw -> calibrated
30m         0.092 -> 0.016           0.106 -> 0.083
1h          0.092 -> 0.016           0.106 -> 0.083
3h          0.062 -> 0.024           0.107 -> 0.099
6h          0.057 -> 0.038           0.111 -> 0.110
```

The raw model is overconfident at short horizons (ECE 0.092) and isotonic cuts it ~5.8×.
At 6h the raw probabilities are already nearly honest, so there is little to fix and Brier
is flat. After calibration every horizon sits at ECE ≤ 0.04 — the quoted probabilities can
be trusted as frequencies, which matters the moment anyone acts on the number.

## Accepted limits

- **Single temporal holdout.** One 8-week test window = one verdict, no variance estimate,
  one war regime. Honest and leak-safe, but not robustness-proof. → walk-forward CV, Phase 2.
- **Backtest overstates live performance.** Not leakage (the timestamp guard is correct) but
  data *availability*: the most recent rows are complete in the historical CSV yet would be
  missing/partial live, because sources publish with lag (ragged right edge). The real
  product — forecasting the actual next 6h — must be scored separately. → operational/vintage
  eval, Phase 2.
- **6h horizon ceiling.** Deliberate domain cut; beyond 6h timing erodes. Not a falloff in
  the table — the model simply stops there.
- **No native extrapolation / no free uncertainty bands** (point estimate; quantile LightGBM
  is roadmap). A patches long-horizon trend.
- **30m == 1h** until a sub-hourly grid exists.
- **Decoy tagging dead (#8); threat_map needs rework (#9).**

## Roadmap pointer

Full vision and phasing in [STRUCTURE.md](../STRUCTURE.md). Next concrete steps, in order of
honesty-payoff: (1) operational/vintage eval to quantify the backtest-vs-live gap;
(2) walk-forward backtest for variance + drift; (3) Phase-2 duration model (survival,
lifelines). The open call is whether a simple `forecast_now()` entrypoint + incomplete-data
caveat lands in this PR or defers to Phase 2.
