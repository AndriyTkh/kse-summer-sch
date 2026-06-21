# STRUCTURE.md

> The independent, authoritative description of the project: complete goal, tech stack, repo
> structure, constraints, and key design decisions. Self-contained — does not depend on any
> other doc. The repo structure section is **living**: update it as modules are added.

---

## 1. Goal (complete)

Time-series analysis of air-raid alerts in Ukraine. Two prediction tasks:

1. **Forecasting** — probability of an air-raid alert per oblast over a short horizon (≤6h),
   output as an oblast × horizon probability heatmap (30m / 1h / 3h / 6h).
2. **Duration** — given an alert started, time until all-clear.

Defense value: shelter/resource planning and early warning. Deliverable framing —
*"short-horizon (≤6h) per-oblast alert probability, with OSINT launch / strategic-aviation
leading indicators."*

---

## 2. Constraints

- **Time:** 2-day mini pet-project. Forces a hard MVP cut (see §6).
- **Skill stack:** classical + tabular ML solid; advanced deep learning (RNN family) is stretch only.
- **Hardware:** RTX 4060 laptop, 8GB VRAM. MVP is CPU-only; GPU idle unless the C/GRU comparison runs.
- **No scraping in MVP:** all data must be bulk/CSV/API. Telegram scraping is roadmap (Phase 3).
- **Domain limits (not bugs):**
  - Event-timing horizon caps ~6h — alerts are adversary decisions, not a natural process.
  - War is non-stationary (2022 ≠ 2025 Shahed-swarm era). Concept drift > sample count.

---

## 3. Tech stack

```
core:     pandas numpy lightgbm scikit-learn matplotlib
baseline: prophet
phase 2:  lifelines
stretch:  pytorch (C/GRU), statsmodels (E/anomaly)
```

CPU-only for MVP.

---

## 4. Data layers (launch → alert → impact)

| Layer | Dataset | Captures | Timing | Grid |
|---|---|---|---|---|
| Alert (target) | **Vadimkin** air-raid-sirens-dataset | alert on/off per oblast, start/end | real-time | hourly |
| Launch | **piterfm massive-attacks** | per-wave type/count/origin, time_start/end | per wave | hourly |
| Tempo | **piterfm missile_attacks_daily** | daily national launch totals | daily | daily (A) |
| Impact (opt, Phase 2) | **UCDP GED** | geolocated fatal events per oblast | annual release (ends 2024) | static prior |
| Real-time geo | **TG monitor channels** (roadmap) | live "N drones → oblast Y" | real-time | Phase 3 |

Sources:
- https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
- https://www.kaggle.com/datasets/piterfm/massive-missile-attacks-on-ukraine/data
- https://github.com/PetroIvaniuk/2022-Ukraine-Russia-War-Dataset
- UCDP GED: https://ucdp.uu.se/downloads/ (CC-BY, bulk CSV, no gate) — Phase 2 impact prior
  - (ACLED dropped: commercial-license clause + registration gate + tier caps; UCDP cleaner)

---

## 5. Models

| Model | Tech | Resolution | Role |
|---|---|---|---|
| **B** | LightGBM | hourly | **Core.** 4 direct models (30m/1h/3h/6h) → oblast×horizon heatmap |
| **A** | Prophet | daily | Long-horizon load/calendar baseline; B must beat it |
| **Survival** | lifelines | per-event | Alert duration (time-to-all-clear) |
| **C** | GRU (PyTorch) | hourly | Comparison only — expected to lose; report breadth |
| **E** | IsolationForest / STL | hourly | Anomaly flag, accuracy ablation (optional) |

### Features
```
lags / calendar / region
threat-type channels:  ballistic | air-cruise | sea-cruise | drone | kinzhal
drone split:           strike (Shahed-136/131) vs decoy (Gerbera) vs recon (Orlan/ZALA)
                       (decoy channel empty in current data — see PLAN #8; recon added data-driven)
launch_place origin:   Engels/Olenya... (structured strategic-aviation signal)
drone tempo:           daily launch count (feeds A)

optional / Phase 2:
  target geo (MVP):    piterfm massive-attacks `target` + `launch_place` — coarse oblast signal, no extra dataset
  UCDP propensity:     per-oblast static impact prior (CC-BY, Phase 2; replaces ACLED)
  anomaly flag:        IsolationForest / STL
```
Threat-type is the highest-leverage feature after raw lags: each type powers a different horizon
(ballistic/kinzhal → 30m–1h; air-cruise from bomber bases → 3h–6h; drones → duration + spread).

---

## 6. Phasing

```
PHASE 1 — MVP (2 days):   B forecasting + threat-features + A baseline              ✅ DONE
PHASE 2 — evaluation:      walk-forward backtest (rolling-origin CV) replacing the   ✅ DONE
                           single temporal holdout: slide the train/test cut forward,
                           score many folds -> mean ± spread + drift across war regimes
PHASE 2 — nowcast:         forecast_now operational entrypoint — train on all data,   ✅ DONE
                           emit next-6h per-oblast calibrated probabilities at the
                           grid edge. Ragged-right-edge caveat carried in output.
PHASE 2 — operational eval: forward/live forecast scored SEPARATELY from backtest.  ✅ DONE (simulated)
                           Backtest overstates live perf: not leakage (timestamp guard
                           is correct) but DATA AVAILABILITY — recent rows complete in
                           the historical CSV are missing/partial live (source publish
                           lag = ragged right edge / nowcast problem). Simulated vintage:
                           degrade threat/tempo features by zeroing last N hours of the
                           test fold, sweep multiple lag scenarios (3/6/12/24h), report
                           backtest-vs-degraded PR-AUC gap per horizon. Real vintage
                           snapshots (Phase 3 data pipeline) will replace the simulation.
PHASE 2 — duration:        survival (lifelines), reuses Phase-1 covariates           ✅ DONE
                           Kaplan-Meier baseline + Cox PH with hourly features at
                           alert start. Censoring-aware (issue #7). Accuracy capped
                           until Phase-3 per-oblast swarm counts.
PHASE 3+ — roadmap:        TG real-time scrape · nowcast tier · quantile intervals ·
                           auto-retrain (drift) · C/TCN/TFT compare · Hawkes ·
                           spatial wave-propagation · multi-channel OSINT fusion
```

Phase-2 duration accuracy is capped until Phase-3 real-time per-oblast counts land
(swarm size at alert start is the primary duration driver; bulk data is national-only).

---

## 7. Repo structure (living — update as modules are added)

```
kse-summer-sch/
├── CLAUDE.md
├── claude/
│   ├── STRUCTURE.md
│   ├── PLAN.md
│   └── archive/            deprecated PLAN/STRUCTURE versions (only on explicit say-so)
├── data/                   raw downloads (gitignored)              [planned]
├── artifacts/              models, plots, metrics (gitignored)     [planned]
├── src/
│   ├── config.py           paths, grid, horizons, oblast codelist + aliases
│   ├── loaders.py          Vadimkin alerts + massive-attacks + missile_daily
│   ├── index.py            master hourly UTC grid + leak-guard join
│   ├── threat_map.py       model → threat-type table (7 channels, real-data verified)
│   ├── features.py         lags, calendar, threat channels; UCDP prior [Phase 2]
│   ├── model_b.py          4 direct LightGBM
│   ├── model_a.py          Prophet baseline
│   ├── forecast.py         operational nowcast — forecast_now at the grid edge
│   ├── operational_eval.py simulated vintage eval — backtest-vs-live gap quantification
│   ├── survival.py         Phase-2 duration: KM + Cox PH alert time-to-all-clear
│   └── evaluate.py         temporal split + walk-forward CV, PR-AUC, calibration, heatmap
├── run_mvp.py              Phase-1 headline: single-holdout B + A + artifacts
├── run_walkforward.py      Phase-2: rolling-origin CV (mean ± spread + drift)
├── run_forecast.py         Phase-2: emit next-6h per-oblast nowcast (calibrated)
├── run_operational_eval.py Phase-2: sweep source-lag scenarios, report PR-AUC gap
├── run_survival.py         Phase-2: KM + Cox PH duration model, C-index + MAE
├── tests/                  86 passing, 6 skipped (Prophet) — full pipeline + Phase-2
│   ├── test_threat_map.py
│   ├── test_index.py
│   ├── test_loaders.py
│   ├── test_features.py
│   ├── test_model_a.py
│   ├── test_model_b.py
│   ├── test_evaluate.py
│   ├── test_walk_forward.py    Phase-2: rolling-origin folds + B eval/summary
│   ├── test_forecast.py        Phase-2: forecast_now edge nowcast
│   ├── test_operational_eval.py Phase-2: degradation + gap direction
│   └── test_survival.py        Phase-2: survival dataset, KM, Cox, temporal split
├── notebooks/
│   └── eda.ipynb                                                   [planned]
└── requirements.txt
```

Mark modules `[planned]` until they exist; drop the tag once built.

---

## 8. Key design decisions

### Problem framing
Surveyed 5 problem types (forecasting, anomaly, duration, spatio-temporal clustering,
classification). Chose **forecasting (core) + duration (phase 2)**. Classification dropped as a
standalone task — covered by defense scouting; threat-type instead enters as a *feature*.

### Why B (LightGBM) over C
Engineered features (threat-type, launch_place, counts) hand the wave structure to trees
pre-computed — exactly the long-pattern signal C would have to learn from raw sequence. Richer
data therefore **widens B's lead**. B is also CPU-light, drift-robust, interpretable, and
absorbs A's seasonality via calendar features.

### C (alternative design, kept as comparison)
If pursued: a sequence model on the hourly grid.
- **RNN family:** RNN (vanishing gradient, forgets long past) → LSTM (3 gates + cell state, long
  memory, more params, overfit risk on small data) → **GRU** (2 gates reset/update, no separate
  cell state, lighter, faster, less data). Pick GRU for this scale.
- Edge = autonomous sequence memory for long patterns — but that edge is mostly *absorbed* once
  threat-type/launch features are engineered, so C is expected to lose here.
- Roadmap deep alternatives: **TCN** (1D conv, parallel, ≥ GRU often), **TFT** (multi-horizon +
  attention interpretability + intervals). Both use the 4060.

### Other rejected / deferred methods
- **D — Hawkes self-exciting point process:** conceptually ideal for attack-wave cascades; niche/hard. Roadmap-only.
- **E — anomaly (IsolationForest/STL):** optional accuracy ablation, not core.

### Multi-horizon = direct method
One model per horizon (target shifted to its distance), not recursive (recursive compounds error,
garbage by ~24h).

### Resolution per model
A daily (smooth seasonal), B hourly (sharp wave timestamps), survival per-event. Same raw
resampled to each grid — not a conflict.

### Extrapolation vs horizon (distinct)
Extrapolation = output value range (trees cap at training max; A extends trend). Horizon = time
forward (B sharp ~6h; A long but seasonal-average only).

### Feature precision is capped by training-history resolution
Not by inference-time data. Real-time signals need matched-resolution history OR a separate
hybrid/nowcast layer → daily/hourly bulk for MVP; real-time TG = roadmap.

### Strategic-aviation leading indicators
MiG-31K airborne → Kinzhal ballistic ~10–40min (near-deterministic, country-wide) powers short
horizons; Tu-95/Tu-160 from Engels → cruise wave 3–6h lead powers long horizons. Folded into
structured data via the launch dataset's `launch_place` — no scrape needed for MVP.

### Accepted drawbacks
- No native extrapolation (A patches trend at long horizon).
- No free uncertainty bands (point estimate; quantile LightGBM = roadmap).
- 6h event-timing ceiling (domain limit).
- Single temporal holdout: one 8wk test window = one verdict, no variance estimate,
  one war regime. Honest + leak-safe but not robustness-proof; walk-forward = Phase 2.
