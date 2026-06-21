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
PHASE 1 — MVP (2 days):   B forecasting + threat-features + A baseline
PHASE 2 — duration:        survival (lifelines), reuses Phase-1 covariates
PHASE 2 — evaluation:      walk-forward backtest (rolling-origin CV) replacing the
                           single temporal holdout: slide the train/test cut forward,
                           score many folds -> mean ± spread + drift across war regimes
PHASE 2 — operational eval: forward/live forecast scored SEPARATELY from backtest.
                           Backtest overstates live perf: not leakage (timestamp guard
                           is correct) but DATA AVAILABILITY — recent rows complete in
                           the historical CSV are missing/partial live (source publish
                           lag = ragged right edge / nowcast problem). Honest test =
                           real data VINTAGE: snapshot the sources at forecast time
                           (e.g. before overnight refresh) into a separate dataset,
                           predict next 6h, validate vs the later-updated source. The
                           stale snapshot carries the true ragged edge -> no synthetic
                           lag-masking needed. Report backtest-vs-operational gap as a
                           headline (the real live capability).
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
│   ├── loaders.py          massive-attacks + missile_daily done; alerts stubbed (UCDP Phase 2)
│   ├── index.py            master hourly UTC grid + leak-guard join
│   ├── threat_map.py       model → threat-type table (7 channels, real-data verified)
│   ├── features.py         lags, calendar, threat channels [planned]; UCDP prior [Phase 2]
│   ├── model_b.py          4 direct LightGBM                       [planned]
│   ├── model_a.py          Prophet baseline                        [planned]
│   ├── evaluate.py         temporal split, PR-AUC, calibration, heatmap [planned]
│   └── export_predictions.py  runs A+B → predictions.json + metrics.json
├── viz/                    React + MapLibre dashboard (build passes)
│   ├── public/             ukraine-oblasts.geojson + generated JSONs
│   └── src/                App, AlertMap, HorizonToggle, MetricsPanel, RegionPanel
├── tests/                  threat_map + index + loaders (46 passing)
│   ├── test_threat_map.py
│   ├── test_index.py
│   └── test_loaders.py
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

---

## 9. Visualization dashboard (Phase 2)

Interactive map-based UI for viewing predictions and model accuracy.

### Goal
Display per-oblast alert probabilities on a Ukraine map with live alert overlay,
switchable prediction horizons, and a metrics panel comparing models A and B.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Python export script (src/export_predictions.py)   │
│  runs both models → predictions.json + metrics.json │
└──────────────────────┬──────────────────────────────┘
                       │  static JSON files
┌──────────────────────▼──────────────────────────────┐
│  React + MapLibre GL  (viz/)                        │
│  ├── choropleth map (oblast polygons)               │
│  ├── horizon toggle (30m / 1h / 2h / 3h / 6h)      │
│  ├── live alert overlay (alerts.in.ua API)          │
│  └── side panel (metrics / region detail)           │
└─────────────────────────────────────────────────────┘
```

Zero-backend: the frontend reads pre-computed JSON and fetches live alerts
client-side. No inference at serve time.

### Data flow

**Pre-compute layer** (`src/export_predictions.py`):
- Loads trained B models (all horizons) and A model.
- Runs inference on latest available data for every oblast × horizon.
- Exports:
  - `predictions.json` — `{ oblast: { "30m": p, "1h": p, ... }, ... }` with
    timestamps and model source (A or B).
  - `metrics.json` — aggregate PR-AUC, calibration error, per-oblast accuracy,
    B-vs-A delta per horizon.
- Re-run to refresh (manual or cron); stale predictions are timestamped so the
  UI can show data age.

**Live alert layer** (client-side fetch):
- Source: `https://alerts.in.ua` API (free tier, JSON).
- Polled on page load + interval (~60s).
- Alerting oblasts get a distinct visual treatment (hatched fill / bold outline)
  layered on top of the probability choropleth.

### UI components

| Component | View | Content |
|---|---|---|
| **Map** | always | Ukraine choropleth, fill-color = predicted probability (green → yellow → red gradient). Active alerts overlaid as hatched/outlined regions. |
| **Horizon toggle** | always | Switch between 30m / 1h / 2h / 3h / 6h — re-colors the map per selected horizon. |
| **Metrics panel** | no region selected | Aggregate accuracy: PR-AUC (B and A), calibration curve, B-vs-A delta table per horizon, data freshness timestamp. |
| **Region panel** | oblast clicked | Per-oblast detail: probability per horizon (bar chart), historical hit rate, recent alert timeline, B-vs-A comparison for that oblast. |

### Map details

- **GeoJSON source:** Ukraine ADM1 oblast boundaries (public domain / OSM-derived).
  Stored as `viz/public/ukraine-oblasts.geojson`.
- **Color scale:** continuous diverging, anchored: 0.0 = green (#22c55e),
  0.5 = yellow (#eab308), 1.0 = red (#ef4444). Probability is the fill-opacity
  driver so low-risk regions stay subtle.
- **Active alert styling:** oblasts with a current alert get a diagonal hatch
  pattern overlay + thicker border (2px → 4px), independent of probability color.
  This separates "model thinks risk is high" from "alert is already active."
- **Interaction:** click an oblast to select it (opens region panel); click
  again or click empty space to deselect (returns to metrics panel).

### Tech stack (frontend)

```
react (vite)          — build tooling
maplibre-gl           — map rendering
react-map-gl          — React wrapper for MapLibre
recharts              — charts in the side panel (calibration curve, bar charts)
```

### File layout

```
viz/
├── public/
│   ├── ukraine-oblasts.geojson
│   ├── predictions.json        (generated by export script)
│   └── metrics.json            (generated by export script)
├── src/
│   ├── App.tsx
│   ├── components/
│   │   ├── AlertMap.tsx         MapLibre choropleth + alert overlay
│   │   ├── HorizonToggle.tsx    horizon selector (30m–6h)
│   │   ├── MetricsPanel.tsx     aggregate accuracy (no region selected)
│   │   └── RegionPanel.tsx      per-oblast detail (region selected)
│   ├── hooks/
│   │   ├── usePredictions.ts    load + parse predictions.json
│   │   ├── useMetrics.ts        load + parse metrics.json
│   │   └── useAlerts.ts         poll alerts.in.ua API
│   └── utils/
│       └── colorScale.ts        probability → fill color mapping
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### Accepted limitations
- **Stale predictions:** static JSON means forecasts age until re-export.
  Timestamp shown in UI; could add FastAPI live-inference backend later.
- **Alert API dependency:** if alerts.in.ua is down or rate-limited, the
  overlay degrades gracefully (shows predictions only, no alert layer).
- **No auth / no deployment target in MVP:** runs locally (`npm run dev`).
  Deployment (Vercel/GH Pages) is trivial but not scoped here.
