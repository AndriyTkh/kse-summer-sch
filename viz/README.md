# viz — Air-Raid Forecast Dashboard

React + MapLibre choropleth for the per-oblast alert-probability forecast
(STRUCTURE.md §9). Zero-backend: reads pre-computed JSON + fetches live alerts
client-side.

## Run

```bash
npm install
npm run dev        # dev server (http://localhost:5173)
npm run build      # type-check + production bundle -> dist/
npm run preview    # serve the built bundle
```

## Data

The dashboard reads these files from `public/`. Phase-1 (`predictions`/`metrics`)
is the cheap headline export; each Phase-2 file is produced by its **own** script
so the heavy rolling-origin / degradation sweeps stay off the default refresh.
Every Phase-2 file is **optional** — if it's missing the matching UI panel just
hides (toggle disabled). The UI runs standalone off the committed JSON.

| File | Produced by | Feeds | Cost |
|---|---|---|---|
| `ukraine-oblasts.geojson` | committed | map polygons (keyed by `code`) | — |
| `predictions.json` | `python -m src.export_predictions` | map · "Backtest" prediction source | ~1 min |
| `metrics.json` | `python -m src.export_predictions` | stats · "Holdout" method | (same run) |
| `nowcast.json` | `python scripts/runs/run_forecast.py` | map · "Nowcast" prediction source | ~1 min |
| `walkforward.json` | `python scripts/runs/run_walkforward.py` | stats · "Walk-forward" method | ~2 min |
| `operational.json` | `python scripts/runs/run_operational_eval.py` | stats · "Operational" method | ~1 min |
| `survival.json` | `python scripts/runs/run_survival.py` | stats · "Duration" method | ~1 min |
| `intervals.json` | `python scripts/runs/run_phase3.py` | stats · Bq quantile bands | ~1 min |

> ⚠️ Each script trains LightGBM over the full ~1 M-row hourly grid (walk-forward
> retrains 4× by design) → 95 % CPU for the noted minutes. Run only the slice you
> need to refresh. They are **independent** — order doesn't matter, nothing
> overwrites another's file.

**One-go:** `python run.py` (repo root) installs deps, regenerates every JSON below
except walk-forward, then launches this dashboard. `--no-install` / `--no-viz` /
`--skip-compute` narrow that. Walk-forward stays manual (dominant cost for tiny gain).

Or regenerate one slice at a time from the repo root (prefix `PYTHONUTF8=1`; on
Windows use `.venv/Scripts/python`). The partials live under `scripts/runs/`:

```bash
PYTHONUTF8=1 python -m src.export_predictions             # backtest + holdout + onset (default)
PYTHONUTF8=1 python scripts/runs/run_forecast.py          # nowcast prediction source
PYTHONUTF8=1 python scripts/runs/run_walkforward.py       # walk-forward CV stats
PYTHONUTF8=1 python scripts/runs/run_operational_eval.py  # backtest-vs-live gap stats
PYTHONUTF8=1 python scripts/runs/run_survival.py          # alert-duration stats
PYTHONUTF8=1 python scripts/runs/run_phase3.py            # Bq quantile intervals
```

### UI switches

- **Prediction** (topbar): `Backtest` (held-out test window) ↔ `Nowcast`
  (operational next-6h edge forecast). Re-colours the map + region panel.
- **Evaluation method** (stats page tabs): `Holdout` · `Walk-forward` ·
  `Operational` · `Duration`. Disabled tabs mean that script hasn't been run.

> `ukraine-oblasts.geojson` holds **real ADM1 boundaries** (27 features: 24
> oblasts + Crimea + Kyiv City + Sevastopol) from
> [geoBoundaries](https://www.geoboundaries.org) gbOpen, simplified and rounded
> to 4 decimals (~470 KB). Each feature carries only `code` (the app's contract)
> and a display `name`. The choropleth renders over a CARTO dark raster basemap
> (no API key; CARTO + OpenStreetMap attribution shown on-map) for coastline /
> neighbour / sea context.

## Live alerts

The active-alert overlay polls [alerts.in.ua](https://alerts.in.ua), which needs a
token. Set it to enable the overlay:

```bash
echo "VITE_ALERTS_TOKEN=your_token" > .env.local
```

Without a token (or on a network error) the overlay degrades gracefully — the map
shows predictions only and the status bar reads "live alerts unavailable".

## Layout

```
src/
├── App.tsx                       shell: topbar, map, sidebar, legend; source/method state
├── types.ts                      shared JSON shapes (PredictionSource + Phase-2 blocks)
├── components/
│   ├── AlertMap.tsx              MapLibre choropleth + alert outline overlay
│   ├── HorizonToggle.tsx         30m / 1h / 3h / 6h selector
│   ├── PredictionSourceToggle.tsx  Backtest ↔ Nowcast map source
│   ├── MetricsPanel.tsx          stats page: Holdout/Walk-forward/Operational/Duration tabs
│   └── RegionPanel.tsx           per-oblast detail (region selected)
├── hooks/
│   ├── usePredictions.ts         load predictions.json (backtest)
│   ├── useMetrics.ts             load metrics.json (holdout)
│   ├── useJson.ts                generic loader; optional 404 → null (Phase-2 files)
│   └── useAlerts.ts              poll alerts.in.ua (token-gated)
└── utils/colorScale.ts           probability -> color, oblast display names
```
