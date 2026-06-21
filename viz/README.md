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

The dashboard reads three files from `public/`:

| File | Produced by | Contents |
|---|---|---|
| `ukraine-oblasts.geojson` | committed | ADM1 oblast polygons keyed by `code` |
| `predictions.json` | `python -m src.export_predictions` | latest per-oblast probability per horizon |
| `metrics.json` | `python -m src.export_predictions` | aggregate + per-oblast PR-AUC / calibration, B-vs-A |

`predictions.json` / `metrics.json` currently hold **sample data** so the UI runs
standalone. Regenerate real values from the repo root once models are trained:

```bash
python -m src.export_predictions
```

> The bundled `ukraine-oblasts.geojson` is a **simplified placeholder** (hex-ish
> shapes from oblast bounding boxes) — the network egress in the build env blocked
> GADM/Natural-Earth. Swap in accurate boundaries for production; the `code`
> property is the only contract the app relies on.

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
├── App.tsx                 shell: topbar, map, sidebar, legend
├── components/
│   ├── AlertMap.tsx        MapLibre choropleth + alert outline overlay
│   ├── HorizonToggle.tsx   30m / 1h / 3h / 6h selector
│   ├── MetricsPanel.tsx    aggregate accuracy (no region selected)
│   └── RegionPanel.tsx     per-oblast detail (region selected)
├── hooks/
│   ├── usePredictions.ts   load predictions.json
│   ├── useMetrics.ts       load metrics.json
│   └── useAlerts.ts        poll alerts.in.ua (token-gated)
└── utils/colorScale.ts     probability -> color, oblast display names
```
