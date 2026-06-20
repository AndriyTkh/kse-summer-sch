# PLAN.md — Current Stage: Phase 1 MVP

> Scope: **B forecasting + threat-features + A baseline.** Everything else → [STRUCTURE.md](STRUCTURE.md) roadmap.
> Cut line is firm. If a task threatens the 2-day budget, drop it to Phase 2, do not expand MVP.

---

## MVP definition of done

- [ ] 4 direct LightGBM models (30m/1h/3h/6h) predicting per-oblast alert probability.
- [ ] Prophet daily baseline; documented comparison (B beats A at short horizon).
- [ ] Threat-type features wired from the launch dataset.
- [ ] Leak-safe temporal evaluation with PR-AUC + calibration.
- [ ] Oblast × horizon probability heatmap.

---

## Build order

### Day 1 — data + core model
1. **Loaders** — Vadimkin (alerts), piterfm massive-attacks (launches), missile_daily (tempo).
2. **Master index** — hourly, UTC, oblast × hour grid from war start → cut.
   - Convert all sources Kyiv-local → UTC (watch DST).
   - **Leak guard:** every feature at row `t` uses only data with timestamp `< t`; target = future window `t→t+H`.
3. **Threat mapping table** — `model` → {ballistic, air-cruise, sea-cruise, drone-strike, drone-decoy, kinzhal}. Test it (combos, Cyrillic/Latin, typos).
4. **Feature pipeline** — lags, calendar, region, threat channels. Geo signal from piterfm `target`/`launch_place` (no impact dataset in MVP; UCDP propensity → Phase 2).
5. **B** — 4 direct LightGBM, one target shift per horizon.
6. **A** — Prophet daily baseline.

### Day 2 — eval + polish
7. **Temporal split** — train early, test last N weeks. Never random.
8. **Metrics** — PR-AUC, reliability/calibration plot (isotonic if miscalibrated).
9. **Heatmap** — oblast × horizon probabilities.
10. **Writeup** — B vs A comparison, accepted limits, roadmap pointer.
11. *(buffer)* — stubbed Phase-2 survival hook if time remains.

---

## Issue resolutions (locked)

| # | Issue | Resolution |
|---|---|---|
| 1 | Timestamp alignment | Master hourly UTC index; lag-only joins; tested join fn verified on a known date |
| 2 | Impact dataset | **Dropped from MVP.** ACLED → commercial-license/registration friction. Geo covered by piterfm `target`/`launch_place`. Phase 2: **UCDP GED** (CC-BY) as static per-oblast prior |
| 3 | massive = "massive" only | Missing hour = **0 launches** (absence = no wave); daily fills tempo |
| 4 | Geo mismatch | Normalize all to **ADM1 oblast** codelist; raion/hromada dropped in MVP |
| 5 | Class imbalance | Metric = **PR-AUC + calibration**, not accuracy; `scale_pos_weight` |
| 6 | Non-stationarity | **Temporal split mandatory** + recency weighting |
| 7 | Survival censoring | lifelines censored-flag — **Phase 2**, not MVP |
| 8 | Decoy drones | **Separate `drone-decoy` category** in mapping. ⚠️ Verify `model` tags decoys on download; if untagged early-war, they fall under strike (accept, note) |
| 9 | `model` free-text cleanup | Mapping table built + tested first; time buffered |
| 10 | Calibration honesty | Reliability plot; isotonic/Platt if needed |
| 11 | Scope creep | MVP locked; overflow → Phase 2 |

---

## Target file layout

```
kse-summer-sch/
├── claude/
│   ├── STRUCTURE.md
│   ├── PLAN.md
│   └── archive/
├── data/                  # raw downloads (gitignored)
├── src/
│   ├── loaders.py         # Vadimkin, massive-attacks, missile_daily
│   ├── index.py           # master hourly UTC grid + leak-guard join
│   ├── threat_map.py      # model → threat-type table
│   ├── features.py        # lags, calendar, threat channels (UCDP prior → Phase 2)
│   ├── model_b.py         # 4 direct LightGBM
│   ├── model_a.py         # Prophet baseline
│   └── evaluate.py        # temporal split, PR-AUC, calibration, heatmap
├── notebooks/
│   └── eda.ipynb
└── requirements.txt
```

---

## Next action
Scaffold the above (loaders + master index + threat map + B + A + eval), leak-guard baked in.
