# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

Time-series analysis of air-raid alerts in Ukraine. Predict short-horizon (≤6h) per-oblast
alert probability, and (phase 2) alert duration. AI-assisted defense pet-project, 2-day MVP.

## Where things live

- **[claude/STRUCTURE.md](claude/STRUCTURE.md)** — target full structure: all phases, data layers, models, features, roadmap. The complete vision.
- **[claude/PLAN.md](claude/PLAN.md)** — current implementation stage (Phase 1 MVP). Build order, locked issue resolutions, file layout. **This is the active to-do.**
- **claude/archive/** — deprecated PLAN/STRUCTURE versions. **Only move files here when the user explicitly says so.**

Read STRUCTURE.md for the *why*, PLAN.md for the *what now*. When they conflict, ask.

## Scope discipline

- MVP cut line is firm: **B (LightGBM forecasting) + threat-features + A (Prophet baseline)**.
- Anything outside that → it is roadmap (STRUCTURE.md), not MVP. Do not expand MVP; overflow drops to Phase 2.
- If a task threatens the 2-day budget, flag it and defer rather than cut corners.

## Working conventions

- **Leak guard is non-negotiable:** every feature at row `t` uses only data with timestamp `< t`; targets are future windows. Evaluation split is **temporal, never random**.
- Metrics: **PR-AUC + calibration** (data is rare-positive; accuracy is misleading).
- All datasets are bulk/CSV/API — **no Telegram scraping in MVP** (it is Phase 3).
- Master time grid: **hourly, UTC** (convert Kyiv-local, watch DST).
- Verify the `model`-field decoy tagging on data download (open issue #8 in PLAN.md).

## Stack

`pandas numpy lightgbm prophet scikit-learn matplotlib` (CPU). `lifelines` phase 2.
`pytorch` only for the optional C/GRU comparison. RTX 4060 8GB idle unless C is built.
