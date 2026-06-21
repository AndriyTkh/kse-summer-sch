# Air-Raid Alert Forecasting — Ukraine

Short-horizon (≤6h) **per-oblast air-raid alert probability** for Ukraine, plus alert
**duration** (time-to-all-clear). Built as a 2-day AI-assisted pet-project for the
**KSE AI Agentic Summer School, Stage 2**.

Defense value: shelter / resource planning and early warning, using only bulk/CSV/API
data (no scraping) and OSINT launch + strategic-aviation leading indicators.

> **Reviewers, start here:** this README has two jobs — (1) tell you how to run the code,
> and (2) show you **how the project was built with AI** (where the logs are, how they're
> organized, and which conversations carry the real human steering). See
> [The AI process](#the-ai-process--how-this-was-built) below.

---

## What it does

| Task | Model | Output |
|---|---|---|
| **Forecast** (core) | 4 direct LightGBM models (30m / 1h / 3h / 6h) | per-oblast × horizon alert probability heatmap |
| **Onset** | LightGBM | probability an alert *starts* in the window (pivot from "is an alert on") |
| **Baseline** | Prophet (daily) | long-horizon seasonal/calendar load — B must beat it |
| **Duration** | lifelines (KM + Cox PH) | time until all-clear given an alert started |
| **Intervals** | quantile LightGBM (Bq) | q10 / q50 / q90 uncertainty bands |

Results render in an interactive **React + MapLibre** dashboard (`viz/`): Ukraine
choropleth coloured by predicted probability, live alert overlay, horizon toggle, and a
metrics panel (PR-AUC + calibration; data is rare-positive so accuracy is misleading).

**Leak guard is non-negotiable:** every feature at row `t` uses only data with timestamp
`< t`; targets are future windows; the evaluation split is **temporal, never random**.

---

## Quick start

Requirements: **Python 3.11+** and **Node.js** (for the dashboard).

```bash
python run.py
```

One command: installs deps → computes all dashboard JSON → launches the viz dev server.

Useful flags:

```bash
python run.py --no-install     # skip pip/npm install
python run.py --no-viz         # compute JSON only, don't launch the dashboard
python run.py --skip-compute   # just launch on the JSON already in viz/public/
```

Tests (86 passing, 6 skipped — Prophet):

```bash
python -m pytest
```

> Raw data (`data/`) and trained artifacts (`artifacts/`) are gitignored. The repo ships
> the **generated `viz/public/*.json`**, so `python run.py --skip-compute` shows the
> dashboard without a full data download.

---

## Repo map

The two planning docs are the spine of the project — read these for the *why*:

- **[claude/STRUCTURE.md](claude/STRUCTURE.md)** — the complete, authoritative vision:
  goal, data layers, model menu, design decisions, full roadmap.
- **[claude/PLAN.md](claude/PLAN.md)** — the active to-do: MVP cut line, build order,
  locked issue resolutions.
- **[CLAUDE.md](CLAUDE.md)** — working conventions handed to the AI on every session.

```
src/        loaders, feature engineering, models (B / Bq / A / survival), eval, export
scripts/    runs/  standalone partial entrypoints; data_freshness + update_data
viz/        React + MapLibre dashboard (reads pre-computed JSON, zero backend)
tests/      full pipeline + Phase-2/3 coverage
run.py      one-go driver (install → compute → serve)
_sessions/  exported AI conversation logs  ← see next section
```

Full module-by-module tree lives in [claude/STRUCTURE.md §7](claude/STRUCTURE.md).

---

## The AI process — how this was built

This project was developed almost entirely through AI pair-programming (Claude Code). Per
Stage-2 requirements, **the full conversation history is included in this repo** so the
*process* is auditable, not just the result.

### Where the logs are

```
_sessions/
├── INDEX.md            ← ranked table of every conversation
├── important/          ← the decision-driving sessions (hand-picked)
└── <date>_<id>.md      ← one readable Markdown transcript per session
```

Each transcript is a cleaned render of a real Claude Code session: human prompts,
assistant replies, tool calls (one-line summaries), and collapsible tool results /
thinking. The raw `.jsonl` event streams Claude Code stores locally are far too noisy to
read directly — `_sessions/` is the human-readable export of them.

### How to read them

Open **[`_sessions/INDEX.md`](_sessions/INDEX.md)** first. Every session is ranked by
**human input** — the number of characters the human actually typed (steering, pushback,
corrections), not autonomous grind. The higher a session sits, the more it represents a
human guiding/challenging the model rather than rubber-stamping output.

The **highest-signal conversations** (architecture planning, the data-reality analysis
that reshaped the feature set, validation/calibration fixes) are copied into
**`_sessions/important/`** so a reviewer can read the few that matter without wading
through all 21.

> **Why a structure file may be pasted into the submission form instead of the full log:**
> the complete transcripts total ~800 KB — far past what the submission form accepts. The
> planning docs in **`claude/`** (STRUCTURE.md + PLAN.md) are the condensed, human-readable
> record of the architecture the AI was guided toward, and double as the digestible
> "what we decided and why" narrative. For the *full* logs, read `_sessions/`.

### Regenerating the logs

The export is reproducible:

```bash
python scripts/export_sessions.py          # write _sessions/ + INDEX.md
python scripts/export_sessions.py --list    # just print the human-input ranking
```

The script ([scripts/export_sessions.py](scripts/export_sessions.py)) finds this repo's
Claude Code sessions under `~/.claude/projects/`, matches them by working directory,
renders each to Markdown, and ranks them by typed-prompt volume. The `important/` folder
is curated by hand and is **not** overwritten by re-running the export.

---

## Scope & honesty notes

- MVP cut line was firm: **LightGBM forecasting + threat-type features + Prophet baseline**.
  Everything past that (walk-forward CV, survival, quantile intervals, drift retrain) is
  later-phase work, clearly marked as such in `claude/`.
- **Non-stationarity is the hard part:** the 2022 war is not the 2025 Shahed-swarm era.
  Concept drift outweighs sample count; this is handled with recency weighting + (Phase 3)
  PSI-drift-triggered retraining, and the limits are documented rather than hidden.
- Walk-forward CV is intentionally **excluded** from the one-go `run.py` (dominant compute
  cost for ≤0.005 PR-AUC gain). Run it standalone if needed:
  `python scripts/runs/run_walkforward.py`.
