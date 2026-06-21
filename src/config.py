"""Shared constants: paths, time grid, horizons, oblast codelist.

Single source of truth so every module agrees on grid, timezone, and split point.
"""

from __future__ import annotations

from pathlib import Path

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # raw downloads (gitignored)
ARTIFACTS_DIR = ROOT / "artifacts"  # models, plots, metrics (gitignored)

# --- time grid -----------------------------------------------------------
# Master grid is HOURLY, UTC. All sources are Kyiv-local → convert to UTC.
TZ_LOCAL = "Europe/Kyiv"   # watch DST when converting
TZ_GRID = "UTC"
GRID_FREQ = "1h"
WAR_START = "2022-02-24"   # true war origin (used for day_of_war feature only)
# Modeling-grid origin. A learning-curve sweep (fixed recent-2mo test, growing train depth)
# showed: 1h accuracy plateaus by ~4 months; 6h keeps gaining slowly to ~2.8yr then flattens;
# the ONLY clearly-droppable slice is the 2022 ground-war regime (frontline-LOCAL alerts, not
# the nationwide strategic-strike pattern the model targets), which adds nothing and mildly
# dilutes 6h. Clamping at 2023-07 (~1030d before the test edge) put 6h at its peak (0.930,
# slightly ABOVE full history), left 1h flat, kept per-oblast lift, and cut ~31% of rows.
# Applies to BOTH B and Bq (both build on build_master_index). day_of_war still counts from
# WAR_START. Full history stays loadable by passing oblasts=/start= explicitly. See the
# 2026-06-21 phase-4 write-down for the curve.
GRID_START = "2023-07-01"

# --- horizons (direct multi-horizon: one model per horizon) --------------
# Targets are FUTURE windows t -> t+H. Leak guard: features at t use data < t only.
HORIZONS = ["30m", "1h", "3h", "6h"]
HORIZON_HOURS = {"30m": 0.5, "1h": 1, "3h": 3, "6h": 6}

# --- threat-feature allowlist (Phase 4) ----------------------------------
# DEPRECATED for the current "whether an alert fires" model. The full cross-product
# (7 channels x {launched,waves} x {3,6,24}h = 36 cols) added only ~+1% PR-AUC, and a
# gain probe traced that to the target: predicting a PERSISTENT alert state needs only
# autocorrelation (recent lags), so launch/threat leading-indicators barely help. They
# are LEADING indicators of a NEW strike — exactly what the planned ONSET / time-to-next
# model targets. So threat is dropped here (THREAT_CHANNELS empty -> no threat columns)
# and queued for revival in the onset reframe (see PLAN.md next session). The prune
# allowlist below (launched / {6,24}h / 3 channels) is the validated lean set to bring
# back then — widen the tuples to restore. See the 2026-06-21 phase-4 write-down.
THREAT_CHANNELS: tuple[str, ...] = ()                       # deprecated -> empty
THREAT_VALUES: tuple[str, ...] = ("launched",)             # revival set for onset model
THREAT_WINDOWS: tuple[int, ...] = (6, 24)                  # revival set for onset model

# --- quantile prediction intervals (Phase 3, Model Bq) -------------------
# B outputs a point probability; Bq regresses the continuous alert-FRACTION over
# (t, t+H] at several quantiles -> a calibrated uncertainty band per oblast×horizon.
# Default 0.1/0.5/0.9 => a nominal 80% central interval [q10, q90] around the q50 median.
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
PI_LOW, PI_HIGH = 0.1, 0.9          # band edges (nominal 80% coverage)

# --- drift detection (Phase 3) -------------------------------------------
# Population Stability Index thresholds (industry rule-of-thumb): PSI < 0.1 stable,
# 0.1–0.25 moderate shift, > 0.25 significant. War is non-stationary (issue #6), so
# the auto-retrain loop watches feature PSI between a reference window and live blocks.
DRIFT_PSI_WARN = 0.10
DRIFT_PSI_ALERT = 0.25

# --- auto-retrain walk-forward (Phase 3) ---------------------------------
# Online protocol: score the current model on each forward block, THEN adapt.
RETRAIN_BLOCK_DAYS = 14            # forward evaluation block width
RETRAIN_WINDOW_DAYS = 180         # trailing window each retrain fits on (matches recency half-life)
RETRAIN_PERIOD_BLOCKS = 4         # cadence for the periodic baseline policy

# --- UCDP leak-safe lag --------------------------------------------------
# Impact data released annually; join as-of a lagged static per-oblast prior.
# (Phase 2; UCDP GED replaces dropped ACLED. Kept as generic as-of lag knob.)
UCDP_LAG_DAYS = 7

# --- evaluation ----------------------------------------------------------
# Temporal split only, never random. Hold out the last N weeks as test.
TEST_WEEKS = 8
# Calibration fold (issue #10): the last N weeks of the TRAIN remainder, held out
# to fit isotonic out-of-fold. Three-way temporal order: train_fit < calib < test.
CALIB_WEEKS = 4
# Purge gap (hours) dropped from the TRAIN side of every split. Targets span
# t -> t+H, so train rows within max-H of the cut carry labels that peek past it
# into the held-out fold (label bleed). Drop them => zero-bleed purged split.
PURGE_HOURS = max(HORIZON_HOURS.values())  # 6 (longest horizon)

# --- walk-forward backtest (Phase 2) ------------------------------------
# Rolling-origin CV: slide the test window back N times -> a variance estimate +
# drift across war regimes. The single holdout gives one verdict only (one window,
# one regime, no spread). Each fold reuses TEST_WEEKS / PURGE_HOURS above.
WALK_FORWARD_FOLDS = 4

# --- oblast codelist (ADM1) ---------------------------------------------
# Normalize every source to these codes. Raion/hromada dropped in MVP.
# Canonical ADM1 set: 24 oblasts + Kyiv city + AR Crimea + Sevastopol = 27.
# Loaders map each source's region naming onto these slugs.
OBLAST_CODES: list[str] = [
    "cherkaska",
    "chernihivska",
    "chernivetska",
    "crimea",          # AR Crimea
    "dnipropetrovska",
    "donetska",
    "ivano-frankivska",
    "kharkivska",
    "khersonska",
    "khmelnytska",
    "kyivska",         # Kyiv oblast
    "kyiv-city",       # Kyiv city
    "kirovohradska",
    "luhanska",
    "lvivska",
    "mykolaivska",
    "odeska",
    "poltavska",
    "rivnenska",
    "sevastopol",      # Sevastopol city
    "sumska",
    "ternopilska",
    "vinnytska",
    "volynska",
    "zakarpatska",
    "zaporizka",
    "zhytomyrska",
]

# Fast identity lookup: a stripped name that already equals a canonical code
# (e.g. "Cherkaska oblast" -> "cherkaska") is valid as-is.
OBLAST_CODE_SET: frozenset[str] = frozenset(OBLAST_CODES)

# Oblasts excluded from the modeling grid: occupied territory the siren dataset does
# NOT cover -> near-zero alert base rate (crimea 0.0%, sevastopol 0.0%, luhanska 0.2%).
# They are dead rows (no signal, unpredictable), not a metric-inflation problem. The
# near-PERMANENT frontline (dnipropetrovska ~82%, kharkivska/donetska ~67%) is KEPT and
# instead handled by per-oblast PR-AUC-vs-base-rate (lift) reporting so its trivial
# high base rate can't hide inside the aggregate. OBLAST_CODES stays the full canonical
# 27 (loaders/normalization need it); the grid is built from MODEL_OBLASTS.
EXCLUDED_OBLASTS: frozenset[str] = frozenset({"crimea", "sevastopol", "luhanska"})
MODEL_OBLASTS: list[str] = [o for o in OBLAST_CODES if o not in EXCLUDED_OBLASTS]

# Source region name -> ADM1 code (issue #4). Keys are lowercased, " oblast" stripped.
# Covers the `affected region` vocabulary in the Kaggle wave file + common city aliases.
# "ukraine" -> None (national marker, not an oblast). Cities fold into their oblast.
OBLAST_ALIASES: dict[str, str] = {
    "cherkasy": "cherkaska",
    "chernihiv": "chernihivska",
    "chernivtsi": "chernivetska",
    "crimea": "crimea",
    "dnipro": "dnipropetrovska",
    "dnipropetrovsk": "dnipropetrovska",
    "kryvyi rih": "dnipropetrovska",      # city in Dnipropetrovska
    "donetsk": "donetska",
    "ivano-frankivsk": "ivano-frankivska",
    "kharkiv": "kharkivska",
    "kherson": "khersonska",
    "khmelnytskyi": "khmelnytska",
    "kirovohrad": "kirovohradska",
    "kyiv oblast": "kyivska",
    "kyiv": "kyiv-city",                  # bare "Kyiv" = the city
    "kyiv city": "kyiv-city",             # sirens file label
    "luhansk": "luhanska",
    "lutsk": "volynska",                  # Lutsk = capital of Volyn
    "lviv": "lvivska",
    "mykolaiv": "mykolaivska",
    "odesa": "odeska",
    "poltava": "poltavska",
    "rivne": "rivnenska",
    "sevastopol": "sevastopol",
    "sumy": "sumska",
    "ternopil": "ternopilska",
    "vinnytsia": "vinnytska",
    "volyn": "volynska",
    "zakarpattia": "zakarpatska",
    "zaporizhzhia": "zaporizka",
    "zhytomyr": "zhytomyrska",
}
