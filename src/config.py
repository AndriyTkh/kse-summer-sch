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
WAR_START = "2022-02-24"   # grid origin

# --- horizons (direct multi-horizon: one model per horizon) --------------
# Targets are FUTURE windows t -> t+H. Leak guard: features at t use data < t only.
HORIZONS = ["30m", "1h", "3h", "6h"]
HORIZON_HOURS = {"30m": 0.5, "1h": 1, "3h": 3, "6h": 6}

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
