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

# --- ACLED leak-safe lag -------------------------------------------------
# Impact data coded ~days later; join as-of t-7d rolling impact.
ACLED_LAG_DAYS = 7

# --- evaluation ----------------------------------------------------------
# Temporal split only, never random. Hold out the last N weeks as test.
TEST_WEEKS = 8

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
