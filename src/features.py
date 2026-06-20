"""Feature pipeline on the master grid. Leak-safe by construction.

All features at row t derive from data strictly before t.

Channels:
  lags          — recent alert history per oblast (t-1, t-3, t-6, t-24, rolling means)
  calendar      — hour, dow, is_night, day-of-war (absorbs A's seasonality)
  region        — oblast id / one-hot / frontline proximity
  threat        — per-type launch counts/recency (ballistic|air-cruise|sea-cruise|
                  drone-strike|drone-decoy|kinzhal) from massive-attacks
  launch_place  — strategic-aviation origin (Engels/Olenya...) leading indicator
  tempo         — daily national launch count (missile_daily)
  acled         — per-oblast rolling impact, 7-day-lagged (issue #2)
"""

from __future__ import annotations

import pandas as pd

from . import config


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alert-history lags + rolling means per oblast. Only t-k, k>=1."""
    raise NotImplementedError


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, day-of-week, night flag, day-of-war. Deterministic from ts."""
    raise NotImplementedError


def add_threat_features(df: pd.DataFrame, waves: pd.DataFrame) -> pd.DataFrame:
    """Per-threat-type launch counts + recency, lag-joined (< t)."""
    raise NotImplementedError


def add_tempo_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Daily national launch tempo, shifted to avoid same-day leak."""
    raise NotImplementedError


def add_acled_features(df: pd.DataFrame, acled: pd.DataFrame) -> pd.DataFrame:
    """Per-oblast rolling impact propensity, joined as-of t - ACLED_LAG_DAYS."""
    raise NotImplementedError


def build_feature_matrix(grid: pd.DataFrame, sources: dict) -> pd.DataFrame:
    """Run all feature builders in order, return model-ready matrix.

    `sources` carries waves/daily/acled frames. Asserts no column references >= t.
    """
    raise NotImplementedError
