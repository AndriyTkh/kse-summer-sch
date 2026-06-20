"""Model A — Prophet daily baseline.

Long-horizon load/calendar seasonal baseline. B must beat A at short horizon
(that comparison is the deliverable, PLAN MVP done-criteria).

Daily resolution (smooth seasonal), per-oblast or national. Captures weekly/holiday
load and trend; no sharp wave timing — that is B's job.
"""

from __future__ import annotations

import pandas as pd

from . import config


def to_daily(grid: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly alert grid -> daily alert rate per oblast (Prophet `ds`/`y`)."""
    raise NotImplementedError


def fit(daily: pd.DataFrame):
    """Fit Prophet on the training span only (temporal split). Returns model(s)."""
    raise NotImplementedError


def predict(model, future: pd.DataFrame) -> pd.DataFrame:
    """Daily baseline forecast for the held-out span."""
    raise NotImplementedError
