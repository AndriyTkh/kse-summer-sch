"""Master hourly UTC grid + leak-guard joins.

The grid is the spine: (oblast x hour) from WAR_START -> cut. Everything joins onto it.

LEAK GUARD (non-negotiable):
  - Every feature at row t uses ONLY data with timestamp < t.
  - Targets are FUTURE windows t -> t+H.
  - Joins are as-of/lag-only; verified on a known date (issue #1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_master_index(end=None, oblasts=None, start=None) -> pd.DataFrame:
    """Empty (oblast x hour) grid, UTC, hourly, from `start` (WAR_START) to `end`.

    Returns a DataFrame with a MultiIndex (oblast, ts_utc) and no feature columns.
    `end` defaults to now (UTC, floored to the hour).
    """
    start = pd.Timestamp(start or config.WAR_START, tz=config.TZ_GRID)
    if end is None:
        end = pd.Timestamp.now(tz=config.TZ_GRID).floor(config.GRID_FREQ)
    else:
        end = pd.Timestamp(end, tz=config.TZ_GRID)

    oblasts = list(oblasts) if oblasts is not None else list(config.OBLAST_CODES)
    if not oblasts:
        raise ValueError("No oblasts: populate config.OBLAST_CODES or pass oblasts=")

    hours = pd.date_range(start, end, freq=config.GRID_FREQ, tz=config.TZ_GRID)
    idx = pd.MultiIndex.from_product([oblasts, hours], names=["oblast", "ts_utc"])
    return pd.DataFrame(index=idx)


def expand_alerts_to_grid(grid: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """Mark each (oblast, hour) cell alert=1 if an alert interval overlaps it.

    `alerts` columns: [oblast, start_utc, end_utc] (tz-aware UTC). A cell covering
    [ts, ts+1h) is positive if any interval intersects it. This is the RAW label;
    per-horizon target shifting happens in model_b (never here — keep leak surface small).

    Vectorized: the grid is sorted to (oblast, ts) so each oblast's hours form a sorted,
    contiguous block; per interval we `searchsorted` the slice of cells it overlaps and
    fill it — O(intervals * log hours), no per-cell Python. Overlap rule is unchanged:
    cell [ts, ts+step) overlaps [start, end) iff ts < end and ts+step > start, i.e.
    ts in (start - step, end). Assumes the regular hourly grid from build_master_index.
    """
    out = grid.sort_index()
    n = len(out)
    flag = np.zeros(n, dtype="int8")
    if alerts.empty:
        out["alert"] = flag
        return out

    step = pd.Timedelta(config.GRID_FREQ)
    obl = out.index.get_level_values("oblast")
    ts_vals = out.index.get_level_values("ts_utc").values   # datetime64[ns] UTC
    pos = np.arange(n)
    # Per oblast: absolute positions + its sorted hour array (one O(n) pass per oblast).
    blocks = {}
    for ob in pd.unique(obl):
        m = obl == ob
        blocks[ob] = (pos[m], ts_vals[m])

    for row in alerts.itertuples(index=False):
        block = blocks.get(row.oblast)
        if block is None:
            continue
        bpos, hrs = block
        lo_key = (pd.Timestamp(row.start_utc) - step).to_datetime64()  # ts > start-step
        hi_key = pd.Timestamp(row.end_utc).to_datetime64()             # ts < end
        lo = np.searchsorted(hrs, lo_key, side="right")
        hi = np.searchsorted(hrs, hi_key, side="left")
        if hi > lo:
            flag[bpos[lo:hi]] = 1

    out["alert"] = flag
    return out


def asof_join(
    grid: pd.DataFrame,
    source: pd.DataFrame,
    *,
    on: str,
    lag="0h",
    by: str | None = None,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Leak-safe backward as-of join.

    Attaches, for each grid row at time t, the most recent `source` row whose
    timestamp is STRICTLY before (t - lag). Enforced via:
      - shifting source time forward by `lag` (so UCDP's release delay is honored), and
      - `allow_exact_matches=False` -> strict `<`, so nothing at exactly t leaks.

    `on`   : timestamp column in `source` (tz-aware UTC).
    `lag`  : Timedelta/str; 0h = plain "< t". UCDP uses config.UCDP_LAG_DAYS.
    `by`   : optional group key present in both (e.g. 'oblast').
    `cols` : source columns to bring over (default: all but keys).
    """
    lag = pd.Timedelta(lag)
    left = grid.reset_index()  # exposes ts_utc (+ oblast) as columns
    right = source.copy()

    right["_key"] = pd.to_datetime(right[on]) + lag
    right = right.sort_values("_key")
    left = left.sort_values("ts_utc")

    keep = cols if cols is not None else [c for c in right.columns if c not in {on, "_key", by}]
    right = right[["_key"] + ([by] if by else []) + keep]

    merged = pd.merge_asof(
        left,
        right,
        left_on="ts_utc",
        right_on="_key",
        by=by,
        direction="backward",
        allow_exact_matches=False,  # strict < : no same-timestamp leak
    )
    merged = merged.drop(columns=["_key"])
    return merged.set_index(["oblast", "ts_utc"]).sort_index()
