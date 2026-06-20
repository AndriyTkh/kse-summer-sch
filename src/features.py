"""Feature pipeline on the master grid. Leak-safe by construction.

All features at row t derive from data STRICTLY before t (issue #1). Lags use
`shift(k>=1)`; rolling windows use `shift(1).rolling(...)` (current cell excluded);
threat/tempo sources are floored to the hour and shifted so nothing at exactly t leaks.

Channels:
  lags          — recent alert history per oblast (t-1,3,6,24 + rolling means, since-last)
  calendar      — hour, dow, is_night, is_weekend, day-of-war (absorbs A's seasonality)
  threat        — per-type launch/wave counts over trailing windows (ballistic|air-cruise|
                  sea-cruise|drone-strike|drone-recon|drone-decoy|kinzhal); national
                  (no affected-region) waves broadcast to all oblasts
  tempo         — national daily launch tempo, previous-day (no same-day leak)
  ucdp          — per-oblast impact prior, lagged (issue #2) — NO-OP (Phase 2, source not wired)

Grid contract: MultiIndex (oblast, ts_utc), hourly UTC, with a raw `alert` column
(from index.expand_alerts_to_grid). Per-horizon target shifting lives in model_b.
"""

from __future__ import annotations

import pandas as pd

from . import config

# Trailing windows (hours) for rolling alert history and threat activity.
_LAG_HOURS = (1, 3, 6, 24)
_ROLL_HOURS = (3, 6, 24, 168)        # 168 = 7d
_THREAT_WINDOWS = (3, 6, 24)
_NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))   # 22:00–05:59 UTC


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alert-history lags + rolling means per oblast. Only t-k, k>=1 (no leak)."""
    out = df.sort_index()
    g = out.groupby(level="oblast", sort=False)["alert"]

    for k in _LAG_HOURS:
        out[f"alert_lag_{k}h"] = g.shift(k)
    for w in _ROLL_HOURS:
        # shift(1) first -> window covers [t-w, t-1], current cell excluded.
        out[f"alert_roll_{w}h"] = g.shift(1).rolling(w, min_periods=1).mean()

    # Hours since the most recent alert==1 strictly before t (capped by series start).
    out["hours_since_alert"] = _hours_since_last_positive(g.shift(1))
    return out


def _hours_since_last_positive(shifted: pd.Series) -> pd.Series:
    """Per oblast, count hours since the last 1 in `shifted` (already lagged by 1).

    Resets to 0 on a positive; increments by 1 each hour otherwise. NaN before any
    history. Implemented with a grouped cumulative trick (no per-row Python loop).
    """
    def _per_oblast(s: pd.Series) -> pd.Series:
        is_on = (s == 1).fillna(False)
        # group id increments at each positive; within a group, position = hours since.
        grp = is_on.cumsum()
        pos = s.groupby(grp).cumcount()
        # before the first positive, grp==0 -> distance undefined; leave as the running count.
        return pos.where(grp > 0)
    return shifted.groupby(level="oblast", sort=False, group_keys=False).apply(_per_oblast)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, day-of-week, night flag, weekend, day-of-war. Deterministic from ts."""
    out = df
    ts = out.index.get_level_values("ts_utc")
    out["hour"] = ts.hour
    out["dow"] = ts.dayofweek
    out["is_night"] = ts.hour.isin(_NIGHT_HOURS).astype("int8")
    out["is_weekend"] = (ts.dayofweek >= 5).astype("int8")
    war_start = pd.Timestamp(config.WAR_START, tz=config.TZ_GRID)
    out["day_of_war"] = ((ts - war_start).days).astype("int32")
    return out


def _explode_waves(waves: pd.DataFrame, oblasts) -> pd.DataFrame:
    """Wave rows -> (oblast, ts_utc[hour], channel, launched, waves) long frame.

    National waves (empty oblasts list) broadcast to every oblast. Each (channel)
    in a wave's channel set contributes a row. `launched` NaN -> 0.
    """
    all_obl = list(oblasts)
    recs: list[tuple] = []
    for r in waves.itertuples(index=False):
        chans = r.channels or ()
        if not chans:
            continue
        obs = r.oblasts if r.oblasts else all_obl
        hour = r.time_start_utc.floor("h")
        launched = 0.0 if pd.isna(r.launched) else float(r.launched)
        for ob in obs:
            for ch in chans:
                recs.append((ob, hour, ch, launched))
    long = pd.DataFrame(recs, columns=["oblast", "ts_utc", "channel", "launched"])
    long["waves"] = 1
    return long


def add_threat_features(df: pd.DataFrame, waves: pd.DataFrame) -> pd.DataFrame:
    """Per-channel launch/wave counts over trailing windows, lag-shifted (< t)."""
    out = df.sort_index()
    oblasts = out.index.get_level_values("oblast").unique()
    long = _explode_waves(waves, oblasts)
    if long.empty:
        return out

    hourly = (
        long.groupby(["oblast", "ts_utc", "channel"])
        .agg(launched=("launched", "sum"), waves=("waves", "sum"))
        .reset_index()
    )
    # Wide: one column block per channel, reindexed onto the grid's (oblast, ts) hours.
    wide = hourly.pivot_table(
        index=["oblast", "ts_utc"], columns="channel",
        values=["launched", "waves"], fill_value=0,
    )
    wide.columns = [f"thr_{ch}_{val}" for val, ch in wide.columns]
    wide = wide.reindex(out.index, fill_value=0)

    g = wide.groupby(level="oblast", sort=False)
    for col in wide.columns:
        shifted = g[col].shift(1)                      # exclude current hour
        sg = shifted.groupby(level="oblast", sort=False)
        for w in _THREAT_WINDOWS:
            out[f"{col}_{w}h"] = sg.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
    return out


def add_tempo_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """National daily launch tempo, mapped by PREVIOUS UTC day (no same-day leak)."""
    out = df
    d = daily.set_index(daily["date_utc"].dt.floor("D")).sort_index()
    # Previous-day tempo for a cell at day D = tempo on D-1 (shift the daily series).
    prev = d[["launched", "destroyed", "n_waves"]].shift(1)
    prev = prev.add_prefix("tempo_prev_")
    prev["tempo_roll7_launched"] = d["launched"].shift(1).rolling(7, min_periods=1).mean()

    day = out.index.get_level_values("ts_utc").floor("D")
    mapped = prev.reindex(day).reset_index(drop=True)
    mapped.index = out.index
    for c in mapped.columns:
        out[c] = mapped[c]
    return out


def add_ucdp_features(df: pd.DataFrame, ucdp=None) -> pd.DataFrame:
    """No-op: UCDP is Phase 2 (issue #2 stays wired in spec only)."""
    return df


def build_feature_matrix(grid: pd.DataFrame, sources: dict) -> pd.DataFrame:
    """Run all feature builders in order, return model-ready matrix.

    `grid` must already carry the raw `alert` column (index.expand_alerts_to_grid).
    `sources` keys: 'waves' (massive attacks), 'daily' (missile tempo), optional 'ucdp'.
    Leak-safety is structural (shifts only); see module docstring.
    """
    if "alert" not in grid.columns:
        raise ValueError("grid needs raw 'alert' column — run expand_alerts_to_grid first")

    out = add_lag_features(grid)
    out = add_calendar_features(out)
    if sources.get("waves") is not None:
        out = add_threat_features(out, sources["waves"])
    if sources.get("daily") is not None:
        out = add_tempo_features(out, sources["daily"])
    out = add_ucdp_features(out, sources.get("ucdp"))
    return out
