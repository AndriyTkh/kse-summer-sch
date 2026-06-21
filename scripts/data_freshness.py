"""Report coverage span + staleness of each raw dataset. Read-only."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
NOW = pd.Timestamp.now(tz="UTC")


def _span(name: str, ts: pd.Series) -> None:
    ts = ts.dropna()
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")  # naive source -> treat as UTC for the lag math
    lag = NOW - ts.max()
    print(
        f"{name:9s} rows={len(ts):>7d}  "
        f"min={ts.min():%Y-%m-%d}  max={ts.max():%Y-%m-%d %H:%M}  "
        f"lag={lag.days}d {lag.components.hours}h"
    )


def main() -> None:
    print(f"now (UTC): {NOW:%Y-%m-%d %H:%M}\n")

    a = pd.read_csv(DATA / "official_data_en.csv", usecols=["started_at"])
    _span("alerts", pd.to_datetime(a["started_at"], utc=True, errors="coerce"))

    m = pd.read_csv(DATA / "missile_attacks_daily.csv", usecols=["time_start"])
    _span("missiles", pd.to_datetime(m["time_start"], format="mixed", errors="coerce"))

    u = pd.read_csv(DATA / "ged251_ukraine.csv", usecols=["year"])
    yr = int(u["year"].max())
    print(f"ucdp      rows={len(u):>7d}  years={int(u['year'].min())}-{yr}  "
          f"lag={NOW.year - yr}y (annual release)")


if __name__ == "__main__":
    main()
