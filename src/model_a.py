"""Model A — Prophet daily baseline.

Long-horizon load/calendar seasonal baseline. B must beat A at short horizon
(that comparison is the deliverable, PLAN MVP done-criteria).

Daily resolution (smooth seasonal), per-oblast. Captures weekly/holiday/yearly
load and trend; no sharp wave timing — that is B's job. A has NO horizon
resolution: its forecast is a per-day alert rate, broadcast identically across
every horizon and every hour of that day. That flatness is the point — it is
the floor B must beat.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config


def _quiet_prophet() -> None:
    """Silence the per-fit cmdstanpy/prophet INFO banner (27 fits = noise wall).

    Set inside fit() not at import: prophet reconfigures these loggers when it is
    imported, so a module-level setLevel gets clobbered.
    """
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)


def to_daily(grid: pd.DataFrame) -> pd.DataFrame:
    """Hourly alert grid -> per-oblast daily alert rate, long Prophet frame.

    Returns columns [oblast, ds, y]: `y` = mean active-alert fraction over the day's
    hours (in [0, 1]), `ds` = tz-naive day (Prophet rejects tz-aware timestamps).
    """
    ts = grid.index.get_level_values("ts_utc")
    ob = grid.index.get_level_values("oblast")
    s = grid["alert"].groupby([ob, ts.floor("D")]).mean()
    s.index.names = ["oblast", "ds"]
    out = s.rename("y").reset_index()
    out["ds"] = out["ds"].dt.tz_localize(None)
    return out


def fit(daily: pd.DataFrame) -> dict:
    """Fit one Prophet per oblast on the training span. Returns {oblast: model}.

    Weekly + yearly seasonality (war tempo has both); daily seasonality off — the
    series is already daily. Caller must pass TRAIN rows only (temporal split).
    """
    from prophet import Prophet

    _quiet_prophet()
    models = {}
    for ob, g in daily.groupby("oblast"):
        m = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        m.fit(g[["ds", "y"]])
        models[ob] = m
    return models


def predict(models: dict, dates: pd.DataFrame) -> pd.DataFrame:
    """Daily baseline forecast per oblast. `dates` = long frame [oblast, ds].

    Returns [oblast, ds, yhat] with yhat clipped to [0, 1] (it is an alert rate,
    Prophet's linear trend can drift outside). Oblasts absent from `models` are skipped.
    """
    rows = []
    for ob, g in dates.groupby("oblast"):
        m = models.get(ob)
        if m is None:
            continue
        fc = m.predict(g[["ds"]])
        out = pd.DataFrame({"oblast": ob, "ds": fc["ds"].values, "yhat": fc["yhat"].values})
        rows.append(out)
    res = pd.concat(rows, ignore_index=True)
    res["yhat"] = res["yhat"].clip(0.0, 1.0)
    return res


def baseline_for_grid(train_grid: pd.DataFrame, test_grid: pd.DataFrame, horizons=None) -> pd.DataFrame:
    """Fit A on `train_grid`, forecast, broadcast to `test_grid`'s hourly index.

    Returns a frame indexed exactly like `test_grid` (oblast, ts_utc) with one column
    per horizon, all holding that row's day-level baseline rate (A is horizon-flat).
    This is the `a_pred` shape `evaluate.compare_b_vs_a` expects.
    """
    horizons = horizons or config.HORIZONS
    models = fit(to_daily(train_grid))

    ts = test_grid.index.get_level_values("ts_utc")
    ob = test_grid.index.get_level_values("oblast")
    days = pd.DataFrame({"oblast": ob, "ds": ts.floor("D").tz_localize(None)})
    daily_pred = predict(models, days.drop_duplicates())

    # Map each test row's (oblast, day) to its forecast rate, then fan out to horizons.
    key = days.merge(daily_pred, on=["oblast", "ds"], how="left")["yhat"].to_numpy()
    out = pd.DataFrame(index=test_grid.index)
    for h in horizons:
        out[h] = key
    return out
