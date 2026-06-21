"""Phase-2 duration model — time-to-all-clear via survival analysis (lifelines).

Given an alert started in oblast X at time T, predict how long until all-clear.
Uses a Cox proportional-hazards model (covariates from the hourly feature matrix at
alert-start time) or a non-parametric Kaplan-Meier baseline.

Unit of observation: one alert EVENT (start, end, duration_hours), not the hourly grid.
Censoring (issue #7): if an alert is still active at the data cutoff, it is right-censored
(observed=0) — we know it lasted *at least* that long, but not when it ended.

Accuracy cap: per-oblast swarm size at alert start is the primary duration driver, but
bulk data is national-only → this cap holds until Phase-3 real-time per-oblast counts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_survival_dataset(
    alerts: pd.DataFrame,
    fm: pd.DataFrame,
    data_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Alert events + hourly features at alert start -> survival-ready frame.

    Each row = one alert event with:
      - duration_hours: (end - start) in hours (or time-to-cutoff if censored)
      - observed: 1 if the alert ended (uncensored), 0 if still active at cutoff
      - covariates: feature columns from `fm` at the alert-start hour

    `alerts` is the output of loaders.load_alerts (columns: oblast, start_utc, end_utc).
    `fm` is the hourly feature matrix (MultiIndex oblast x ts_utc). We snap each alert's
    start to its floor hour and look up the feature row — leak-safe because features at
    row t use only data with timestamp < t (structural guarantee from features.py).

    `data_cutoff` defaults to the max timestamp in fm. Alerts starting after the cutoff
    are dropped (no features available).
    """
    if data_cutoff is None:
        data_cutoff = fm.index.get_level_values("ts_utc").max()
    data_cutoff = pd.Timestamp(data_cutoff)

    feature_cols = [c for c in fm.columns if c != "alert" and not c.startswith("target_")]

    rows = []
    for r in alerts.itertuples(index=False):
        start_hour = r.start_utc.floor("h")
        key = (r.oblast, start_hour)
        if key not in fm.index:
            continue

        if pd.isna(r.end_utc) or r.end_utc > data_cutoff:
            duration = (data_cutoff - r.start_utc).total_seconds() / 3600
            observed = 0
        else:
            duration = (r.end_utc - r.start_utc).total_seconds() / 3600
            observed = 1

        if duration <= 0:
            continue

        feat_row = fm.loc[key]
        if isinstance(feat_row, pd.DataFrame):
            feat_row = feat_row.iloc[0]

        row = {"oblast": r.oblast, "start_utc": r.start_utc,
               "duration_hours": duration, "observed": observed}
        for c in feature_cols:
            row[c] = feat_row[c]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["duration_hours"] = out["duration_hours"].clip(lower=1e-6)
    return out


def km_baseline(surv_df: pd.DataFrame):
    """Kaplan-Meier non-parametric survival curve (no covariates).

    Returns a fitted KaplanMeierFitter. The median survival time is the headline
    "typical alert duration" number.
    """
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=surv_df["duration_hours"],
        event_observed=surv_df["observed"],
    )
    return kmf


def cox_model(surv_df: pd.DataFrame, covariates: list[str] | None = None):
    """Cox proportional-hazards model for alert duration.

    Covariates default to all numeric feature columns (excluding metadata).
    Returns a fitted CoxPHFitter. Key outputs:
      - .summary: per-covariate hazard ratios + significance
      - .predict_median(X): predicted median duration for new alerts
      - .concordance_index_: discrimination (C-index, like AUC for survival)

    Penalizer is set to regularize against overfitting on many features.
    """
    from lifelines import CoxPHFitter

    _META = {"oblast", "start_utc", "duration_hours", "observed"}
    if covariates is None:
        covariates = [c for c in surv_df.columns
                      if c not in _META and surv_df[c].dtype in ("float64", "int64", "int8", "int32", "float32")]

    df = surv_df[["duration_hours", "observed"] + covariates].dropna()
    if df.empty or df["observed"].sum() == 0:
        raise ValueError("No uncensored events — cannot fit Cox model")

    # Drop near-constant columns (variance < epsilon) that cause convergence failures
    var = df[covariates].var()
    keep = var[var > 1e-8].index.tolist()
    df = df[["duration_hours", "observed"] + keep]

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col="duration_hours", event_col="observed")
    return cph


def predict_duration(cph, surv_df: pd.DataFrame, covariates: list[str] | None = None) -> pd.Series:
    """Predicted median time-to-all-clear for each alert event.

    Returns a Series aligned to surv_df's index with predicted median duration in hours.
    Events where the model can't estimate a median (survival never crosses 0.5) get NaN.
    """
    _META = {"oblast", "start_utc", "duration_hours", "observed"}
    if covariates is None:
        covariates = [c for c in surv_df.columns
                      if c not in _META and c in cph.params_.index]

    X = surv_df[covariates].fillna(0)
    median = cph.predict_median(X)
    median = median.replace([np.inf, -np.inf], np.nan)
    if isinstance(median, pd.DataFrame):
        median = median.iloc[:, 0]
    median.index = surv_df.index
    return median.rename("predicted_median_hours")


def temporal_split_events(
    surv_df: pd.DataFrame,
    test_weeks: int = config.TEST_WEEKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split on alert events (by start_utc). Never random."""
    cut = surv_df["start_utc"].max() - pd.Timedelta(weeks=test_weeks)
    train = surv_df[surv_df["start_utc"] < cut]
    test = surv_df[surv_df["start_utc"] >= cut]
    return train, test
