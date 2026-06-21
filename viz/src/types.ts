import type { Horizon } from "./utils/colorScale";

/** Anything the map / region panel can colour: per-oblast per-horizon probability. */
export interface PredictionSource {
  generated_utc: string;
  horizons: string[];
  predictions: Record<string, Record<Horizon, number | null>>;
}

/** Operational next-6h edge forecast (run_forecast.py → nowcast.json). */
export interface NowcastData extends PredictionSource {
  origin_utc: string;
  calibrated: boolean;
}

/** Walk-forward rolling-origin CV summary (run_walkforward.py → walkforward.json). */
export interface WalkForwardData {
  generated_utc: string;
  n_folds: number;
  test_weeks: number;
  purge_hours: number;
  by_horizon: Record<
    string,
    {
      n_folds: number;
      pr_auc_mean: number;
      pr_auc_std: number;
      pr_auc_min: number;
      pr_auc_max: number;
      base_mean: number;
      lift_mean: number;
    }
  >;
}

/** Backtest-vs-live PR-AUC gap sweep (run_operational_eval.py → operational.json). */
export interface OperationalData {
  generated_utc: string;
  test_weeks: number;
  lag_hours: number[];
  by_horizon: Record<
    string,
    {
      pr_auc_full: number;
      by_lag: Record<string, { pr_auc_degraded: number; gap_pct: number }>;
    }
  >;
}

/** Quantile prediction intervals — Model Bq (run_phase3.py → intervals.json). */
export interface IntervalData {
  generated_utc: string;
  forecast_base_utc: string;
  test_weeks: number;
  horizons: string[];
  nominal_coverage: number;
  by_horizon: Record<
    string,
    { pinball: number; coverage: number; width: number; base_fraction: number }
  >;
  predictions: Record<
    string,
    Record<Horizon, { q10: number; q50: number; q90: number } | null>
  >;
}

/** Drift-triggered auto-retrain trajectory (run_phase3.py → drift.json). */
export interface DriftData {
  generated_utc: string;
  horizon: string;
  psi_warn: number;
  psi_alert: number;
  block_days: number;
  policies: Record<
    string,
    { mean_pinball: number; retrains: number; mean_coverage: number }
  >;
  trajectory: {
    block_start: string;
    never: number;
    periodic: number;
    drift: number;
    drift_retrained: boolean;
  }[];
}

/** Alert-duration survival model (run_survival.py → survival.json). */
export interface SurvivalData {
  generated_utc: string;
  km_median_hours: number;
  cox_c_index: number;
  test_mae_hours: number | null;
  n_events: number;
  n_uncensored: number;
  test_n_valid: number;
  top_hazard: { covariate: string; exp_coef: number; p: number }[];
  km_curve: { t: number[]; s: number[] };
}
