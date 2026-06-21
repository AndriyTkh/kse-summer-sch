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
