import { useState, useEffect } from "react";

export interface AggregateMetrics {
  base_rate: number;
  pr_auc_b: number;
  pr_auc_a: number;
  lift: number;
  ece_b: number;
  ece_a: number;
  brier_b: number;
  brier_a: number;
}

export interface OblastHorizonMetrics {
  pr_auc_b: number;
  base_rate: number;
  mean_pred: number;
  n_samples: number;
}

export interface CalibrationCurve {
  frac_pos: number[];
  mean_pred: number[];
}

export interface MetricsData {
  generated_utc: string;
  test_weeks: number;
  aggregate: Record<string, AggregateMetrics>;
  per_oblast: Record<string, Record<string, OblastHorizonMetrics>>;
  calibration_curves: Record<string, CalibrationCurve>;
}

export function useMetrics() {
  const [data, setData] = useState<MetricsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/metrics.json")
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  return { data, error };
}
