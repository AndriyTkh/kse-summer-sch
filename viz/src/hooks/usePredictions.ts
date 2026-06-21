import { useState, useEffect } from "react";
import type { Horizon } from "../utils/colorScale";

export interface PredictionsData {
  generated_utc: string;
  forecast_base_utc: string;
  horizons: string[];
  predictions: Record<string, Record<Horizon, number | null>>;
}

export function usePredictions() {
  const [data, setData] = useState<PredictionsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/predictions.json")
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  return { data, error };
}
