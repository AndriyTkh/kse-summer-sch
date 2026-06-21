import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import type { PredictionSource } from "../types";
import type { MetricsData } from "../hooks/useMetrics";
import { HORIZONS, DISPLAY_NAMES, probabilityToColor } from "../utils/colorScale";
import type { Horizon } from "../utils/colorScale";

interface Props {
  code: string;
  predictions: PredictionSource;
  metrics: MetricsData;
  horizon: Horizon;
  alerting: boolean;
  onClose: () => void;
}

export function RegionPanel({ code, predictions, metrics, horizon, alerting, onClose }: Props) {
  const preds = predictions.predictions[code] ?? {};
  const obMetrics = metrics.per_oblast[code] ?? {};

  const barData = HORIZONS.map((h) => ({
    horizon: h,
    prob: preds[h as Horizon] ?? 0,
  }));

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>{DISPLAY_NAMES[code] ?? code}</h2>
        <button className="close-btn" onClick={onClose}>✕</button>
      </div>

      {alerting && <div className="alert-badge">● ACTIVE AIR ALERT</div>}

      <h3>Predicted alert probability</h3>
      <div className="chart">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={barData} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="horizon" stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
              formatter={(v) => Number(v).toFixed(3)}
            />
            <Bar dataKey="prob" name="P(alert)">
              {barData.map((d, i) => (
                <Cell
                  key={i}
                  fill={probabilityToColor(d.prob)}
                  stroke={d.horizon === horizon ? "#e2e8f0" : "none"}
                  strokeWidth={d.horizon === horizon ? 2 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h3>Per-horizon accuracy (B)</h3>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th>PR-AUC</th>
            <th>Mean pred</th>
            <th>Base rate</th>
          </tr>
        </thead>
        <tbody>
          {HORIZONS.map((h) => {
            const m = obMetrics[h];
            return (
              <tr key={h} className={h === horizon ? "row-active" : ""}>
                <td>{h}</td>
                <td className="num strong">{m ? m.pr_auc_b.toFixed(3) : "—"}</td>
                <td className="num">{m ? m.mean_pred.toFixed(3) : "—"}</td>
                <td className="num dim">{m ? m.base_rate.toFixed(3) : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p className="footnote">
        Mean pred vs base rate shows the model's average risk estimate against the
        observed alert frequency for this oblast over the test window.
      </p>
    </div>
  );
}
