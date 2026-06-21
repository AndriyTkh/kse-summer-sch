import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { MetricsData } from "../hooks/useMetrics";
import { HORIZONS } from "../utils/colorScale";
import type { Horizon } from "../utils/colorScale";

interface Props {
  metrics: MetricsData;
  horizon: Horizon;
}

export function MetricsPanel({ metrics, horizon }: Props) {
  const agg = metrics.aggregate;
  const cal = metrics.calibration_curves[horizon];

  const calData =
    cal?.mean_pred.map((mp, i) => ({
      pred: mp,
      observed: cal.frac_pos[i],
      perfect: mp,
    })) ?? [];

  return (
    <div className="panel">
      <h2>Model evaluation</h2>
      <p className="subtitle">
        Temporal holdout · last {metrics.test_weeks} weeks · B (LightGBM) vs A (Prophet)
      </p>

      <h3>PR-AUC by horizon</h3>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th>B</th>
            <th>A</th>
            <th>Lift</th>
            <th>Base</th>
          </tr>
        </thead>
        <tbody>
          {HORIZONS.map((h) => {
            const m = agg[h];
            if (!m) return null;
            return (
              <tr key={h} className={h === horizon ? "row-active" : ""}>
                <td>{h}</td>
                <td className="num strong">{m.pr_auc_b.toFixed(3)}</td>
                <td className="num">{m.pr_auc_a.toFixed(3)}</td>
                <td className="num lift">{m.lift.toFixed(2)}×</td>
                <td className="num dim">{m.base_rate.toFixed(3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <h3>Calibration ({horizon})</h3>
      <div className="chart">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={calData} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis
              dataKey="pred"
              type="number"
              domain={[0, 1]}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
              label={{ value: "predicted", position: "insideBottom", offset: -4, fill: "#64748b", fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              stroke="#64748b"
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="observed" stroke="#22d3ee" strokeWidth={2} dot={{ r: 2 }} name="observed" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <h3>Calibration error & Brier ({horizon})</h3>
      <table className="metrics-table">
        <thead>
          <tr><th></th><th>B</th><th>A</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>ECE</td>
            <td className="num strong">{agg[horizon]?.ece_b.toFixed(3)}</td>
            <td className="num">{agg[horizon]?.ece_a.toFixed(3)}</td>
          </tr>
          <tr>
            <td>Brier</td>
            <td className="num strong">{agg[horizon]?.brier_b.toFixed(3)}</td>
            <td className="num">{agg[horizon]?.brier_a.toFixed(3)}</td>
          </tr>
        </tbody>
      </table>

      <p className="footnote">
        Lower ECE/Brier is better; B is isotonic-calibrated. Click an oblast for regional detail.
      </p>
    </div>
  );
}
