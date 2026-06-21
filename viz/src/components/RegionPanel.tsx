import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, AreaChart, Area, ReferenceLine,
} from "recharts";
import type { PredictionSource, OnsetData } from "../types";
import type { MetricsData } from "../hooks/useMetrics";
import {
  HORIZONS, DISPLAY_NAMES, probabilityToColor,
  HORIZON_HOURS, onsetTiming, timingColor, ONSET_TAU,
} from "../utils/colorScale";
import type { Horizon } from "../utils/colorScale";

interface Props {
  code: string;
  predictions: PredictionSource;
  metrics: MetricsData;
  horizon: Horizon;
  alerting: boolean;
  onClose: () => void;
  onsetMode: boolean;
  onset: OnsetData | null;
}

export function RegionPanel({
  code, predictions, metrics, horizon, alerting, onClose, onsetMode, onset,
}: Props) {
  const preds = predictions.predictions[code] ?? {};
  const obMetrics = metrics.per_oblast[code] ?? {};

  const barData = HORIZONS.map((h) => ({
    horizon: h,
    prob: preds[h as Horizon] ?? 0,
  }));

  if (onsetMode && onset) {
    return (
      <OnsetRegion code={code} onset={onset} horizon={horizon} alerting={alerting} onClose={onClose} />
    );
  }

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

interface OnsetRegionProps {
  code: string;
  onset: OnsetData;
  horizon: Horizon;
  alerting: boolean;
  onClose: () => void;
}

/** Onset / timing detail: the P(new alert by T+n) distribution + time-to-alert + skill. */
function OnsetRegion({ code, onset, horizon, alerting, onClose }: OnsetRegionProps) {
  const preds = onset.predictions[code] ?? {};
  const obMetrics = onset.per_oblast[code] ?? {};
  const timing = onsetTiming(preds);

  // Cumulative onset CDF over time, anchored at the origin (0h -> 0) for a clean curve.
  const cdf = [
    { hours: 0, label: "now", p: 0 },
    ...HORIZONS.map((h) => ({
      hours: HORIZON_HOURS[h],
      label: h,
      p: preds[h as Horizon] ?? 0,
    })),
  ];
  const noData = preds[HORIZONS[HORIZONS.length - 1] as Horizon] == null;

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>{DISPLAY_NAMES[code] ?? code}</h2>
        <button className="close-btn" onClick={onClose}>✕</button>
      </div>

      {alerting && <div className="alert-badge">● ACTIVE AIR ALERT</div>}

      {noData ? (
        <p className="footnote">No onset prediction for this oblast (excluded from the model grid).</p>
      ) : (
        <>
          <div className="eta-banner" style={{ borderColor: timingColor(timing.etaHours) }}>
            <span className="eta-dot" style={{ background: timingColor(timing.etaHours) }} />
            New alert likely within <strong>{timing.etaLabel}</strong>
            <span className="eta-sub">
              (P≥{ONSET_TAU} crossing · {(timing.reach * 100).toFixed(0)}% within 6h)
            </span>
          </div>

          <h3>P(new alert starts by T+n)</h3>
          <div className="chart">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={cdf} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
                <CartesianGrid stroke="#1e293b" />
                <XAxis
                  dataKey="hours" type="number" domain={[0, 6]}
                  ticks={[0, 0.5, 1, 3, 6]} unit="h"
                  stroke="#64748b" tick={{ fontSize: 11 }}
                />
                <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
                  labelFormatter={(v) => `within ${v}h`}
                  formatter={(v) => [Number(v).toFixed(3), "P(onset)"]}
                />
                <ReferenceLine y={ONSET_TAU} stroke="#94a3b8" strokeDasharray="4 3"
                  label={{ value: `τ=${ONSET_TAU}`, fill: "#94a3b8", fontSize: 10, position: "insideTopLeft" }} />
                {Number.isFinite(timing.etaHours) && (
                  <ReferenceLine x={timing.etaHours} stroke={timingColor(timing.etaHours)}
                    label={{ value: timing.etaLabel, fill: timingColor(timing.etaHours), fontSize: 10, position: "top" }} />
                )}
                <Area type="monotone" dataKey="p" stroke="#38bdf8"
                  fill="#38bdf8" fillOpacity={0.18} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <h3>Onset skill (quiet-state rows)</h3>
          <table className="metrics-table">
            <thead>
              <tr><th>Horizon</th><th>PR-AUC</th><th>Lift</th><th>Base rate</th></tr>
            </thead>
            <tbody>
              {HORIZONS.map((h) => {
                const m = obMetrics[h];
                const lift = m && m.base_rate ? m.pr_auc / m.base_rate : null;
                return (
                  <tr key={h} className={h === horizon ? "row-active" : ""}>
                    <td>{h}</td>
                    <td className="num strong">{m ? m.pr_auc.toFixed(3) : "—"}</td>
                    <td className="num">{lift != null ? `${lift.toFixed(2)}×` : "—"}</td>
                    <td className="num dim">{m ? m.base_rate.toFixed(3) : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <p className="footnote">
            Onset = a NEW alert starting from a quiet state (no alert active now). The curve is
            the cumulative probability of onset by each lead time; colour marks the soonest
            time it crosses τ={ONSET_TAU}. Lift = PR-AUC over base rate — the honest signal
            (vs the persistence-inflated "is an alert active" risk view).
          </p>
        </>
      )}
    </div>
  );
}
