import { useState } from "react";
import {
  LineChart, Line, Area, ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { MetricsData } from "../hooks/useMetrics";
import type {
  WalkForwardData, OperationalData, SurvivalData, IntervalData, DriftData,
} from "../types";
import { HORIZONS } from "../utils/colorScale";
import type { Horizon } from "../utils/colorScale";

type Method = "holdout" | "walkforward" | "operational" | "duration" | "intervals" | "drift";

interface Props {
  metrics: MetricsData;
  horizon: Horizon;
  walkForward: WalkForwardData | null;
  operational: OperationalData | null;
  survival: SurvivalData | null;
  intervals: IntervalData | null;
  drift: DriftData | null;
}

export function MetricsPanel({
  metrics, horizon, walkForward, operational, survival, intervals, drift,
}: Props) {
  const [method, setMethod] = useState<Method>("holdout");

  const tabs: { id: Method; label: string; enabled: boolean }[] = [
    { id: "holdout", label: "Holdout", enabled: true },
    { id: "walkforward", label: "Walk-forward", enabled: !!walkForward },
    { id: "operational", label: "Operational", enabled: !!operational },
    { id: "duration", label: "Duration", enabled: !!survival },
    { id: "intervals", label: "Intervals", enabled: !!intervals },
    { id: "drift", label: "Drift", enabled: !!drift },
  ];
  const active = tabs.find((t) => t.id === method)?.enabled ? method : "holdout";

  return (
    <div className="panel">
      <h2>Model evaluation</h2>

      <div className="method-tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={t.id === active ? "active" : ""}
            disabled={!t.enabled}
            title={t.enabled ? "" : "Run the matching run_*.py to generate this"}
            onClick={() => setMethod(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {active === "holdout" && <Holdout metrics={metrics} horizon={horizon} />}
      {active === "walkforward" && walkForward && (
        <WalkForward data={walkForward} horizon={horizon} />
      )}
      {active === "operational" && operational && (
        <Operational data={operational} horizon={horizon} />
      )}
      {active === "duration" && survival && <Duration data={survival} />}
      {active === "intervals" && intervals && (
        <Intervals data={intervals} horizon={horizon} />
      )}
      {active === "drift" && drift && <Drift data={drift} />}
    </div>
  );
}

// ── Single temporal holdout (Phase 1) ───────────────────────────────────────
function Holdout({ metrics, horizon }: { metrics: MetricsData; horizon: Horizon }) {
  const agg = metrics.aggregate;
  const cal = metrics.calibration_curves[horizon];
  const calData =
    cal?.mean_pred.map((mp, i) => ({ pred: mp, observed: cal.frac_pos[i] })) ?? [];

  return (
    <>
      <p className="subtitle">
        Temporal holdout · last {metrics.test_weeks} weeks · B (LightGBM) vs A (Prophet)
      </p>

      <h3>PR-AUC by horizon</h3>
      <table className="metrics-table">
        <thead>
          <tr><th>Horizon</th><th>B</th><th>A</th><th>Lift</th><th>Base</th></tr>
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
            <XAxis dataKey="pred" type="number" domain={[0, 1]} stroke="#64748b"
              tick={{ fontSize: 11 }}
              label={{ value: "predicted", position: "insideBottom", offset: -4, fill: "#64748b", fontSize: 11 }} />
            <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#475569" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="observed" stroke="#22d3ee" strokeWidth={2} dot={{ r: 2 }} name="observed" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <h3>Calibration error & Brier ({horizon})</h3>
      <table className="metrics-table">
        <thead><tr><th></th><th>B</th><th>A</th></tr></thead>
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
    </>
  );
}

// ── Walk-forward rolling-origin CV (Phase 2) ────────────────────────────────
function WalkForward({ data, horizon }: { data: WalkForwardData; horizon: Horizon }) {
  return (
    <>
      <p className="subtitle">
        Rolling-origin CV · {data.n_folds} folds × {data.test_weeks}wk · purge {data.purge_hours}h ·
        PR-AUC mean ± spread across war regimes
      </p>
      <h3>PR-AUC by horizon (B)</h3>
      <table className="metrics-table">
        <thead>
          <tr><th>Horizon</th><th>Mean</th><th>± Std</th><th>Min</th><th>Max</th><th>Lift</th></tr>
        </thead>
        <tbody>
          {HORIZONS.map((h) => {
            const r = data.by_horizon[h];
            if (!r) return null;
            return (
              <tr key={h} className={h === horizon ? "row-active" : ""}>
                <td>{h}</td>
                <td className="num strong">{r.pr_auc_mean.toFixed(3)}</td>
                <td className="num dim">±{r.pr_auc_std.toFixed(3)}</td>
                <td className="num">{r.pr_auc_min.toFixed(3)}</td>
                <td className="num">{r.pr_auc_max.toFixed(3)}</td>
                <td className="num lift">{r.lift_mean.toFixed(2)}×</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="footnote">
        Tight std ⇒ the single holdout wasn't a lucky window; spread ⇒ regime drift.
      </p>
    </>
  );
}

// ── Operational eval: backtest-vs-live gap (Phase 2) ────────────────────────
function Operational({ data, horizon }: { data: OperationalData; horizon: Horizon }) {
  return (
    <>
      <p className="subtitle">
        Backtest-vs-live PR-AUC gap · test {data.test_weeks}wk · source-lag sweep.
        Positive % = backtest overstates live performance (ragged right edge).
      </p>
      <h3>PR-AUC gap % by horizon × source lag</h3>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Horizon</th>
            <th>Full</th>
            {data.lag_hours.map((l) => <th key={l}>{l}h</th>)}
          </tr>
        </thead>
        <tbody>
          {HORIZONS.map((h) => {
            const r = data.by_horizon[h];
            if (!r) return null;
            return (
              <tr key={h} className={h === horizon ? "row-active" : ""}>
                <td>{h}</td>
                <td className="num strong">{r.pr_auc_full.toFixed(3)}</td>
                {data.lag_hours.map((l) => {
                  const cell = r.by_lag[String(l)];
                  const g = cell?.gap_pct;
                  return (
                    <td key={l} className="num" style={{ color: g != null && g > 0 ? "#f87171" : "#94a3b8" }}>
                      {g != null ? `${g >= 0 ? "+" : ""}${g.toFixed(2)}%` : "—"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="footnote">
        "Full" is the backtest PR-AUC; each lag column zeroes the last N h of launch/tempo
        features to simulate publish lag at inference time.
      </p>
    </>
  );
}

// ── Duration: survival model (Phase 2) ──────────────────────────────────────
function Duration({ data }: { data: SurvivalData }) {
  const curve = data.km_curve.t.map((t, i) => ({ t, s: data.km_curve.s[i] }));
  return (
    <>
      <p className="subtitle">
        Alert time-to-all-clear · Kaplan-Meier baseline + Cox PH · {data.n_events.toLocaleString()} events
        ({data.n_uncensored.toLocaleString()} uncensored)
      </p>
      <table className="metrics-table">
        <tbody>
          <tr><td>KM median duration</td><td className="num strong">{data.km_median_hours.toFixed(2)} h</td></tr>
          <tr><td>Cox C-index</td><td className="num strong">{data.cox_c_index.toFixed(3)}</td></tr>
          <tr>
            <td>Test MAE</td>
            <td className="num">{data.test_mae_hours != null ? `${data.test_mae_hours.toFixed(2)} h` : "—"}</td>
          </tr>
        </tbody>
      </table>

      <h3>KM survival curve</h3>
      <div className="chart">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="t" type="number" stroke="#64748b" tick={{ fontSize: 11 }}
              label={{ value: "hours", position: "insideBottom", offset: -4, fill: "#64748b", fontSize: 11 }} />
            <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
              formatter={(v) => Number(v).toFixed(3)} />
            <Line type="stepAfter" dataKey="s" stroke="#a78bfa" strokeWidth={2} dot={false} name="P(active)" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <h3>Top hazard ratios</h3>
      <table className="metrics-table">
        <thead><tr><th>Covariate</th><th>exp(coef)</th><th>p</th></tr></thead>
        <tbody>
          {data.top_hazard.map((h) => (
            <tr key={h.covariate}>
              <td className="cov">{h.covariate}</td>
              <td className="num strong">{h.exp_coef.toFixed(2)}</td>
              <td className="num dim">{h.p < 0.001 ? "<0.001" : h.p.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="footnote">
        exp(coef) &gt; 1 ⇒ covariate shortens alerts (raises hazard of all-clear). C-index is
        capped until Phase-3 per-oblast swarm counts.
      </p>
    </>
  );
}

// ── Quantile prediction intervals: Model Bq (Phase 3) ───────────────────────
function Intervals({ data, horizon }: { data: IntervalData; horizon: Horizon }) {
  // Mean band across oblasts per horizon (latest forecast base) for the chart.
  const band = HORIZONS.map((h) => {
    const vals = Object.values(data.predictions)
      .map((p) => p[h])
      .filter((v): v is { q10: number; q50: number; q90: number } => v != null);
    if (!vals.length) return { h, q50: 0, band: [0, 0] as [number, number] };
    const mean = (k: "q10" | "q50" | "q90") =>
      vals.reduce((s, v) => s + v[k], 0) / vals.length;
    return { h, q50: mean("q50"), band: [mean("q10"), mean("q90")] as [number, number] };
  });
  const nominalPct = (data.nominal_coverage * 100).toFixed(0);

  return (
    <>
      <p className="subtitle">
        Model Bq · quantile regression on alert-FRACTION over (t, t+H] · nominal {nominalPct}%
        band [q10, q90] · temporal holdout {data.test_weeks}wk
      </p>

      <h3>Band quality by horizon</h3>
      <table className="metrics-table">
        <thead>
          <tr><th>Horizon</th><th>Pinball</th><th>Coverage</th><th>Width</th><th>Base frac</th></tr>
        </thead>
        <tbody>
          {HORIZONS.map((h) => {
            const m = data.by_horizon[h];
            if (!m) return null;
            const off = Math.abs(m.coverage - data.nominal_coverage) > 0.1;
            return (
              <tr key={h} className={h === horizon ? "row-active" : ""}>
                <td>{h}</td>
                <td className="num strong">{m.pinball.toFixed(4)}</td>
                <td className="num" style={{ color: off ? "#f87171" : "#94a3b8" }}>
                  {(m.coverage * 100).toFixed(0)}%
                </td>
                <td className="num">{m.width.toFixed(3)}</td>
                <td className="num dim">{m.base_fraction.toFixed(3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <h3>Mean band across oblasts (latest)</h3>
      <div className="chart">
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={band} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="h" stroke="#64748b" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 1]} stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
              formatter={(v) => (Array.isArray(v) ? v.map((x) => Number(x).toFixed(3)).join(" – ") : Number(v).toFixed(3))} />
            <Area dataKey="band" stroke="none" fill="#f59e0b" fillOpacity={0.18} name="q10–q90" />
            <Line type="monotone" dataKey="q50" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} name="median" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <p className="footnote">
        30m/1h collapse to the next-hour binary (k=1) — bands degenerate there, honest. 3h/6h
        carry the real intensity spread. Coverage near {nominalPct}% ⇒ well-calibrated bands.
      </p>
    </>
  );
}

// ── Drift-triggered auto-retrain trajectory (Phase 3) ───────────────────────
function Drift({ data }: { data: DriftData }) {
  const traj = data.trajectory.map((p) => ({
    ...p,
    label: new Date(p.block_start).toISOString().slice(0, 10),
  }));
  const retrainBlocks = traj.filter((p) => p.drift_retrained).map((p) => p.label);

  return (
    <>
      <p className="subtitle">
        Walk-forward {data.horizon} · {data.block_days}-day blocks · PSI trigger
        (warn {data.psi_warn}, alert {data.psi_alert}) · frozen vs periodic vs drift-retrain
      </p>

      <h3>Policy summary</h3>
      <table className="metrics-table">
        <thead>
          <tr><th>Policy</th><th>Mean pinball</th><th>Retrains</th><th>Mean cover</th></tr>
        </thead>
        <tbody>
          {[["never", "frozen"], ["periodic", "periodic"], ["drift", "drift"]].map(([p, label]) => {
            const m = data.policies[p];
            if (!m) return null;
            return (
              <tr key={p} className={p === "drift" ? "row-active" : ""}>
                <td>{label}</td>
                <td className="num strong">{m.mean_pinball.toFixed(4)}</td>
                <td className="num">{m.retrains}</td>
                <td className="num dim">{(m.mean_coverage * 100).toFixed(0)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <h3>Pinball trajectory (dotted = drift retrain)</h3>
      <div className="chart">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={traj} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
            <CartesianGrid stroke="#1e293b" />
            <XAxis dataKey="label" stroke="#64748b" tick={{ fontSize: 10 }} minTickGap={24} />
            <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 6 }}
              formatter={(v) => Number(v).toFixed(4)} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {retrainBlocks.map((b) => (
              <ReferenceLine key={b} x={b} stroke="#475569" strokeDasharray="3 3" />
            ))}
            <Line type="monotone" dataKey="never" stroke="#64748b" strokeWidth={1.5} dot={false} name="frozen" />
            <Line type="monotone" dataKey="periodic" stroke="#22d3ee" strokeWidth={1.5} dot={false} name="periodic" />
            <Line type="monotone" dataKey="drift" stroke="#f59e0b" strokeWidth={2} dot={false} name="drift" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="footnote">
        Frozen is the never-retrain floor; drift adapts only when feature PSI breaches the band.
        Lower pinball under the 2022→2025 regime shift ⇒ adaptation pays for its retrains.
      </p>
    </>
  );
}
