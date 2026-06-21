import { useState } from "react";
import { AlertMap } from "./components/AlertMap";
import { HorizonToggle } from "./components/HorizonToggle";
import { MetricsPanel } from "./components/MetricsPanel";
import { RegionPanel } from "./components/RegionPanel";
import { usePredictions } from "./hooks/usePredictions";
import { useMetrics } from "./hooks/useMetrics";
import { useAlerts } from "./hooks/useAlerts";
import type { Horizon } from "./utils/colorScale";
import "./App.css";

export default function App() {
  const { data: predictions, error: predErr } = usePredictions();
  const { data: metrics, error: metErr } = useMetrics();
  const alerts = useAlerts();
  const [horizon, setHorizon] = useState<Horizon>("1h");
  const [selected, setSelected] = useState<string | null>(null);

  if (predErr || metErr) {
    return (
      <div className="loading error">
        Failed to load data: {predErr ?? metErr}
      </div>
    );
  }
  if (!predictions || !metrics) {
    return <div className="loading">Loading predictions…</div>;
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Ukraine Air-Raid Forecast</h1>
          <span className="tagline">Per-oblast alert probability · ≤6h horizon</span>
        </div>
        <HorizonToggle value={horizon} onChange={setHorizon} />
        <div className="status">
          <div>
            Forecast base:{" "}
            <strong>{new Date(predictions.forecast_base_utc).toUTCString().slice(5, 22)} UTC</strong>
          </div>
          <div className={alerts.available ? "live on" : "live off"}>
            {alerts.available
              ? `● ${alerts.active.size} active alerts`
              : "○ live alerts unavailable"}
          </div>
        </div>
      </header>

      <div className="main">
        <AlertMap
          predictions={predictions}
          horizon={horizon}
          activeAlerts={alerts.active}
          selected={selected}
          onSelect={setSelected}
        />
        <aside className="sidebar">
          {selected ? (
            <RegionPanel
              code={selected}
              predictions={predictions}
              metrics={metrics}
              horizon={horizon}
              alerting={alerts.active.has(selected)}
              onClose={() => setSelected(null)}
            />
          ) : (
            <MetricsPanel metrics={metrics} horizon={horizon} />
          )}
        </aside>
      </div>

      <div className="legend">
        <span>Low risk</span>
        <div className="legend-bar" />
        <span>High risk</span>
        <span className="legend-alert">⎯⎯ dashed red = active alert</span>
      </div>
    </div>
  );
}
