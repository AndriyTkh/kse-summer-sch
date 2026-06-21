import { useState } from "react";
import { AlertMap } from "./components/AlertMap";
import { HorizonToggle } from "./components/HorizonToggle";
import { MetricsPanel } from "./components/MetricsPanel";
import { RegionPanel } from "./components/RegionPanel";
import { PredictionSourceToggle } from "./components/PredictionSourceToggle";
import type { PredSource } from "./components/PredictionSourceToggle";
import { usePredictions } from "./hooks/usePredictions";
import { useMetrics } from "./hooks/useMetrics";
import { useAlerts } from "./hooks/useAlerts";
import { useJson } from "./hooks/useJson";
import type {
  NowcastData, WalkForwardData, OperationalData, SurvivalData, PredictionSource,
  IntervalData, DriftData,
} from "./types";
import type { Horizon } from "./utils/colorScale";
import "./App.css";

export default function App() {
  const { data: predictions, error: predErr } = usePredictions();
  const { data: metrics, error: metErr } = useMetrics();
  const { data: nowcast } = useJson<NowcastData>("/nowcast.json", true);
  const { data: walkForward } = useJson<WalkForwardData>("/walkforward.json", true);
  const { data: operational } = useJson<OperationalData>("/operational.json", true);
  const { data: survival } = useJson<SurvivalData>("/survival.json", true);
  const { data: intervals } = useJson<IntervalData>("/intervals.json", true);
  const { data: drift } = useJson<DriftData>("/drift.json", true);
  const alerts = useAlerts();
  const [horizon, setHorizon] = useState<Horizon>("1h");
  const [selected, setSelected] = useState<string | null>(null);
  const [source, setSource] = useState<PredSource>("backtest");

  if (predErr || metErr) {
    return <div className="loading error">Failed to load data: {predErr ?? metErr}</div>;
  }
  if (!predictions || !metrics) {
    return <div className="loading">Loading predictions…</div>;
  }

  const useNowcast = source === "nowcast" && !!nowcast;
  const activePred: PredictionSource = useNowcast ? nowcast! : predictions;
  const baseLabel = useNowcast ? nowcast!.origin_utc : predictions.forecast_base_utc;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Ukraine Air-Raid Forecast</h1>
          <span className="tagline">Per-oblast alert probability · ≤6h horizon</span>
        </div>
        <PredictionSourceToggle
          value={source}
          onChange={setSource}
          nowcastAvailable={!!nowcast}
        />
        <HorizonToggle value={horizon} onChange={setHorizon} />
        <div className="status">
          <div>
            {useNowcast ? "Nowcast origin: " : "Forecast base: "}
            <strong>{new Date(baseLabel).toUTCString().slice(5, 22)} UTC</strong>
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
          predictions={activePred}
          horizon={horizon}
          activeAlerts={alerts.active}
          selected={selected}
          onSelect={setSelected}
        />
        <aside className="sidebar">
          {selected ? (
            <RegionPanel
              code={selected}
              predictions={activePred}
              metrics={metrics}
              horizon={horizon}
              alerting={alerts.active.has(selected)}
              onClose={() => setSelected(null)}
            />
          ) : (
            <MetricsPanel
              metrics={metrics}
              horizon={horizon}
              walkForward={walkForward}
              operational={operational}
              survival={survival}
              intervals={intervals}
              drift={drift}
            />
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
