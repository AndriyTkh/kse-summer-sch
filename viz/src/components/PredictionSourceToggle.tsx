export type PredSource = "backtest" | "nowcast";

interface Props {
  value: PredSource;
  onChange: (s: PredSource) => void;
  nowcastAvailable: boolean;
}

const LABELS: Record<PredSource, string> = {
  backtest: "Backtest",
  nowcast: "Nowcast",
};

const HINTS: Record<PredSource, string> = {
  backtest: "Held-out test window (last test weeks) — what the model scored on unseen history.",
  nowcast: "Operational next-6h edge forecast — B retrained on all data, calibrated.",
};

export function PredictionSourceToggle({ value, onChange, nowcastAvailable }: Props) {
  const sources: PredSource[] = nowcastAvailable ? ["backtest", "nowcast"] : ["backtest"];
  return (
    <div className="horizon-toggle">
      <span className="horizon-label">Prediction</span>
      <div className="horizon-buttons">
        {sources.map((s) => (
          <button
            key={s}
            className={s === value ? "active" : ""}
            onClick={() => onChange(s)}
            title={HINTS[s]}
          >
            {LABELS[s]}
          </button>
        ))}
      </div>
    </div>
  );
}
