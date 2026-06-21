export type MapMode = "risk" | "onset";

interface Props {
  value: MapMode;
  onChange: (m: MapMode) => void;
  onsetAvailable: boolean;
}

const LABELS: Record<MapMode, string> = {
  risk: "Risk",
  onset: "Timing",
};

const HINTS: Record<MapMode, string> = {
  risk: "Model B — P(an alert is ACTIVE in the next H). Coloured by probability.",
  onset: "Onset model — when a NEW alert is likely to START. Coloured by soonest time-to-alert.",
};

export function ModeToggle({ value, onChange, onsetAvailable }: Props) {
  const modes: MapMode[] = onsetAvailable ? ["risk", "onset"] : ["risk"];
  return (
    <div className="horizon-toggle">
      <span className="horizon-label">Map mode</span>
      <div className="horizon-buttons">
        {modes.map((m) => (
          <button
            key={m}
            className={m === value ? "active" : ""}
            onClick={() => onChange(m)}
            title={HINTS[m]}
          >
            {LABELS[m]}
          </button>
        ))}
      </div>
    </div>
  );
}
