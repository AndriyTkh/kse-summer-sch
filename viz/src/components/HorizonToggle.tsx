import { HORIZONS } from "../utils/colorScale";
import type { Horizon } from "../utils/colorScale";

interface Props {
  value: Horizon;
  onChange: (h: Horizon) => void;
}

export function HorizonToggle({ value, onChange }: Props) {
  return (
    <div className="horizon-toggle">
      <span className="horizon-label">Forecast horizon</span>
      <div className="horizon-buttons">
        {HORIZONS.map((h) => (
          <button
            key={h}
            className={h === value ? "active" : ""}
            onClick={() => onChange(h)}
          >
            {h}
          </button>
        ))}
      </div>
    </div>
  );
}
