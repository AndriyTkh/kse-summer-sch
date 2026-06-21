export function probabilityToColor(p: number): string {
  const clamped = Math.max(0, Math.min(1, p));
  if (clamped <= 0.5) {
    const t = clamped / 0.5;
    const r = Math.round(34 + t * (234 - 34));
    const g = Math.round(197 - t * (197 - 179));
    const b = Math.round(94 - t * (94 - 8));
    return `rgb(${r},${g},${b})`;
  }
  const t = (clamped - 0.5) / 0.5;
  const r = Math.round(234 + t * (239 - 234));
  const g = Math.round(179 - t * (179 - 68));
  const b = Math.round(8 + t * (68 - 8));
  return `rgb(${r},${g},${b})`;
}

export function probabilityToOpacity(p: number): number {
  return 0.3 + Math.min(1, p) * 0.6;
}

export const HORIZONS = ["30m", "1h", "3h", "6h"] as const;
export type Horizon = (typeof HORIZONS)[number];

export const HORIZON_HOURS: Record<Horizon, number> = {
  "30m": 0.5, "1h": 1, "3h": 3, "6h": 6,
};

// --- onset / timing mode -------------------------------------------------
// The onset model gives a CDF: P(a NEW alert starts within H). "Time to alert" is the
// SOONEST horizon whose cumulative onset probability crosses a threshold τ; if none do,
// onset is unlikely inside the 6h window (treated as "calm", eta = Infinity).
export const ONSET_TAU = 0.5;

export interface OnsetTiming {
  etaHours: number;        // soonest horizon crossing τ, or Infinity
  etaLabel: string;        // "≤1h", "≤3h", "≤6h", ">6h"
  reach: number;           // CDF at the longest horizon = overall onset likelihood in window
}

/** Soonest-crossing time-to-alert from a per-horizon onset CDF. */
export function onsetTiming(
  preds: Partial<Record<Horizon, number | null>> | undefined,
  tau = ONSET_TAU,
): OnsetTiming {
  let etaHours = Infinity;
  let reach = 0;
  for (const h of HORIZONS) {
    const p = preds?.[h];
    if (p == null) continue;
    reach = p; // CDF is non-decreasing; last seen = widest horizon
    if (etaHours === Infinity && p >= tau) etaHours = HORIZON_HOURS[h];
  }
  const etaLabel =
    etaHours <= 1 ? "≤1h" : etaHours <= 3 ? "≤3h" : etaHours <= 6 ? "≤6h" : ">6h";
  return { etaHours, etaLabel, reach };
}

// Discrete time-to-alert palette: sooner = hotter. Calm (no τ-crossing) = green.
export function timingColor(etaHours: number): string {
  if (etaHours <= 1) return "rgb(239,68,68)";   // red — imminent
  if (etaHours <= 3) return "rgb(249,115,22)";  // orange
  if (etaHours <= 6) return "rgb(234,179,8)";   // yellow
  return "rgb(34,197,94)";                       // green — calm
}

export const TIMING_LEGEND: { label: string; color: string }[] = [
  { label: "≤1h", color: "rgb(239,68,68)" },
  { label: "≤3h", color: "rgb(249,115,22)" },
  { label: "≤6h", color: "rgb(234,179,8)" },
  { label: ">6h", color: "rgb(34,197,94)" },
];

export const DISPLAY_NAMES: Record<string, string> = {
  cherkaska: "Cherkasy",
  chernihivska: "Chernihiv",
  chernivetska: "Chernivtsi",
  crimea: "Crimea",
  dnipropetrovska: "Dnipropetrovsk",
  donetska: "Donetsk",
  "ivano-frankivska": "Ivano-Frankivsk",
  kharkivska: "Kharkiv",
  khersonska: "Kherson",
  khmelnytska: "Khmelnytskyi",
  kyivska: "Kyiv Oblast",
  "kyiv-city": "Kyiv City",
  kirovohradska: "Kirovohrad",
  luhanska: "Luhansk",
  lvivska: "Lviv",
  mykolaivska: "Mykolaiv",
  odeska: "Odesa",
  poltavska: "Poltava",
  rivnenska: "Rivne",
  sevastopol: "Sevastopol",
  sumska: "Sumy",
  ternopilska: "Ternopil",
  vinnytska: "Vinnytsia",
  volynska: "Volyn",
  zakarpatska: "Zakarpattia",
  zaporizka: "Zaporizhzhia",
  zhytomyrska: "Zhytomyr",
};
