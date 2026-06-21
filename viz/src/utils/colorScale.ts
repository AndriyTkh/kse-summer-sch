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
