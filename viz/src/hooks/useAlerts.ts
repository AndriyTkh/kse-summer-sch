import { useState, useEffect } from "react";

// Maps alerts.in.ua location_title (UA names) -> our oblast codes.
const NAME_TO_CODE: Record<string, string> = {
  "Черкаська область": "cherkaska",
  "Чернігівська область": "chernihivska",
  "Чернівецька область": "chernivetska",
  "Автономна Республіка Крим": "crimea",
  "Дніпропетровська область": "dnipropetrovska",
  "Донецька область": "donetska",
  "Івано-Франківська область": "ivano-frankivska",
  "Харківська область": "kharkivska",
  "Херсонська область": "khersonska",
  "Хмельницька область": "khmelnytska",
  "Київська область": "kyivska",
  "м. Київ": "kyiv-city",
  "Кіровоградська область": "kirovohradska",
  "Луганська область": "luhanska",
  "Львівська область": "lvivska",
  "Миколаївська область": "mykolaivska",
  "Одеська область": "odeska",
  "Полтавська область": "poltavska",
  "Рівненська область": "rivnenska",
  "м. Севастополь": "sevastopol",
  "Сумська область": "sumska",
  "Тернопільська область": "ternopilska",
  "Вінницька область": "vinnytska",
  "Волинська область": "volynska",
  "Закарпатська область": "zakarpatska",
  "Запорізька область": "zaporizka",
  "Житомирська область": "zhytomyrska",
};

const POLL_MS = 60_000;

// alerts.in.ua needs a token. Set VITE_ALERTS_TOKEN to enable the live overlay;
// without it (or on a network error) the overlay degrades to empty.
const TOKEN = import.meta.env.VITE_ALERTS_TOKEN as string | undefined;
const API_URL = "https://api.alerts.in.ua/v1/alerts/active.json";

export interface AlertsState {
  active: Set<string>;
  available: boolean;
  lastFetch: Date | null;
}

export function useAlerts() {
  const [state, setState] = useState<AlertsState>({
    active: new Set(),
    available: false,
    lastFetch: null,
  });

  useEffect(() => {
    if (!TOKEN) return; // no token -> overlay stays empty, map still works

    let cancelled = false;

    async function poll() {
      try {
        const res = await fetch(`${API_URL}?token=${TOKEN}`);
        if (!res.ok) throw new Error(`alerts API ${res.status}`);
        const json = await res.json();
        const active = new Set<string>();
        for (const a of json.alerts ?? []) {
          if (a.alert_type !== "air_raid") continue;
          const code = NAME_TO_CODE[a.location_title];
          if (code) active.add(code);
        }
        if (!cancelled) {
          setState({ active, available: true, lastFetch: new Date() });
        }
      } catch {
        if (!cancelled) {
          setState((s) => ({ ...s, available: false, lastFetch: new Date() }));
        }
      }
    }

    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return state;
}
