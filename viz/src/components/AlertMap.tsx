import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { probabilityToColor } from "../utils/colorScale";
import type { Horizon } from "../utils/colorScale";
import type { PredictionsData } from "../hooks/usePredictions";

interface Props {
  predictions: PredictionsData;
  horizon: Horizon;
  activeAlerts: Set<string>;
  selected: string | null;
  onSelect: (code: string | null) => void;
}

// Minimal raster-free style: just a background + our GeoJSON. No external tiles
// (keeps the dashboard fully offline / no API key, per STRUCTURE §9).
const BASE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {},
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#0b1120" } },
  ],
};

export function AlertMap({ predictions, horizon, activeAlerts, selected, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Init map once.
  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASE_STYLE,
      center: [31.5, 48.8],
      zoom: 4.7,
      attributionControl: false,
    });
    mapRef.current = map;

    map.on("load", async () => {
      const geo = await fetch("/ukraine-oblasts.geojson").then((r) => r.json());
      // Inject per-feature props the layers read.
      for (const f of geo.features) {
        f.properties.prob = 0;
        f.properties.fillColor = "#1e293b";
        f.properties.alerting = false;
      }
      map.addSource("oblasts", { type: "geojson", data: geo });

      map.addLayer({
        id: "oblast-fill",
        type: "fill",
        source: "oblasts",
        paint: {
          "fill-color": ["get", "fillColor"],
          "fill-opacity": [
            "interpolate", ["linear"], ["get", "prob"],
            0, 0.35, 1, 0.92,
          ],
        },
      });

      map.addLayer({
        id: "oblast-alert",
        type: "line",
        source: "oblasts",
        filter: ["==", ["get", "alerting"], true],
        paint: {
          "line-color": "#f87171",
          "line-width": 3.5,
          "line-dasharray": [2, 1.5],
        },
      });

      map.addLayer({
        id: "oblast-border",
        type: "line",
        source: "oblasts",
        paint: {
          "line-color": "#334155",
          "line-width": ["case", ["boolean", ["get", "selected"], false], 3, 0.8],
        },
      });

      map.addLayer({
        id: "oblast-label",
        type: "symbol",
        source: "oblasts",
        layout: {
          "text-field": ["get", "name"],
          "text-size": 10,
        },
        paint: {
          "text-color": "#e2e8f0",
          "text-halo-color": "#0b1120",
          "text-halo-width": 1.2,
        },
      });

      map.on("click", "oblast-fill", (e) => {
        const code = e.features?.[0]?.properties?.code as string | undefined;
        if (code) onSelect(code);
      });
      map.on("mouseenter", "oblast-fill", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "oblast-fill", () => {
        map.getCanvas().style.cursor = "";
      });

      setLoaded(true);
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-paint when horizon / alerts / selection / data change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;
    const src = map.getSource("oblasts") as maplibregl.GeoJSONSource | undefined;
    if (!src) return;

    fetch("/ukraine-oblasts.geojson")
      .then((r) => r.json())
      .then((geo) => {
        for (const f of geo.features) {
          const code = f.properties.code as string;
          const p = predictions.predictions[code]?.[horizon as Horizon] ?? 0;
          const prob = p ?? 0;
          f.properties.prob = prob;
          f.properties.fillColor = probabilityToColor(prob);
          f.properties.alerting = activeAlerts.has(code);
          f.properties.selected = code === selected;
        }
        src.setData(geo);
      });
  }, [predictions, horizon, activeAlerts, selected, loaded]);

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map" />
      <button
        className="deselect-hint"
        style={{ display: selected ? "block" : "none" }}
        onClick={() => onSelect(null)}
      >
        ✕ clear selection
      </button>
    </div>
  );
}
