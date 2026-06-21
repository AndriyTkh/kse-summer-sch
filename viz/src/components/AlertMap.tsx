import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { probabilityToColor, onsetTiming, timingColor } from "../utils/colorScale";
import type { Horizon } from "../utils/colorScale";
import type { PredictionSource, OnsetData } from "../types";
import type { MapMode } from "./ModeToggle";

interface Props {
  predictions: PredictionSource;
  horizon: Horizon;
  activeAlerts: Set<string>;
  selected: string | null;
  onSelect: (code: string | null) => void;
  mode: MapMode;
  onset: OnsetData | null;
}

// Base style: dark background + a CARTO dark raster basemap underneath the
// oblast choropleth, so regions read against real coastlines / neighbours /
// the Black & Azov seas. CARTO basemaps need no API key (attribution required).
const BASE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    carto: {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png",
        "https://b.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png",
        "https://c.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png",
        "https://d.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png",
      ],
      tileSize: 256,
      attribution:
        '© <a href="https://carto.com/attributions">CARTO</a> · © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#0b1120" } },
    { id: "carto", type: "raster", source: "carto", paint: { "raster-opacity": 0.7 } },
  ],
};

export function AlertMap({ predictions, horizon, activeAlerts, selected, onSelect, mode, onset }: Props) {
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
      attributionControl: { compact: true },
    });
    mapRef.current = map;

    map.on("load", async () => {
      const geo = await fetch("/ukraine-oblasts.geojson").then((r) => r.json());
      // promoteId lets us address features by `code` via setFeatureState — so
      // re-paints touch only changed features (no refetch / no setData of the
      // ~470KB source on every horizon/selection change).
      map.addSource("oblasts", { type: "geojson", data: geo, promoteId: "code" });

      map.addLayer({
        id: "oblast-fill",
        type: "fill",
        source: "oblasts",
        paint: {
          "fill-color": ["coalesce", ["feature-state", "fillColor"], "#1e293b"],
          // Lower floor than before so low-risk oblasts let the basemap show
          // through; high-risk stays near-opaque.
          "fill-opacity": [
            "interpolate", ["linear"], ["coalesce", ["feature-state", "prob"], 0],
            0, 0.15, 1, 0.85,
          ],
        },
      });

      map.addLayer({
        id: "oblast-alert",
        type: "line",
        source: "oblasts",
        paint: {
          "line-color": "#f87171",
          "line-width": 3.5,
          "line-dasharray": [2, 1.5],
          // feature-state can't drive layer `filter`, so toggle visibility via
          // opacity instead.
          "line-opacity": ["case", ["boolean", ["feature-state", "alerting"], false], 1, 0],
        },
      });

      map.addLayer({
        id: "oblast-border",
        type: "line",
        source: "oblasts",
        paint: {
          "line-color": ["case", ["boolean", ["feature-state", "selected"], false], "#e2e8f0", "#64748b"],
          "line-width": ["case", ["boolean", ["feature-state", "selected"], false], 3, 1],
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

  // Re-paint when horizon / alerts / selection / data change. Mutates only
  // feature-state per oblast — no refetch, no full-source setData.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loaded) return;

    const onsetMode = mode === "onset" && !!onset;
    // Source of truth for which oblasts to paint (both share the same code set).
    const codes = Object.keys((onsetMode ? onset! : predictions).predictions);

    for (const code of codes) {
      let prob: number;
      let fillColor: string;
      if (onsetMode) {
        // Timing mode: colour by SOONEST time-to-alert; opacity by overall onset
        // likelihood in the 6h window so calm/unlikely zones fade to the basemap.
        const t = onsetTiming(onset!.predictions[code]);
        fillColor = timingColor(t.etaHours);
        prob = t.reach;
      } else {
        prob = predictions.predictions[code]?.[horizon as Horizon] ?? 0;
        fillColor = probabilityToColor(prob);
      }
      map.setFeatureState(
        { source: "oblasts", id: code },
        {
          prob,
          fillColor,
          alerting: activeAlerts.has(code),
          selected: code === selected,
        },
      );
    }
  }, [predictions, horizon, activeAlerts, selected, loaded, mode, onset]);

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
