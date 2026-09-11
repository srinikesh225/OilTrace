import { useEffect } from "react";
import {
  MapContainer,
  Rectangle,
  Polygon,
  Polyline,
  CircleMarker,
  Tooltip,
  useMap,
} from "react-leaflet";

const COLORS = {
  footprint: "#5b7183",
  slick: "#e0a94a", // detected slick — amber
  release: "#2f9e8f", // backtracked release area — teal
  track: "#8a9bab", // ordinary vessel track
  culprit: "#d1495b", // top-ranked vessel
  selected: "#2f6fed", // currently selected vessel
};

function FitToScene({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (!bounds) return;
    map.fitBounds(
      [
        [bounds.min_lat, bounds.min_lon],
        [bounds.max_lat, bounds.max_lon],
      ],
      { padding: [24, 24] }
    );
  }, [bounds, map]);
  return null;
}

const ring = (pts) => pts.map((p) => [p.lat, p.lon]);

export default function Map({ data, topMmsi, selectedMmsi, onSelect }) {
  const b = data?.scene?.bounds;
  const center = b
    ? [(b.min_lat + b.max_lat) / 2, (b.min_lon + b.max_lon) / 2]
    : [56, 11.75];

  return (
    <MapContainer
      center={center}
      zoom={8}
      zoomControl={true}
      style={{ height: "100%", width: "100%", background: "#0d2230" }}
    >
      {b && <FitToScene bounds={b} />}

      {/* Scene footprint */}
      {b && (
        <Rectangle
          bounds={[
            [b.min_lat, b.min_lon],
            [b.max_lat, b.max_lon],
          ]}
          pathOptions={{
            color: COLORS.footprint,
            weight: 1,
            fill: false,
            dashArray: "4 4",
          }}
        />
      )}

      {/* Detected slick polygon */}
      {data?.detected_slick && (
        <Polygon
          positions={ring(data.detected_slick.boundary)}
          pathOptions={{ color: COLORS.slick, weight: 2, fillOpacity: 0.35 }}
        >
          <Tooltip sticky>
            Detected slick · bearing {data.detected_slick.bearing_deg}° ·{" "}
            {data.detected_slick.area_km2} km²
          </Tooltip>
        </Polygon>
      )}

      {/* Other detections that passed the wind gate but are not the primary
          slick — drawn faint so nothing is hidden, but clearly secondary. */}
      {data?.other_detections?.map((p) => (
        <Polygon
          key={p.polygon_id}
          positions={ring(p.boundary)}
          pathOptions={{
            color: COLORS.slick,
            weight: 1,
            dashArray: "3 4",
            fillOpacity: 0.08,
            opacity: 0.6,
          }}
        >
          <Tooltip sticky>
            Other detection ({p.polygon_id}) · not attributed this run
          </Tooltip>
        </Polygon>
      ))}

      {/* Backtracked release area */}
      {data?.release && (
        <Polygon
          positions={ring(data.release.release_polygon)}
          pathOptions={{
            color: COLORS.release,
            weight: 2,
            dashArray: "6 4",
            fillOpacity: 0.15,
          }}
        >
          <Tooltip sticky>
            Estimated release area · {data.release.release_time}
          </Tooltip>
        </Polygon>
      )}

      {/* Vessel tracks */}
      {data?.all_vessels?.map((v) => {
        const isTop = v.mmsi === topMmsi;
        const isSel = v.mmsi === selectedMmsi;
        const color = isSel
          ? COLORS.selected
          : isTop
          ? COLORS.culprit
          : COLORS.track;
        const weight = isSel ? 5 : isTop ? 4 : 2;
        return (
          <Polyline
            key={v.mmsi}
            positions={v.positions.map((p) => [p.lat, p.lon])}
            pathOptions={{ color, weight, opacity: isTop || isSel ? 0.95 : 0.6 }}
            eventHandlers={{ click: () => onSelect && onSelect(v.mmsi) }}
          >
            <Tooltip sticky>
              {v.name} · MMSI {v.mmsi}
              {isTop ? " · top candidate" : ""}
            </Tooltip>
          </Polyline>
        );
      })}

      {/* Matched positions for ranked candidates */}
      {data?.ranked_candidates?.map((c) => (
        <CircleMarker
          key={c.mmsi}
          center={[c.matched_position.lat, c.matched_position.lon]}
          radius={c.mmsi === topMmsi ? 8 : 6}
          pathOptions={{
            color: c.mmsi === selectedMmsi ? COLORS.selected : COLORS.culprit,
            weight: 2,
            fillColor: "#ffffff",
            fillOpacity: 0.9,
          }}
          eventHandlers={{ click: () => onSelect && onSelect(c.mmsi) }}
        >
          <Tooltip>
            {c.name} matched here · {c.matched_position.time}
          </Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
