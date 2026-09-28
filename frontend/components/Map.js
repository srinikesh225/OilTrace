import { useEffect } from "react";
import {
  MapContainer,
  Pane,
  GeoJSON,
  Rectangle,
  Polygon,
  Polyline,
  CircleMarker,
  Tooltip,
  useMap,
} from "react-leaflet";

// Offline geographic context: Danish/Swedish land clipped to REGION_BOUNDS.
// Imported as a bundled module, so it ships inside the app JS and causes no
// runtime network request (no tiles, no remote GeoJSON, no CDN).
import landGeoJson from "../data/land.json";

const COLORS = {
  footprint: "#5b7183",
  slick: "#e0a94a", // detected slick — amber
  release: "#2f9e8f", // backtracked release area — teal
  track: "#8a9bab", // ordinary vessel track
  culprit: "#d1495b", // top-ranked vessel
  selected: "#2f6fed", // currently selected vessel
};

// Muted land against the dark sea background; a thin coastline. Deliberately
// low-contrast so it never competes with the slick/release/track vectors.
// `interactive: false` keeps it from intercepting clicks meant for the tracks.
const LAND_STYLE = {
  fillColor: "#1c3a35",
  fillOpacity: 1,
  color: "#3c6459",
  weight: 0.8,
  interactive: false,
};

// Collect every rendered vector's lat/lon points: scene footprint, detected
// slick, other detections, release area, all vessel tracks, and matched
// candidate positions. The map is fit to the union of these so the useful
// geometry fills the viewport instead of sitting tiny in the centre. Bounds are
// derived only from result data (never from screen size or hardcoded coords).
function collectLatLngs(data) {
  const pts = [];
  const b = data?.scene?.bounds;
  if (b) {
    pts.push([b.min_lat, b.min_lon], [b.max_lat, b.max_lon]);
  }
  const pushRing = (r) => r && r.forEach((p) => pts.push([p.lat, p.lon]));
  if (data?.detected_slick) pushRing(data.detected_slick.boundary);
  data?.other_detections?.forEach((o) => pushRing(o.boundary));
  if (data?.release) pushRing(data.release.release_polygon);
  data?.all_vessels?.forEach((v) =>
    v.positions?.forEach((p) => pts.push([p.lat, p.lon]))
  );
  data?.ranked_candidates?.forEach((c) =>
    c.matched_position &&
    pts.push([c.matched_position.lat, c.matched_position.lon])
  );
  return pts;
}

// A stable signature so the fit re-runs whenever the displayed scene/result
// changes (and only then), never reusing the previous scene's bounds.
function sceneKey(data) {
  return [
    data?.scene?.scene_id,
    data?.detected_slick?.polygon_id,
    data?.release?.release_time,
    data?.all_vessels?.length,
    data?.ranked_candidates?.length,
  ].join("|");
}

function FitToScene({ data }) {
  const map = useMap();
  const key = sceneKey(data);
  useEffect(() => {
    if (!map) return;
    const pts = collectLatLngs(data);
    if (!pts.length) return;
    // animate:false — the fit runs on every scene switch; animating it causes
    // visible jank on mobile. maxZoom guards against over-zooming a tiny tile.
    map.fitBounds(pts, { padding: [24, 24], animate: false, maxZoom: 12 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, map]);
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
      {data && <FitToScene data={data} />}

      {/* Land geography — bottom layer. A dedicated pane with a z-index below
          the overlay pane guarantees the land always sits beneath every
          vector, regardless of mount order or scene switches. Rendered once
          from the static import; the sea is the map background behind it. */}
      <Pane name="landbase" style={{ zIndex: 250 }}>
        <GeoJSON data={landGeoJson} style={() => LAND_STYLE} interactive={false} />
      </Pane>

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
