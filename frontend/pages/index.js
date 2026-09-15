import Head from "next/head";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

// Backend API base URL. NEXT_PUBLIC_* is inlined into the browser bundle at
// build time; set NEXT_PUBLIC_API_URL to the backend's domain in production
// (e.g. https://api.example.com). Falls back to the local dev backend, so
// local development needs no environment variables.
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Leaflet touches `window`, so the map is client-only.
const Map = dynamic(() => import("../components/Map"), {
  ssr: false,
  loading: () => <div className="map-loading">Loading map…</div>,
});

function ScoreBar({ label, value, weight }) {
  const pct = Math.round(value * 100);
  return (
    <div className="scorebar">
      <div className="scorebar-head">
        <span>{label}</span>
        <span className="scorebar-val">
          {value.toFixed(2)}
          {/* Weight comes from the evidence bundle; shown only when known,
              never a hardcoded stand-in that could diverge from the backend. */}
          {weight != null && <span className="weight"> w={weight}</span>}
        </span>
      </div>
      <div className="scorebar-track">
        <div className="scorebar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// The scenes the backend exposes via POST /analyze {scene}. Normal and Calm
// are mandatory; Ambiguous is shown because the backend supports it.
const SCENES = [
  { id: "normal", label: "Normal" },
  { id: "ambiguous", label: "Ambiguous" },
  { id: "calm", label: "Calm" },
];

// The exact set of scene ids the request body may carry: "normal" |
// "ambiguous" | "calm". Anything else (e.g. a React event forwarded by a bare
// onClick={runAnalysis}) must be rejected before serialising, not stringified.
const VALID_SCENES = SCENES.map((s) => s.id);

// Tags a failure so the UI can tell a client-side mistake (bad scene value)
// apart from the request never reaching / being rejected by the backend.
class AnalysisError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = "AnalysisError";
    this.kind = kind; // "client" | "http"
  }
}

// Describe a rejected scene value without serialising it — the whole point is
// that the value may be a circular object (a synthetic event) that JSON.stringify
// would choke on. Name it by type so the thrown error is actionable.
function describeScene(value) {
  if (typeof value === "string") return `"${value}"`;
  if (value && typeof value === "object") {
    return `a ${value.constructor?.name || "object"} value`;
  }
  return String(value);
}

export default function Home() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [weights, setWeights] = useState(null);
  const [scene, setScene] = useState("normal");

  // Call the existing POST /analyze with the scene in the body (existing API
  // contract — no new endpoint). Defaults to the current scene.
  async function runAnalysis(sceneName = scene) {
    setLoading(true);
    setError(null);
    try {
      // Guard the request body before it is serialised. If sceneName is not a
      // known scene id (the classic bug: a bare onClick forwards the click
      // event here), fail loudly and name the bad value instead of letting
      // JSON.stringify hit a circular structure.
      if (!VALID_SCENES.includes(sceneName)) {
        throw new AnalysisError(
          "client",
          `Invalid scene ${describeScene(sceneName)}. Expected one of ${VALID_SCENES.join(", ")}.`
        );
      }

      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene: sceneName }),
      });
      if (!res.ok) {
        throw new AnalysisError("http", `API returned ${res.status}`);
      }
      const json = await res.json();
      setData(json);
      setSelected(json.ranked_candidates?.[0]?.mmsi ?? null);
      // Pull weights from the evidence bundle for the score breakdown labels.
      const ev = await fetch(`${API}/evidence/${json.evidence_id}`);
      if (ev.ok) {
        const bundle = await ev.json();
        setWeights(bundle.config_used || null);
      }
    } catch (e) {
      // A client-side error (bad scene value) is our bug to fix, not a backend
      // outage — say so plainly. HTTP and network failures both mean the
      // request didn't succeed against the API, so they share the API message.
      if (e instanceof AnalysisError && e.kind === "client") {
        setError(`Couldn't build the analysis request: ${e.message}`);
      } else if (e instanceof AnalysisError && e.kind === "http") {
        setError(`The OILTRACE API returned an error. (${e.message})`);
      } else {
        setError(
          `Could not reach the OILTRACE API. Is the backend running? (${e.message})`
        );
      }
    } finally {
      setLoading(false);
    }
  }

  function selectScene(sceneName) {
    if (sceneName === scene || loading) return;
    setScene(sceneName);
    runAnalysis(sceneName);
  }

  useEffect(() => {
    runAnalysis("normal");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const topMmsi = data?.ranked_candidates?.[0]?.mmsi ?? null;

  // A slick rejected by the wind gate: no primary slick, but a filter-stage
  // rejection carries the backend's exact reason. Shown verbatim, never rewritten.
  const filterRejection = data?.rejected?.find((r) => r.stage === "filter");
  const slickRejected = data && !data.detected_slick && filterRejection;

  return (
    <>
      <Head>
        <title>OILTRACE — slick attribution</title>
      </Head>

      <div className="app">
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true" />
            <div>
              <h1>OILTRACE</h1>
              <p className="tagline">
                Trace a radar-observed oil slick back to a candidate vessel
              </p>
            </div>
          </div>
          <div className="topbar-right">
            <div className="scene-switch" role="group" aria-label="Scene">
              {SCENES.map((s) => (
                <button
                  key={s.id}
                  className={`scene-btn${scene === s.id ? " active" : ""}`}
                  onClick={() => selectScene(s.id)}
                  disabled={loading}
                  aria-pressed={scene === s.id}
                >
                  {s.label}
                </button>
              ))}
            </div>
            {data?.scene && (
              <div className="scene-meta">
                <div>
                  <span className="k">Scene</span>
                  <span className="v">{data.scene.scene_id}</span>
                </div>
                <div>
                  <span className="k">Acquired</span>
                  <span className="v">{data.scene.acquired_at}</span>
                </div>
                {data.wind_at_scene && (
                  <div>
                    <span className="k">Wind at scene</span>
                    <span className="v">
                      {data.wind_at_scene.speed_ms} m/s @{" "}
                      {data.wind_at_scene.dir_deg}°
                    </span>
                  </div>
                )}
              </div>
            )}
            <button
              className="run"
              onClick={() => runAnalysis(scene)}
              disabled={loading}
            >
              {loading ? "Running…" : "Re-run analysis"}
            </button>
          </div>
        </header>

        {data?.data_source === "SYNTHETIC_FIXTURE" && (
          <div className="banner provenance" role="note">
            <strong>Synthetic sample data.</strong>{" "}
            {data.provenance_note ||
              "Generated fixture — results demonstrate pipeline mechanics, not a real detection."}
          </div>
        )}

        {slickRejected && (
          <div className="banner rejection" role="status">
            <strong>Slick rejected at the wind gate.</strong>{" "}
            {/* Verbatim backend reason — not rewritten by the frontend. */}
            {filterRejection.reason}
          </div>
        )}

        {error && <div className="banner error">{error}</div>}

        <main className="layout">
          <section className="map-col" aria-label="Map">
            <Map
              data={data}
              topMmsi={topMmsi}
              selectedMmsi={selected}
              onSelect={setSelected}
            />
            {data?.other_detections?.length > 0 && (
              <div className="other-note" role="note">
                {data.other_detections.length} other detection
                {data.other_detections.length > 1 ? "s" : ""} passed the wind
                gate but {data.other_detections.length > 1 ? "are" : "is"} not
                the primary slick (not attributed this run).
              </div>
            )}
            <ul className="legend">
              <li>
                <span className="sw" style={{ background: "#e0a94a" }} />
                Detected slick
              </li>
              <li>
                <span className="sw" style={{ background: "#2f9e8f" }} />
                Release area (backtracked)
              </li>
              <li>
                <span className="sw" style={{ background: "#d1495b" }} />
                Top candidate track
              </li>
              <li>
                <span className="sw" style={{ background: "#8a9bab" }} />
                Other vessel tracks
              </li>
              <li>
                <span
                  className="sw"
                  style={{ background: "#1c3a35", border: "1px solid #3c6459" }}
                />
                Land (Denmark / Sweden)
              </li>
            </ul>
          </section>

          <aside className="panel-col">
            <div className="panel">
              <h2>
                Ranked candidates
                {data?.ranked_candidates && (
                  <span className="count">
                    {data.ranked_candidates.length}
                  </span>
                )}
              </h2>
              {data?.ranked_candidates?.length ? (
                <ul className="candidates">
                  {data.ranked_candidates.map((c) => {
                    const open = selected === c.mmsi;
                    return (
                      <li
                        key={c.mmsi}
                        className={`candidate${open ? " open" : ""}${
                          c.mmsi === topMmsi ? " top" : ""
                        }`}
                      >
                        <button
                          className="candidate-head"
                          onClick={() => setSelected(open ? null : c.mmsi)}
                          aria-expanded={open}
                        >
                          <span className="rank">#{c.rank}</span>
                          <span className="cname">
                            {c.name}
                            <span className="mmsi">MMSI {c.mmsi}</span>
                          </span>
                          <span className="total">
                            {c.score.total.toFixed(2)}
                          </span>
                        </button>

                        {c.has_ais_gap && (
                          <p className="flag" title={c.ais_gap_note}>
                            AIS reporting gap — possible dark vessel
                          </p>
                        )}

                        {open && (
                          <div className="breakdown">
                            <ScoreBar
                              label="Time overlap"
                              value={c.score.time_score}
                              weight={weights?.SCORE_WEIGHT_TIME}
                            />
                            <ScoreBar
                              label="Spatial (depth inside area)"
                              value={c.score.spatial_score}
                              weight={weights?.SCORE_WEIGHT_SPATIAL}
                            />
                            <ScoreBar
                              label="Course match"
                              value={c.score.course_score}
                              weight={weights?.SCORE_WEIGHT_COURSE}
                            />
                            <dl className="facts">
                              <div>
                                <dt>Matched at</dt>
                                <dd>{c.matched_position.time}</dd>
                              </div>
                              <div>
                                <dt>Heading</dt>
                                <dd>{c.heading_deg}°</dd>
                              </div>
                              {c.ais_gap_note && (
                                <div>
                                  <dt>Note</dt>
                                  <dd>{c.ais_gap_note}</dd>
                                </div>
                              )}
                            </dl>
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <p className="empty">No candidate vessels matched.</p>
              )}
            </div>

            <div className="panel">
              <h2>
                Rejected detections
                {data?.rejected && (
                  <span className="count">{data.rejected.length}</span>
                )}
              </h2>
              {data?.rejected?.length ? (
                <ul className="rejects">
                  {data.rejected.map((r, i) => (
                    <li key={i}>
                      <span className="rej-id">{r.polygon_id}</span>
                      <span className={`rej-stage stage-${r.stage}`}>
                        {r.stage}
                      </span>
                      <p>{r.reason}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty">Nothing rejected.</p>
              )}
            </div>

            {data?.evidence_id && (
              <a
                className="evidence-link"
                href={`${API}/evidence/${data.evidence_id}`}
                target="_blank"
                rel="noreferrer"
              >
                Open full evidence bundle (JSON)
              </a>
            )}
          </aside>
        </main>

        <footer className="disclaimer">
          Detection is not attribution. Attribution is not proof.
        </footer>
      </div>
    </>
  );
}
