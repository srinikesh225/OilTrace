"""
OILTRACE API.

Wires the five stages into one chain and exposes three endpoints:

  POST /analyze         run the whole chain on the bundled scene
  GET  /candidates      just the ranked candidate list
  GET  /evidence/{id}   the full evidence bundle for a run

The chain runs on bundled sample files only — no database, no network. A fixed
random seed makes every run reproducible, so the same scene always yields the
same result.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Literal, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .models import AnalyzeResponse, SceneMeta
from .stages import explain as explain_stage
from .stages import filter as filter_stage
from .stages import name as name_stage
from .stages import rewind as rewind_stage
from .stages.name import FileShipSource
from .stages.segmenters import make_segmenter

app = FastAPI(
    title="OILTRACE",
    description="Trace a satellite-observed oil slick back to the ship that "
                "released it. Prototype skeleton — bundled sample data only.",
    version="0.1.0",
)

# The frontend runs on a separate origin (its own domain in production). Only
# the explicitly allowed origins may call the API — never a wildcard. The list
# comes from ALLOWED_ORIGINS (comma-separated), defaulting to the local dev
# frontend. See config.allowed_origins().
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of the most recent run. No database by design (day one).
# The lock serialises the (CPU-bound, shared-state) pipeline so concurrent
# requests can't interleave writes to the cache.
_LAST_RESULT: dict | None = None
_EVIDENCE: dict[str, dict] = {}
_PIPELINE_LOCK = threading.Lock()

# The segmenter is built once and reused so the model backend loads its weights
# a single time (not per request). Chosen by config.SEGMENTER_BACKEND.
_SEGMENTER = None


def _get_segmenter():
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = make_segmenter()
    return _SEGMENTER


class AnalyzeRequest(BaseModel):
    """Optional body for POST /analyze. Restricting `scene` to the known names
    (a Literal) both documents the choices and blocks any arbitrary path."""
    scene: Literal["normal", "ambiguous", "calm"] = config.DEFAULT_SCENE


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(scene_name: str = config.DEFAULT_SCENE) -> tuple[AnalyzeResponse, dict]:
    """Run SEE -> FILTER -> REWIND -> NAME -> EXPLAIN on a named bundled scene."""
    paths = config.scene_paths(scene_name)  # validates the name; no arbitrary paths
    scene = SceneMeta(**_load_json(paths["meta"]))
    wind_samples = _load_json(paths["wind"])
    ship_source = FileShipSource(paths["ships"])

    # 1. SEE — detect slick polygons (threshold or model backend).
    segmenter = _get_segmenter()
    see_out, see_rejections = segmenter.segment(scene.scene_id, paths["tif"])

    # 2. FILTER — wind gate.
    filter_out = filter_stage.apply_wind_gate(see_out, scene.acquired_at, wind_samples)
    all_rejections = see_rejections + filter_out.rejected

    # Choose the primary slick as the largest accepted polygon (not merely the
    # first). Any other accepted polygons are surfaced as other_detections so
    # they are never silently dropped.
    # PROTOTYPE: only the primary slick is attributed; multi-slick attribution
    # (one release area + candidate set per slick) is future work.
    accepted = sorted(filter_out.accepted, key=lambda p: p.area_px, reverse=True)
    detected_slick = accepted[0] if accepted else None
    other_detections = accepted[1:]

    rewind_out = None
    explain_out = None
    evidence_bundle = None
    all_vessels = ship_source.fetch()

    if detected_slick is not None:
        # 3. REWIND — backward drift to the release area.
        rewind_out = rewind_stage.rewind(
            detected_slick, scene.acquired_at, wind_samples, filter_out.wind_at_scene)

        # 4. NAME — vessels in the release area at the release time.
        name_out, all_vessels = name_stage.name_vessels(rewind_out, ship_source)

        # 5. EXPLAIN — score, rank, evidence bundle.
        explain_out, evidence_bundle = explain_stage.explain(
            name_out, rewind_out, scene, see_out, filter_out)

    ranked = explain_out.ranked_candidates if explain_out else []
    evidence_id = explain_out.evidence_id if explain_out else scene.scene_id

    response = AnalyzeResponse(
        scene=scene,
        detected_slick=detected_slick,
        other_detections=other_detections,
        release=rewind_out,
        ranked_candidates=ranked,
        separation_flag=explain_out.separation_flag if explain_out else None,
        joint_candidates=explain_out.joint_candidates if explain_out else [],
        score_gap=explain_out.score_gap if explain_out else None,
        rejected=all_rejections,
        all_vessels=all_vessels,
        wind_at_scene=filter_out.wind_at_scene,
        scene_name=scene_name,
        evidence_id=evidence_id,
        data_source=config.DATA_SOURCE,
        provenance_note=config.PROVENANCE_NOTE,
        disclaimer=config.DISCLAIMER,
    )

    if evidence_bundle is None:
        evidence_bundle = {
            "evidence_id": evidence_id,
            "scene": scene_name,
            "scene_id": scene.scene_id,
            "scene_timestamp": scene.acquired_at,
            "data_source": config.DATA_SOURCE,
            "provenance_note": config.PROVENANCE_NOTE,
            "note": "No slick survived the detector/wind gate; nothing to attribute.",
            "rejections": [r.model_dump() for r in all_rejections],
            "disclaimer": config.DISCLAIMER,
        }

    return response, evidence_bundle


def _run_and_cache(scene_name: str = config.DEFAULT_SCENE) -> tuple[dict, dict, float]:
    """Run the pipeline under the lock and update the cache. Input problems
    (missing/corrupt sample files) become a clean HTTP 500 with a readable
    message instead of a raw stack trace."""
    global _LAST_RESULT
    try:
        with _PIPELINE_LOCK:
            start = time.perf_counter()
            response, bundle = run_pipeline(scene_name)
            elapsed = time.perf_counter() - start
            _LAST_RESULT = response.model_dump()
            _EVIDENCE[bundle["evidence_id"]] = bundle
            return _LAST_RESULT, bundle, elapsed
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"A required sample file is missing: {e}. "
                   f"Run `python -m scripts.make_sample_data` in backend/.",
        )
    except Exception as e:  # e.g. a corrupt GeoTIFF, malformed JSON
        raise HTTPException(
            status_code=500,
            detail=f"OILTRACE could not process the bundled scene ({type(e).__name__}: {e}).",
        )


def _ensure_run():
    if _LAST_RESULT is None:
        _run_and_cache()


@app.get("/")
def root():
    return {"service": "OILTRACE", "version": "0.1.0",
            "data_source": config.DATA_SOURCE, "disclaimer": config.DISCLAIMER}


@app.get("/health")
def health():
    """Lightweight liveness/readiness probe for hosting platforms. Does no
    pipeline work, no image processing, no network calls — just confirms the
    app is up and reports which scene fixtures are actually present on disk."""
    return {"status": "ok", "scenes": config.available_scenes()}


@app.post("/analyze")
def analyze(body: Optional[AnalyzeRequest] = Body(default=None)):
    """Run the whole chain on the requested scene and return the full result.

    The optional body selects the scene (`normal`, `ambiguous`, `calm`). With no
    body it defaults to `normal`, so existing callers are unaffected. An unknown
    scene name is rejected by the request model (HTTP 422)."""
    scene_name = body.scene if body is not None else config.DEFAULT_SCENE
    result, _bundle, elapsed = _run_and_cache(scene_name)
    out = dict(result)
    # Surface the runtime so the <10s budget is visible, not just asserted.
    out["_runtime_seconds"] = round(elapsed, 3)
    return out


@app.get("/candidates")
def candidates():
    """Just the ranked candidate list from the most recent run."""
    _ensure_run()
    return {
        "scene_id": _LAST_RESULT["scene"]["scene_id"],
        "ranked_candidates": _LAST_RESULT["ranked_candidates"],
        "data_source": config.DATA_SOURCE,
        "disclaimer": config.DISCLAIMER,
    }


@app.get("/evidence/{evidence_id}")
def evidence(evidence_id: str):
    """The full evidence bundle for a given run id."""
    _ensure_run()
    bundle = _EVIDENCE.get(evidence_id)
    if bundle is None:
        raise HTTPException(status_code=404,
                            detail=f"No evidence bundle for id '{evidence_id}'.")
    return bundle


if __name__ == "__main__":
    # Production/container entrypoint: bind 0.0.0.0 on $PORT (default 8000) so
    # the app is reachable on the container's network interface, not just
    # loopback. `python -m app.main` uses this.
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=config.port())
