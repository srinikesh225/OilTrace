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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .models import AnalyzeResponse, SceneMeta
from .stages import explain as explain_stage
from .stages import filter as filter_stage
from .stages import name as name_stage
from .stages import rewind as rewind_stage
from .stages.name import FileShipSource
from .stages.see import Segmenter

app = FastAPI(
    title="OILTRACE",
    description="Trace a satellite-observed oil slick back to the ship that "
                "released it. Prototype skeleton — bundled sample data only.",
    version="0.1.0",
)

# The frontend is served from a different origin in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of the most recent run. No database by design (day one).
# The lock serialises the (CPU-bound, shared-state) pipeline so concurrent
# requests can't interleave writes to the cache.
_LAST_RESULT: dict | None = None
_EVIDENCE: dict[str, dict] = {}
_PIPELINE_LOCK = threading.Lock()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline() -> tuple[AnalyzeResponse, dict]:
    """Run SEE -> FILTER -> REWIND -> NAME -> EXPLAIN on the bundled scene."""
    scene = SceneMeta(**_load_json(config.SCENE_META_JSON))
    wind_samples = _load_json(config.WIND_JSON)
    ship_source = FileShipSource(config.SHIPS_JSON)

    # 1. SEE — detect slick polygons.
    segmenter = Segmenter()
    see_out, see_rejections = segmenter.segment(scene.scene_id, config.SCENE_TIF)

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
        rejected=all_rejections,
        all_vessels=all_vessels,
        wind_at_scene=filter_out.wind_at_scene,
        evidence_id=evidence_id,
        data_source=config.DATA_SOURCE,
        provenance_note=config.PROVENANCE_NOTE,
        disclaimer=config.DISCLAIMER,
    )

    if evidence_bundle is None:
        evidence_bundle = {
            "evidence_id": evidence_id,
            "scene_id": scene.scene_id,
            "scene_timestamp": scene.acquired_at,
            "data_source": config.DATA_SOURCE,
            "provenance_note": config.PROVENANCE_NOTE,
            "note": "No slick survived the detector/wind gate; nothing to attribute.",
            "rejections": [r.model_dump() for r in all_rejections],
            "disclaimer": config.DISCLAIMER,
        }

    return response, evidence_bundle


def _run_and_cache() -> tuple[dict, dict, float]:
    """Run the pipeline under the lock and update the cache. Input problems
    (missing/corrupt sample files) become a clean HTTP 500 with a readable
    message instead of a raw stack trace."""
    global _LAST_RESULT
    try:
        with _PIPELINE_LOCK:
            start = time.perf_counter()
            response, bundle = run_pipeline()
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


@app.post("/analyze")
def analyze():
    """Run the whole chain and return the full result, rejections included."""
    result, _bundle, elapsed = _run_and_cache()
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
