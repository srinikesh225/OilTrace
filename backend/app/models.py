"""
Pydantic models for every stage's input and output.

The pipeline is a chain of pure functions. Each stage takes one of these
models in and returns another, so the data crossing every boundary is typed,
validated, and serialisable to JSON without extra work.

Coordinates are always (lat, lon) in decimal degrees to match Leaflet.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --- Shared geometry --------------------------------------------------------

class GeoPoint(BaseModel):
    lat: float
    lon: float


class Bounds(BaseModel):
    """Axis-aligned lat/lon box: the scene footprint / region of interest."""
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


# --- Scene ------------------------------------------------------------------

class SceneMeta(BaseModel):
    scene_id: str
    acquired_at: str  # ISO 8601 timestamp of acquisition
    bounds: Bounds


# --- Stage 1: see (segmentation) --------------------------------------------

class SlickPolygon(BaseModel):
    """One dark region the detector found in the radar scene."""
    polygon_id: str
    boundary: list[GeoPoint]        # ordered ring in lat/lon
    area_px: float                  # contour area in image pixels
    area_km2: float                 # rough real-world area
    centroid: GeoPoint
    bearing_deg: float              # long-axis bearing, 0=N, 90=E, degrees


class SeeOutput(BaseModel):
    scene_id: str
    polygons: list[SlickPolygon]


# --- Stage 2: filter (wind gate) --------------------------------------------

class WindSample(BaseModel):
    time: str          # ISO 8601
    speed_ms: float    # wind speed, metres per second
    dir_deg: float     # direction the wind blows FROM, 0=N, 90=E


class Rejection(BaseModel):
    """A polygon that did not survive a gate, with the human reason why."""
    polygon_id: str
    stage: str
    reason: str


class FilterOutput(BaseModel):
    accepted: list[SlickPolygon]
    rejected: list[Rejection]
    wind_at_scene: WindSample


# --- Stage 3: rewind (backward drift) ---------------------------------------

class RewindOutput(BaseModel):
    polygon_id: str                 # the slick this release area belongs to
    release_polygon: list[GeoPoint]  # convex hull of backtracked particles
    particle_cloud: list[GeoPoint]   # final backtracked particle positions
    release_time: str               # estimated release time, ISO 8601
    slick_bearing_deg: float        # carried through for course scoring


# --- Stage 4: name (ship matching) ------------------------------------------

class ShipPosition(BaseModel):
    time: str
    lat: float
    lon: float


class Vessel(BaseModel):
    mmsi: str
    name: str
    positions: list[ShipPosition]


class CandidateMatch(BaseModel):
    """A vessel whose track intersects the release area within the window."""
    mmsi: str
    name: str
    matched_position: ShipPosition
    heading_deg: float              # vessel course at the matched position
    dist_inside_km: float           # distance from polygon edge, inward
    time_delta_min: float           # |matched time - release time| in minutes
    has_ais_gap: bool               # a reporting gap longer than the threshold
    max_gap_min: float              # longest gap in this vessel's track


class NameOutput(BaseModel):
    candidates: list[CandidateMatch]


# --- Stage 5: explain (scoring) ---------------------------------------------

class ScoreBreakdown(BaseModel):
    time_score: float
    spatial_score: float
    course_score: float
    total: float


class ScoredCandidate(BaseModel):
    rank: int
    mmsi: str
    name: str
    score: ScoreBreakdown
    matched_position: ShipPosition
    heading_deg: float
    has_ais_gap: bool
    ais_gap_note: Optional[str] = None


class ExplainOutput(BaseModel):
    scene_id: str
    ranked_candidates: list[ScoredCandidate]
    evidence_id: str
    disclaimer: str


# --- Top-level API response -------------------------------------------------

class AnalyzeResponse(BaseModel):
    scene: SceneMeta
    detected_slick: Optional[SlickPolygon]
    # Extra polygons that passed the wind gate but are not the primary slick
    # this run. Surfaced (never silently dropped) so a multi-slick scene is
    # visible even though only the primary is attributed today.
    other_detections: list[SlickPolygon] = Field(default_factory=list)
    release: Optional[RewindOutput]
    ranked_candidates: list[ScoredCandidate]
    rejected: list[Rejection]
    all_vessels: list[Vessel]           # so the frontend can draw every track
    wind_at_scene: WindSample
    evidence_id: str
    data_source: str                    # e.g. SYNTHETIC_FIXTURE
    provenance_note: str                # plain-language "this is sample data"
    disclaimer: str = Field(...)
