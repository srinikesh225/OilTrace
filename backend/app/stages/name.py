"""
Stage 4 — NAME: which vessels were in the release area at the release time.

For every vessel we test each reported position against the backtracked
release polygon inside a time window around the estimated release time. A
vessel that falls inside becomes a candidate, tagged with the exact position
and timestamp that matched, its local heading, how deep inside the polygon it
was, and whether its track has an AIS reporting gap (a possible dark-vessel
event).

The vessel data comes through a ShipSource interface, the twin of see.py's
Segmenter: swap FileShipSource for a live AIS feed later and nothing else in
this stage changes.

# PROTOTYPE: naive point-in-polygon over stored AIS points. The real version
# reconciles interpolated tracks, resolves gaps, dedupes MMSI spoofing, and
# pulls from a satellite/coastal AIS service instead of a bundled JSON file.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from datetime import datetime

from shapely.geometry import Point, Polygon

from .. import config
from ..models import CandidateMatch, NameOutput, RewindOutput, Vessel


# --- Ship source interface --------------------------------------------------

class ShipSource(ABC):
    """Provides the vessel tracks to match against. The concrete source is
    chosen at the composition root (main.py); this stage depends only on the
    interface."""

    @abstractmethod
    def fetch(self) -> list[Vessel]:
        ...


class FileShipSource(ShipSource):
    """Loads vessels from a bundled ships.json file."""

    def __init__(self, path):
        self.path = path

    def fetch(self) -> list[Vessel]:
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Vessel(**s) for s in raw]


# --- helpers ----------------------------------------------------------------

def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _bearing(lat1, lon1, lat2, lon2) -> float:
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _max_gap_minutes(positions) -> float:
    times = sorted(_parse(p.time) for p in positions)
    gap = 0.0
    for a, b in zip(times, times[1:]):
        gap = max(gap, (b - a).total_seconds() / 60.0)
    return gap


# --- stage ------------------------------------------------------------------

def name_vessels(rewind: RewindOutput, ship_source: ShipSource
                 ) -> tuple[NameOutput, list[Vessel]]:
    vessels = ship_source.fetch()

    poly = Polygon([(p.lon, p.lat) for p in rewind.release_polygon])
    release_dt = _parse(rewind.release_time)
    window_s = config.MATCH_TIME_WINDOW_HOURS * 3600.0

    candidates: list[CandidateMatch] = []
    for v in vessels:
        positions = v.positions
        max_gap = _max_gap_minutes(positions)
        has_gap = max_gap > config.AIS_GAP_FLAG_MINUTES

        # Find the in-polygon position closest in time to the release moment.
        best = None
        best_dt_delta = None
        for i, p in enumerate(positions):
            pt_time = _parse(p.time)
            if abs((pt_time - release_dt).total_seconds()) > window_s:
                continue
            if not poly.contains(Point(p.lon, p.lat)):
                continue
            delta = abs((pt_time - release_dt).total_seconds())
            if best_dt_delta is None or delta < best_dt_delta:
                best_dt_delta = delta
                best = (i, p)

        if best is None:
            continue

        i, p = best
        # Heading from this position to the next report (fall back to previous).
        if i + 1 < len(positions):
            nxt = positions[i + 1]
            heading = _bearing(p.lat, p.lon, nxt.lat, nxt.lon)
        elif i > 0:
            prv = positions[i - 1]
            heading = _bearing(prv.lat, prv.lon, p.lat, p.lon)
        else:
            heading = 0.0

        # Depth inside the polygon: distance from the matched point to the ring.
        # (The point is always inside here, so this is the inward depth.)
        pt = Point(p.lon, p.lat)
        edge_deg = poly.exterior.distance(pt)
        dist_inside_km = edge_deg * config.KM_PER_DEG_LAT * math.cos(math.radians(p.lat))

        candidates.append(CandidateMatch(
            mmsi=v.mmsi,
            name=v.name,
            matched_position=p,
            heading_deg=round(heading, 1),
            dist_inside_km=round(dist_inside_km, 2),
            time_delta_min=round(best_dt_delta / 60.0, 1),
            has_ais_gap=has_gap,
            max_gap_min=round(max_gap, 1),
        ))

    return NameOutput(candidates=candidates), vessels
