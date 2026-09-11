"""
Stage 3 — REWIND: backward drift to the release area.

Scatter N_PARTICLES inside the accepted slick polygon, then walk each particle
backwards in hourly steps for SLICK_AGE_HOURS. Each step moves the particle
upwind (opposite the slick's forward drift) at WIND_DRIFT_FACTOR times the wind
speed, plus a small per-particle random jitter so the cloud spreads the way a
real ensemble would.

# PROTOTYPE: no ocean currents, no Stokes drift, no diffusion tensor yet —
# OpenDrift replaces this whole stage later.

The release area is the convex hull of the final particle positions.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from .. import config
from ..models import GeoPoint, RewindOutput, SlickPolygon, WindSample


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _step_latlon(lat, lon, bearing_deg, dist_m):
    b = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(b)) / config.METERS_PER_DEG_LAT
    dlon = (dist_m * math.sin(b)) / (config.METERS_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def _scatter_in_polygon(poly: Polygon, n: int, rng) -> list[tuple[float, float]]:
    """Rejection-sample n points (lon, lat) uniformly inside a polygon."""
    minx, miny, maxx, maxy = poly.bounds
    pts = []
    guard = 0
    while len(pts) < n and guard < n * 200:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
        guard += 1
    # If the polygon is a sliver, fall back to the centroid so we never hang.
    while len(pts) < n:
        c = poly.centroid
        pts.append((c.x, c.y))
    return pts


def rewind(slick: SlickPolygon, scene_time: str, wind_samples: list[dict],
           wind_at_scene: WindSample) -> RewindOutput:
    rng = np.random.default_rng(config.RANDOM_SEED)

    # Build the slick polygon in (lon, lat) for shapely.
    ring = [(p.lon, p.lat) for p in slick.boundary]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)

    seeds = _scatter_in_polygon(poly, config.N_PARTICLES, rng)

    scene_dt = _parse(scene_time)
    # The SLICK_AGE_HOURS hourly wind samples just before the scene, newest first.
    def wind_for_hour(dt: datetime) -> WindSample:
        nearest = min(wind_samples,
                      key=lambda w: abs((_parse(w["time"]) - dt).total_seconds()))
        return WindSample(time=nearest["time"], speed_ms=nearest["speed_ms"],
                          dir_deg=nearest["dir_deg"])

    # Walk every particle backwards hour by hour, over the SLICK_AGE_HOURS
    # strictly before the scene (scene-1h .. scene-SLICK_AGE_HOURS).
    cloud = []  # (lon, lat)
    for (lon, lat) in seeds:
        cur_lat, cur_lon = lat, lon
        for h in range(1, config.SLICK_AGE_HOURS + 1):
            dt = scene_dt - timedelta(hours=h)
            w = wind_for_hour(dt)
            dist = config.WIND_DRIFT_FACTOR * w.speed_ms * 3600.0
            # Backward = move toward where the wind blows FROM (upwind).
            cur_lat, cur_lon = _step_latlon(cur_lat, cur_lon, w.dir_deg, dist)
            # Per-step random jitter to spread the ensemble (sigma from config).
            jitter_bearing = rng.uniform(0, 360)
            jitter_dist = abs(rng.normal(0, config.DRIFT_JITTER_SIGMA_M))
            cur_lat, cur_lon = _step_latlon(cur_lat, cur_lon, jitter_bearing, jitter_dist)
        cloud.append((cur_lon, cur_lat))

    hull = MultiPoint(cloud).convex_hull
    if hull.geom_type != "Polygon":
        # Degenerate hull (all points collinear); give it a little body.
        hull = hull.buffer(0.01)

    release_polygon = [GeoPoint(lat=y, lon=x) for x, y in hull.exterior.coords]
    particle_cloud = [GeoPoint(lat=y, lon=x) for x, y in cloud]
    release_time = _iso(scene_dt - timedelta(hours=config.SLICK_AGE_HOURS))

    return RewindOutput(
        polygon_id=slick.polygon_id,
        release_polygon=release_polygon,
        particle_cloud=particle_cloud,
        release_time=release_time,
        slick_bearing_deg=slick.bearing_deg,
    )
