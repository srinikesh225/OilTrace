"""
Stage 1 — SEE: find candidate slick polygons in the radar scene.

# PROTOTYPE: this is a fixed threshold + contours, the real version is a U-Net.

The detector is wrapped in a Segmenter class exposing segment(). Swapping in a
learned model later means replacing this one class; nothing downstream changes
because the output contract (SeeOutput) stays the same.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import rasterio

from .. import config
from ..models import GeoPoint, Rejection, SeeOutput, SlickPolygon


def _bearing(p1, p2) -> float:
    """Bearing in degrees (0=N, 90=E) from geographic point p1 to p2."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


class Segmenter:
    """Finds dark elongated regions in a single-band radar GeoTIFF."""

    def __init__(self, threshold: int = config.DARK_THRESHOLD,
                 min_area_px: float = config.MIN_SLICK_AREA_PX):
        self.threshold = threshold
        self.min_area_px = min_area_px

    def segment(self, scene_id: str, tif_path) -> tuple[SeeOutput, list[Rejection]]:
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            transform = src.transform

        def px_to_latlon(col: float, row: float) -> GeoPoint:
            # rasterio.xy returns (x=lon, y=lat) for the pixel centre.
            lon, lat = rasterio.transform.xy(transform, row, col)
            return GeoPoint(lat=lat, lon=lon)

        # Rough metres-per-pixel so we can report an area in km^2.
        # (One degree lat ~ KM_PER_DEG_LAT; use scene centre for the lon scale.)
        h, w = band.shape
        deg_per_px_y = abs(transform.e)
        deg_per_px_x = abs(transform.a)
        centre_lat = transform.f + transform.e * (h / 2.0)
        km_per_px_y = deg_per_px_y * config.KM_PER_DEG_LAT
        km_per_px_x = deg_per_px_x * config.KM_PER_DEG_LAT * math.cos(math.radians(centre_lat))
        km2_per_px = km_per_px_x * km_per_px_y

        # Dark pixels: oil is darker than the bright sea return.
        mask = (band < self.threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons: list[SlickPolygon] = []
        rejections: list[Rejection] = []
        idx = 0
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area_px:
                # Small compact contours are treated as noise / look-alikes.
                # Never silently dropped: we carry the reason forward.
                idx += 1
                rejections.append(Rejection(
                    polygon_id=f"poly-{idx}",
                    stage="see",
                    reason=(f"Contour area {area:.0f}px below minimum "
                            f"{self.min_area_px:.0f}px — treated as noise / look-alike."),
                ))
                continue

            idx += 1
            poly_id = f"poly-{idx}"

            # Boundary ring in lat/lon.
            boundary = [px_to_latlon(float(pt[0][0]), float(pt[0][1])) for pt in c]

            # Centroid from image moments.
            m = cv2.moments(c)
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            centroid = px_to_latlon(cx, cy)

            # Long-axis bearing from the minimum-area rectangle. minAreaRect
            # gives ((cx,cy),(wid,hei),angle). Take the longer side's direction.
            (bx, by), (bw, bh), bangle = cv2.minAreaRect(c)
            # Endpoints of the long axis in pixel space.
            if bw >= bh:
                length, ang = bw, bangle
            else:
                length, ang = bh, bangle + 90.0
            ar = math.radians(ang)
            half = length / 2.0
            p_a = px_to_latlon(bx + half * math.cos(ar), by + half * math.sin(ar))
            p_b = px_to_latlon(bx - half * math.cos(ar), by - half * math.sin(ar))
            bearing = _bearing((p_a.lat, p_a.lon), (p_b.lat, p_b.lon))
            # Normalise to 0-180: a line has no direction, only an axis.
            bearing = bearing % 180.0

            polygons.append(SlickPolygon(
                polygon_id=poly_id,
                boundary=boundary,
                area_px=area,
                area_km2=round(area * km2_per_px, 3),
                centroid=centroid,
                bearing_deg=round(bearing, 1),
            ))

        return SeeOutput(scene_id=scene_id, polygons=polygons), rejections
