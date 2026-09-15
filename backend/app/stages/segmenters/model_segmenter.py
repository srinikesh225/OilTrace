"""
Stage SEE — ModelSegmenter: learned oil-spill detector behind the Segmenter
interface (drop-in alternative to app/stages/see.py::Segmenter).

Runs a pretrained ResNet50 DeepLabV3+ (5 classes: sea_surface, oil_spill,
oil_spill_look_alike, ship, land) and turns the oil_spill class-probability map
into the SAME SlickPolygon geometry the threshold Segmenter produces, using the
same contour/area/bearing conventions (the bearing helper is reused directly).
oil_spill_look_alike (class 2) is returned separately via SeeOutput.

Model + weights + preprocessing come from the isolated evaluation under
scratch/model_eval/ (verified there). The preprocessing is reused, not
reimplemented.

THIRD-PARTY MODEL: github.com/AbhishekRS4/HTSM_Oil_Spill_Segmentation (MIT).
See THIRD_PARTY_LICENSES.md at the repository root for full attribution.
"""

from __future__ import annotations

import math
import sys

import cv2
import numpy as np
import rasterio

from ... import config
from ...models import GeoPoint, Rejection, SeeOutput, SlickPolygon
from ..see import _bearing  # reuse the exact bearing method (do not reinvent)

# The verified model+preprocessing lives in scratch/model_eval. Import it by
# path so no preprocessing logic is duplicated here.
_EVAL_DIR = config.BACKEND_DIR.parent / "scratch" / "model_eval"


class ModelSegmenter:
    """Learned segmenter with the same interface as see.Segmenter:
    segment(scene_id, tif_path) -> (SeeOutput, list[Rejection]).

    The checkpoint is loaded ONCE in __init__. Any construction/availability
    failure raises here so the caller (segmenters.make_segmenter) can fall back
    to the threshold segmenter with a clear warning. Failures are NOT caught
    around inference, so real bugs stay visible.
    """

    def __init__(self, weights_path: str = None,
                 oil_class_index: int = None,
                 lookalike_class_index: int = None,
                 oil_prob_threshold: float = None,
                 min_area_px: float = None):
        self.weights_path = weights_path or config.MODEL_WEIGHTS_PATH
        self.oil_class_index = config.OIL_CLASS_INDEX if oil_class_index is None else oil_class_index
        self.lookalike_class_index = (config.LOOKALIKE_CLASS_INDEX
                                      if lookalike_class_index is None else lookalike_class_index)
        self.oil_prob_threshold = (config.OIL_PROB_THRESHOLD
                                   if oil_prob_threshold is None else oil_prob_threshold)
        self.min_area_px = config.MIN_SLICK_AREA_PX if min_area_px is None else min_area_px

        # Load the verified model wrapper (this pulls torch and the checkpoint
        # exactly once). Raises on any failure.
        if str(_EVAL_DIR) not in sys.path:
            sys.path.insert(0, str(_EVAL_DIR))
        from oil_model import OilSpillModel  # noqa: E402
        self._model = OilSpillModel(self.weights_path)

    # --- geometry: reproduce see.Segmenter's contour -> SlickPolygon exactly ---
    def _mask_to_polygons(self, mask_u8, transform, id_prefix, km2_per_px):
        def px_to_latlon(col, row):
            lon, lat = rasterio.transform.xy(transform, row, col)
            return GeoPoint(lat=lat, lon=lon)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons, rejections = [], []
        idx = 0
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area_px:
                idx += 1
                rejections.append(Rejection(
                    polygon_id=f"{id_prefix}-{idx}",
                    stage="see",
                    reason=(f"Model region area {area:.0f}px below minimum "
                            f"{self.min_area_px:.0f}px — dropped."),
                ))
                continue
            idx += 1
            boundary = [px_to_latlon(float(p[0][0]), float(p[0][1])) for p in c]
            m = cv2.moments(c)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            centroid = px_to_latlon(cx, cy)
            (bx, by), (bw, bh), bangle = cv2.minAreaRect(c)
            if bw >= bh:
                length, ang = bw, bangle
            else:
                length, ang = bh, bangle + 90.0
            ar = math.radians(ang)
            half = length / 2.0
            p_a = px_to_latlon(bx + half * math.cos(ar), by + half * math.sin(ar))
            p_b = px_to_latlon(bx - half * math.cos(ar), by - half * math.sin(ar))
            bearing = _bearing((p_a.lat, p_a.lon), (p_b.lat, p_b.lon)) % 180.0
            polygons.append(SlickPolygon(
                polygon_id=f"{id_prefix}-{idx}",
                boundary=boundary,
                area_px=area,
                area_km2=round(area * km2_per_px, 3),
                centroid=centroid,
                bearing_deg=round(bearing, 1),
            ))
        return polygons, rejections

    def segment(self, scene_id: str, tif_path) -> tuple[SeeOutput, list[Rejection]]:
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            transform = src.transform
        h, w = band.shape

        # km^2 per pixel, exactly as see.Segmenter computes it.
        centre_lat = transform.f + transform.e * (h / 2.0)
        km_per_px_y = abs(transform.e) * config.KM_PER_DEG_LAT
        km_per_px_x = abs(transform.a) * config.KM_PER_DEG_LAT * math.cos(math.radians(centre_lat))
        km2_per_px = km_per_px_x * km_per_px_y

        # ADAPTATION (documented): the model expects a 650x1250 3-band 8-bit
        # dataset-format image. OILTRACE's scene is a single-band GeoTIFF of
        # arbitrary size (the bundled demo is 1000x1000). We resize to the
        # native model size and replicate the band to 3 channels, run the model,
        # then resize the class-probability maps back to the scene grid so the
        # geo-transform maps contours to correct lat/lon. For a native
        # 650x1250 SAR patch this resize is a no-op.
        from oil_model import NATIVE_H, NATIVE_W  # noqa: E402
        band_u8 = band if band.dtype == np.uint8 else np.clip(band, 0, 255).astype(np.uint8)
        resized = cv2.resize(band_u8, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([resized, resized, resized], axis=-1)

        probs = self._model.class_probs(rgb)  # (5,650,1250) softmax
        oil_prob = cv2.resize(probs[self.oil_class_index], (w, h), interpolation=cv2.INTER_LINEAR)
        look_prob = cv2.resize(probs[self.lookalike_class_index], (w, h), interpolation=cv2.INTER_LINEAR)

        oil_mask = (oil_prob >= self.oil_prob_threshold).astype(np.uint8) * 255
        look_mask = (look_prob >= self.oil_prob_threshold).astype(np.uint8) * 255

        oil_polys, rejections = self._mask_to_polygons(oil_mask, transform, "poly", km2_per_px)
        look_polys, _ = self._mask_to_polygons(look_mask, transform, "look", km2_per_px)

        out = SeeOutput(scene_id=scene_id, polygons=oil_polys, look_alike_polygons=look_polys)
        return out, rejections
