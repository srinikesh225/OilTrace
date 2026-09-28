"""
Segmenter backends for stage SEE.

    Segmenter interface  (segment(scene_id, tif_path) -> (SeeOutput, [Rejection]))
       |
       +-- ThresholdSegmenter  (the original fixed-threshold detector, see.py)
       +-- ModelSegmenter      (ResNet50 DeepLabV3+, model_segmenter.py)

make_segmenter() picks the backend from config.SEGMENTER_BACKEND. The threshold
detector is the default and is used verbatim (its class lives in see.py and is
not modified). If the model backend is requested but the checkpoint is missing
or fails to load, we fall back to the threshold detector with a clear warning —
the app never crashes just because model weights are absent.
"""

from __future__ import annotations

import logging
import os

from ... import config
# The original threshold detector IS the ThresholdSegmenter; alias it so both
# backends are named consistently without touching see.py.
from ..see import Segmenter as ThresholdSegmenter

logger = logging.getLogger("oiltrace.segmenter")


def _threshold_with_provenance(requested, fallback, reason):
    """ThresholdSegmenter tagged with provenance so the pipeline can report which
    backend actually ran (and whether a model->threshold fallback happened)."""
    seg = ThresholdSegmenter()
    seg.provenance = {
        "segmenter_backend": "threshold",
        "segmenter_requested": requested,
        "segmenter_fallback": bool(fallback),
        "segmenter_fallback_reason": reason,
        "model_weights_id": None,
        "oil_prob_threshold": None,
    }
    return seg


def make_segmenter():
    """Return the configured segmenter. Falls back to ThresholdSegmenter only on
    model construction/availability failure (not on inference errors). The
    returned instance carries a `.provenance` dict describing what actually ran."""
    backend = (config.SEGMENTER_BACKEND or "threshold").strip().lower()

    if backend == "model":
        try:
            from .model_segmenter import ModelSegmenter
            seg = ModelSegmenter()
            seg.provenance = {
                "segmenter_backend": "model",
                "segmenter_requested": "model",
                "segmenter_fallback": False,
                "segmenter_fallback_reason": None,
                "model_weights_id": os.path.basename(config.MODEL_WEIGHTS_PATH),
                "oil_prob_threshold": config.OIL_PROB_THRESHOLD,
            }
            logger.info("Segmenter backend: model (%s)", config.MODEL_WEIGHTS_PATH)
            return seg
        except Exception as e:
            # Narrow, intentional fallback: initialization/availability only.
            logger.warning(
                "ModelSegmenter unavailable; falling back to ThresholdSegmenter "
                "because %s: %s", type(e).__name__, e)
            return _threshold_with_provenance("model", True, f"{type(e).__name__}: {e}")

    if backend != "threshold":
        logger.warning("Unknown SEGMENTER_BACKEND=%r; using threshold.", backend)
        return _threshold_with_provenance(backend, False,
                                          f"unknown SEGMENTER_BACKEND={backend!r}; used threshold")
    return _threshold_with_provenance("threshold", False, None)


__all__ = ["ThresholdSegmenter", "make_segmenter"]
