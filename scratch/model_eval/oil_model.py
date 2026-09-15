"""
Reusable, verified oil-spill model + preprocessing (importable form of the
Phase-1 reference implementation in reference_infer.py).

This exposes the SAME preprocessing that was verified in the isolated
evaluation (ImagePadder anchor padding -> /255 -> -mean(0.5185) -> /std(0.197)
-> NCHW -> softmax) so the OILTRACE ModelSegmenter can call it directly instead
of duplicating the logic. Nothing here imports backend/app.

Model: ResNet50 DeepLabV3+ from github.com/AbhishekRS4/HTSM_Oil_Spill_Segmentation
(MIT). See THIRD_PARTY_LICENSES.md at the repo root.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(HERE, "source", "src", "training")
SAMPLE_DIR = os.path.join(HERE, "source", "src", "sample_padding_image_for_inference")
STATS_JSON = os.path.join(TRAINING_DIR, "image_stats.json")
DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "oil_spill_seg_resnet_50_deeplab_v3+_80.pt")

# Authors' class definitions and colour map (source/src/app.py).
CLASS_NAMES = ["sea_surface", "oil_spill", "oil_spill_look_alike", "ship", "land"]
NUM_CLASSES = 5
# The model's native input size for the dataset images (650x1250), padded to
# 672x1280 by the anchor padder, then cropped back after inference.
NATIVE_H, NATIVE_W = 650, 1250


class OilSpillModel:
    """Loads the checkpoint ONCE and exposes class_probs(). Raises on any load
    failure so the caller can decide on a fallback (it never silently degrades)."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS):
        # Imports are done here (not at module top) so importing this module is
        # cheap and only pulls torch when a model is actually constructed.
        import torch
        import torch.nn.functional as F

        if TRAINING_DIR not in sys.path:
            sys.path.insert(0, TRAINING_DIR)
        from seg_models import ResNet50DeepLabV3Plus
        from image_preprocessing import ImagePadder
        from logger_utils import load_dict_from_json

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"model weights not found: {weights_path}")

        self._torch = torch
        self._F = F
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pretrained=False: the ImageNet encoder weights are fully overwritten by
        # the checkpoint, so this only removes a network download (see the
        # deviation log in results/evaluation_record.json).
        model = ResNet50DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=False)
        try:
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
        except Exception:
            state = torch.load(weights_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()
        self.model = model

        self._padder = ImagePadder(SAMPLE_DIR)
        self._stats = load_dict_from_json(STATS_JSON)

    def class_probs(self, image_uint8: np.ndarray) -> np.ndarray:
        """image_uint8: HxWx3 uint8 at the native (650x1250) size.
        Returns softmax class probabilities, shape (NUM_CLASSES, 650, 1250)."""
        torch, F = self._torch, self._F
        if image_uint8.shape[:2] != (NATIVE_H, NATIVE_W):
            raise ValueError(
                f"expected {NATIVE_H}x{NATIVE_W} input, got {image_uint8.shape[:2]}")

        # Verified preprocessing (verbatim from the reference app.py).
        padded = self._padder.pad_image(image_uint8)          # 672x1280x3
        x = padded / 255.0
        x = x - self._stats["mean"]
        x = x / self._stats["std"]
        x = np.expand_dims(x, axis=0)
        x = np.transpose(x, (0, 3, 1, 2))                     # NCHW
        t = torch.tensor(x).float().to(self.device)

        with torch.no_grad():
            logits = self.model(t)
            probs = F.softmax(logits, dim=1)                 # (1,5,672,1280)
        probs = probs.squeeze(0).cpu().numpy()               # (5,672,1280)

        # Crop away the padding to the native region (as the reference does).
        ph, pw = probs.shape[1], probs.shape[2]
        probs = probs[:, 11:ph - 11, 15:pw - 15]             # (5,650,1250)
        return probs
