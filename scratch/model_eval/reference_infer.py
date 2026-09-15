"""
Phase 1 — Author reference inference baseline (ISOLATED).

Faithfully reproduces the authors' inference procedure from
source/src/app.py::run_inference, reusing the repository's OWN preprocessing
and model modules (no reimplementation). Runs on the author-provided sample
image source/src/sample_padding_image_for_inference/img_0814.jpg.

Nothing here imports or touches backend/app/.

Documented deviations from the reference (see EXACT DEVIATION LOG in report):
  1. ResNet50DeepLabV3Plus(pretrained=False): the reference passes
     pretrained=True, which downloads ImageNet ResNet-50 weights for the
     encoder. Those weights are fully overwritten by load_state_dict of the
     trained checkpoint, so this is behaviourally identical for inference and
     removes a network dependency (offline requirement).
  2. torch.load(..., weights_only=...): torch 2.14 defaults weights_only=True;
     we try that first (safe) and fall back to False if the checkpoint needs it.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(HERE, "source", "src", "training")
SAMPLE_DIR = os.path.join(HERE, "source", "src", "sample_padding_image_for_inference")
SAMPLE_IMG = os.path.join(SAMPLE_DIR, "img_0814.jpg")
STATS_JSON = os.path.join(TRAINING_DIR, "image_stats.json")
WEIGHTS = os.path.join(HERE, "weights", "oil_spill_seg_resnet_50_deeplab_v3+_80.pt")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# The reference modules use bare imports (from decoder_models import ...), so
# the training dir must be importable.
sys.path.insert(0, TRAINING_DIR)
from seg_models import ResNet50DeepLabV3Plus          # noqa: E402
from image_preprocessing import ImagePadder           # noqa: E402
from logger_utils import load_dict_from_json          # noqa: E402

# Authors' class colour map (source/src/app.py).
LABEL_TO_COLOR = {
    0: np.array([0, 0, 0]),        # sea_surface
    1: np.array([0, 255, 255]),    # oil_spill
    2: np.array([255, 0, 0]),      # oil_spill_look_alike
    3: np.array([153, 76, 0]),     # ship
    4: np.array([0, 153, 0]),      # land
}
CLASS_NAMES = ["sea_surface", "oil_spill", "oil_spill_look_alike", "ship", "land"]
NUM_CLASSES = 5


def main():
    device = torch.device("cpu")

    # --- model ---
    model = ResNet50DeepLabV3Plus(num_classes=NUM_CLASSES, pretrained=False)
    try:
        state = torch.load(WEIGHTS, map_location=device, weights_only=True)
        loaded_with = "weights_only=True"
    except Exception:
        state = torch.load(WEIGHTS, map_location=device, weights_only=False)
        loaded_with = "weights_only=False"
    missing, unexpected = model.load_state_dict(state, strict=True), None
    model.to(device)
    model.eval()

    # --- input (author sample) ---
    image_array = np.array(Image.open(SAMPLE_IMG))     # 650x1250x3 uint8, as in app.py
    dict_stats = load_dict_from_json(STATS_JSON)

    # --- authors' padding + preprocessing (verbatim from app.py) ---
    image_padder = ImagePadder(SAMPLE_DIR)
    image_padded = image_padder.pad_image(image_array)
    image_preprocessed = image_padded / 255.0
    image_preprocessed = image_preprocessed - dict_stats["mean"]
    image_preprocessed = image_preprocessed / dict_stats["std"]
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    image_preprocessed = np.transpose(image_preprocessed, (0, 3, 1, 2))  # NCHW
    image_tensor = torch.tensor(image_preprocessed).float().to(device)

    # --- inference (CPU); time the forward+softmax+argmax, excluding a warmup ---
    with torch.no_grad():
        _ = model(image_tensor)            # warmup (excluded from timing)
        t0 = time.perf_counter()
        pred_logits = model(image_tensor)
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
        infer_s = time.perf_counter() - t0

    output_shape = tuple(pred_logits.shape)            # (1,5,672,1280)
    pred_label_arr = np.squeeze(pred_label.cpu().numpy())  # (672,1280)

    # --- colourise then crop out the padding (verbatim from app.py) ---
    one_hot = np.eye(NUM_CLASSES)[pred_label_arr]
    mask = np.zeros((pred_label_arr.shape[0], pred_label_arr.shape[1], 3))
    for c in range(NUM_CLASSES):
        mask += one_hot[:, :, c].reshape(*pred_label_arr.shape, 1) * LABEL_TO_COLOR[c].reshape(1, 3)
    mask = mask.astype(np.uint8)
    ph, pw = pred_label_arr.shape
    mask_cropped = mask[11:ph - 11, 15:pw - 15]        # (650,1250,3)
    label_cropped = pred_label_arr[11:ph - 11, 15:pw - 15]  # (650,1250)

    Image.fromarray(mask_cropped, "RGB").save(os.path.join(RESULTS, "reference_mask.png"))

    # --- pixel counts on the final (cropped) mask ---
    total = label_cropped.size
    counts = {CLASS_NAMES[c]: int((label_cropped == c).sum()) for c in range(NUM_CLASSES)}

    report = {
        "weights_loaded_with": loaded_with,
        "input_image": os.path.relpath(SAMPLE_IMG, HERE),
        "input_dimensions": list(image_array.shape),
        "model_input_dimensions": list(image_tensor.shape),
        "output_shape": list(output_shape),
        "final_mask_dimensions": list(mask_cropped.shape),
        "inference_time_s": round(infer_s, 4),
        "timing_note": "forward+softmax+argmax on CPU, one warmup pass excluded",
        "total_pixels": int(total),
        "pixel_counts": counts,
        "pixel_percentages": {k: round(100.0 * v / total, 4) for k, v in counts.items()},
        "counts_sum_ok": sum(counts.values()) == total,
        "stats_used": dict_stats,
        "torch": torch.__version__,
    }
    with open(os.path.join(RESULTS, "reference_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
