"""
Phase 2 — oil_spill IoU / precision / recall against a HELD-OUT LABELLED test
set.

This script does NOT fabricate metrics. It first establishes whether a valid
held-out labelled test set exists, and only computes metrics if it does. For
this model the labelled dataset is gated (not public), so it reports the
evaluation as blocked instead of inventing numbers (see task sections 20, 21,
25, 39).

If you obtain the labelled test set (input SAR images + 5-class ground-truth
masks that were NOT used to train the model), place it under
scratch/model_eval/data/testset/ as pairs and re-run: this script will then
compute aggregate pixel-level TP/FP/FN for the oil_spill class only.
"""

from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source")
TESTSET = os.path.join(HERE, "data", "testset")   # where a real labelled set would go
CLASS_NAMES = ["sea_surface", "oil_spill", "oil_spill_look_alike", "ship", "land"]


def find_labelled_testset():
    """Return (images, labels) pairs if a valid held-out labelled test set is
    present, else None. A valid pair is an input image + a ground-truth 5-class
    label mask. The repository's images/pred_mask_*.png are model PREDICTIONS,
    not ground truth, and must never be used as labels."""
    inputs = sorted(glob.glob(os.path.join(TESTSET, "images", "*")))
    labels = sorted(glob.glob(os.path.join(TESTSET, "labels", "*")))
    if inputs and labels and len(inputs) == len(labels):
        return list(zip(inputs, labels))
    return None


def audit_repository():
    """What the source repo actually ships, to justify the blocked verdict."""
    pred_masks = sorted(glob.glob(os.path.join(SRC, "images", "pred_mask_*.png")))
    sample_inputs = sorted(glob.glob(
        os.path.join(SRC, "src", "sample_padding_image_for_inference", "*")))
    return {
        "repo_images_dir": [os.path.basename(p) for p in pred_masks],
        "repo_images_dir_nature": "model PREDICTIONS (pred_mask_*), NOT ground-truth labels",
        "sample_inputs": [os.path.basename(p) for p in sample_inputs],
        "sample_inputs_have_labels": False,
        "dataset_public": False,
        "dataset_note": ("The dataset (https://m4d.iti.gr/oil-spill-detection-dataset/) "
                         "is gated/not public; the repo README states the original "
                         "images are not uploaded, only their predictions."),
    }


def main():
    pairs = find_labelled_testset()
    audit = audit_repository()

    if pairs is None:
        result = {
            "status": "NO VALID HELD-OUT LABELLED TEST SET AVAILABLE",
            "oil_iou": None, "oil_precision": None, "oil_recall": None,
            "overall_pixel_accuracy": None, "class_balance": None,
            "num_images": 0,
            "reason": (
                "No (input image, 5-class ground-truth mask) pairs are available. "
                "The labelled oil-spill dataset is gated and not bundled; the only "
                "images in the repo are the authors' predicted masks (not ground "
                "truth) plus one unlabelled sample input (img_0814.jpg). Using the "
                "authors' predicted masks as 'labels' would be pseudo-ground-truth "
                "derived from the model itself, which is explicitly disallowed."
            ),
            "leakage_note": (
                "Because no evaluation labels exist, no IoU/precision/recall is "
                "reported. OIL_PROB_THRESHOLD (0.5) was chosen as a neutral, untuned "
                "default and was NOT selected against any test set, so there is no "
                "train/test or threshold-selection leakage to report."
            ),
            "how_to_unblock": (
                "Place a held-out labelled test set under "
                "scratch/model_eval/data/testset/{images,labels}/ (labels as 5-class "
                "index masks, 0..4) and re-run. This script then computes aggregate "
                "pixel-level TP/FP/FN for the oil_spill class (index 1) only."
            ),
            "repository_audit": audit,
        }
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        with open(os.path.join(HERE, "results", "measurement_blocked.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    # --- If a valid labelled test set were present, metrics would be computed
    #     here as aggregate pixel-level TP/FP/FN for the oil_spill class. ---
    raise SystemExit("Labelled test set found — implement aggregate metric here "
                     "(intentionally not reached without real labels).")


if __name__ == "__main__":
    main()
