"""Spatial-coherence analysis of the reference mask + overlay for visual check."""
import os
import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SAMPLE = os.path.join(HERE, "source", "src", "sample_padding_image_for_inference", "img_0814.jpg")

COLOR_TO_LABEL = {
    (0, 0, 0): 0, (0, 255, 255): 1, (255, 0, 0): 2, (153, 76, 0): 3, (0, 153, 0): 4,
}
NAMES = ["sea_surface", "oil_spill", "oil_spill_look_alike", "ship", "land"]

mask = np.array(Image.open(os.path.join(RESULTS, "reference_mask.png")).convert("RGB"))
label = np.zeros(mask.shape[:2], np.uint8)
for color, lid in COLOR_TO_LABEL.items():
    label[np.all(mask == np.array(color), axis=-1)] = lid

print("=== connected-component analysis (non-sea classes) ===")
for c in (1, 2, 3):
    m = (label == c).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    ncomp = n - 1
    if ncomp == 0:
        print(f"{NAMES[c]:20s}: 0 pixels"); continue
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    # bounding box of the largest component
    bi = 1 + int(np.argmax(areas))
    x, y, w, h = stats[bi, 0], stats[bi, 1], stats[bi, 2], stats[bi, 3]
    print(f"{NAMES[c]:20s}: {int(m.sum())}px in {ncomp} component(s); "
          f"largest={biggest}px ({100.0*biggest/max(1,int(m.sum())):.0f}% of class), "
          f"bbox=({x},{y},{w}x{h})")

# Overlay: grayscale input with oil (cyan) and ship (red) dilated for visibility.
gray = cv2.cvtColor(np.array(Image.open(SAMPLE).convert("RGB")), cv2.COLOR_RGB2GRAY)
ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
k = np.ones((5, 5), np.uint8)
oil = cv2.dilate((label == 1).astype(np.uint8), k)
look = cv2.dilate((label == 2).astype(np.uint8), k)
ship = cv2.dilate((label == 3).astype(np.uint8), k)
ov[oil > 0] = [0, 255, 255]
ov[look > 0] = [255, 0, 0]
ov[ship > 0] = [255, 128, 0]
Image.fromarray(ov).save(os.path.join(RESULTS, "reference_overlay.png"))
print("\nsaved results/reference_overlay.png (oil=cyan, look-alike=red, ship=orange, dilated 5x5)")
