import os, numpy as np
from PIL import Image

# Exact mapping from the author's inference.py dict_label_to_color_mapping
COLOR2IDX = {(0,0,0):0, (0,255,255):1, (255,0,0):2, (153,76,0):3, (0,153,0):4}

src, dst = "data/test/labels", "data/test/labels_1D"
os.makedirs(dst, exist_ok=True)
stray_total = 0; px_total = 0
for f in sorted(os.listdir(src)):
    a = np.array(Image.open(f"{src}/{f}").convert("RGB")).astype(np.int32)
    out = np.full(a.shape[:2], 255, dtype=np.uint8)
    for rgb, idx in COLOR2IDX.items():
        m = (a[:,:,0]==rgb[0]) & (a[:,:,1]==rgb[1]) & (a[:,:,2]==rgb[2])
        out[m] = idx
    stray = int((out==255).sum())
    if stray:
        # nearest-colour snap for the handful of compression-artefact pixels
        ys, xs = np.where(out==255)
        cols = np.array(list(COLOR2IDX.keys()))
        idxs = np.array(list(COLOR2IDX.values()))
        d = np.linalg.norm(a[ys,xs][:,None,:] - cols[None,:,:], axis=2)
        out[ys,xs] = idxs[d.argmin(axis=1)]
    stray_total += stray; px_total += out.size
    Image.fromarray(out).save(f"{dst}/{f}")
print(f"converted {len(os.listdir(dst))} masks")
print(f"stray pixels snapped to nearest class: {stray_total:,} of {px_total:,} ({100*stray_total/px_total:.6f}%)")
