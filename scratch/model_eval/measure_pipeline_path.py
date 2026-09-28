"""Measure OILTRACE's own ModelSegmenter chain, not the reference argmax path.
Replicates backend/app/stages/segmenters/model_segmenter.py::segment exactly:
  single-band read -> resize to 650x1250 -> replicate to 3ch -> class_probs
  -> oil_prob >= OIL_PROB_THRESHOLD -> contour filter at MIN_SLICK_AREA_PX
"""
import os, sys, json, time
import numpy as np, cv2, torch, torch.nn.functional as F
from skimage.io import imread
sys.path.insert(0, "repo/src/training")
from seg_models import ResNet50DeepLabV3Plus
from image_preprocessing import ImagePadder

W = "/mnt/user-data/uploads/SIH/oiltrace/scratch/model_eval/weights/oil_spill_seg_resnet_50_deeplab_v3+_80.pt"
MEAN, STD = 0.5185, 0.197
OIL, OIL_PROB_THRESHOLD, MIN_SLICK_AREA_PX = 1, 0.5, 1500
NATIVE_H, NATIVE_W = 650, 1250

dev = torch.device("cpu")
model = ResNet50DeepLabV3Plus(num_classes=5, pretrained=False)
model.load_state_dict(torch.load(W, map_location=dev, weights_only=True), strict=True)
model.to(dev).eval()
padder = ImagePadder("data/train/images")

def class_probs(rgb_u8):
    p = padder.pad_image(rgb_u8.copy())
    x = ((p/255.0 - MEAN)/STD)[None].transpose(0,3,1,2)
    with torch.no_grad():
        pr = F.softmax(model(torch.tensor(x).float()), dim=1).squeeze(0).numpy()
    return pr[:, 11:pr.shape[1]-11, 15:pr.shape[2]-15]

def area_filter(mask_u8):
    cnts,_ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = [c for c in cnts if cv2.contourArea(c) >= MIN_SLICK_AREA_PX]
    out = np.zeros_like(mask_u8)
    if keep: cv2.drawContours(out, keep, -1, 255, thickness=cv2.FILLED)
    return out, len(cnts), len(keep)

stats = {k: np.zeros(3, dtype=np.int64) for k in ("argmax","thresh","pipeline")}  # inter, pred, true
chan_diff = 0; dropped_total = 0; kept_total = 0
imgs = sorted(os.listdir("data/test/images")); t0=time.time()
for i,f in enumerate(imgs):
    img = imread(f"data/test/images/{f}")
    if not np.array_equal(img[:,:,0], img[:,:,1]): chan_diff += 1
    band = img[:,:,0]                                    # single-band read
    resized = cv2.resize(band, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_LINEAR)
    rgb = np.stack([resized]*3, axis=-1)                 # replicate to 3ch
    probs = class_probs(rgb)
    gt = (imread(f"data/test/labels_1D/{f.replace('.jpg','.png')}") == OIL)

    m_arg  = (np.argmax(probs, axis=0) == OIL)
    m_thr  = (probs[OIL] >= OIL_PROB_THRESHOLD)
    m_pipe_u8, n_all, n_keep = area_filter((m_thr.astype(np.uint8))*255)
    dropped_total += n_all-n_keep; kept_total += n_keep
    m_pipe = m_pipe_u8 > 0

    for k, m in (("argmax",m_arg), ("thresh",m_thr), ("pipeline",m_pipe)):
        stats[k] += np.array([np.logical_and(m,gt).sum(), m.sum(), gt.sum()], dtype=np.int64)
    if (i+1)%25==0: print(f"  {i+1}/{len(imgs)} {time.time()-t0:.0f}s", flush=True)

res={"n_test_images":len(imgs),"runtime_s":round(time.time()-t0,1),
     "images_where_rgb_channels_differ":chan_diff,
     "contours_kept":kept_total,"contours_dropped_below_1500px":dropped_total,
     "config":{"OIL_PROB_THRESHOLD":OIL_PROB_THRESHOLD,"MIN_SLICK_AREA_PX":MIN_SLICK_AREA_PX}}
for k,(inter,pred,true) in stats.items():
    res[k]={"oil_iou_global":round(float(inter/(pred+true-inter)),5),
            "precision":round(float(inter/max(pred,1)),5),
            "recall":round(float(inter/max(true,1)),5)}
os.makedirs("results",exist_ok=True)
json.dump(res, open("results/pipeline_path_measurement.json","w"), indent=2)
print(json.dumps(res, indent=2))
