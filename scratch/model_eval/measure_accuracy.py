import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from skimage.io import imread
import torchvision.transforms as transforms

sys.path.insert(0, "repo/src/training")
from seg_models import ResNet50DeepLabV3Plus
from image_preprocessing import ImagePadder

CLASSES = ["sea_surface","oil_spill","oil_spill_look_alike","ship","land"]
OIL = 1
WEIGHTS = sys.argv[1]
MEAN, STD = 0.5185, 0.197
PAD_T, PAD_B, PAD_L, PAD_R = 11, 11, 15, 15

dev = torch.device("cpu")
model = ResNet50DeepLabV3Plus(num_classes=5, pretrained=False)   # checkpoint overwrites encoder
try:    state = torch.load(WEIGHTS, map_location=dev, weights_only=True)
except Exception: state = torch.load(WEIGHTS, map_location=dev, weights_only=False)
if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
model.load_state_dict(state, strict=True); model.to(dev).eval()
print(f"loaded: {os.path.basename(WEIGHTS)}  (strict=True)", flush=True)

padder = ImagePadder("data/train/images")
tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
       transforms.Normalize(mean=[MEAN]*3, std=[STD]*3)])

imgs = sorted(os.listdir("data/test/images"))
Cp = np.zeros((5,5), dtype=np.int64)   # padded  (author-matching)
Cc = np.zeros((5,5), dtype=np.int64)   # cropped (honest)
pi_p, pi_c, acc_p = [], [], []
t0 = time.time()

for i, f in enumerate(imgs):
    img = imread(f"data/test/images/{f}")
    lab = imread(f"data/test/labels_1D/{f.replace('.jpg','.png')}")
    x = tf(padder.pad_image(img.copy())).unsqueeze(0).to(dev, dtype=torch.float)
    with torch.no_grad():
        pred = torch.argmax(F.softmax(model(x), dim=1), dim=1)[0].cpu().numpy().astype(np.int64)
    tp = padder.pad_label(lab).astype(np.int64)
    tc = lab.astype(np.int64)
    pc = pred[PAD_T:pred.shape[0]-PAD_B, PAD_L:pred.shape[1]-PAD_R]
    np.add.at(Cp, (tp.ravel(), pred.ravel()), 1)
    np.add.at(Cc, (tc.ravel(), pc.ravel()), 1)
    acc_p.append(float((tp==pred).mean()))
    for store, t, p in ((pi_p, tp, pred), (pi_c, tc, pc)):
        ious = np.full(5, np.nan)
        for c in range(5):
            ti, pri = (t==c), (p==c)
            if ti.sum()==0: continue
            inter = np.logical_and(ti,pri).sum()
            ious[c] = inter/(ti.sum()+pri.sum()-inter)
        store.append(ious)
    if (i+1) % 20 == 0: print(f"  {i+1}/{len(imgs)}  {time.time()-t0:.0f}s", flush=True)

def summarise(C, per_img, tag):
    per_img = np.array(per_img)
    inter = np.diag(C).astype(float)
    union = C.sum(1) + C.sum(0) - np.diag(C)
    g = inter/union
    prec = inter/np.maximum(C.sum(0),1); rec = inter/np.maximum(C.sum(1),1)
    return {
      "per_image_mean_iou": {CLASSES[c]: round(float(np.nanmean(per_img[:,c])),5) for c in range(5)},
      "global_iou":         {CLASSES[c]: round(float(g[c]),5) for c in range(5)},
      "mean_iou_per_image_method": round(float(np.nanmean(np.nanmean(per_img,axis=0))),5),
      "mean_iou_global":    round(float(np.nanmean(g)),5),
      "oil": {"global_iou": round(float(g[OIL]),5),
              "per_image_mean_iou": round(float(np.nanmean(per_img[:,OIL])),5),
              "precision": round(float(prec[OIL]),5),
              "recall": round(float(rec[OIL]),5),
              "images_containing_oil": int(np.sum(~np.isnan(per_img[:,OIL])))},
      "overall_pixel_accuracy": round(float(np.diag(C).sum()/C.sum()),5),
      "class_balance_pct": {CLASSES[c]: round(100*float(C.sum(1)[c]/C.sum()),4) for c in range(5)},
      "baseline_all_sea_pixel_accuracy": round(float(C.sum(1)[0]/C.sum()),5),
    }

res = {"weights": os.path.basename(WEIGHTS), "n_test_images": len(imgs),
       "runtime_s": round(time.time()-t0,1),
       "authors_published_oil_iou": 0.61549, "authors_published_mean_iou": 0.64868,
       "padded_author_matching": summarise(Cp, pi_p, "padded"),
       "cropped_650x1250": summarise(Cc, pi_c, "cropped")}
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/iou_measurement.json","w"), indent=2)
print(json.dumps(res, indent=2))
