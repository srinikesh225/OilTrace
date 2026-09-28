#!/usr/bin/env python
"""Inspect the labelled SAR oil-spill dataset (report only; no evaluation)."""
from __future__ import annotations
import os, sys, json, collections
import numpy as np
from skimage.io import imread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8")
    except Exception: pass

D = r"C:\Users\srini\Downloads\archive\oil-spill"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
SPLITS = ["train", "test"]

# reference colour map (authors' app.py) -> class index
REF_RGB = {(0,0,0):0, (0,255,255):1, (255,0,0):2, (153,76,0):3, (0,153,0):4}
CLASS = ["sea_surface","oil_spill","oil_spill_look_alike","ship","land"]

report = {"dataset_path": D, "splits": {}}

# --- 1. matching ---
print("== 1. MATCHING ==")
for sp in SPLITS:
    imgs = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(D,sp,"images"))}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(D,sp,"labels"))}
    missing_mask = sorted(imgs - lbls); missing_img = sorted(lbls - imgs)
    print(f"  {sp}: {len(imgs)} images, {len(lbls)} labels; "
          f"images_without_mask={len(missing_mask)} labels_without_image={len(missing_img)} "
          f"matched={len(imgs & lbls)}")
    report["splits"][sp] = {"n_images":len(imgs),"n_labels":len(lbls),
        "matched":len(imgs&lbls),"images_without_mask":missing_mask[:10],
        "labels_without_image":missing_img[:10]}

# --- 2. image properties (sample) ---
print("\n== 2. IMAGE PROPERTIES ==")
img_dims=set(); img_channels=set(); img_dtypes=set(); img_min=255; img_max=0
n_sampled=0
for sp in SPLITS:
    idir=os.path.join(D,sp,"images")
    for f in sorted(os.listdir(idir))[:60]:
        a=imread(os.path.join(idir,f)); n_sampled+=1
        img_dims.add(a.shape[:2]); img_dtypes.add(str(a.dtype))
        img_channels.add(1 if a.ndim==2 else a.shape[2])
        img_min=min(img_min,int(a.min())); img_max=max(img_max,int(a.max()))
        if a.ndim==3 and a.shape[2]>=3:
            gray_equal = bool((a[...,0]==a[...,1]).all() and (a[...,1]==a[...,2]).all())
        else: gray_equal=True
print(f"  sampled {n_sampled} images")
print(f"  dims present: {sorted(img_dims)}")
print(f"  channels: {sorted(img_channels)}  dtype: {sorted(img_dtypes)}  value range: [{img_min},{img_max}]")
print(f"  RGB channels identical (grayscale-in-RGB) on last sample: {gray_equal}")
report["image_properties"]={"dims":sorted(str(d) for d in img_dims),
    "channels":sorted(img_channels),"dtype":sorted(img_dtypes),
    "value_range":[img_min,img_max],"rgb_grayscale":gray_equal,"n_sampled":n_sampled}

# --- 3. mask properties: unique values + pixel counts across ALL masks ---
print("\n== 3. MASK PROPERTIES (all masks) ==")
mask_dims=set(); mask_channels=set(); mask_dtypes=set()
scalar_counter=collections.Counter()   # if masks are single-channel index
rgb_counter=collections.Counter()      # if masks are RGB colour
n_masks=0
for sp in SPLITS:
    ldir=os.path.join(D,sp,"labels")
    for f in sorted(os.listdir(ldir)):
        m=imread(os.path.join(ldir,f)); n_masks+=1
        mask_dtypes.add(str(m.dtype)); mask_dims.add(m.shape[:2])
        if m.ndim==2:
            mask_channels.add(1)
            vals,cnts=np.unique(m,return_counts=True)
            for v,c in zip(vals,cnts): scalar_counter[int(v)]+=int(c)
        else:
            ch=m.shape[2]; mask_channels.add(ch)
            rgb=m[...,:3].reshape(-1,3)
            uniq,cnts=np.unique(rgb,axis=0,return_counts=True)
            for u,c in zip(uniq,cnts): rgb_counter[tuple(int(x) for x in u)]+=int(c)
print(f"  masks read: {n_masks}")
print(f"  dims present: {sorted(mask_dims)}  channels: {sorted(mask_channels)}  dtype: {sorted(mask_dtypes)}")
total_px=sum(scalar_counter.values())+sum(rgb_counter.values())
if scalar_counter:
    print("  SINGLE-CHANNEL index values -> pixel counts:")
    for v in sorted(scalar_counter):
        c=scalar_counter[v]; print(f"    value {v:>3}: {c:>12,} px  ({100*c/total_px:.3f}%)")
if rgb_counter:
    print("  RGB colour values -> pixel counts:")
    for rgb in sorted(rgb_counter, key=lambda k:-rgb_counter[k]):
        c=rgb_counter[rgb]; mapped=REF_RGB.get(rgb,"UNMAPPED")
        name=CLASS[mapped] if isinstance(mapped,int) else "??"
        print(f"    RGB {str(rgb):>16}: {c:>12,} px ({100*c/total_px:.3f}%) -> class {mapped} {name}")
report["mask_properties"]={"n_masks":n_masks,"dims":sorted(str(d) for d in mask_dims),
    "channels":sorted(mask_channels),"dtype":sorted(mask_dtypes),
    "scalar_value_counts":{str(k):v for k,v in sorted(scalar_counter.items())},
    "rgb_value_counts":{str(k):v for k,v in sorted(rgb_counter.items(),key=lambda x:-x[1])}}

# --- 5. sample pairs ---
print("\n== 5. SAMPLE PAIRS ==")
picks=[("train","img_0001"),("train","img_0500"),("test",None)]
saved=[]
for i,(sp,stem) in enumerate(picks,1):
    idir=os.path.join(D,sp,"images"); ldir=os.path.join(D,sp,"labels")
    if stem is None:
        stem=os.path.splitext(sorted(os.listdir(idir))[0])[0]
    ip=os.path.join(idir,stem+".jpg"); lp=os.path.join(ldir,stem+".png")
    if not (os.path.isfile(ip) and os.path.isfile(lp)):
        # fallback to first available
        stem=os.path.splitext(sorted(os.listdir(idir))[0])[0]
        ip=os.path.join(idir,stem+".jpg"); lp=os.path.join(ldir,stem+".png")
    img=imread(ip); mask=imread(lp)
    fig,ax=plt.subplots(1,2,figsize=(11,3.4))
    ax[0].imshow(img,cmap="gray" if img.ndim==2 else None); ax[0].set_title(f"{sp}/{stem} image",fontsize=9); ax[0].axis("off")
    ax[1].imshow(mask); ax[1].set_title(f"{sp}/{stem} mask",fontsize=9); ax[1].axis("off")
    p=os.path.join(OUT,f"dataset_sample_{i:02d}.png")
    plt.tight_layout(); plt.savefig(p,dpi=110); plt.close(); saved.append(p)
    print(f"  saved {p}  ({sp}/{stem})")
report["sample_pairs"]=saved

with open(os.path.join(HERE,"results","dataset_inspection.json"),"w",encoding="utf-8") as f:
    json.dump(report,f,indent=2)
print("\nWrote results/dataset_inspection.json")
