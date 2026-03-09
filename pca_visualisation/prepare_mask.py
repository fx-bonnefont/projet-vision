import os
import cv2
import numpy as np
import glob

ROOT = "/home/ericl/projet-vision/data/DOTA"
ROOT2 = "/home/ericl/projet-vision/data/DOTA_PLANES_TILED"

IMG_DIR = os.path.join(ROOT)
LBL_DIR = os.path.join(ROOT, "labels")
OUT_DIR = os.path.join(ROOT, "masks_planes_center")

splits = ["train", "test"]

os.makedirs(OUT_DIR, exist_ok=True)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_polygons(label_path):
    polys = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
            cls = parts[8]  # classe DOTA
            if cls.lower() != "plane":
                continue  # ignorer tout sauf les avions

            coords = list(map(float, parts[:8]))
            poly = np.array(coords, dtype=np.int32).reshape(-1, 2)
            polys.append(poly)
    return polys

def polygon_center(poly):
    # centre géométrique simple (moyenne des points)
    cx = int(np.mean(poly[:, 0]))
    cy = int(np.mean(poly[:, 1]))
    
    return cx, cy

for split in splits:
    print(f"Processing split: {split}")

    img_dir = os.path.join(IMG_DIR, split, "image")
    lbl_dir = os.path.join(LBL_DIR, split)
    out_dir = os.path.join(OUT_DIR, split)

    ensure_dir(out_dir)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    for img_path in img_paths:
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]

        label_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(label_path):
            print(f"Warning: no label for {fname}")
            continue

        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        mask = np.zeros((H, W), dtype=np.uint8)

        polys = load_polygons(label_path)

        for poly in polys:
            cx, cy = polygon_center(poly)
            if 0 <= cx < W and 0 <= cy < H:
                mask[cy, cx] = 255
            #print("Centres:", cx, cy)
            #cv2.fillPoly(mask, [poly], 255)

        out_path = os.path.join(out_dir, base + ".png")
        cv2.imwrite(out_path, mask)

    print(f"Done: {split}")