import os
import cv2
import glob

ROOT = "data/DOTA"
OUT = os.path.join(ROOT, "DOTA_PLANES_TILED")
TILE = 512
tiles_info = {}

splits = ["train", "test"]
global_id = 1

for split in splits:
    img_dir = os.path.join(ROOT, "images", split)
    out_img_dir = os.path.join(OUT, split, "image")
    os.makedirs(out_img_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print("Cannot read", img_path)
            continue

        H, W = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]

        tile_id = 0
        for y in range(0, H, TILE):
            for x in range(0, W, TILE):
                tile = img[y:y+TILE, x:x+TILE]

                if tile.shape[0] < TILE or tile.shape[1] < TILE:
                    continue

                out_name = f"{base}_x{x}_y{y}.png"
                cv2.imwrite(os.path.join(out_img_dir, out_name), tile)
                tile_id += 1