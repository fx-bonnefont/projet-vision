import os
import cv2
import numpy as np
import glob

ROOT = "/home/ericl/projet-vision/data"

TILES_DIR = os.path.join(ROOT, "DOTA_PLANES_TILED")
LABELS_DIR = os.path.join(ROOT, "DOTA/labels")

splits = ["train", "test"]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_polygons(label_path):
    polys = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            cls = parts[8].lower()
            if cls != "plane":
                continue

            coords = list(map(float, parts[:8]))
            poly = np.array(coords, dtype=np.float32).reshape(-1, 2)
            polys.append(poly)
    return polys

def polygon_center(poly):
    cx = np.mean(poly[:, 0])
    cy = np.mean(poly[:, 1])
    return cx, cy

for split in splits:
    print(f"\n=== Processing {split} ===")

    img_dir = os.path.join(TILES_DIR, split, "image")
    out_dir = os.path.join(TILES_DIR, split, "mask")
    lbl_dir = os.path.join(LABELS_DIR, split)

    ensure_dir(out_dir)

    tile_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    for tile_path in tile_paths:
        fname = os.path.basename(tile_path)
        base = os.path.splitext(fname)[0]

        # Exemple : P1414_obj_20 → image originale = P1414
        original_id = base.split("_")[0]

        label_path = os.path.join(lbl_dir, original_id + ".txt")
        if not os.path.exists(label_path):
            continue

        # Charger la tuile
        tile = cv2.imread(tile_path)
        H, W = tile.shape[:2]  # normalement 512x512

        # Extraire offset de la tuile
        # Format attendu : Pxxxx_obj_20 → obj_20 → 20
        parts = base.split("_")
        if len(parts) < 3:
            continue

        tile_type = parts[1]  # obj ou bg
        tile_index = int(parts[2])  # numéro de tuile

        # Chaque tuile fait 512x512
        TILE_SIZE = 512

        # Calcul de l’offset global dans l’image originale
        # Hypothèse : tuilage en grille row-major
        # Exemple : obj_20 → row = 20 // ncols, col = 20 % ncols
        # On doit connaître la taille originale
        # On la lit depuis l’image originale
        original_img_path = os.path.join(ROOT, "DOTA", split, "image", original_id + ".png")
        original_img = cv2.imread(original_img_path)
        if original_img is None:
            continue

        H0, W0 = original_img.shape[:2]
        ncols = W0 // TILE_SIZE

        row = tile_index // ncols
        col = tile_index % ncols

        x_offset = col * TILE_SIZE
        y_offset = row * TILE_SIZE

        # Charger les polygones de l’image originale
        polys = load_polygons(label_path)

        # Masque vide
        mask = np.zeros((H, W), dtype=np.uint8)

        for poly in polys:
            cx, cy = polygon_center(poly)

            # Convertir coordonnées globales → locales dans la tuile
            cx_local = int(cx - x_offset)
            cy_local = int(cy - y_offset)

            # Vérifier si le centre tombe dans la tuile
            if 0 <= cx_local < W and 0 <= cy_local < H:
                mask[cy_local, cx_local] = 255

        # Sauvegarde
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, mask)

    print(f"Done {split}")