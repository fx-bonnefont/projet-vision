from pathlib import Path
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import re

# --- Définition des dossiers ---
base_dir = Path("attack_results")

dirs = [
    base_dir / "04_03-patch16_0_0_divulse-stern",
    base_dir / "04_03-patch16_0_496_divulse-stern",
    base_dir / "04_03-patch16_496_0_divulse-stern",
    base_dir / "04_03-patch16_496_496_divulse-stern",
]

patch_name = "patch.png"

# --- Extraction automatique des positions (0,0), (0,480), etc. ---
def extract_pos(path):
    # Cherche les deux nombres après "patch32_"
    m = re.search(r"patch16_(\d+)_(\d+)", str(path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

positions = [extract_pos(d) for d in dirs]

# --- Chargement des patchs ---
patches = [Image.open(d / patch_name).convert("RGB") for d in dirs]

# Harmonisation des tailles
w, h = patches[0].size
patches = [p.resize((w, h)) for p in patches]

# --- Fonction différence ---
def diff(a, b):
    return ImageChops.difference(a, b)

# --- Création de la grille 4×4 ---
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for row in range(4):
    ref = patches[row]
    pos_ref = positions[row]

    # Colonne 0 : patch de référence
    axes[row, 0].imshow(ref)
    axes[row, 0].set_title(f"patch 16×16 en pos {pos_ref}")
    axes[row, 0].axis("off")

    # Colonnes 1–3 : différences avec les autres patchs
    col_idx = 1
    for j in range(4):
        if j == row:
            continue

        d = diff(ref, patches[j])
        pos_j = positions[j]

        axes[row, col_idx].imshow(d)
        axes[row, col_idx].set_title(
            f"Diff pos {pos_ref} vs pos {pos_j}"
        )
        axes[row, col_idx].axis("off")

        col_idx += 1

plt.tight_layout()
plt.show()