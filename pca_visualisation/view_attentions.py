import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
# from torchvision import transforms
from PIL import Image
from dataset import PreTiledDataset, mask_to_centers
from pca_visualisation.train_patch5 import AdversarialPatch

# ------------------------------------------------------------
# 1. Importer ton AttentionViewer
# ------------------------------------------------------------
from pca_visualisation.model import AttentionViewer   # <-- adapte le nom du fichier

# ------------------------------------------------------------
# 2. Charger le modèle DINOv3 small-plus (HuggingFace)
# ------------------------------------------------------------
viewer = AttentionViewer(model_size="small-plus")
viewer.eval()

IMAGE = "P2550_obj_6"
DATA_DIR = "data/DOTA/DOTA_PLANES_TILED"

# # ------------------------------------------------------------
# # 3. Préparer l’image
# # ------------------------------------------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225)
#     )
# ])

def load_single_image(image_id, data_dir):
    """
    Charge une seule tuile depuis PreTiledDataset.
    Retourne :
      - img : (1,3,H,W)
      - meta : dict contenant 'centers', 'id', etc.
    """
    dataset = PreTiledDataset(data_dir, split="test")

    filename = image_id + ".png"
    if filename not in dataset.images:
        raise ValueError(f"Image {filename} non trouvée dans {data_dir}")

    idx = dataset.images.index(filename)
    img, _, meta = dataset[idx]
    return img.unsqueeze(0), meta

def visu_embeddings(cfg):
    device = cfg.get("device")# , "cuda" if torch.cuda.is_available() else "cpu")

    # Charger l’image
    img, meta = load_single_image(cfg["image_id"], cfg["data_dir"])
    print("Image shape:", img.shape)
    px, py = cfg["px"], cfg["py"]
    size = cfg["size"]

    # img est en (C, H, W)
    #img = img.squeeze(0)
    _, C, H, W = img.shape

    x1, x2 = px, min(px + size, W)
    y1, y2 = py, min(py + size, H)

    img[:, :, y1:y2, x1:x2] = 0.0
    img = img.to(device)
    # Charger l’attaquant
    attacker = AdversarialPatch(
        checkpoint_path=cfg["checkpoint"],
        device=device,
        patch_size=cfg["size"],
        px=cfg["px"],
        py=cfg["py"],
        threshold=cfg["threshold"],
        model_size=cfg.get("model_size", "small-plus"),
    ).to(device)

    patch_data = torch.load(cfg["path"], map_location=device)
    attacker.patch.data.copy_(patch_data)

    return img, meta

cfg2 = {
    "name": "patch22_divulse",
    "path": "attack_results/18_02-patch22_divulse-stern/patch.pt",
    "size": 22,
    "px": 11,
    "py": 11,
    "threshold": 0.3,
    "checkpoint": "runs/18_02-12_57_10_SMALL-PLUS_divulse-stern.pth",
    "image_id": "P2550_obj_6",
    "data_dir": "data/DOTA/DOTA_PLANES_TILED",
    "model_size": "small-plus",
    "device": "cpu",
    "return_embeddings": True
}

img, meta = load_single_image(IMAGE, DATA_DIR)

img_patched, meta = visu_embeddings(cfg2)


x = img
x_patched = img_patched

# ------------------------------------------------------------
# 4. Extraire les cartes d’attention via le backbone HuggingFace
# ------------------------------------------------------------
with torch.no_grad():
    attn_maps_clean = viewer(x)   # liste : [layer][B, heads, tokens, tokens]
    attn_maps_patched = viewer(x_patched)
    attn_maps = [attn_p - attn_c for attn_p, attn_c in zip(attn_maps_patched, attn_maps_clean)]

# ------------------------------------------------------------
# 5. On prend la dernière couche
# ------------------------------------------------------------
attn = attn_maps[-1][0]  # shape: [num_heads, tokens, tokens]
num_heads = attn.shape[0]

# Retirer les tokens spéciaux (register tokens + class token)
# DINOv3 small-plus a 4 register tokens + 1 class token
num_special = 1 + viewer.backbone.num_register_tokens
attn = attn[:, num_special:, num_special:]

# Nombre de patchs
num_patches = int(np.sqrt(attn.shape[-1]))

# ------------------------------------------------------------
# 6. Visualisation des cartes d’attention
# ------------------------------------------------------------
fig, axes = plt.subplots(1, num_heads, figsize=(3*num_heads, 3))

for h in range(num_heads):
    a = attn[h].mean(0).reshape(num_patches, num_patches)
    a = a / a.max()

    axes[h].imshow(a.cpu(), cmap="inferno")
    axes[h].set_title(f"Tête {h}")
    axes[h].axis("off")

plt.tight_layout()
plt.show()
