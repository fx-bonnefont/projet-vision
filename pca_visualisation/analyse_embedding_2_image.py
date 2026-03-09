import os
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()

from sklearn.decomposition import PCA

from model import SegDino
from train_patch5 import AdversarialPatch
from dataset import PreTiledDataset

from inference import find_peaks, mask_to_centers

from compare_embeddings import get_dino_embeddings

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
IMAGE_ID = "P2550_obj_6"
DATA_DIR = "/home/ericl/projet-vision/data/DOTA/DOTA_PLANES_TILED"
PATCH_DIR = "attack_results/19_02-patch64_divulse-stern"
PATCH_SIZE = 64
CHECKPOINT = "runs/18_02-12_57_10_SMALL-PLUS_divulse-stern.pth"
MODEL_SIZE = "small-plus"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


OUTPUT_DIR = os.path.join(PATCH_DIR, "a_PCA")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Saving figures to:", OUTPUT_DIR)
# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------
def load_single_image(image_id):
    dataset = PreTiledDataset(DATA_DIR, split="test")
    idx = dataset.images.index(image_id + ".png")
    img, _, meta = dataset[idx]
    return img.unsqueeze(0), meta

def extract_clean_and_patch_embeddings(attacker, image_id):
    img, meta = load_single_image(image_id)
    img = img.to(DEVICE)

    # Heatmap clean → centres clean
    heatmap_clean = attacker.predict_heatmaps(img)[0, 0].detach().cpu().numpy()
    centers_clean = find_peaks(heatmap_clean, threshold=attacker.threshold)
    centers_list = [centers_clean]

    # Patch
    img_patch = attacker.apply_patch(img, centers_list)

    # Embeddings
    feats_clean = get_dino_embeddings(attacker, img, MODEL_SIZE)
    feats_patch = get_dino_embeddings(attacker, img_patch, MODEL_SIZE)

    return feats_clean, feats_patch

# ---------------------------------------------------------
# PCA ANALYSIS
# ---------------------------------------------------------
def compute_pca(feats_clean, feats_patch, layer):
    f1 = feats_clean[layer][0].cpu().numpy()
    f2 = feats_patch[layer][0].cpu().numpy()
    D = f2 - f1
    pca = PCA(n_components=2)
    pca.fit(D)
    return pca, D

def subspace_angles(pcaA, pcaB, k=2):
    """
    Calcule les angles principaux entre les sous-espaces PCA de dimension k.
    pcaA.components_ : (k, C)
    pcaB.components_ : (k, C)
    """
    UA = pcaA.components_[:k]      # (k, C)
    UB = pcaB.components_[:k]      # (k, C)

    # Produit croisé
    M = UA @ UB.T                  # (k, k)

    # Valeurs singulières
    sigma = np.linalg.svd(M, compute_uv=False)

    # Angles principaux
    angles = np.arccos(np.clip(sigma, -1.0, 1.0))

    return angles, sigma

def cross_project(pca, feats_clean, feats_patch, layer):
    f1 = feats_clean[layer][0].cpu().numpy()
    f2 = feats_patch[layer][0].cpu().numpy()
    proj_clean = pca.transform(f1)
    proj_patch = pca.transform(f2)
    return proj_clean, proj_patch

def plot_cross_projection(projA_clean, projA_patch, projB_clean, projB_patch, layer, out_path):
    plt.figure(figsize=(7,7))
    plt.scatter(projA_clean[:,0], projA_clean[:,1], s=8, c='blue', label="A clean")
    plt.scatter(projA_patch[:,0], projA_patch[:,1], s=8, c='red', label="A patch")
    plt.scatter(projB_clean[:,0], projB_clean[:,1], s=8, c='cyan', label="B clean")
    plt.scatter(projB_patch[:,0], projB_patch[:,1], s=8, c='orange', label="B patch")
    plt.legend()
    plt.title(f"Cross PCA projection — Layer {layer}")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

def main():
    attacker = AdversarialPatch(
        checkpoint_path=CHECKPOINT,
        device=DEVICE,
        patch_size=PATCH_SIZE,
        px=0, py=0,
        threshold=0.3,
        model_size=MODEL_SIZE,
    ).to(DEVICE)

    patch_data = torch.load(os.path.join(PATCH_DIR, "patch.pt"), map_location=DEVICE)
    attacker.patch.data.copy_(patch_data)

    # --- Images A et B ---
    IMAGE_A = "P2550_obj_6"
    IMAGE_B = "P2550_obj_17"

    featsA_clean, featsA_patch = extract_clean_and_patch_embeddings(attacker, IMAGE_A)
    featsB_clean, featsB_patch = extract_clean_and_patch_embeddings(attacker, IMAGE_B)

    layer = 3

    # PCA de A
    pcaA, DA = compute_pca(featsA_clean, featsA_patch, layer)
    projA_clean_A, projA_patch_A = cross_project(pcaA, featsA_clean, featsA_patch, layer)
    projB_clean_A, projB_patch_A = cross_project(pcaA, featsB_clean, featsB_patch, layer)

    plot_cross_projection(
        projA_clean_A, projA_patch_A,
        projB_clean_A, projB_patch_A,
        layer,
        os.path.join(OUTPUT_DIR, f"cross_A_layer{layer}.png")
    )

    # PCA de B
    pcaB, DB = compute_pca(featsB_clean, featsB_patch, layer)
    projA_clean_B, projA_patch_B = cross_project(pcaB, featsA_clean, featsA_patch, layer)
    projB_clean_B, projB_patch_B = cross_project(pcaB, featsB_clean, featsB_patch, layer)

    plot_cross_projection(
        projA_clean_B, projA_patch_B,
        projB_clean_B, projB_patch_B,
        layer,
        os.path.join(OUTPUT_DIR, f"cross_B_layer{layer}.png")
    )

    angles, sigma = subspace_angles(pcaA, pcaB, k=2)

    theta_min = np.degrees(angles.min())
    theta_max = np.degrees(angles.max())
    theta_mean = np.degrees(angles.mean())

    print(f"\n=== Subspace affinity (Layer {layer}) ===")
    print(f"  σ = {sigma}")
    print(f"  angles (deg) = {np.degrees(angles)}")
    print(f"  θ_min = {theta_min:.2f}°")
    print(f"  θ_max = {theta_max:.2f}°")
    print(f"  θ_mean = {theta_mean:.2f}°")

if __name__ == "__main__":
    main()