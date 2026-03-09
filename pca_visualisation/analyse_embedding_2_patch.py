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
IMAGE_ID = "P2550_obj_17"
DATA_DIR = "/home/ericl/projet-vision/data/DOTA/DOTA_PLANES_TILED"
PATCH_DIR_1 = "attack_results/19_02-patch64_divulse-stern"
PATCH_DIR_2 = "attack_results/21_02-patch16_16_divulse-stern"
PATCH_SIZE_1 = 64
PATCH_SIZE_2 = 16
CHECKPOINT = "runs/18_02-12_57_10_SMALL-PLUS_divulse-stern.pth"
MODEL_SIZE = "small-plus"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


OUTPUT_DIR = os.path.join(PATCH_DIR_1, "a_PCA")
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

def extract_clean_and_patch_embeddings_for_patch(attacker, image_id, patch_tensor):
    img, meta = load_single_image(image_id)
    img = img.to(DEVICE)

    # Charger le patch dans l'attaquant
    attacker.patch.data.copy_(patch_tensor)

    # Heatmap clean → centres clean
    heatmap_clean = attacker.predict_heatmaps(img)[0, 0].detach().cpu().numpy()
    centers_clean = find_peaks(heatmap_clean, threshold=attacker.threshold)
    centers_list = [centers_clean]

    # Image patchée
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

def plot_principal_components_maps(pca, D, layer, prefix, out_dir):
    """
    pca : PCA fitted on D
    D   : (1024, C) perturbation matrix
    prefix : "patch1" ou "patch2"
    """
    proj = pca.transform(D)  # (1024, 2)

    for pc in [0, 1]:
        pc_map = proj[:, pc].reshape(32, 32)

        plt.figure(figsize=(5,5))
        plt.imshow(pc_map, cmap="inferno")
        plt.title(f"{prefix} — Layer {layer} — PC{pc}")
        plt.axis("off")

        out_path = os.path.join(out_dir, f"{prefix}_layer{layer}_PC{pc}.png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved PC{pc} map for {prefix}: {out_path}")

def plot_cross_projection(
    proj1_clean, proj1_patch,
    proj2_clean, proj2_patch,
    layer, out_path,
    theta_min=None, theta_mean=None, theta_max=None,
    patch_size_1=None, patch_size_2=None,
    img_name=None
):
    plt.figure(figsize=(7,7))

    label1_clean = f"Image origine {img_name})" if img_name else "Image origine"
    label1_patch = f"Patch1 patch (size={patch_size_1})" if patch_size_1 else "Patch1 patch"
    # label2_clean = f"Patch2 clean (size={patch_size_2})" if patch_size_2 else "Patch2 clean"
    label2_patch = f"Patch2 patch (size={patch_size_2})" if patch_size_2 else "Patch2 patch"

    plt.scatter(proj1_clean[:,0], proj1_clean[:,1], s=8, c='blue',   label=label1_clean)
    plt.scatter(proj1_patch[:,0], proj1_patch[:,1], s=8, c='red',    label=label1_patch)
    # plt.scatter(proj2_clean[:,0], proj2_clean[:,1], s=8, c='cyan',   label=label2_clean)
    plt.scatter(proj2_patch[:,0], proj2_patch[:,1], s=8, c='orange', label=label2_patch)

    if theta_mean is not None:
        subtitle = f"θ_mean={theta_mean:.1f}°, θ_min={theta_min:.1f}°, θ_max={theta_max:.1f}°"
        plt.title(f"Cross PCA projection — Layer {layer}\n{subtitle}")
    else:
        plt.title(f"Cross PCA projection — Layer {layer}")

    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

def main():
    attacker_1 = AdversarialPatch(
        checkpoint_path=CHECKPOINT,
        device=DEVICE,
        patch_size=PATCH_SIZE_1,
        px=0, py=0,
        threshold=0.3,
        model_size=MODEL_SIZE,
    ).to(DEVICE)

    attacker_2 = AdversarialPatch(
        checkpoint_path=CHECKPOINT,
        device=DEVICE,
        patch_size=PATCH_SIZE_2,
        px=0, py=0,
        threshold=0.3,
        model_size=MODEL_SIZE,
    ).to(DEVICE)

    # Deux patchs différents
    patch1 = torch.load(os.path.join(PATCH_DIR_1, "patch.pt"), map_location=DEVICE)
    patch2 = torch.load(os.path.join(PATCH_DIR_2, "patch.pt"), map_location=DEVICE)

    IMAGE = "P2550_obj_6"
    layer = 3

    # Image + patch1
    feats_clean_1, feats_patch1 = extract_clean_and_patch_embeddings_for_patch(
        attacker_1, IMAGE, patch1
    )
    # Image + patch2 (clean recalculé, mais tu peux aussi réutiliser feats_clean_1)
    feats_clean_2, feats_patch2 = extract_clean_and_patch_embeddings_for_patch(
        attacker_2, IMAGE, patch2
    )

    # PCA sur perturbation 1
    pca1, D1 = compute_pca(feats_clean_1, feats_patch1, layer)
    proj1_clean_1, proj1_patch_1 = cross_project(pca1, feats_clean_1, feats_patch1, layer)
    proj2_clean_1, proj2_patch_1 = cross_project(pca1, feats_clean_2, feats_patch2, layer)

    # PCA sur perturbation 2
    pca2, D2 = compute_pca(feats_clean_2, feats_patch2, layer)
    proj1_clean_2, proj1_patch_2 = cross_project(pca2, feats_clean_1, feats_patch1, layer)
    proj2_clean_2, proj2_patch_2 = cross_project(pca2, feats_clean_2, feats_patch2, layer)

    # Angles entre sous-espaces
    angles, sigma = subspace_angles(pca1, pca2, k=2)
    theta_min = np.degrees(angles.min())
    theta_max = np.degrees(angles.max())
    theta_mean = np.degrees(angles.mean())

    print(f"\n=== Subspace affinity (Layer {layer}) ===")
    print(f"  σ = {sigma}")
    print(f"  angles (deg) = {np.degrees(angles)}")
    print(f"  θ_min = {theta_min:.2f}°")
    print(f"  θ_max = {theta_max:.2f}°")
    print(f"  θ_mean = {theta_mean:.2f}°")

    # Plots avec angles dans le titre
    plot_cross_projection(
        proj1_clean_1, proj1_patch_1,
        proj2_clean_1, proj2_patch_1,
        layer,
        os.path.join(OUTPUT_DIR, f"cross_PCA1_layer{layer}.png"),
        theta_min, theta_mean, theta_max,
        patch_size_1=PATCH_SIZE_1,
        patch_size_2=PATCH_SIZE_2,
        img_name=IMAGE_ID
    )

    plot_cross_projection(
        proj1_clean_2, proj1_patch_2,
        proj2_clean_2, proj2_patch_2,
        layer,
        os.path.join(OUTPUT_DIR, f"cross_PCA2_layer{layer}.png"),
        theta_min, theta_mean, theta_max,
        patch_size_1=PATCH_SIZE_1,
        patch_size_2=PATCH_SIZE_2,
        img_name=IMAGE_ID
    )

    plot_principal_components_maps(pca1, D1, layer, prefix="patch1", out_dir=OUTPUT_DIR)
    plot_principal_components_maps(pca2, D2, layer, prefix="patch2", out_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()