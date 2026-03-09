import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()

from sklearn.decomposition import PCA

from pca_visualisation.model import SegDino
from pca_visualisation.train_patch5 import AdversarialPatch
from dataset import PreTiledDataset

from inference import find_peaks, mask_to_centers

from pca_visualisation.compare_embeddings import get_dino_embeddings

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


# ---------------------------------------------------------
# PCA ANALYSIS
# ---------------------------------------------------------
def analyze_subspace(feat_clean, feat_patch, layer):
    """
    feat_clean[layer] : (1, 1024, C)
    feat_patch[layer] : (1, 1024, C)
    """
    f1 = feat_clean[layer][0].cpu().numpy()   # (1024, C)
    f2 = feat_patch[layer][0].cpu().numpy()

    D = f2 - f1  # perturbation

    pca = PCA(n_components=10)
    pca.fit(D)

    print(f"\n=== PCA variance explained (Layer {layer}) ===")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i}: {v:.4f}")

    return pca, D


def visualize_pc_maps(pca, D, layer):
    """
    Affiche PC0 et PC1 sous forme de cartes 32×32
    """
    proj = pca.transform(D)  # (1024, 10)

    for pc in [0, 1]:
        pc_map = proj[:, pc].reshape(32, 32)
        plt.figure(figsize=(5,5))
        plt.imshow(pc_map, cmap="inferno")
        plt.title(f"Layer {layer} — PC{pc} spatial map")
        plt.axis("off")
        plt.savefig(os.path.join(OUTPUT_DIR, f"figure_layer{layer}_pc{pc}.png"))
        plt.close()


def visualize_pca_scatter(pca, D, layer):
    proj = pca.transform(D)[:, :2]

    plt.figure(figsize=(6,6))
    plt.scatter(proj[:,0], proj[:,1], s=8, c='red')
    plt.title(f"Layer {layer} — PCA scatter (PC0 vs PC1)")
    plt.xlabel("PC0")
    plt.ylabel("PC1")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_layer{layer}.png"))
    plt.close()


def visualize_clean_vs_patch_pca(feat_clean, feat_patch, layer):
    f1 = feat_clean[layer][0].cpu().numpy()   # (1024, 384)
    f2 = feat_patch[layer][0].cpu().numpy()

    D = f2 - f1

    pca = PCA(n_components=2)
    pca.fit(np.concatenate([f1, f2], axis=0))

    proj_clean = pca.transform(f1)
    proj_patch = pca.transform(f2)

    plt.figure(figsize=(6,6))
    plt.scatter(proj_clean[:,0], proj_clean[:,1], s=8, c='blue', label="clean")
    plt.scatter(proj_patch[:,0], proj_patch[:,1], s=8, c='red', label="patch")
    plt.title(f"Layer {layer} — Clean vs Patch PCA")
    plt.xlabel("PC0")
    plt.ylabel("PC1")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"clean_vs_patch_layer{layer}.png"))
    plt.close()

def visualize_topk_channels(feat_clean, feat_patch, layer, k=10):
    f1 = feat_clean[layer][0].cpu().numpy()   # (1024, 384)
    f2 = feat_patch[layer][0].cpu().numpy()

    D = np.abs(f2 - f1)  # (1024, 384)
    channel_energy = D.mean(axis=0)  # importance par canal

    topk = np.argsort(channel_energy)[-k:][::-1]

    plt.figure(figsize=(10,4))
    plt.bar(range(k), channel_energy[topk])
    plt.xticks(range(k), topk, rotation=45)
    plt.title(f"Layer {layer} — Top {k} perturbed channels")
    plt.ylabel("Mean absolute difference")
    plt.savefig(os.path.join(OUTPUT_DIR, f"topk_channels_layer{layer}.png"))
    plt.close()

    print(f"Top-{k} channels for layer {layer}: {topk}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("\n=== Loading attacker ===")
    attacker = AdversarialPatch(
        checkpoint_path=CHECKPOINT,
        device=DEVICE,
        patch_size=PATCH_SIZE,
        px=10,
        py=10,
        threshold=0.3,
        model_size=MODEL_SIZE,
    ).to(DEVICE)

    # Load patch
    patch_data = torch.load(os.path.join(PATCH_DIR, "patch.pt"), map_location=DEVICE)
    attacker.patch.data.copy_(patch_data)
    print("Patch loaded.")

    # Load image
    print(f"\n=== Loading image {IMAGE_ID} ===")
    img, meta = load_single_image(IMAGE_ID)
    img = img.to(DEVICE)

    # Clean embeddings
    print("\n=== Extracting clean embeddings ===")
    feats_clean = get_dino_embeddings(attacker, img, MODEL_SIZE)

    # Patched embeddings
    print("\n=== Extracting patched embeddings ===")
    heatmap_clean = attacker.predict_heatmaps(img)[0, 0].detach().cpu().numpy()

    # 2) Trouver les centres clean
    centers_clean = find_peaks(heatmap_clean, threshold=attacker.threshold)

    # 3) Construire centers_list pour un batch de taille 1
    centers_list = [centers_clean]

    img_patch = attacker.apply_patch(img, centers_list)
    feats_patch = get_dino_embeddings(attacker, img_patch, MODEL_SIZE)

    # ---------------------------------------------------------
    # PCA SUBSPACE ANALYSIS
    # ---------------------------------------------------------
    for layer in range(4):
        print(f"\n\n########## LAYER {layer} ##########")

        pca, D = analyze_subspace(feats_clean, feats_patch, layer)

        visualize_pc_maps(pca, D, layer)
        visualize_pca_scatter(pca, D, layer)
        visualize_clean_vs_patch_pca(feats_clean, feats_patch, layer)
        visualize_topk_channels(feats_clean, feats_patch, layer, k=10)


if __name__ == "__main__":
    main()
