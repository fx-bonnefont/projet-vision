"""
Evaluate an adversarial patch on SegDino center detection.

Applies a trained patch to test tiles and reports per-image recall
degradation compared to clean predictions.
"""
"""
Fast evaluation of an adversarial patch on SegDino.
Optimized version: 3x–10x faster.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import argparse
import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PreTiledDataset, mask_to_centers
from inference import find_peaks, match_centers
from train import segdino_collate

from pca_visualisation.train_patch5 import AdversarialPatch

from pca_visualisation.model import DINOv3Backbone, LAYER_INDICES

DATA_DIR = "data/DOTA/DOTA_PLANES_TILED"

# ---------------------------------------------------------
# DINO embedding extraction (compatible with SegDino)
# ---------------------------------------------------------

def get_dino_embeddings(attacker, images, model_size):
    """
    Retourne une liste de 4 tenseurs (B, 1024, C)
    correspondant aux couches utilisées par SegDino.
    """
    layer_ids = LAYER_INDICES[model_size]

    with torch.no_grad():
        feats = attacker.backbone.get_intermediate_layers(images, layer_ids)

    return feats

def visualize_all_with_centers(
    img_clean, img_patch,
    feats_clean, feats_patch,
    centers_clean, centers_patch,
    tile_id, save_dir
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # --- 1) Image clean + centres ---
    clean_np = img_clean[0].permute(1,2,0).cpu().numpy()
    clean_np = (clean_np - clean_np.min()) / (clean_np.max() - clean_np.min())
    axes[0].imshow(clean_np)
    for (x, y) in centers_clean:
        axes[0].scatter(x, y, s=40, c="red")
    axes[0].set_title("Clean + centres")
    axes[0].axis("off")

    # --- 2) Image patchée + centres ---
    patch_np = img_patch[0].permute(1,2,0).cpu().numpy()
    patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min())
    axes[1].imshow(patch_np)
    for (x, y) in centers_patch:
        axes[1].scatter(x, y, s=40, c="blue")
    axes[1].set_title("Patchée + centres")
    axes[1].axis("off")

    # --- 3–6) Différences L2 par couche ---
    for li in range(4):
        f1 = feats_clean[li][0]
        f2 = feats_patch[li][0]

        C = f1.shape[-1]
        f1_2d = f1.permute(1,0).reshape(C, 32, 32)
        f2_2d = f2.permute(1,0).reshape(C, 32, 32)

        diff_map = (f1_2d - f2_2d).norm(dim=0).cpu().numpy()

        axes[li+2].imshow(diff_map, cmap="inferno")
        axes[li+2].set_title(f"Diff L2 Layer {li}")
        axes[li+2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"vis_{tile_id}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved visualization: {save_path}")


def compare_embeddings(feats_clean, feats_patch):
    """
    Compare clean vs patch embeddings.
    Retourne L2 diff et cosine similarity par couche.
    """
    l2_diffs = []
    cos_sims = []

    for f1, f2 in zip(feats_clean, feats_patch):
        # L2
        l2 = (f1 - f2).norm(dim=-1).mean().item()
        l2_diffs.append(l2)

        # Cosine
        cos = torch.nn.functional.cosine_similarity(
            f1.flatten(1), f2.flatten(1)
        ).mean().item()
        cos_sims.append(cos)

    return l2_diffs, cos_sims


def main():
    parser = argparse.ArgumentParser(description="Evaluate an adversarial patch on SegDino")
    parser.add_argument("-c", required=True, help="checkpoint path")
    parser.add_argument("--patch_dir", required=True, help="path to saved (patch.pt)")
    parser.add_argument("-b", type=int, default=8, help="batch size")
    parser.add_argument("--px", type=int, default=0)
    parser.add_argument("--py", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("-model", default="small-plus")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Load attacker + patch
    # ---------------------------------------------------------
    attacker = AdversarialPatch(
        checkpoint_path=args.c,
        device=device,
        patch_size=args.patch_size,
        px=args.px, py=args.py,
        threshold=args.threshold,
        model_size=args.model,
    ).to(device)

    patch_data = torch.load(os.path.join(args.patch_dir, "patch.pt"), map_location=device)
    attacker.patch.data.copy_(patch_data)
    print(f"Loaded patch from {args.patch_dir}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    full_dataset = PreTiledDataset(DATA_DIR, split="test")
    obj_indices = [
        i for i, fname in enumerate(full_dataset.images)
        if len(mask_to_centers(cv2.imread(full_dataset.mask_dir + "/" + fname, 0))) > 0
    ]
    dataset = torch.utils.data.Subset(full_dataset, obj_indices)
    print(f"Tiles with objects: {len(dataset)}/{len(full_dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=4,
        collate_fn=segdino_collate,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 1) PRECOMPUTE CLEAN HEATMAPS
    # ---------------------------------------------------------
    print("Precomputing clean heatmaps...")
    clean_cache = {}

    with torch.no_grad():
        for images, _, metas, _ in tqdm(loader, desc="clean"):
            images = images.to(device)
            heatmaps_clean = attacker.predict_heatmaps(images)

            for b, meta in enumerate(metas):
                clean_cache[meta["id"]] = heatmaps_clean[b, 0].cpu()

    # ---------------------------------------------------------
    # 2) EVALUATION LOOP (patched + embeddings + visualisation)
    # ---------------------------------------------------------
    print("Evaluating patched predictions...")

    image_stats = {}

    with torch.no_grad():
        for images, _, metas, _ in tqdm(loader, desc="patched"):
            images = images.to(device)
            B = len(metas)

            centers_list = [
                find_peaks(clean_cache[meta["id"]].numpy().astype("float32"),
                        threshold=args.threshold)
                for meta in metas
            ]
            # Heatmaps patchées
            heatmaps_patched = attacker.predict_heatmaps_patched(images, centers_list)

            # Embeddings clean et patchés (batch complet)
            feats_clean = get_dino_embeddings(attacker, images, attacker.model_size)
            feats_patch = get_dino_embeddings(attacker, attacker.apply_patch(images, centers_list), attacker.model_size)

            # Comparaison numérique (batch complet)
            l2_diffs, cos_sims = compare_embeddings(feats_clean, feats_patch)

            # --- Boucle par image ---
            for b in range(B):
                meta = metas[b]
                tile_id = meta["id"]

                # --- Affichage des différences d'embeddings ---
                # print(f"\nEmbeddings diff for tile {tile_id}:")
                # for li, (l2, cos) in enumerate(zip(l2_diffs, cos_sims)):
                #     print(f"  Layer {li}: L2={l2:.4f}, Cos={cos:.4f}")

                # --- Heatmaps clean et patchées ---
                hm_clean = clean_cache[tile_id].numpy().astype("float32")
                hm_patched = heatmaps_patched[b, 0].cpu().numpy().astype("float32")

                centers_clean = find_peaks(hm_clean, threshold=args.threshold)
                centers_patch = find_peaks(hm_patched, threshold=args.threshold)

                # --- Images clean et patchées ---
                img_clean = images[b].unsqueeze(0)
                img_patch = attacker.apply_patch(images[b].unsqueeze(0), centers_list)

                # --- Visualisation complète (clean/patch + centres + 4 couches) ---
                visualize_all_with_centers(
                    img_clean, img_patch,
                    [f[b:b+1] for f in feats_clean],
                    [f[b:b+1] for f in feats_patch],
                    centers_clean, centers_patch,
                    tile_id=tile_id,
                    save_dir=args.patch_dir
                )

                # ---------------------------------------------------------
                # Mise à jour des stats globales
                # ---------------------------------------------------------
                parent = tile_id.split("_")[0]
                gt_centers = meta["centers"]

                if not gt_centers:
                    continue

                if parent not in image_stats:
                    image_stats[parent] = {"gt": 0, "tp_clean": 0, "tp_patched": 0}

                pred_clean = centers_clean
                pred_patched = centers_patch

                tp_c, _, _ = match_centers(gt_centers, pred_clean)
                tp_p, _, _ = match_centers(gt_centers, pred_patched)

                image_stats[parent]["gt"] += len(gt_centers)
                image_stats[parent]["tp_clean"] += tp_c
                image_stats[parent]["tp_patched"] += tp_p


if __name__ == "__main__":
    main()
