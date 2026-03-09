import argparse
import csv
import os
import re
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from dataset import DOTA_MEAN, DOTA_STD, PreTiledDataset, mask_to_centers
from inference import find_peaks, load_checkpoint, match_centers
from train import segdino_collate
from model import DINOv3Backbone, LAYER_INDICES
from train_patch5 import AdversarialPatch, make_run_id, save_patch_image

from sklearn.decomposition import PCA

#from compare_embeddings import get_dino_embeddings

DATA_DIR = "data/DOTA/DOTA_PLANES_TILED"
SAVE_DIR = "attack_results"
MODEL_DIR = "runs/18_02-12_57_10_SMALL-PLUS_divulse-stern.pth"
limit_obj = None
batch_size = 8

RESUME_SAVED_PATCH = "27_02-21_06_39_divulse-stern/patch32_distilled.pt"

def get_dino_embeddings(attacker, images, model_size):
    """
    Retourne une liste de 4 tenseurs (B, 1024, C)
    correspondant aux couches utilisées par SegDino.
    """
    layer_ids = LAYER_INDICES[model_size]

    feats = attacker.backbone.get_intermediate_layers(images, layer_ids)

    return feats

def build_pca_reference(attacker64, dataloader, device):
    feats = []
    for imgs, _, metas in tqdm(dataloader, desc="Step PCA ref", leave=False):
        imgs = imgs.to(device)
        # Forward DINO en autocast
        with autocast("cuda"):
            stats = eval_patch_2(attacker64, imgs, metas, return_embeddings=True)
            f = stats["feats_patch"][3]  # (B, C, Hf, Wf)

        B = f.shape[0]
        # Conversion : float32 CPU pour sklearn PCA
        feats.append(f.reshape(B, -1).detach().float().cpu().numpy())

    # Concat final (N, D)
    feats = np.concatenate(feats, axis=0)

    # PCA sklearn → CPU float32
    pca = PCA(n_components=2)
    pca.fit(feats)

    return pca

def eval_patch_2(attacker, imgs, metas, threshold=0.3, return_embeddings=False):
    """
    imgs : (B,3,H,W)
    metas : liste de dicts, len = B
    """
    def _to_float32_numpy(hm):
        if hasattr(hm, "detach"):  # PyTorch tensor
            return hm.detach().float().cpu().numpy()
        return hm.astype("float32")  # numpy array

    B = imgs.shape[0]

    # --- Ground truth ---
    gt_centers_batch = [m["centers"] for m in metas]

    # --- Clean heatmaps ---
    heatmaps_clean = attacker.predict_heatmaps(imgs)[:, 0].detach().cpu().numpy()
    pred_clean_batch = [
            find_peaks(_to_float32_numpy(hm), threshold=threshold)
            for hm in heatmaps_clean
            ]

    # --- Patch application ---
    centers_list = pred_clean_batch
    imgs_patch = attacker.apply_patch(imgs, centers_list)
    # print("imgs_patch requires_grad:", imgs_patch.requires_grad)
    # print("patch requires_grad:", attacker.patch.requires_grad)

    # --- Patched heatmaps ---
    heatmaps_patched = attacker.predict_heatmaps(imgs_patch)[:, 0].detach().cpu().numpy()
    pred_patched_batch = [
            find_peaks(_to_float32_numpy(hm), threshold=threshold)
            for hm in heatmaps_patched
        ]
    # --- Matching ---
    tp_clean_batch = []
    tp_patched_batch = []

    for gt, pc, pp in zip(gt_centers_batch, pred_clean_batch, pred_patched_batch):
        tp_clean, _, _ = match_centers(gt, pc)
        tp_patched, _, _ = match_centers(gt, pp)
        tp_clean_batch.append(tp_clean)
        tp_patched_batch.append(tp_patched)

    stats = {
        "gt": [len(gt) for gt in gt_centers_batch],
        "tp_clean": tp_clean_batch,
        "tp_patched": tp_patched_batch,
        "pred_clean": pred_clean_batch,
        "pred_patched": pred_patched_batch,
        "img_patch": imgs_patch,
    }

    # --- Embeddings optionnels ---
    if return_embeddings:
        feats_clean = get_dino_embeddings(attacker, imgs, attacker.model_size)
        feats_patch = get_dino_embeddings(attacker, imgs_patch, attacker.model_size)
        stats["feats_clean"] = feats_clean
        stats["feats_patch"] = feats_patch

    return stats

def train_patch32(
    attacker32, attacker64,
    dataloader,
    pca_mean, pca_components,
    device="cuda",
    steps=100,
    lr=1e-1,
    lambda_dir=0.1
):
    optimizer = torch.optim.Adam([attacker32.patch], lr=lr)
    for step in tqdm(range(steps), desc="Steps"):
        total_loss = 0.0
        for imgs, _, metas in tqdm(dataloader, desc=f"Step {step}", leave=False):
            imgs = imgs.to(device)
            with autocast("cuda"):
                # --- Forward pour patch 32 ---
                stats32 = eval_patch_2(attacker32, imgs, metas, return_embeddings=True)
                f32 = stats32["feats_patch"][3]      # (B, C, Hf, Wf)
                fclean = stats32["feats_clean"][3]   # (B, C, Hf, Wf)
                # --- Forward pour patch 64 ---
                stats64 = eval_patch_2(attacker64, imgs, metas, return_embeddings=True)
                f64 = stats64["feats_patch"][3]      # (B, C, Hf, Wf)

                B = f32.shape[0]
                f32_flat = f32.reshape(B, -1)        # (B, D)
                f64_flat = f64.reshape(B, -1)        # (B, D)
                f32_flat = torch.nn.functional.normalize(f32_flat, dim=1)
                f64_flat = torch.nn.functional.normalize(f64_flat, dim=1)

                # --- PCA loss différentiable ---
                pca32 = pca_project_torch(f32_flat, pca_mean, pca_components)     # (B,2)
                pca64_ref = pca_project_torch(f64_flat, pca_mean, pca_components) # (B,2)
                loss_pca = ((pca32 - pca64_ref)**2).mean()

                # --- Directional loss (batch) ---
                loss_dir = 0.0
                for i in range(B):
                    d32 = f32[i] - fclean[i]
                    d64 = f64[i] - fclean[i]
                    d32 = torch.nn.functional.normalize(d32.flatten(), dim=0)
                    d64 = torch.nn.functional.normalize(d64.flatten(), dim=0)

                    loss_dir += 1 - torch.nn.functional.cosine_similarity(
                        d32.flatten(), d64.flatten(), dim=0
                    )
                loss_dir /= B

                # --- Loss totale ---
                loss = loss_pca + lambda_dir * loss_dir
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Step {step:04d} | Loss = {total_loss:.4f}")
        if step == 0:
            best_total_loss = total_loss

        if total_loss <= best_total_loss:
            best_total_loss = total_loss
            torch.save(attacker32.patch.data, "patch32_distilled.pt")
            torch.save(attacker32.patch.data.cpu(), patch_path)
            save_patch_image(attacker32.patch.data.cpu(), image_path)
            tqdm.write(f"  → best patch saved (recall {total_loss:.2f})")

    print("Training finished.")

def pca_project_torch(x, pca_mean, pca_components):
    # x : (B, D)
    x_centered = x - pca_mean
    return x_centered @ pca_components.T   # (B, 2)

device = "cuda"

run_id = make_run_id(MODEL_DIR)
run_dir = os.path.join(SAVE_DIR, run_id)
os.makedirs(run_dir, exist_ok=True)
patch_path = os.path.join(run_dir, "patch32_distilled.pt")
image_path = os.path.join(run_dir, "patch32_distilled.png")
best_total_loss = float("inf")

# Charger patch 64×64
attacker64 = AdversarialPatch(
        checkpoint_path=MODEL_DIR,
        device=device,
        patch_size=64,
        px=10, py=10,
        # threshold=args.threshold,
        # temperature=args.temperature,
    ).to(device)

loaded_patch = torch.load(os.path.join(SAVE_DIR, "19_02-patch64_divulse-stern/patch.pt"), map_location=device)
attacker64.patch.data.copy_(loaded_patch)

# Créer patch 32×32
attacker32 = AdversarialPatch(
        checkpoint_path=MODEL_DIR,
        device=device,
        patch_size=32,
        px=10, py=10,
        # threshold=args.threshold,
        # temperature=args.temperature,
    ).to(device)

if RESUME_SAVED_PATCH is not None:
    print(f"Resuming from patch: {RESUME_SAVED_PATCH}")
    loaded_patch = torch.load(os.path.join(SAVE_DIR, RESUME_SAVED_PATCH), map_location=device)
    attacker32.patch.data.copy_(loaded_patch)

full_dataset = PreTiledDataset(DATA_DIR, split="test")
# Keep only tiles that contain objects
obj_indices = []
for i, fname in enumerate(full_dataset.images):
    mask = cv2.imread(os.path.join(full_dataset.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
    if mask is not None and len(mask_to_centers(mask)) > 0:
        obj_indices.append(i)
if limit_obj:
    obj_indices = obj_indices[:limit_obj]
dataset = torch.utils.data.Subset(full_dataset, obj_indices)
print(f"Tiles with objects: {len(dataset)}/{len(full_dataset)}")

loader = DataLoader(
    dataset,
    batch_size=batch_size, shuffle=True, num_workers=2,
    collate_fn=segdino_collate,
)

# Construire PCA de référence
pca64 = build_pca_reference(attacker64, loader, device)
print("PCA reference done")
pca_mean = torch.tensor(pca64.mean_, dtype=torch.float32, device=device)        # (D,)
pca_components = torch.tensor(pca64.components_, dtype=torch.float32, device=device)  # (2, D)
print("PCA reference 2 done")

# Entraîner patch 32×32
train_patch32(
    attacker32,
    attacker64,
    loader,
    pca_mean,
    pca_components,
    device=device,
    steps=300,
    lr=5e-2,
    lambda_dir=0.01,
)


# Sauvegarde
# torch.save(attacker32.patch.data, "patch32_distilled.pt")
# torch.save(attacker32.patch.data.cpu(), patch_path)
# save_patch_image(attacker32.patch.data.cpu(), image_path)