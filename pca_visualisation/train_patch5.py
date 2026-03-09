"""
Adversarial patch training for SegDino center detection.

Trains a small RGB patch (e.g. 16x16) that, when placed on tiles,
minimizes the number of detected centers (false negative attack).
"""
import argparse
import csv
import os
import re
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from dataset import DOTA_MEAN, DOTA_STD, PreTiledDataset, mask_to_centers
from inference import find_peaks, load_checkpoint, match_centers
from train import segdino_collate
from model import DINOv3Backbone

DATA_DIR = "data/DOTA/DOTA_PLANES_TILED"
SAVE_DIR = "attack_results"

# Valid pixel range in normalized space
NORM_MIN = [(0.0 - m) / s for m, s in zip(DOTA_MEAN, DOTA_STD)]
NORM_MAX = [(1.0 - m) / s for m, s in zip(DOTA_MEAN, DOTA_STD)]


class AdversarialPatch(nn.Module):
    def __init__(self, checkpoint_path, device, patch_size=16, px=0, py=0, threshold=0.3, temperature=10.0, model_size='small-plus'):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.patch_size = patch_size
        self.px = px
        self.py = py
        self.model_size = model_size

        # Load and freeze the full model
        self.model = load_checkpoint(checkpoint_path, device)
        self.model.eval()
        self.backbone = self.model.backbone
        self.backbone.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        # Trainable patch (only learnable parameter)
        self.patch = nn.Parameter(torch.randn(3, patch_size, patch_size) * 0.01)

    def apply_patch(self, images, centers_list):
        B, C, H, W = images.shape
        P = self.patch_size

        x = images.clone()

        patch_clamped = torch.stack([
            self.patch[c].clamp(NORM_MIN[c], NORM_MAX[c]) for c in range(3)
        ])

        max_x = W - P
        max_y = H - P

        # -----------------------------
        # POSITION LOGIC
        # -----------------------------
        # px
        if self.px is None:
            px = torch.randint(0, max_x + 1, (B,), device=images.device)
        else:
            px = torch.full((B,), self.px, device=images.device, dtype=torch.long)

        # py
        if self.py is None:
            py = torch.randint(0, max_y + 1, (B,), device=images.device)
        else:
            py = torch.full((B,), self.py, device=images.device, dtype=torch.long)

        # -----------------------------
        # FORBIDDEN CIRCLES AROUND MULTIPLE TARGETS
        # -----------------------------
        r = 10  # rayon interdit

        # px, py : (B,)
        # centers_list : liste de listes de centres [(x,y), ...]

        invalid = torch.zeros(B, dtype=torch.bool, device=images.device)

        for i in range(B):
            centers = centers_list[i]  # liste de tuples
            if len(centers) == 0:
                continue

            # Convertir en tenseur (N,2)
            c = torch.tensor(centers, device=images.device)  # (N,2)
            cx = c[:, 0].view(-1, 1)
            cy = c[:, 1].view(-1, 1)

            # Distance entre le coin du patch et tous les centres
            dx = px[i] - cx
            dy = py[i] - cy
            dist2 = dx*dx + dy*dy

            if (dist2 < r*r).any():
                invalid[i] = True

        # -----------------------------
        # REJECTION SAMPLING
        # -----------------------------
        while invalid.any():
            idxs = invalid.nonzero(as_tuple=True)[0]
            n = len(idxs)

            # régénérer px, py pour les images invalides
            px[idxs] = torch.randint(0, max_x + 1, (n,), device=images.device)
            py[idxs] = torch.randint(0, max_y + 1, (n,), device=images.device)

            # recalculer invalid
            invalid[:] = False
            for i in range(B):
                centers = centers_list[i]
                if len(centers) == 0:
                    continue

                c = torch.tensor(centers, device=images.device)
                cx = c[:, 0].view(-1, 1)
                cy = c[:, 1].view(-1, 1)

                dx = px[i] - cx
                dy = py[i] - cy
                dist2 = dx*dx + dy*dy

                if (dist2 < r*r).any():
                    invalid[i] = True

        # -----------------------------
        # APPLY PATCH
        # -----------------------------
        y_offsets = torch.arange(P, device=images.device).view(1, 1, P, 1)
        x_offsets = torch.arange(P, device=images.device).view(1, 1, 1, P)

        ys = py.view(B, 1, 1, 1) + y_offsets
        xs = px.view(B, 1, 1, 1) + x_offsets

        ys = ys.expand(B, C, P, P)
        xs = xs.expand(B, C, P, P)

        b_idx = torch.arange(B, device=images.device).view(B, 1, 1, 1).expand(B, C, P, P)
        c_idx = torch.arange(C, device=images.device).view(1, C, 1, 1).expand(B, C, P, P)

        x[b_idx, c_idx, ys, xs] = patch_clamped.view(1, C, P, P)

        self.last_px = px
        self.last_py = py

        return x

    def forward(self, images, centers_list):
        patched = self.apply_patch(images, centers_list)
        logits = self.model(patched)
        pred = torch.sigmoid(logits)
        soft_count = torch.sigmoid(self.temperature * (pred - self.threshold)).sum()
        return soft_count


    def predict_heatmaps(self, images):
        """Return sigmoid heatmaps for a batch (no grad)."""
        with torch.no_grad():
            logits = self.model(images)
            return torch.sigmoid(logits)

    def predict_heatmaps_patched(self, images, centers_list):
        """Return sigmoid heatmaps for patched batch (no grad)."""
        with torch.no_grad(), autocast("cuda"):
            patched = self.apply_patch(images, centers_list)
            logits = self.model(patched)
            return torch.sigmoid(logits)


def save_patch_image(patch_param, path):
    """Save the patch as a viewable PNG."""
    import torchvision.utils as vutils
    with torch.no_grad(), autocast("cuda"):
        patch = torch.stack([
            patch_param[c].clamp(NORM_MIN[c], NORM_MAX[c]) for c in range(3)
        ])
        # Denormalize to [0, 1]
        mean = torch.tensor(DOTA_MEAN, device=patch.device).view(3, 1, 1)
        std = torch.tensor(DOTA_STD, device=patch.device).view(3, 1, 1)
        patch_rgb = (patch * std + mean).clamp(0, 1)
        vutils.save_image(patch_rgb, path)


def make_run_id(checkpoint_path):
    """Extract codename from checkpoint and combine with current timestamp."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    # Extract the codename part (e.g. "funning-almanac" from "06_02-09_48_12_SMALL_funning-almanac")
    m = re.search(r"[a-z]+-[a-z]+$", stem)
    codename = m.group() if m else "patch"
    timestamp = datetime.now().strftime("%d_%m-%H_%M_%S")
    return f"{timestamp}_{codename}"


def main():
    parser = argparse.ArgumentParser(description="Train an adversarial patch on SegDino")
    parser.add_argument("-c", required=True, help="checkpoint path")
    parser.add_argument("-e", type=int, default=50, help="epochs")
    parser.add_argument("-b", type=int, default=4, help="batch size")
    parser.add_argument("-l", type=float, default=0.1, help="learning rate")
    parser.add_argument("--px", type=int, default=None, help="patch x position (None = random)")
    parser.add_argument("--py", type=int, default=None, help="patch y position (None = random)")
    parser.add_argument("--patch-size", type=int, default=16, help="patch size")
    parser.add_argument("--threshold", type=float, default=0.3, help="detection threshold")
    parser.add_argument("--temperature", type=float, default=10.0, help="sigmoid temperature")
    parser.add_argument("--limit", type=int, default=None, help="max number of tiles to load")
    parser.add_argument("--resume-patch", type=str, default=None, help="Path to a previously saved patch (.pt) to resume training")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    run_id = make_run_id(args.c)
    run_dir = os.path.join(SAVE_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    attacker = AdversarialPatch(
        checkpoint_path=args.c,
        device=device,
        patch_size=args.patch_size,
        px=args.px, py=args.py,
        threshold=args.threshold,
        temperature=args.temperature,
    ).to(device)

    if args.resume_patch is not None:
        print(f"Resuming from patch: {args.resume_patch}")
        loaded_patch = torch.load(os.path.join(SAVE_DIR, args.resume_patch), map_location=device)
        attacker.patch.data.copy_(loaded_patch)


    optimizer = torch.optim.Adam([attacker.patch], lr=args.l)
    scaler = GradScaler("cuda")

    full_dataset = PreTiledDataset(DATA_DIR, split="test")
    # Keep only tiles that contain objects
    obj_indices = []
    for i, fname in enumerate(full_dataset.images):
        mask = cv2.imread(os.path.join(full_dataset.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is not None and len(mask_to_centers(mask)) > 0:
            obj_indices.append(i)
    if args.limit:
        obj_indices = obj_indices[:args.limit]
    dataset = torch.utils.data.Subset(full_dataset, obj_indices)
    print(f"Tiles with objects: {len(dataset)}/{len(full_dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.b, shuffle=True, num_workers=2,
        collate_fn=segdino_collate,
    )

    print(f"Patch {args.patch_size}x{args.patch_size} at ({args.px}, {args.py})")
    print(f"Threshold: {args.threshold}, Temperature: {args.temperature}")
    print(f"Device: {device}, Epochs: {args.e}, LR: {args.l}")
    if args.limit:
        print(f"Limit: {args.limit} tiles")

    print("Precomputing clean predictions...")

    clean_preds = {}
    clean_gt = {}

    with torch.no_grad(), autocast("cuda"):
        for images, _, metas in loader:
            images = images.to(device)
            heatmaps_clean = attacker.predict_heatmaps(images)

            for i, meta in enumerate(metas):
                idx = meta["index"]
                hm_clean = heatmaps_clean[i, 0].float().cpu().numpy()
                pred_clean = find_peaks(hm_clean, threshold=args.threshold)

                clean_preds[idx] = pred_clean
                clean_gt[idx] = meta["centers"]

    # CSV log
    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "loss", "centres_clean", "centres_patched",
                      "recall_clean", "recall_patched",
                      "batch_size", "num_tiles"])
    n_tiles = len(dataset)
    best_recall_patched = float("inf")
    patch_path = os.path.join(run_dir, "patch.pt")
    image_path = os.path.join(run_dir, "patch.png")

    # prev_unattackable = None
    last_unattackable = []   # liste des sets des 10 derniers epochs
    WINDOW = 10

    for epoch in range(1, args.e + 1):
        
    # ============================
    #   TRAINING LOOP (FAST)
    # ============================
        total_loss = 0.0
        n_batches = 0
        
        for images, _, metas in tqdm(loader, desc=f"epoch {epoch}/{args.e}", leave=False):
            images = images.to(device)

            batch_centers = []
            for meta in metas:
                idx = meta["index"]
                centers = clean_gt[idx]      # liste de centres [(x,y), ...]
                batch_centers.append(centers)

            optimizer.zero_grad()

            # Forward en mixed precision
            with autocast("cuda"):
                soft_count = attacker(images, batch_centers)

            # Backward en mixed precision
            scaler.scale(soft_count).backward()

            # Optimisation
            scaler.step(optimizer)
            scaler.update()

            total_loss += soft_count.item()
            n_batches += 1

        avg_loss = total_loss / n_batches


        # ============================
        #   EVALUATION LOOP (SLOW)
        # ============================
        total_centers_clean = 0
        total_centers_patched = 0
        total_tp_clean = 0
        total_tp_patched = 0
        total_gt = 0
        unattackable = set()
        with torch.no_grad(), autocast("cuda"):
            for images, _, metas in tqdm(loader, desc=f"test {epoch}/{args.e}", leave=False):
                images = images.to(device)

                batch_centers = []
                for meta in metas:
                    idx = meta["index"]
                    centers = clean_gt[idx]
                    batch_centers.append(centers)

                # Clean
                heatmaps_clean = attacker.predict_heatmaps(images)

                # Patched
                heatmaps_patched = attacker.predict_heatmaps_patched(images, batch_centers)
                
                # Metrics
                for i, meta in enumerate(metas):
                    idx = meta["index"]
                    gt_centers = clean_gt[idx]

                    hm_clean = heatmaps_clean[i, 0].float().cpu().numpy()
                    hm_patched = heatmaps_patched[i, 0].float().cpu().numpy()


                    pred_clean = clean_preds[idx]
                    pred_patched = find_peaks(hm_patched, threshold=args.threshold)

                    total_centers_clean += len(pred_clean)
                    total_centers_patched += len(pred_patched)

                    if len(pred_patched) >= len(pred_clean):
                        unattackable.add(idx)

                    if gt_centers:
                        tp_c, _, _ = match_centers(gt_centers, pred_clean)
                        tp_p, _, _ = match_centers(gt_centers, pred_patched)
                        total_tp_clean += tp_c
                        total_tp_patched += tp_p
                        total_gt += len(gt_centers)

        recall_clean = total_tp_clean / total_gt if total_gt > 0 else 0.0
        recall_patched = total_tp_patched / total_gt if total_gt > 0 else 0.0

        # ============================
        #   PRINT METRICS
        # ============================
        tqdm.write(
            f"epoch {epoch:3d} | loss: {avg_loss:.3f} "
            f"| centres: {total_centers_clean}→{total_centers_patched} "
            f"| recall: {recall_clean:.2f}→{recall_patched:.2f}"
        )

        current_unattackable = unattackable

        last_unattackable.append(current_unattackable)

        # Garder seulement les 10 derniers
        if len(last_unattackable) > WINDOW:
            last_unattackable.pop(0)

        # Calculer la stabilité glissante
        if len(last_unattackable) > 1:
            # Intersection de tous les ensembles du buffer
            stable = set.union(*last_unattackable)
            num_stable = len(stable)

            tqdm.write(
                f"Images résistantes stables sur les {len(last_unattackable)} derniers epochs : {len(unattackable)}" # num_stable}"
            )
        else:
            tqdm.write("Pas encore assez d'epochs pour une mesure glissante")

        # prev_unattackable = current_unattackable

        # ============================
        #   SAVE BEST PATCH
        # ============================
        if recall_patched < best_recall_patched:
            best_recall_patched = recall_patched
            torch.save(attacker.patch.data.cpu(), patch_path)
            save_patch_image(attacker.patch.data.cpu(), image_path)
            tqdm.write(f"  → best patch saved (recall {recall_patched:.2f})")

        if total_centers_patched == 0:
            tqdm.write("Early stop: 0 centres detected with patch.")
            break

    csv_file.close()
    print(f"\nBest patch: {patch_path} (recall {best_recall_patched:.2f})")
    print(f"Patch image: {image_path}")
    print(f"Metrics CSV: {csv_path}")

if __name__ == "__main__":
    main()
