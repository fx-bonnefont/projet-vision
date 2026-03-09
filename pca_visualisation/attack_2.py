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

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PreTiledDataset, mask_to_centers
from inference import find_peaks, match_centers
from train import segdino_collate
from pca_visualisation.train_patch import AdversarialPatch

DATA_DIR = "data/DOTA/DOTA_PLANES_TILED"


def main():
    parser = argparse.ArgumentParser(description="Evaluate an adversarial patch on SegDino")
    parser.add_argument("-c", required=True, help="checkpoint path")
    parser.add_argument("--patch_dir", required=True, help="path to saved (patch.pt)")
    parser.add_argument("-b", type=int, default=8, help="batch size (higher = faster)")
    parser.add_argument("--px", type=int, default=0)
    parser.add_argument("--py", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    # GPU optimizations
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    attacker = AdversarialPatch(
        checkpoint_path=args.c,
        device=device,
        patch_size=args.patch_size,
        px=args.px, py=args.py,
        threshold=args.threshold,
    ).to(device)

    # Load patch
    patch_file_name = 'patch.pt'
    patch_dir = args.patch_dir
    patch_data = torch.load(os.path.join(patch_dir,patch_file_name), map_location=device)
    attacker.patch.data.copy_(patch_data)
    print(f"Loaded patch from {args.patch_dir}")

    

    # Dataset
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
    # 1) PRECOMPUTE CLEAN HEATMAPS (biggest speedup)
    # ---------------------------------------------------------
    print("Precomputing clean heatmaps...")
    clean_cache = {}

    with torch.no_grad():
        for images, _, metas, _ in tqdm(loader, desc="clean"):
            images = images.to(device)
            heatmaps_clean = attacker.predict_heatmaps(images)

            for i, meta in enumerate(metas):
                clean_cache[meta["id"]] = heatmaps_clean[i, 0].cpu()

    # ---------------------------------------------------------
    # 2) EVALUATION LOOP (patched only)
    # ---------------------------------------------------------
    print("Evaluating patched predictions...")

    image_stats = {}

    with torch.no_grad():
        for images, _, metas, _ in tqdm(loader, desc="patched"):
            images = images.to(device)
            heatmaps_patched = attacker.predict_heatmaps_patched(images)

            for i, meta in enumerate(metas):
                parent = meta["id"].split("_")[0]
                gt_centers = meta["centers"]
                if not gt_centers:
                    continue

                if parent not in image_stats:
                    image_stats[parent] = {"gt": 0, "tp_clean": 0, "tp_patched": 0}

                hm_clean = clean_cache[meta["id"]].numpy()
                hm_patched = heatmaps_patched[i, 0].cpu().numpy()

                pred_clean = find_peaks(hm_clean, threshold=args.threshold)
                pred_patched = find_peaks(hm_patched, threshold=args.threshold)

                tp_c, _, _ = match_centers(gt_centers, pred_clean)
                tp_p, _, _ = match_centers(gt_centers, pred_patched)

                image_stats[parent]["gt"] += len(gt_centers)
                image_stats[parent]["tp_clean"] += tp_c
                image_stats[parent]["tp_patched"] += tp_p

    # ---------------------------------------------------------
    # 3) PRINT RESULTS
    # ---------------------------------------------------------
    total_gt = total_tp_clean = total_tp_patched = 0
    header = f"{'Image':<10}| {'GT':>4} | {'clean':>5} | {'patched':>7} | recall clean → patched"
    print(f"\n{header}")
    print("-" * len(header))

    for parent in sorted(image_stats):
        s = image_stats[parent]
        r_clean = s["tp_clean"] / s["gt"]
        r_patched = s["tp_patched"] / s["gt"]
        print(f"{parent:<10}| {s['gt']:>4} | {s['tp_clean']:>5} | {s['tp_patched']:>7} |   {r_clean:.2f} → {r_patched:.2f}")

        total_gt += s["gt"]
        total_tp_clean += s["tp_clean"]
        total_tp_patched += s["tp_patched"]

    r_clean = total_tp_clean / total_gt
    r_patched = total_tp_patched / total_gt
    print("-" * len(header))
    print(f"{'TOTAL':<10}| {total_gt:>4} | {total_tp_clean:>5} | {total_tp_patched:>7} |   {r_clean:.2f} → {r_patched:.2f}")

    #patch_dir = os.path.dirname(args.patch_dir)
    csv_out = os.path.join(patch_dir, "evaluation_results.csv")

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Image", "GT", "TP_clean", "TP_patched", "Recall_clean", "Recall_patched"])

        # Per-image rows
        for parent in sorted(image_stats):
            s = image_stats[parent]
            r_clean = s["tp_clean"] / s["gt"]
            r_patched = s["tp_patched"] / s["gt"]
            writer.writerow([parent, s["gt"], s["tp_clean"], s["tp_patched"], f"{r_clean:.4f}", f"{r_patched:.4f}"])

        # Total row
        writer.writerow([])
        writer.writerow(["TOTAL", total_gt, total_tp_clean, total_tp_patched, f"{r_clean:.4f}", f"{r_patched:.4f}"])

    print(f"\nCSV saved to: {csv_out}")


if __name__ == "__main__":
    main()
