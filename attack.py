"""
Evaluate an adversarial patch on SegDino center detection.

Applies a trained patch to test tiles and reports per-image recall
degradation compared to clean predictions.
"""
import argparse

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PreTiledDataset, mask_to_centers
from inference import find_peaks, match_centers
from train import segdino_collate
from train_patch import AdversarialPatch

DATA_DIR = "segdata/DOTA/DOTA_PLANES_TILED"


def main():
    parser = argparse.ArgumentParser(description="Evaluate an adversarial patch on SegDino")
    parser.add_argument("-c", required=True, help="checkpoint path")
    parser.add_argument("--patch", required=True, help="path to saved patch .pt")
    parser.add_argument("-b", type=int, default=4, help="batch size")
    parser.add_argument("--px", type=int, default=0, help="patch x position")
    parser.add_argument("--py", type=int, default=0, help="patch y position")
    parser.add_argument("--patch-size", type=int, default=16, help="patch size")
    parser.add_argument("--threshold", type=float, default=0.3, help="detection threshold")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    attacker = AdversarialPatch(
        checkpoint_path=args.c,
        device=device,
        patch_size=args.patch_size,
        px=args.px, py=args.py,
        threshold=args.threshold,
    ).to(device)

    # Load trained patch
    patch_data = torch.load(args.patch, map_location=device, weights_only=True)
    attacker.patch.data.copy_(patch_data)
    print(f"Loaded patch from {args.patch} ({args.patch_size}x{args.patch_size} at ({args.px},{args.py}))")

    full_dataset = PreTiledDataset(DATA_DIR, split="test")
    obj_indices = []
    for i, fname in enumerate(full_dataset.images):
        mask = cv2.imread(full_dataset.mask_dir + "/" + fname, cv2.IMREAD_GRAYSCALE)
        if mask is not None and len(mask_to_centers(mask)) > 0:
            obj_indices.append(i)
    dataset = torch.utils.data.Subset(full_dataset, obj_indices)
    print(f"Tiles with objects: {len(dataset)}/{len(full_dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.b, shuffle=False, num_workers=2,
        collate_fn=segdino_collate,
    )

    # Per-image aggregation: {parent_image: {gt, tp_clean, tp_patched}}
    image_stats = {}

    for images, _, metas in tqdm(loader, desc="eval"):
        images = images.to(device)
        heatmaps_clean = attacker.predict_heatmaps(images)
        heatmaps_patched = attacker.predict_heatmaps_patched(images)

        for i, meta in enumerate(metas):
            gt_centers = meta["centers"]
            if not gt_centers:
                continue

            parent = meta["id"].split("_")[0]
            if parent not in image_stats:
                image_stats[parent] = {"gt": 0, "tp_clean": 0, "tp_patched": 0}

            hm_clean = heatmaps_clean[i, 0].cpu().numpy()
            hm_patched = heatmaps_patched[i, 0].cpu().numpy()

            pred_clean = find_peaks(hm_clean, threshold=args.threshold)
            pred_patched = find_peaks(hm_patched, threshold=args.threshold)

            tp_c, _, _ = match_centers(gt_centers, pred_clean)
            tp_p, _, _ = match_centers(gt_centers, pred_patched)

            image_stats[parent]["gt"] += len(gt_centers)
            image_stats[parent]["tp_clean"] += tp_c
            image_stats[parent]["tp_patched"] += tp_p

    # Display results
    total_gt = total_tp_clean = total_tp_patched = 0
    header = f"{'Image':<10}| {'GT':>4} | {'clean':>5} | {'patched':>7} | recall clean \u2192 patched"
    print(f"\n{header}")
    print("-" * len(header))

    for parent in sorted(image_stats):
        s = image_stats[parent]
        r_clean = s["tp_clean"] / s["gt"] if s["gt"] > 0 else 0.0
        r_patched = s["tp_patched"] / s["gt"] if s["gt"] > 0 else 0.0
        print(f"{parent:<10}| {s['gt']:>4} | {s['tp_clean']:>5} | {s['tp_patched']:>7} |   {r_clean:.2f} \u2192 {r_patched:.2f}")
        total_gt += s["gt"]
        total_tp_clean += s["tp_clean"]
        total_tp_patched += s["tp_patched"]

    r_clean = total_tp_clean / total_gt if total_gt > 0 else 0.0
    r_patched = total_tp_patched / total_gt if total_gt > 0 else 0.0
    print("-" * len(header))
    print(f"{'TOTAL':<10}| {total_gt:>4} | {total_tp_clean:>5} | {total_tp_patched:>7} |   {r_clean:.2f} \u2192 {r_patched:.2f}")


if __name__ == "__main__":
    main()
