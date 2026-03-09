"""
SegDino Inference for center detection.
"""
import argparse
import csv
import os
import re

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
from tqdm import tqdm

from dataset import DOTA_MEAN, DOTA_STD, mask_to_centers
from model import SegDino

# Hardcoded configuration
DATA_DIR = "data/DOTA/DOTA_PLANES/train"
SAVE_DIR = "inference_one_picture"
TILE_SIZE = 512
STRIDE = 384
THRESHOLD = 0.3
MATCH_RADIUS = 20.0


def extract_checkpoint_id(ckpt_path: str) -> str:
    """Extract identifier from checkpoint path."""
    basename = os.path.basename(ckpt_path)
    name = os.path.splitext(basename)[0]
    name = re.sub(r'^(best_model_|checkpoint_)', '', name)
    return name


def match_centers(gt_centers: list, pred_centers: list, match_radius: float = MATCH_RADIUS) -> tuple:
    """Match predicted centers to ground truth using nearest neighbor."""
    if len(gt_centers) == 0 and len(pred_centers) == 0:
        return 0, 0, 0
    if len(gt_centers) == 0:
        return 0, len(pred_centers), 0
    if len(pred_centers) == 0:
        return 0, 0, len(gt_centers)

    gt_arr = np.array(gt_centers)
    pred_arr = np.array(pred_centers)

    distances = cdist(pred_arr, gt_arr)

    matched_gt = set()
    tp = 0
    for i in range(len(pred_centers)):
        min_dist_idx = np.argmin(distances[i])
        if distances[i, min_dist_idx] <= match_radius and min_dist_idx not in matched_gt:
            tp += 1
            matched_gt.add(min_dist_idx)

    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp

    return tp, fp, fn


def compute_metrics_at_threshold(all_gt_centers: list, all_heatmaps: list, threshold: float) -> dict:
    """Compute classification metrics at a given threshold."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for gt_centers, heatmap in zip(all_gt_centers, all_heatmaps):
        pred_centers = find_peaks(heatmap, threshold=threshold)
        tp, fp, fn = match_centers(gt_centers, pred_centers)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def save_metrics_csv(save_dir: str, metrics_at_thresholds: list):
    """Save classification metrics to CSV."""
    csv_path = os.path.join(save_dir, 'metrics.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# SegDino Inference Metrics'])
        writer.writerow([])

        # Find best threshold by F1
        best_m = max(metrics_at_thresholds, key=lambda x: x['f1'])
        writer.writerow(['## Best threshold by F1:', f"{best_m['threshold']:.2f}"])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', f"{best_m['precision']:.4f}"])
        writer.writerow(['Recall', f"{best_m['recall']:.4f}"])
        writer.writerow(['F1 Score', f"{best_m['f1']:.4f}"])
        writer.writerow(['TP', best_m['tp']])
        writer.writerow(['FP', best_m['fp']])
        writer.writerow(['FN', best_m['fn']])
        writer.writerow([])

        writer.writerow(['## Threshold sweep'])
        writer.writerow(['Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
        for m in metrics_at_thresholds:
            writer.writerow([
                f"{m['threshold']:.2f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                m['tp'], m['fp'], m['fn']
            ])

    return csv_path


def get_gaussian_weight_map(size, device):
    """Gaussian weight map for smooth stitching."""
    sigma = size / 2.0
    ax = torch.arange(size, device=device) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.max()


def predict_sliding_window(model, img, device='cuda'):
    """Inference using sliding window with Gaussian blending."""
    H, W = img.shape[:2]

    pad_h = (TILE_SIZE - H % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - W % TILE_SIZE) % TILE_SIZE
    if (H + pad_h - TILE_SIZE) % STRIDE != 0:
        pad_h += STRIDE - ((H + pad_h - TILE_SIZE) % STRIDE)
    if (W + pad_w - TILE_SIZE) % STRIDE != 0:
        pad_w += STRIDE - ((W + pad_w - TILE_SIZE) % STRIDE)

    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    H_pad, W_pad = img_padded.shape[:2]

    weight_map = get_gaussian_weight_map(TILE_SIZE, device)
    prob_map = torch.zeros((H_pad, W_pad), device=device)
    count_map = torch.zeros((H_pad, W_pad), device=device)

    coords = []
    for y in range(0, H_pad - TILE_SIZE + 1, STRIDE):
        for x in range(0, W_pad - TILE_SIZE + 1, STRIDE):
            coords.append((y, x))

    for y, x in coords:
        tile = img_padded[y:y+TILE_SIZE, x:x+TILE_SIZE]
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
        tile_t = TF.normalize(tile_t, DOTA_MEAN, DOTA_STD)
        tile_t = tile_t.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tile_t)
            prob = torch.sigmoid(logits).squeeze()

        prob_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += prob * weight_map
        count_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += weight_map

    final_prob = prob_map / (count_map + 1e-8)
    final_prob = final_prob[:H, :W]
    return final_prob.cpu().numpy()


def find_peaks(heatmap: np.ndarray, threshold: float = THRESHOLD, min_distance: int = 10) -> list:
    """Find local maxima in heatmap."""
    local_max = maximum_filter(heatmap, size=min_distance)
    peaks_mask = (heatmap == local_max) & (heatmap > threshold)
    coords = np.argwhere(peaks_mask)
    return [(int(x), int(y)) for y, x in coords]


def draw_centers(img, centers: list, color, radius: int = 8, thickness: int = 2):
    """Draw tapered crosses at center locations."""
    out = img.copy()
    inner_radius = max(2, radius // 4)
    inner_thickness = max(1, thickness // 3)

    for cx, cy in centers:
        # Outer segments (thick)
        cv2.line(out, (cx - radius, cy), (cx - inner_radius, cy), color, thickness)
        cv2.line(out, (cx + inner_radius, cy), (cx + radius, cy), color, thickness)
        cv2.line(out, (cx, cy - radius), (cx, cy - inner_radius), color, thickness)
        cv2.line(out, (cx, cy + inner_radius), (cx, cy + radius), color, thickness)

        # Inner segments (thin)
        cv2.line(out, (cx - inner_radius, cy), (cx + inner_radius, cy), color, inner_thickness)
        cv2.line(out, (cx, cy - inner_radius), (cx, cy + inner_radius), color, inner_thickness)

    return out


def load_checkpoint(ckpt_path: str, device: str):
    """Load checkpoint and instantiate model."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        model_size = config["model_size"]
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("Legacy checkpoint format not supported. Please retrain with the new system.")

    import transformers
    transformers.logging.set_verbosity_error()
    model = SegDino(model_size=model_size, freeze_backbone=True).to(device)
    transformers.logging.set_verbosity_warning()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="SegDino Inference")
    parser.add_argument("-c", required=True, help="checkpoint path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    ckpt_id = extract_checkpoint_id(args.c)
    save_dir = os.path.join(SAVE_DIR, ckpt_id)
    os.makedirs(save_dir, exist_ok=True)
    model = load_checkpoint(args.c, device)

    # Locate data directories
    img_dir = os.path.join(DATA_DIR, "image")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(DATA_DIR, "images")

    mask_dir = os.path.join(DATA_DIR, "mask")
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(DATA_DIR, "masks")

    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at {img_dir}")
        return

    # files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and not f.startswith("._")])

    all_gt_centers = []
    all_heatmaps = []
    # Ici modif EL
    # for fname in tqdm(files, desc="Inference"):
    fname = "P0183.png"
    img_path = os.path.join(img_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {fname}")
        #continue

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is not None:
        gt_mask = (gt_mask > 127).astype(np.uint8)
    else:
        gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    prob_map = predict_sliding_window(model, img, device=device)

    thickness = max(2, int(max(img.shape[:2]) / 500))
    radius = max(8, int(max(img.shape[:2]) / 200))

    gt_centers = mask_to_centers(gt_mask * 255)
    pred_centers = find_peaks(prob_map, threshold=THRESHOLD)

    all_gt_centers.append(gt_centers)
    all_heatmaps.append(prob_map)

    vis_gt = draw_centers(img, gt_centers, (0, 255, 0), radius=radius, thickness=thickness)
    vis_pred = draw_centers(img, pred_centers, (0, 165, 255), radius=radius, thickness=thickness)

    cv2.putText(vis_gt, f"GT: {len(gt_centers)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(vis_pred, f"Pred: {len(pred_centers)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

    combined = cv2.hconcat([vis_gt, vis_pred])

    H, W = combined.shape[:2]
    if max(H, W) > 4000:
        scale = 4000 / max(H, W)
        combined = cv2.resize(combined, (int(W * scale), int(H * scale)))

    cv2.imwrite(os.path.join(save_dir, fname), combined)
    # fin modif EL
    print(f"\nVisualizations saved to: {save_dir}")

    # Compute metrics
    # if len(all_heatmaps) > 0:
    #     print("\nComputing metrics...")
    #     thresholds = [i / 10.0 for i in range(1, 10)]
    #     metrics_at_thresholds = []

    #     for thresh in tqdm(thresholds, desc="Threshold sweep"):
    #         metrics = compute_metrics_at_threshold(all_gt_centers, all_heatmaps, threshold=thresh)
    #         metrics_at_thresholds.append(metrics)

    #     csv_path = save_metrics_csv(save_dir, metrics_at_thresholds)
    #     print(f"Metrics saved to: {csv_path}")

    #     best_m = max(metrics_at_thresholds, key=lambda x: x['f1'])
    #     print(f"\n=== Best results (threshold={best_m['threshold']:.2f}) ===")
    #     print(f"  Precision: {best_m['precision']:.4f}")
    #     print(f"  Recall:    {best_m['recall']:.4f}")
    #     print(f"  F1 Score:  {best_m['f1']:.4f}")
    #     print(f"  TP: {best_m['tp']}, FP: {best_m['fp']}, FN: {best_m['fn']}")


if __name__ == "__main__":
    main()
