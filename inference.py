"""
SegDino Inference with Sliding Window.

Supports both segmentation (mask) and center detection modes.
Processes full images using tiled prediction and generates side-by-side visualizations.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import maximum_filter
from tqdm import tqdm

from dataset import DOTA_MEAN, DOTA_STD, mask_to_centers
from model import DECODERS, SegDino


def get_gaussian_weight_map(size, device):
    """Generates a Gaussian weight map for smooth stitching."""
    sigma = size / 2.0
    ax = torch.arange(size, device=device) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.max()


def predict_sliding_window(model, img, tile_size=512, stride=384, device='cuda', batch_size=16):
    """
    Inference on a large image using batched sliding window with Gaussian blending.

    Returns:
        Probability map (H, W) in [0, 1]
    """
    H, W = img.shape[:2]

    # Padding to ensure full coverage
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if (H + pad_h - tile_size) % stride != 0:
        pad_h += stride - ((H + pad_h - tile_size) % stride)
    if (W + pad_w - tile_size) % stride != 0:
        pad_w += stride - ((W + pad_w - tile_size) % stride)

    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    H_pad, W_pad = img_padded.shape[:2]

    # Gaussian weight map for blending
    weight_map = get_gaussian_weight_map(tile_size, device)

    # Accumulators
    prob_map = torch.zeros((H_pad, W_pad), device=device)
    count_map = torch.zeros((H_pad, W_pad), device=device)

    # Collect all tile coordinates
    coords = []
    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            coords.append((y, x))

    # Process in batches
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]
        batch_tiles = []

        for y, x in batch_coords:
            tile = img_padded[y:y+tile_size, x:x+tile_size]
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
            tile_t = TF.normalize(tile_t, DOTA_MEAN, DOTA_STD)
            batch_tiles.append(tile_t)

        batch_tensor = torch.stack(batch_tiles).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits)

        for j, (y, x) in enumerate(batch_coords):
            prob = probs[j].squeeze()
            prob_map[y:y+tile_size, x:x+tile_size] += prob * weight_map
            count_map[y:y+tile_size, x:x+tile_size] += weight_map

    final_prob = prob_map / (count_map + 1e-8)
    final_prob = final_prob[:H, :W]
    return final_prob.cpu().numpy()


def find_peaks(heatmap: np.ndarray, threshold: float = 0.3, min_distance: int = 10) -> list:
    """
    Find local maxima in a heatmap.

    Args:
        heatmap: 2D array with values in [0, 1]
        threshold: Minimum peak value
        min_distance: Minimum distance between peaks

    Returns:
        List of (x, y) peak coordinates
    """
    # Apply local maximum filter
    local_max = maximum_filter(heatmap, size=min_distance)

    # Find peaks: local maxima above threshold
    peaks_mask = (heatmap == local_max) & (heatmap > threshold)

    # Get coordinates
    coords = np.argwhere(peaks_mask)
    peaks = [(int(x), int(y)) for y, x in coords]  # Convert to (x, y)

    return peaks


def draw_bboxes(img, mask, color, thickness=2):
    """Draw bounding boxes around mask regions (for segmentation mode)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_centers(img, centers: list, color, radius: int = 8, thickness: int = 2):
    """
    Draw crosses at center locations (for center detection mode).

    Args:
        img: BGR image
        centers: List of (x, y) coordinates
        color: BGR color tuple
        radius: Size of the cross
        thickness: Line thickness

    Returns:
        Image with crosses drawn
    """
    out = img.copy()
    for cx, cy in centers:
        # Draw cross
        cv2.line(out, (cx - radius, cy), (cx + radius, cy), color, thickness)
        cv2.line(out, (cx, cy - radius), (cx, cy + radius), color, thickness)
        # Draw circle around
        cv2.circle(out, (cx, cy), radius, color, thickness)
    return out


def load_checkpoint(ckpt_path: str, device: str, model_size_override: str = None, decoder_override: str = None):
    """
    Load checkpoint and instantiate model with correct config.

    Returns:
        tuple: (model, config_dict)
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    # New format: checkpoint contains config
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        model_size = model_size_override or config["model_size"]
        decoder_name = decoder_override or config["decoder_name"]
        target_type = config.get("target_type", "mask")
        state_dict = checkpoint["model_state_dict"]
        print(f"[Checkpoint] backbone={model_size}, decoder={decoder_name}, target={target_type}")
    # Legacy format
    else:
        if model_size_override is None:
            raise ValueError("Legacy checkpoint. Please specify --model_size.")
        model_size = model_size_override
        decoder_name = decoder_override or "heavy_unet"
        target_type = "mask"
        state_dict = checkpoint
        config = {"model_size": model_size, "decoder_name": decoder_name, "target_type": target_type}
        print(f"[Checkpoint] Legacy format: backbone={model_size}, decoder={decoder_name}")

    model = SegDino(
        model_size=model_size,
        decoder_name=decoder_name,
        freeze_backbone=True
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="SegDino Inference")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="segdata/DOTA/DOTA_PLANES/test")
    parser.add_argument("--save_dir", default="inference_results")
    parser.add_argument("--model_size", default=None,
                        choices=["small", "small-plus", "base", "large", "huge", "giant", "large-sat", "giant-sat"])
    parser.add_argument("--decoder", default=None, choices=list(DECODERS.keys()))
    parser.add_argument("--target_type", default=None, choices=["mask", "center"],
                        help="Override target type (auto-detected from checkpoint)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection threshold (for center mode)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    model, config = load_checkpoint(
        args.ckpt, device,
        model_size_override=args.model_size,
        decoder_override=args.decoder
    )

    target_type = args.target_type or config.get("target_type", "mask")
    print(f"Inference mode: {target_type}")

    # Locate data directories
    img_dir = os.path.join(args.data_dir, "image")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(args.data_dir, "images")

    mask_dir = os.path.join(args.data_dir, "mask")
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(args.data_dir, "masks")

    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at {img_dir}")
        return

    # Get file list
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and not f.startswith("._")])
    if args.limit:
        files = files[:args.limit]

    print(f"Processing {len(files)} images...")
    print(f"Tile: {args.tile_size}, Stride: {args.stride}, Threshold: {args.threshold}")

    for fname in tqdm(files, desc="Inference"):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {fname}")
            continue

        # Load ground truth
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = (gt_mask > 127).astype(np.uint8)
        else:
            gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Run inference
        prob_map = predict_sliding_window(
            model, img,
            tile_size=args.tile_size,
            stride=args.stride,
            device=device,
            batch_size=args.batch_size
        )

        # Visualization depends on target type
        thickness = max(2, int(max(img.shape[:2]) / 500))
        radius = max(8, int(max(img.shape[:2]) / 200))

        if target_type == "center":
            # Center detection mode: find peaks and draw crosses
            gt_centers = mask_to_centers(gt_mask * 255)
            pred_centers = find_peaks(prob_map, threshold=args.threshold)

            vis_gt = draw_centers(img, gt_centers, (0, 255, 0), radius=radius, thickness=thickness)
            vis_pred = draw_centers(img, pred_centers, (0, 165, 255), radius=radius, thickness=thickness)

            # Add count overlay
            cv2.putText(vis_gt, f"GT: {len(gt_centers)}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(vis_pred, f"Pred: {len(pred_centers)}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        else:
            # Segmentation mode: draw bounding boxes
            pred_mask = (prob_map > 0.5).astype(np.uint8)
            vis_gt = draw_bboxes(img, gt_mask, (0, 255, 0), thickness=thickness)
            vis_pred = draw_bboxes(img, pred_mask, (0, 165, 255), thickness=thickness)

        # Combine side-by-side
        combined = cv2.hconcat([vis_gt, vis_pred])

        # Resize if needed
        H, W = combined.shape[:2]
        if max(H, W) > 4000:
            scale = 4000 / max(H, W)
            combined = cv2.resize(combined, (int(W * scale), int(H * scale)))

        cv2.imwrite(os.path.join(args.save_dir, fname), combined)

    mode_desc = "crosses at centers" if target_type == "center" else "bounding boxes"
    print(f"\nDone! Visualizations saved to: {args.save_dir}")
    print(f"Format: Left=GT (green {mode_desc}), Right=Pred (orange {mode_desc})")


if __name__ == "__main__":
    main()
