"""
SegDino Inference with Sliding Window.
Processes full images using tiled prediction and generates side-by-side visualizations.
"""
import argparse
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from model import SegDino
from dataset import DOTA_MEAN, DOTA_STD
import torchvision.transforms.functional as TF


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

    Args:
        model: SegDino model
        img: BGR image (H, W, 3)
        tile_size: Size of each tile
        stride: Overlap between tiles
        device: Device to run inference on
        batch_size: Number of tiles to process in parallel

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

        # Prepare batch
        for y, x in batch_coords:
            tile = img_padded[y:y+tile_size, x:x+tile_size]
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
            tile_t = TF.normalize(tile_t, DOTA_MEAN, DOTA_STD)
            batch_tiles.append(tile_t)

        # Stack and run inference
        batch_tensor = torch.stack(batch_tiles).to(device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits)

        # Scatter predictions back to map with Gaussian weights
        for j, (y, x) in enumerate(batch_coords):
            prob = probs[j].squeeze()
            prob_map[y:y+tile_size, x:x+tile_size] += prob * weight_map
            count_map[y:y+tile_size, x:x+tile_size] += weight_map

    # Normalize and crop to original size
    final_prob = prob_map / (count_map + 1e-8)
    final_prob = final_prob[:H, :W]
    return final_prob.cpu().numpy()


def draw_bboxes(img, mask, color, thickness=2):
    """
    Draw bounding boxes around mask regions.

    Args:
        img: BGR image (H, W, 3)
        mask: Binary mask (H, W)
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # Filter tiny noise
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def main():
    parser = argparse.ArgumentParser(description="SegDino Full Image Inference")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="segdata/DOTA/DOTA_PLANES/test",
                        help="Directory containing image/ and mask/ subdirectories")
    parser.add_argument("--save_dir", default="inference_results",
                        help="Output directory for visualizations")
    parser.add_argument("--model_size", default="vit-base",
                        choices=["vit-small", "vit-small-plus", "vit-base", "vit-large",
                                "vit-huge", "vit-giant", "vit-large-sat", "vit-giant-sat"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of images to process (default: all)")
    parser.add_argument("--tile_size", type=int, default=512,
                        help="Tile size for sliding window")
    parser.add_argument("--stride", type=int, default=384,
                        help="Stride for sliding window (smaller = more overlap)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of tiles to process in parallel")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Load model
    print(f"Loading model ({args.model_size})...")
    model = SegDino(model_size=args.model_size, freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Locate data directories (handle both 'image' and 'images' folder names)
    img_dir = os.path.join(args.data_dir, "image")
    if not os.path.exists(img_dir):
        img_dir = os.path.join(args.data_dir, "images")

    mask_dir = os.path.join(args.data_dir, "mask")
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(args.data_dir, "masks")

    if not os.path.exists(img_dir):
        print(f"Error: Image directory not found at {img_dir}")
        print(f"Expected structure: {args.data_dir}/image/ and {args.data_dir}/mask/")
        return

    # Get file list
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and not f.startswith("._")])
    if args.limit:
        files = files[:args.limit]

    print(f"Processing {len(files)} images from {img_dir}...")
    print(f"Tile size: {args.tile_size}, Stride: {args.stride}, Batch: {args.batch_size}")
    print(f"Results will be saved to: {args.save_dir}")

    for fname in tqdm(files, desc="Inference"):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {fname} (read error)")
            continue

        # Load ground truth mask (if available)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = (gt_mask > 127).astype(np.uint8)
        else:
            gt_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Run sliding window inference
        prob_map = predict_sliding_window(
            model, img,
            tile_size=args.tile_size,
            stride=args.stride,
            device=device,
            batch_size=args.batch_size
        )
        pred_mask = (prob_map > 0.5).astype(np.uint8)

        # Generate visualizations
        # Adjust thickness based on image size
        thickness = max(2, int(max(img.shape[:2]) / 500))

        vis_gt = draw_bboxes(img, gt_mask, (0, 255, 0), thickness=thickness)      # Green
        vis_pred = draw_bboxes(img, pred_mask, (0, 165, 255), thickness=thickness) # Orange

        # Combine side-by-side (left=GT, right=Pred)
        combined = cv2.hconcat([vis_gt, vis_pred])

        # Resize if too large (for easier viewing)
        H, W = combined.shape[:2]
        if max(H, W) > 4000:
            scale = 4000 / max(H, W)
            combined = cv2.resize(combined, (int(W*scale), int(H*scale)))

        # Save
        save_path = os.path.join(args.save_dir, fname)
        cv2.imwrite(save_path, combined)

    print(f"\nDone! Results saved to: {args.save_dir}")
    print(f"Visualization format: Left=Ground Truth (green), Right=Prediction (orange)")


if __name__ == "__main__":
    main()
