"""
Inference script: visualize predictions and extract bounding boxes.
Supports multiple backbones via --backbone argument.
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from backbones import list_backbones
from model import load_model, get_image_size, DEFAULT_BACKBONE


# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def preprocess_image(image: np.ndarray, img_size: int) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize
    image = cv2.resize(image, (img_size, img_size))

    # Normalize (ImageNet stats)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    # To tensor [1, 3, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return image


def predict_tiled(model, image: np.ndarray, tile_size: int = 512, device: str = 'cpu') -> np.ndarray:
    """
    Run inference using sliding window (tiling) to preserve resolution.
    Follows SOTA approach (e.g. SAHI, U-Net Tiling) for large images.
    """
    h_img, w_img = image.shape[:2]
    
    # Initialize full-size probability mask
    full_prob_mask = np.zeros((h_img, w_img), dtype=np.float32)
    count_mask = np.zeros((h_img, w_img), dtype=np.float32) # For averaging overlaps
    
    # Config
    stride = tile_size # No overlap for speed, can reduce to tile_size//2 for better boundary fusion
    
    # Tiles Loop
    for y in range(0, h_img, stride):
        for x in range(0, w_img, stride):
            # Coordinates
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w_img)
            y2 = min(y + tile_size, h_img)
            
            # Dimensions of current tile (might be smaller at edges)
            h_tile = y2 - y1
            w_tile = x2 - x1
            
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            # Pad crop to fixed tile_size if needed (for batching/model req)
            # The model expects exactly tile_size (e.g. 512)
            pad_h = tile_size - h_tile
            pad_w = tile_size - w_tile
            
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                
            # Preprocess crop
            input_tensor = preprocess_image(crop, tile_size).to(device)
            
            # Inference
            model.eval()
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits) # [1, 1, 512, 512]
                
            # Extract valid region (remove padding)
            pred_crop = probs[0, 0].cpu().numpy()
            valid_pred = pred_crop[:h_tile, :w_tile]
            
            # Accumulate
            full_prob_mask[y1:y2, x1:x2] += valid_pred
            count_mask[y1:y2, x1:x2] += 1.0

    # Average overlaps (if any)
    full_prob_mask /= np.maximum(count_mask, 1.0)
    
    return full_prob_mask


def predict(model, image: np.ndarray, img_size: int, device: str = 'cpu',
            threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper that selects Tiled Inference if image is large.
    """
    h, w = image.shape[:2]
    
    if h > img_size or w > img_size:
        print(f"  Large image detected ({w}x{h}). Using Sliding Window Inference (Tile={img_size})...")
        prob_mask = predict_tiled(model, image, tile_size=img_size, device=device)
    else:
        # Standard resize for small images
        input_tensor = preprocess_image(image, img_size).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
        prob_mask = probs[0, 0].cpu().numpy()
        
        # Resize mask back to original size if it was resized
        if prob_mask.shape != (h, w):
             prob_mask = cv2.resize(prob_mask, (w, h))

    binary_mask = (prob_mask > threshold).astype(np.uint8)
    return prob_mask, binary_mask


def extract_bboxes(binary_mask: np.ndarray, orig_h: int, orig_w: int, min_area: int = 100) -> list:
    """
    Extract bounding boxes from binary mask using contours.

    Args:
        binary_mask: Binary mask at feature resolution
        orig_h, orig_w: Original image dimensions
        min_area: Minimum contour area to keep

    Returns:
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Scale mask to original resolution
    mask_scaled = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Find contours
    contours, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))

    return bboxes


def visualize(image: np.ndarray, prob_mask: np.ndarray, binary_mask: np.ndarray,
              bboxes: list, output_path: str):
    """
    Create visualization with original image, mask overlay, and bounding boxes.
    """
    orig_h, orig_w = image.shape[:2]

    # Scale masks to original resolution
    prob_mask_scaled = cv2.resize(prob_mask, (orig_w, orig_h))
    binary_mask_scaled = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create heatmap overlay
    heatmap = cv2.applyColorMap((prob_mask_scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)

    # Draw bounding boxes
    image_with_boxes = image_bgr.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create side-by-side visualization
    binary_vis = cv2.cvtColor(binary_mask_scaled * 255, cv2.COLOR_GRAY2BGR)

    top_row = np.hstack([image_bgr, overlay])
    bottom_row = np.hstack([binary_vis, image_with_boxes])
    combined = np.vstack([top_row, bottom_row])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Heatmap', (orig_w + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Binary Mask', (10, orig_h + 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f'Detections ({len(bboxes)})', (orig_w + 10, orig_h + 30), font, 1, (255, 255, 255), 2)

    cv2.imwrite(output_path, combined)
    print(f"Visualization saved to {output_path}")


def run_inference(
    model_path: str,
    image_dir: str,
    output_dir: str = './inference_output',
    backbone_name: str = DEFAULT_BACKBONE,
    num_images: int = 5,
    threshold: float = 0.5,
    device: str | None = None
):
    """Run inference on random images from directory."""

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Get image size for backbone
    img_size = get_image_size(backbone_name)

    # Load model
    model = load_model(
        weights_path=model_path,
        backbone_name=backbone_name,
        device=device
    )

    # Find images
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # Select random images
    selected = random.sample(image_files, min(num_images, len(image_files)))

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Process each image
    for img_path in selected:
        print(f"Processing {img_path.name}...")

        # Load image as RGB
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Could not load image, skipping")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        prob_mask, binary_mask = predict(model, image_rgb, img_size, device, threshold)

        # Extract bounding boxes
        bboxes = extract_bboxes(binary_mask, image.shape[0], image.shape[1])
        print(f"  Found {len(bboxes)} objects")

        # Visualize
        output_path = output_dir / f"pred_{img_path.stem}.png"
        visualize(image_rgb, prob_mask, binary_mask, bboxes, str(output_path))


def main():
    parser = argparse.ArgumentParser(description='Run inference and visualize predictions')
    parser.add_argument('--data', type=str, default=None,
                        help='Root directory containing "images/test" (preferred over --images)')
    parser.add_argument('--model', type=str, default='model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--images', type=str, default=None,
                        help='Specific directory containing images (optional override)')
    parser.add_argument('--output', type=str, default='./inference_output',
                        help='Output directory for visualizations')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=list_backbones(),
                        help='Backbone used during training (auto-detected if saved in model)')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of random images to process')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, mps, cpu)')

    args = parser.parse_args()

    # Determine image directory
    if args.data:
        # Default to test images in data structure
        args.images = os.path.join(args.data, 'images', 'test')
        if not os.path.exists(args.images):
             # Fallback to train if test doesn't exist? Or just error?
             # Let's try train if test not found, just in case
             if os.path.exists(os.path.join(args.data, 'images', 'train')):
                 args.images = os.path.join(args.data, 'images', 'train')
    
    if not args.images:
        parser.error("You must specify either --data (containing images/test) or --images")

    run_inference(
        model_path=args.model,
        image_dir=args.images,
        output_dir=args.output,
        backbone_name=args.backbone,
        num_images=args.num_images,
        threshold=args.threshold,
        device=args.device
    )


if __name__ == '__main__':
    main()
