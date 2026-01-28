"""
Inference script for Multi-Class Segmentation (16 classes).
Visualizes predictions with class-specific colors and labels.
"""
import argparse
import os
import random
from pathlib import Path
import math

import cv2
import numpy as np
import torch

from segmentation import (
    list_backbones,
    load_model,
    get_image_size,
    DEFAULT_BACKBONE,
    DOTA_CLASSES
)

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Generate colors for classes (skip background)
def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        if i == 0:
            colors.append((0, 0, 0)) # Background: Black
        else:
            # Hue based generation
            hue = int(180 * (i / num_classes))
            res = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(res[0]), int(res[1]), int(res[2])))
    return colors

CLASS_COLORS = generate_colors(len(DOTA_CLASSES))


def preprocess_image(image: np.ndarray, img_size: int) -> torch.Tensor:
    """Preprocess image for model input."""
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    return image


def predict_tiled(model, image: np.ndarray, tile_size: int = 512, device: str = 'cpu') -> np.ndarray:
    """
    Run inference using sliding window (tiling) for Multi-Class.
    Returns: Probabilities (C, H, W)
    """
    h_img, w_img = image.shape[:2]
    num_classes = len(DOTA_CLASSES) 
    
    # Initialize full-size probability mask
    full_prob_mask = np.zeros((num_classes, h_img, w_img), dtype=np.float32)
    count_mask = np.zeros((h_img, w_img), dtype=np.float32)
    
    stride = tile_size # No overlap for speed
    
    model.eval()
    
    for y in range(0, h_img, stride):
        for x in range(0, w_img, stride):
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w_img)
            y2 = min(y + tile_size, h_img)
            
            h_tile = y2 - y1
            w_tile = x2 - x1
            
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            # Pad
            pad_h = tile_size - h_tile
            pad_w = tile_size - w_tile
            
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                
            input_tensor = preprocess_image(crop, tile_size).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor) # [1, C, H, W]
                probs = torch.softmax(logits, dim=1) 
                
            # Extract valid region
            pred_crop = probs[0].cpu().numpy() # [C, H, W]
            valid_pred = pred_crop[:, :h_tile, :w_tile]
            
            full_prob_mask[:, y1:y2, x1:x2] += valid_pred
            count_mask[y1:y2, x1:x2] += 1.0

    # Average
    full_prob_mask /= np.maximum(count_mask, 1.0)
    
    return full_prob_mask


def predict(model, image: np.ndarray, img_size: int, device: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Multi-Class Prediction.
    Returns:
        prob_mask: (C, H, W) float
        pred_mask: (H, W) int (Class IDs)
    """
    h, w = image.shape[:2]
    
    if h > img_size or w > img_size:
        print(f"  Large image detected ({w}x{h}). Using Sliding Window Inference (Tile={img_size})...")
        prob_mask = predict_tiled(model, image, tile_size=img_size, device=device)
    else:
        input_tensor = preprocess_image(image, img_size).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1) # [1, C, H, W]
        prob_mask = probs[0].cpu().numpy() # [C, H, W]
        
        # Resize if needed (bilinear for probs)
        if prob_mask.shape[-2:] != (h, w):
             # Resize each channel? Or transpose -> resize -> transpose
             prob_mask_t = prob_mask.transpose(1, 2, 0) # H, W, C
             prob_mask_t = cv2.resize(prob_mask_t, (w, h))
             prob_mask = prob_mask_t.transpose(2, 0, 1) # C, H, W

    # Argmax for class ID
    pred_mask = np.argmax(prob_mask, axis=0).astype(np.uint8)
    
    return prob_mask, pred_mask


def extract_bboxes(pred_mask: np.ndarray, orig_h: int, orig_w: int, min_area: int = 100) -> list:
    """
    Extract bounding boxes for each class separately.
    Returns:
        List of dicts: {'box': (x,y,w,h), 'class_id': int, 'class_name': str}
    """
    # Resize mask to original if needed
    if pred_mask.shape != (orig_h, orig_w):
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    bboxes = []
    
    # Iterate over all object classes (skip 0=background)
    for class_id in range(1, len(DOTA_CLASSES)):
        # Binary mask for this class
        class_mask = (pred_mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
            
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append({
                'box': (x, y, w, h),
                'class_id': class_id,
                'class_name': DOTA_CLASSES[class_id]
            })

    return bboxes


def visualize(image: np.ndarray, prob_mask: np.ndarray, pred_mask: np.ndarray,
              bboxes: list, output_path: str, debug_image: np.ndarray | None = None):
    """
    Visualize Multi-Class predictions.
    """
    orig_h, orig_w = image.shape[:2]

    # 1. Prediction Mask Visualization (Color Coded)
    # Map class IDs to colors
    pred_vis = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    # Iterate and color (inefficient loop but safe for now)
    # faster:
    pred_mask_scaled = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    for cid in range(1, len(DOTA_CLASSES)):
        pred_vis[pred_mask_scaled == cid] = CLASS_COLORS[cid]

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 2. Heatmap: Sum of object probabilities (Foreground Confidence)
    # prob_mask is (C, H, W). Sum channels 1..15
    fg_prob = np.sum(prob_mask[1:], axis=0)
    fg_prob = np.clip(fg_prob, 0, 1)
    fg_prob_scaled = cv2.resize(fg_prob, (orig_w, orig_h))
    heatmap = cv2.applyColorMap((fg_prob_scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)

    # 3. Draw bounding boxes (Prediction)
    image_with_boxes = image_bgr.copy()
    for item in bboxes:
        x, y, w, h = item['box']
        cid = item['class_id']
        color = CLASS_COLORS[cid]
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        # Label
        label = item['class_name']
        cv2.putText(image_with_boxes, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Selection for Top-Left (Ground Truth)
    if debug_image is not None:
        if debug_image.shape[:2] != (orig_h, orig_w):
            debug_image = cv2.resize(debug_image, (orig_w, orig_h))
        top_left_img = debug_image
        tl_label = 'Ground Truth (Debug)'
    else:
        top_left_img = image_bgr
        tl_label = 'Original (No GT)'

    # Layout:
    # GT | Heatmap (FG Prob)
    # Mask (Colored) | Pred (Boxes)
    top_row = np.hstack([top_left_img, overlay])
    bottom_row = np.hstack([pred_vis, image_with_boxes])
    combined = np.vstack([top_row, bottom_row])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thick = 2
    color = (255, 255, 255)
    
    cv2.putText(combined, tl_label, (10, 30), font, scale, color, thick)
    cv2.putText(combined, 'FG Confidence Heatmap', (orig_w + 10, 30), font, scale, color, thick)
    cv2.putText(combined, 'Predicted Class Mask', (10, orig_h + 30), font, scale, color, thick)
    cv2.putText(combined, f'Prediction ({len(bboxes)} objects)', (orig_w + 10, orig_h + 30), font, scale, color, thick)

    cv2.imwrite(output_path, combined)
    print(f"Visualization saved to {output_path}")


def run_inference(
    model_path: str,
    image_dir: str,
    output_dir: str = './inference_output',
    backbone_name: str = DEFAULT_BACKBONE,
    num_images: int = 5,
    threshold: float = 0.5, # Unused in Multi-Class (Using Argmax)
    device: str | None = None
):
    """Run inference on random images from directory."""

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    img_size = get_image_size(backbone_name)

    model = load_model(
        weights_path=model_path,
        backbone_name=backbone_name,
        device=device
    )

    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    selected = random.sample(image_files, min(num_images, len(image_files)))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for img_path in selected:
        print(f"Processing {img_path.name}...")

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prob_mask, pred_mask = predict(model, image_rgb, img_size, device)

        bboxes = extract_bboxes(pred_mask, image.shape[0], image.shape[1])
        print(f"  Found {len(bboxes)} objects")

        # Try to find debug image (Ground Truth)
        debug_dir_path = img_path.parent.as_posix().replace('/images/', '/debug/')
        debug_path = Path(debug_dir_path) / f"visu_{img_path.name}"
        
        debug_image = None
        if debug_path.exists():
            debug_image = cv2.imread(str(debug_path))
            print(f"  Loaded GT debug image: {debug_path.name}")
        else:
             # Fallback
            debug_path_noprefix = Path(debug_dir_path) / img_path.name
            if debug_path_noprefix.exists():
                 debug_image = cv2.imread(str(debug_path_noprefix))

        output_path = output_dir / f"pred_{img_path.stem}.png"
        visualize(image_rgb, prob_mask, pred_mask, bboxes, str(output_path), debug_image)


def main():
    parser = argparse.ArgumentParser(description='Run Multi-Class inference')
    parser.add_argument('--data', type=str, default=None,
                        help='Root directory containing "images/test"')
    parser.add_argument('--model', type=str, default='model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--images', type=str, default=None,
                        help='Specific directory containing images')
    parser.add_argument('--output', type=str, default='./inference_output',
                        help='Output directory')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=list_backbones(),
                        help='Backbone name')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of random images')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, mps, cpu)')

    args = parser.parse_args()

    # Determine image directory
    if args.data:
        args.images = os.path.join(args.data, 'images', 'test')
        if not os.path.exists(args.images):
             if os.path.exists(os.path.join(args.data, 'images', 'train')):
                 args.images = os.path.join(args.data, 'images', 'train')
    
    if not args.images:
        parser.error("You must specify either --data or --images")

    run_inference(
        model_path=args.model,
        image_dir=args.images,
        output_dir=args.output,
        backbone_name=args.backbone,
        num_images=args.num_images,
        device=args.device
    )


if __name__ == '__main__':
    main()
