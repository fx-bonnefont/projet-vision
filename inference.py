"""
SegDino Inference Script.
Generates visualizations (GT vs Pred) for tiled dataset.
"""
import argparse
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from model import SegDino
from dataset import PreTiledDataset, DOTA_MEAN, DOTA_STD

def denormalize(tensor):
    mean = torch.tensor(DOTA_MEAN).view(3, 1, 1)
    std = torch.tensor(DOTA_STD).view(3, 1, 1)
    return tensor * std + mean

def draw_bboxes(img, mask, color, thickness=2):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default="/Volumes/X9Pro/DOTA_PLANES_TILED")
    parser.add_argument("--save_dir", default="inference_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Model
    model = SegDino(model_size="1", freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Data
    ds = PreTiledDataset(args.data_dir, "test")
    # Limit to first 100 for speed
    indices = range(min(100, len(ds)))

    print(f"Running inference on {len(indices)} images...")

    for i in tqdm(indices):
        img_t, mask_t, meta = ds[i]
        
        # Inference
        with torch.no_grad():
            logits = model(img_t.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()

        # Visualization
        img_denorm = denormalize(img_t)
        img_np = np.clip(img_denorm.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        mask_gt = mask_t.squeeze().numpy()
        mask_pred = pred.squeeze().numpy()

        # Draw
        vis_gt = draw_bboxes(img_bgr, mask_gt, (0, 255, 0))   # Green GT
        vis_pred = draw_bboxes(img_bgr, mask_pred, (0, 165, 255)) # Orange Pred
        
        combined = cv2.hconcat([vis_gt, vis_pred])
        cv2.imwrite(os.path.join(args.save_dir, f"{meta['id']}"), combined)

    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()