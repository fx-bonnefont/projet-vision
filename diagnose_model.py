
import torch
import os
import cv2
import numpy as np
import argparse
from model import load_model, get_image_size
from dataset import SegmentationDataset

def diagnose(model_path, data_dir, sample_idx=0):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load Model
    print(f"Loading model: {model_path}")
    try:
        model = load_model(model_path, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Dataset
    print(f"Loading dataset from {data_dir}...")
    # Use image size from model backbone
    img_size = 512 # DINOv3 default
    
    dataset = SegmentationDataset(
        image_dir=os.path.join(data_dir, 'images', 'train'),
        label_dir=os.path.join(data_dir, 'labels', 'train'),
        img_size=img_size
    )
    
    # Get Sample
    if sample_idx >= len(dataset):
        sample_idx = 0
    
    print(f"Analyzing Sample {sample_idx}...")
    image, mask = dataset[sample_idx]
    fname = f"Sample_{sample_idx}" # Placeholder name
    
    # Prepare batch
    x = image.unsqueeze(0).to(device) # [1, 3, H, W]
    y_true = mask.unsqueeze(0).to(device) # [1, 1, H, W]
    
    # Forward
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    
    # Stats
    logits_np = logits.cpu().numpy()
    probs_np = probs.cpu().numpy()
    mask_np = mask.numpy()
    
    print("\n--- STATISTICS ---")
    print(f"Logits: Min={logits_np.min():.4f}, Max={logits_np.max():.4f}, Mean={logits_np.mean():.4f}")
    print(f"Probs : Min={probs_np.min():.4f}, Max={probs_np.max():.4f}, Mean={probs_np.mean():.4f}")
    print(f"GT Mask: Sum={mask_np.sum()} pixels (Content ratio: {mask_np.mean():.4f})")
    
    # Check if dead
    if probs_np.max() < 0.01:
        print("\n⚠️  DIAGNOSIS: The model is predicting ALL BACKGROUND (Zeros).")
        print("    Possible causes: Learning rate too low, pos_weight ineffective, or backbone feature collapse.")
    elif probs_np.min() > 0.99:
        print("\n⚠️  DIAGNOSIS: The model is predicting ALL FOREGROUND (Ones).")
    else:
        print("\n✅  DIAGNOSIS: The model has active predictions (variance exists).")
        
        # Check Dice
        inter = (probs_np * mask_np).sum()
        union = probs_np.sum() + mask_np.sum()
        dice = 2 * inter / (union + 1e-6)
        print(f"    Dice Score on this sample: {dice:.4f}")
        
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--idx', type=int, default=10)
    args = parser.parse_args()
    
    diagnose(args.model, args.data, args.idx)
