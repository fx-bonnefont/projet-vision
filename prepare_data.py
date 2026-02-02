"""
Smart Offline Tiling for DOTA.
Strategy:
1. Positive Pass: Extract 512x512 tiles centered on EVERY plane found in the mask.
2. Negative Pass: Extract random empty tiles (background) to maintain a healthy ratio.
"""

import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm

# Configuration
SOURCE_ROOT = "/Volumes/X9Pro/DOTA_PLANES"
TARGET_ROOT = "/Volumes/X9Pro/DOTA_PLANES_TILED"
TILE_SIZE = 512
SPLITS = ["train", "test"]

# Ratio of background tiles relative to object tiles.
# 0.3 means: for 100 tiles with planes, we generate 30 empty tiles.
NEGATIVE_RATIO = 0.3 

# Minimum area to consider a contour as a valid object (removes noise)
MIN_OBJECT_AREA = 20 

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_safe_crop_coords(center_x, center_y, H, W, tile_size):
    """
    Returns (x1, y1, x2, y2) centered on (center_x, center_y),
    ensuring we stay within image bounds [0, W] and [0, H].
    """
    half = tile_size // 2
    
    x1 = center_x - half
    y1 = center_y - half
    
    # Shift if we go out of bounds (top-left)
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    
    # Calculate end points
    x2 = x1 + tile_size
    y2 = y1 + tile_size
    
    # Shift if we go out of bounds (bottom-right)
    if x2 > W:
        x2 = W
        x1 = W - tile_size
    if y2 > H:
        y2 = H
        y1 = H - tile_size
        
    # Final safety check for images smaller than tile_size
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > W: x2 = W
    if y2 > H: y2 = H
    
    return x1, y1, x2, y2

def process_split(split_name):
    print(f"\nProcessing split: {split_name} (Object-Centric Mode)")
    
    src_img_dir = os.path.join(SOURCE_ROOT, split_name, "image")
    src_mask_dir = os.path.join(SOURCE_ROOT, split_name, "mask")
    
    tgt_img_dir = os.path.join(TARGET_ROOT, split_name, "image")
    tgt_mask_dir = os.path.join(TARGET_ROOT, split_name, "mask")
    
    ensure_dir(tgt_img_dir)
    ensure_dir(tgt_mask_dir)
    
    img_paths = sorted(glob.glob(os.path.join(src_img_dir, "*")))
    
    total_pos = 0
    total_neg = 0
    
    for img_path in tqdm(img_paths):
        # 1. Load Data
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: continue
        
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]
        
        # Find mask
        mask_path = os.path.join(src_mask_dir, base_name + ".png")
        if not os.path.exists(mask_path):
             mask_path = os.path.join(src_mask_dir, fname) # Try same ext
             if not os.path.exists(mask_path): continue
                
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        
        # Handle small images (smaller than tile size)
        if H < TILE_SIZE or W < TILE_SIZE:
            pad_h = max(0, TILE_SIZE - H)
            pad_w = max(0, TILE_SIZE - W)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            H, W = img.shape[:2] # Update dims

        # --------------------------- 
        # PASS 1: Object-Centric Tiles
        # --------------------------- 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_pos_count = 0
        
        for idx, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < MIN_OBJECT_AREA:
                continue
                
            # Get center of the object
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get coords
            x1, y1, x2, y2 = get_safe_crop_coords(cx, cy, H, W, TILE_SIZE)
            
            # Crop
            tile_img = img[y1:y2, x1:x2]
            tile_mask = mask[y1:y2, x1:x2]
            
            # Save
            save_name = f"{base_name}_obj_{idx}.png"
            cv2.imwrite(os.path.join(tgt_img_dir, save_name), tile_img)
            cv2.imwrite(os.path.join(tgt_mask_dir, save_name), tile_mask)
            
            img_pos_count += 1
            total_pos += 1
            
        # --------------------------- 
        # PASS 2: Random Background Tiles
        # --------------------------- 
        # We want roughly NEGATIVE_RATIO * img_pos_count negative tiles
        target_neg = max(1, int(img_pos_count * NEGATIVE_RATIO))
        img_neg_count = 0
        attempts = 0
        max_attempts = target_neg * 10 # Avoid infinite loops
        
        while img_neg_count < target_neg and attempts < max_attempts:
            attempts += 1
            
            # Random top-left
            rx = random.randint(0, max(0, W - TILE_SIZE))
            ry = random.randint(0, max(0, H - TILE_SIZE))
            
            # Crop mask to check if empty
            crop_mask = mask[ry:ry+TILE_SIZE, rx:rx+TILE_SIZE]
            
            if cv2.countNonZero(crop_mask) == 0:
                # It's a valid background tile
                crop_img = img[ry:ry+TILE_SIZE, rx:rx+TILE_SIZE]
                
                save_name = f"{base_name}_bg_{img_neg_count}.png"
                cv2.imwrite(os.path.join(tgt_img_dir, save_name), crop_img)
                cv2.imwrite(os.path.join(tgt_mask_dir, save_name), crop_mask)
                
                img_neg_count += 1
                total_neg += 1

    print(f"Finished {split_name}.")
    print(f"  -> Positive Tiles (Objects): {total_pos}")
    print(f"  -> Negative Tiles (Background): {total_neg}")
    print(f"  -> Total: {total_pos + total_neg}")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source {SOURCE_ROOT} not found.")
        exit(1)
        
    print("Starting Object-Centric Tiling...")
    # Clean up previous dataset to ensure purity? 
    # User might want to do this manually, but for now we overwrite/add.
    
    for split in SPLITS:
        process_split(split)