"""
Dataset for Multi-Class segmentation with DOTA-format annotations.
Generates integer masks (0-15) from oriented bounding boxes.
"""
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# DOTA Class Mapping
# 0 is always background
DOTA_CLASSES = [
    'background',
    'plane', 'ship', 'storage-tank', 'baseball-diamond', 
    'tennis-court', 'basketball-court', 'ground-track-field', 
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
]
assert len(DOTA_CLASSES) == 16, "Expected 16 classes including background"
CLASS_TO_ID = {name: i for i, name in enumerate(DOTA_CLASSES)}


class SegmentationDataset(Dataset):
    """
    Dataset that loads images and generates multi-class masks.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        img_size: int = 518,
        feat_size: int = 37,
        cache_data: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.cache_data = cache_data

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.img_size = img_size
        self.feat_size = feat_size

        # Find all images
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            found = [p for p in self.image_dir.glob(ext) if not p.name.startswith('._')]
            self.image_files.extend(found)
        self.image_files = sorted(self.image_files)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_files)} images")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Classes: {len(DOTA_CLASSES)} (inclusive of background)")

        # Cache data if requested
        self.cached_data = None
        if self.cache_data:
            import os
            from concurrent.futures import ProcessPoolExecutor
            from tqdm import tqdm
            
            scan_workers = max(1, int(os.cpu_count() * 0.8))
            print(f"Caching {len(self.image_files)} images using {scan_workers} workers...")
            
            with ProcessPoolExecutor(max_workers=scan_workers) as executor:
                results = list(tqdm(executor.map(self._load_raw_data, range(len(self.image_files))), 
                                   total=len(self.image_files), desc="Caching"))
            
            self.cached_data = results

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Debug: Print progress for first few items
        if idx < 5:
            print(f"Loading sample {idx+1}/{len(self)}...")
            
        if self.cached_data:
            image_raw, mask_raw = self.cached_data[idx]
        else:
            image_raw, mask_raw = self._load_raw_data(idx)

        # 1. Image: uint8 -> float32 (3,H,W)
        image = image_raw.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # 2. Mask: uint8 (H,W) -> LongTensor (H,W)
        # CrossEntropyLoss expects class indices, NO channel dim for target
        mask = torch.from_numpy(mask_raw).long()

        return image, mask

    def _load_raw_data(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load image and generate a Multi-Class Crop.
        """
        img_path = self.image_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # Load full image with error handling
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"cv2.imread returned None")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"⚠️ Skipping corrupted image {img_path.name}: {e}")
            # Fallback: return a random valid image instead of crashing
            fallback_idx = (idx + 1) % len(self.image_files)
            return self._load_raw_data(fallback_idx)
        
        orig_h, orig_w = image.shape[:2]
        crop_size = self.img_size
        
        # Parse labels to find objects
        objects = [] # list of (class_id, polygon_coords)
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    data = self._detect_format_and_parse(line, orig_h, orig_w)
                    if data is not None:
                        objects.append(data)
        
        # Decide crop coordinates
        if orig_h <= crop_size or orig_w <= crop_size:
            # Pad small images
            pad_h = max(0, crop_size - orig_h)
            pad_w = max(0, crop_size - orig_w)
            if pad_h > 0 or pad_w > 0:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                orig_h, orig_w = image.shape[:2]
                
            x_start = 0
            y_start = 0
        else:
            # Smart Crop: Center on an object 80% of the time
            if len(objects) > 0 and random.random() < 0.8:
                # Pick random object
                _, coords = random.choice(objects)
                center = coords.mean(axis=0)
                cx, cy = int(center[0]), int(center[1])
                cx += random.randint(-100, 100)
                cy += random.randint(-100, 100)
            else:
                cx = random.randint(0, orig_w)
                cy = random.randint(0, orig_h)
                
            x_start = max(0, min(orig_w - crop_size, cx - crop_size // 2))
            y_start = max(0, min(orig_h - crop_size, cy - crop_size // 2))

        # Crop Image
        image_crop = image[y_start:y_start+crop_size, x_start:x_start+crop_size]
        
        # Generate Multi-Class Mask
        # Initialize with 0 (background)
        mask_crop = np.zeros((crop_size, crop_size), dtype=np.uint8)
        
        # Draw objects
        # We assume dataset doesn't have overlapping objects of different classes heavily
        # If overlap, last drawn wins
        for cls_name, coords in objects:
            cls_id = CLASS_TO_ID.get(cls_name)
            if cls_id is None:
                continue # Skip unknown classes (e.g. metadata)
                
            # Shift coords
            coords_shifted = coords.copy()
            coords_shifted[:, 0] -= x_start
            coords_shifted[:, 1] -= y_start
            
            # Fill with Class ID
            cv2.fillPoly(mask_crop, [coords_shifted.astype(np.int32)], int(cls_id))

        return image_crop, mask_crop

    def _parse_dota_line(self, line: str) -> tuple[str, np.ndarray] | None:
        """Parse DOTA format: x1 y1 ... x4 y4 class difficulty"""
        parts = line.strip().split()
        if len(parts) < 9:
            return None

        try:
            # Extract Class Name (8th index, 9th item)
            # x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
            # 0  1  2  3  4  5  6  7  8     9
            cls_name = parts[8]
            
            coords = np.array([
                [float(parts[0]), float(parts[1])],
                [float(parts[2]), float(parts[3])],
                [float(parts[4]), float(parts[5])],
                [float(parts[6]), float(parts[7])]
            ], dtype=np.float32)
            
            return cls_name, coords
        except ValueError:
            return None

    def _detect_format_and_parse(self, line: str, img_h: int, img_w: int) -> tuple[str, np.ndarray] | None:
        if 'imagesource' in line.lower() or 'gsd' in line.lower() or not line.strip():
            return None

        parts = line.strip().split()

        # DOTA format assumption
        if len(parts) >= 8:
            return self._parse_dota_line(line)
        
        return None

def get_dataloader(
    image_dir: str,
    label_dir: str,
    img_size: int = 518,
    feat_size: int = 37,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    cache: bool = False
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the segmentation dataset."""
    dataset = SegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        img_size=img_size,
        feat_size=feat_size,
        cache_data=cache
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
