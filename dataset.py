"""
Dataset for binary segmentation with DOTA-format annotations.
Generates binary masks from oriented bounding boxes.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class SegmentationDataset(Dataset):
    """
    Dataset that loads images and generates binary masks from annotation files.

    Supports:
    - DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty (8 coords, absolute)
    - YOLO format: class x_center y_center w h (normalized 0-1)
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
            # Filter out macOS hidden/metadata files (e.g. ._image.png)
            found = [p for p in self.image_dir.glob(ext) if not p.name.startswith('._')]
            self.image_files.extend(found)
        self.image_files = sorted(self.image_files)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_files)} images")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Feature size: {feat_size}x{feat_size}")

        # Cache data if requested
        self.cached_data = None
        if self.cache_data:
            import os
            from concurrent.futures import ProcessPoolExecutor
            from tqdm import tqdm
            
            # Use 80% of cores for caching to stay responsive
            scan_workers = max(1, int(os.cpu_count() * 0.8))
            print(f"Caching {len(self.image_files)} images in RAM (optimized uint8) using {scan_workers} workers...")
            
            self.cached_data = [None] * len(self.image_files)
            
            # Helper to wrap the method for pickle compatibility if needed, 
            # but _load_raw_data is an instance method. 
            # Best is to use a static helper or just map indices.
            # We can use a lambda or partial if the method is picklable, 
            # but 'self' inside process pool can be tricky with large objects.
            # Since SegmentationDataset is small (just paths), it should be fine.
            
            with ProcessPoolExecutor(max_workers=scan_workers) as executor:
                results = list(tqdm(executor.map(self._load_raw_data, range(len(self.image_files))), 
                                   total=len(self.image_files), desc="Caching"))
            
            self.cached_data = results

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Retrieve raw uint8 data
        if self.cached_data:
            image_raw, mask_raw = self.cached_data[idx]
        else:
            image_raw, mask_raw = self._load_raw_data(idx)

        # On-the-fly normalization (fast on M4)
        # 1. Image: uint8 (H,W,3) -> float32 (3,H,W) normalized
        image = image_raw.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # 2. Mask: uint8 (H,W) -> float32 (1,H,W)
        mask = torch.from_numpy(mask_raw).unsqueeze(0).float()

        return image, mask

    def _load_raw_data(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load image and generate a 512x512 crop (Smart Crop).
        Preserves resolution for small objects.
        """
        img_path = self.image_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # Load full image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # [H, W, 3]
        
        orig_h, orig_w = image.shape[:2]
        crop_size = self.img_size # e.g. 512
        
        # Parse labels to find objects
        object_centers = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # Quick parse for centers
                    coords = self._detect_format_and_parse(line, orig_h, orig_w)
                    if coords is not None:
                        # coords is [4, 2] array of (x,y)
                        center = coords.mean(axis=0) # [center_x, center_y]
                        object_centers.append(center)
        
        # Decide crop coordinates (x, y)
        if orig_h <= crop_size or orig_w <= crop_size:
            # Image smaller than crop: Padding needed mechanism
            # For simplicity, we just resize small images UP to crop_size or PAD
            # Let's simple Resize if small, or padded crop.
            # Easiest: Pad image to ensure it's at least crop_size
            pad_h = max(0, crop_size - orig_h)
            pad_w = max(0, crop_size - orig_w)
            if pad_h > 0 or pad_w > 0:
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                orig_h, orig_w = image.shape[:2] # Update
                
            x_start = 0
            y_start = 0
        else:
            # Determine crop center
            # 80% chance to center on an object if objects exist
            import random
            if len(object_centers) > 0 and random.random() < 0.8:
                center = random.choice(object_centers)
                cx, cy = int(center[0]), int(center[1])
                
                # Jitter center slightly (+- 100 px)
                cx += random.randint(-100, 100)
                cy += random.randint(-100, 100)
            else:
                # Random crop
                cx = random.randint(0, orig_w)
                cy = random.randint(0, orig_h)
                
            # Calculate top-left corner from center
            x_start = max(0, min(orig_w - crop_size, cx - crop_size // 2))
            y_start = max(0, min(orig_h - crop_size, cy - crop_size // 2))

        # Perform Crop on Image
        image_crop = image[y_start:y_start+crop_size, x_start:x_start+crop_size]
        
        # Generate Mask for this specific crop
        # We only draw the mask for the cropped region to save time?
        # NO, 'coords' are absolute. We need to shift them by (x_start, y_start)
        # and flip/clip.
        
        mask_crop = np.zeros((crop_size, crop_size), dtype=np.uint8)
        
        if label_path.exists():
             with open(label_path, 'r') as f:
                for line in f:
                    coords = self._detect_format_and_parse(line, orig_h, orig_w) # Absolute coords
                    if coords is not None:
                        # Shift coordinates to crop frame
                        coords[:, 0] -= x_start
                        coords[:, 1] -= y_start
                        
                        # Fill polygon on the crop mask
                        # cv2.fillPoly handles clipping automatically
                        cv2.fillPoly(mask_crop, [coords.astype(np.int32)], 1)

        return image_crop, mask_crop

    def _parse_dota_line(self, line: str) -> np.ndarray | None:
        """Parse DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 class difficulty"""
        parts = line.strip().split()
        if len(parts) < 8:
            return None

        try:
            coords = np.array([
                [float(parts[0]), float(parts[1])],
                [float(parts[2]), float(parts[3])],
                [float(parts[4]), float(parts[5])],
                [float(parts[6]), float(parts[7])]
            ], dtype=np.float32)
            return coords
        except ValueError:
            return None

    def _parse_yolo_line(self, line: str, img_h: int, img_w: int) -> np.ndarray | None:
        """Parse YOLO format: class x_center y_center w h (normalized)"""
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        try:
            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x_c, w = x_c * img_w, w * img_w
            y_c, h = y_c * img_h, h * img_h

            x1, y1 = x_c - w/2, y_c - h/2
            x2, y2 = x_c + w/2, y_c + h/2

            coords = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)
            return coords
        except ValueError:
            return None

    def _detect_format_and_parse(self, line: str, img_h: int, img_w: int) -> np.ndarray | None:
        """Auto-detect annotation format and parse."""
        if 'imagesource' in line.lower() or 'gsd' in line.lower() or not line.strip():
            return None

        parts = line.strip().split()

        # DOTA format: 8+ values, first 8 are coordinates (absolute)
        if len(parts) >= 8:
            try:
                coords = [float(parts[i]) for i in range(8)]
                if max(coords) > 1:  # Absolute coordinates
                    return self._parse_dota_line(line)
            except ValueError:
                pass

        # YOLO format: 5 values, normalized
        if len(parts) >= 5:
            try:
                vals = [float(parts[i]) for i in range(1, 5)]
                if all(0 <= v <= 1 for v in vals):
                    return self._parse_yolo_line(line, img_h, img_w)
            except ValueError:
                pass

        return None

    def _create_mask(self, label_path: Path, img_h: int, img_w: int) -> np.ndarray:
        """Create binary mask from annotation file at original resolution."""
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if not label_path.exists():
            return mask

        with open(label_path, 'r') as f:
            for line in f:
                coords = self._detect_format_and_parse(line, img_h, img_w)
                if coords is not None:
                    cv2.fillPoly(mask, [coords.astype(np.int32)], 1)

        return mask






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
        pin_memory=False  # Disable for MPS compatibility
    )
