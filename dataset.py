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
        feat_size: int = 37
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.img_size = img_size
        self.feat_size = feat_size

        # Find all images
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(self.image_dir.glob(ext))
        self.image_files = sorted(self.image_files)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_files)} images")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Feature size: {feat_size}x{feat_size}")

    def __len__(self):
        return len(self.image_files)

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

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image: resize, normalize, to tensor."""
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # To tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Create mask at original resolution
        mask = self._create_mask(label_path, orig_h, orig_w)

        # Preprocess image
        image = self._preprocess_image(image)

        # Resize mask to feature size for loss computation
        mask = cv2.resize(mask, (self.feat_size, self.feat_size), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W]

        return image, mask


def get_dataloader(
    image_dir: str,
    label_dir: str,
    img_size: int = 518,
    feat_size: int = 37,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the segmentation dataset."""
    dataset = SegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        img_size=img_size,
        feat_size=feat_size
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
