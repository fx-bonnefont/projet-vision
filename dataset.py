"""
Dataset handling for SegDino.

Supports two target modes:
- 'mask': Binary segmentation masks (original)
- 'center': Gaussian heatmaps at object centers
"""
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

# Calculated statistics for DOTA dataset
DOTA_MEAN = (0.4733, 0.4668, 0.4366)
DOTA_STD = (0.1846, 0.1832, 0.1800)


def mask_to_centers(mask: np.ndarray) -> list:
    """
    Extract object centers from a binary mask using connected components.

    Args:
        mask: Binary mask (H, W) with values in {0, 255} or {0, 1}

    Returns:
        List of (x, y) center coordinates
    """
    # Ensure binary mask
    binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else (mask > 0.5).astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    centers = []
    for i in range(1, num_labels):  # Skip background (label 0)
        cx, cy = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 10:  # Filter tiny noise
            centers.append((int(cx), int(cy)))

    return centers


def generate_gaussian_heatmap(centers: list, size: int = 512, sigma: float = 8.0) -> np.ndarray:
    """
    Generate a heatmap with Gaussian peaks at each center.

    Args:
        centers: List of (x, y) center coordinates
        size: Image size (assumes square)
        sigma: Gaussian standard deviation

    Returns:
        Heatmap (H, W) with values in [0, 1]
    """
    heatmap = np.zeros((size, size), dtype=np.float32)

    if not centers:
        return heatmap

    # Create coordinate grids
    y_grid, x_grid = np.ogrid[0:size, 0:size]

    for cx, cy in centers:
        # Clip to valid range
        cx = np.clip(cx, 0, size - 1)
        cy = np.clip(cy, 0, size - 1)

        # Gaussian centered at (cx, cy)
        gaussian = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, gaussian)  # Take max (don't accumulate)

    return heatmap


class PreTiledDataset(data.Dataset):
    """
    Loads pre-tiled 512x512 images from disk.

    Expected structure:
        root/split/image/*.png
        root/split/mask/*.png

    Args:
        root: Dataset root directory
        split: 'train' or 'test'
        target_type: 'mask' for binary segmentation, 'center' for gaussian heatmaps
        sigma: Gaussian sigma for center mode (default: 8.0)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: str = "mask",
        sigma: float = 8.0
    ):
        self.img_dir = os.path.join(root, split, "image")
        self.mask_dir = os.path.join(root, split, "mask")
        self.target_type = target_type
        self.sigma = sigma

        # Filter hidden files
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png") and not f.startswith("._")
        ])

        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

        mode_str = f"target={target_type}" + (f", sigma={sigma}" if target_type == "center" else "")
        print(f"[{split.upper()}] Loaded {len(self.images)} tiles ({mode_str})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Load and preprocess image and target.

        Returns:
            img_t: Normalized image tensor (3, H, W)
            target_t: Target tensor (1, H, W) - mask or heatmap depending on target_type
            meta: Dict with 'id' and optionally 'centers'
        """
        max_attempts = min(10, len(self.images))

        for attempt in range(max_attempts):
            try:
                fname = self.images[idx]
                img_path = os.path.join(self.img_dir, fname)
                mask_path = os.path.join(self.mask_dir, fname)

                # Load
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if img is None or mask is None:
                    raise ValueError(f"Failed to read {fname}")

                # Ensure 512x512 size
                if img.shape[0] != 512 or img.shape[1] != 512:
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # To tensor and normalize
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_t = TF.normalize(img_t, DOTA_MEAN, DOTA_STD)

                # Generate target based on mode
                meta = {"id": fname}

                if self.target_type == "center":
                    # Extract centers and generate gaussian heatmap
                    centers = mask_to_centers(mask)
                    heatmap = generate_gaussian_heatmap(centers, size=512, sigma=self.sigma)
                    target_t = torch.from_numpy(heatmap).float().unsqueeze(0)
                    meta["centers"] = centers
                    meta["num_objects"] = len(centers)
                else:
                    # Original binary mask
                    mask_t = torch.from_numpy(mask).float() / 255.0
                    target_t = (mask_t > 0.5).float().unsqueeze(0)

                return img_t, target_t, meta

            except Exception as e:
                print(f"[WARN] Error loading {self.images[idx]}: {e}. Trying next...")
                idx = (idx + 1) % len(self.images)

        raise RuntimeError(
            f"Failed to load any valid image after {max_attempts} attempts. "
            f"Check dataset integrity."
        )


class CachedFeaturesDataset(data.Dataset):
    """
    Loads pre-computed backbone features from disk for fast decoder training.

    Expected structure:
        cached_features_dir/model_size/split/*.pt  (features)
        original_data_dir/split/mask/*.png         (masks for target generation)

    Args:
        cached_features_dir: Directory with cached features (e.g., /Volumes/X9Pro/DOTA_PLANES_TILED_CACHED_FEATURES)
        original_data_dir: Directory with original tiled data (for masks)
        model_size: Backbone model size (must match cached features)
        split: 'train' or 'test'
        target_type: 'mask' for binary segmentation, 'center' for gaussian heatmaps
        sigma: Gaussian sigma for center mode (default: 8.0)
    """

    def __init__(
        self,
        cached_features_dir: str,
        original_data_dir: str,
        model_size: str,
        split: str = "train",
        target_type: str = "mask",
        sigma: float = 8.0
    ):
        self.features_dir = os.path.join(cached_features_dir, model_size, split)
        self.mask_dir = os.path.join(original_data_dir, split, "mask")
        self.target_type = target_type
        self.sigma = sigma
        self.model_size = model_size

        # Verify directories exist
        if not os.path.exists(self.features_dir):
            raise ValueError(f"Cached features not found: {self.features_dir}")
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Load metadata
        metadata_path = os.path.join(cached_features_dir, model_size, "metadata.pt")
        if os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path)
            self.embed_dim = self.metadata["embed_dim"]
        else:
            print(f"[WARN] No metadata found at {metadata_path}")
            self.metadata = None
            self.embed_dim = None

        # Get list of cached feature files
        self.feature_files = sorted([
            f for f in os.listdir(self.features_dir)
            if f.endswith(".pt") and not f.startswith("._")
        ])

        if len(self.feature_files) == 0:
            raise ValueError(f"No cached features found in {self.features_dir}")

        mode_str = f"target={target_type}" + (f", sigma={sigma}" if target_type == "center" else "")
        print(f"[{split.upper()}] Loaded {len(self.feature_files)} cached features ({mode_str})")
        if self.metadata:
            print(f"  Backbone: {model_size}, embed_dim: {self.embed_dim}")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx: int):
        """
        Load cached features and generate target.

        Returns:
            features: Cached backbone features (4, embed_dim, H, W) as float32
            target_t: Target tensor (1, H, W) - mask or heatmap depending on target_type
            meta: Dict with 'id' and optionally 'centers'
        """
        max_attempts = min(10, len(self.feature_files))

        for attempt in range(max_attempts):
            try:
                feat_fname = self.feature_files[idx]
                feat_path = os.path.join(self.features_dir, feat_fname)

                # Corresponding mask file (same name but .png)
                mask_fname = feat_fname.replace(".pt", ".png")
                mask_path = os.path.join(self.mask_dir, mask_fname)

                # Load cached features
                features = torch.load(feat_path)
                if features.dtype == torch.float16:
                    features = features.float()

                # Load mask for target generation
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask {mask_fname}")

                # Ensure 512x512 size
                if mask.shape[0] != 512 or mask.shape[1] != 512:
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

                # Generate target based on mode
                meta = {"id": mask_fname}

                if self.target_type == "center":
                    centers = mask_to_centers(mask)
                    heatmap = generate_gaussian_heatmap(centers, size=512, sigma=self.sigma)
                    target_t = torch.from_numpy(heatmap).float().unsqueeze(0)
                    meta["centers"] = centers
                    meta["num_objects"] = len(centers)
                else:
                    mask_t = torch.from_numpy(mask).float() / 255.0
                    target_t = (mask_t > 0.5).float().unsqueeze(0)

                return features, target_t, meta

            except Exception as e:
                print(f"[WARN] Error loading {self.feature_files[idx]}: {e}. Trying next...")
                idx = (idx + 1) % len(self.feature_files)

        raise RuntimeError(
            f"Failed to load any valid features after {max_attempts} attempts. "
            f"Check cache integrity."
        )
