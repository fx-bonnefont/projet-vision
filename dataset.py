"""
Dataset handling for SegDino center detection.
"""
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

# DOTA dataset statistics
DOTA_MEAN = [0.4733, 0.4668, 0.4366]
DOTA_STD = [0.1846, 0.1832, 0.1800]

# Hardcoded configuration
SIGMA = 8.0


def mask_to_centers(mask: np.ndarray, return_areas=False) -> list:
    """Extract object centers from binary mask using connected components."""
    binary = (mask > 127).astype(np.uint8) if mask.max() > 1 else (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    centers = []
    areas = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 10:
            centers.append((int(cx), int(cy)))
            areas.append(area)
    if not return_areas:
        return centers
    else:
        return centers, areas


def generate_gaussian_heatmap(centers: list, size: int = 512, sigma: float = SIGMA) -> np.ndarray:
    """Generate heatmap with Gaussian peaks at each center."""
    heatmap = np.zeros((size, size), dtype=np.float32)

    if not centers:
        return heatmap

    y_grid, x_grid = np.ogrid[0:size, 0:size]

    for cx, cy in centers:
        cx = np.clip(cx, 0, size - 1)
        cy = np.clip(cy, 0, size - 1)
        gaussian = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap


class PreTiledDataset(data.Dataset):
    """
    Loads pre-tiled 512x512 images for center detection.

    Expected structure:
        root/split/image/*.png
        root/split/mask/*.png
    """

    def __init__(self, root: str, split: str = "train", sigma: float = SIGMA, return_empty=True):
        self.img_dir = os.path.join(root, split, "image")
        self.mask_dir = os.path.join(root, split, "mask")
        self.sigma = sigma
        self.return_empty = return_empty
        
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png") and not f.startswith("._")
        ])

        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        max_attempts = min(100, len(self.images))

        for _ in range(max_attempts):
            try:
                fname = self.images[idx]
                img_path = os.path.join(self.img_dir, fname)
                mask_path = os.path.join(self.mask_dir, fname)

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if not self.return_empty and not mask.any():   # all zeros
                    idx = (idx + 1) % len(self.images)
                    continue         # skip this sample
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if img is None or mask is None:
                    raise ValueError(f"Failed to read {fname}")

                if img.shape[0] != 512 or img.shape[1] != 512:
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_t = TF.normalize(img_t, DOTA_MEAN, DOTA_STD)
                
                mask_t = torch.from_numpy(mask)

                centers, areas = mask_to_centers(mask, return_areas=True)
                heatmap = generate_gaussian_heatmap(centers, size=512, sigma=self.sigma)
                target_t = torch.from_numpy(heatmap).float().unsqueeze(0)

                meta = {
                    "id": fname,
                    "centers": centers,
                    "areas": areas,
                    "num_objects": len(centers)
                }
                
                return img_t, target_t, meta, mask_t
 

            except Exception:
                idx = (idx + 1) % len(self.images)

        raise RuntimeError(f"Failed to load any valid image after {max_attempts} attempts.")


class CachedFeaturesDataset(data.Dataset):
    """
    Loads pre-computed backbone features for fast decoder training.

    Expected structure:
        cached_features_dir/model_size/split/*.pt  (features)
        original_data_dir/split/mask/*.png         (masks for target generation)
    """

    def __init__(
        self,
        cached_features_dir: str,
        original_data_dir: str,
        model_size: str,
        split: str = "train",
        sigma: float = SIGMA
    ):
        self.features_dir = os.path.join(cached_features_dir, model_size, split)
        self.mask_dir = os.path.join(original_data_dir, split, "mask")
        self.sigma = sigma
        self.model_size = model_size

        if not os.path.exists(self.features_dir):
            raise ValueError(f"Cached features not found: {self.features_dir}")
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        metadata_path = os.path.join(cached_features_dir, model_size, "metadata.pt")
        if os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path, weights_only=True)
            self.embed_dim = self.metadata["embed_dim"]
        else:
            self.metadata = None
            self.embed_dim = None

        self.feature_files = sorted([
            f for f in os.listdir(self.features_dir)
            if f.endswith(".pt") and not f.startswith("._")
        ])

        if len(self.feature_files) == 0:
            raise ValueError(f"No cached features found in {self.features_dir}")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx: int):
        max_attempts = min(10, len(self.feature_files))

        for attempt in range(max_attempts):
            try:
                feat_fname = self.feature_files[idx]
                feat_path = os.path.join(self.features_dir, feat_fname)

                mask_fname = feat_fname.replace(".pt", ".png")
                mask_path = os.path.join(self.mask_dir, mask_fname)

                features = torch.load(feat_path, weights_only=True)
                if features.dtype == torch.float16:
                    features = features.float()

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask {mask_fname}")

                if mask.shape[0] != 512 or mask.shape[1] != 512:
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

                centers = mask_to_centers(mask)
                heatmap = generate_gaussian_heatmap(centers, size=512, sigma=self.sigma)
                target_t = torch.from_numpy(heatmap).float().unsqueeze(0)

                meta = {
                    "id": mask_fname,
                    "centers": centers,
                    "num_objects": len(centers)
                }
                
                # avoid returning images without plan
                
                return features, target_t, meta


            except Exception:
                idx = (idx + 1) % len(self.feature_files)

        raise RuntimeError(f"Failed to load any valid features after {max_attempts} attempts.")
