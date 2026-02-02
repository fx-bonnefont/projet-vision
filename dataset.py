"""
Dataset handling for SegDino.
"""
import os
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

# Calculated statistics for DOTA dataset
DOTA_MEAN = (0.4733, 0.4668, 0.4366)
DOTA_STD = (0.1846, 0.1832, 0.1800)


class PreTiledDataset(data.Dataset):
    """
    Loads pre-tiled 512x512 images from disk.

    Expected structure:
        root/split/image/*.png
        root/split/mask/*.png
    """

    def __init__(self, root: str, split: str = "train"):
        self.img_dir = os.path.join(root, split, "image")
        self.mask_dir = os.path.join(root, split, "mask")

        # Filter hidden files
        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png") and not f.startswith("._")
        ])

        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

        print(f"[{split.upper()}] Loaded {len(self.images)} tiles from {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Load and preprocess image and mask.

        Includes fallback mechanism with max attempts to handle corrupted files.
        """
        max_attempts = min(10, len(self.images))  # Try up to 10 files

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

                # Ensure 512x512 size (resize if needed)
                if img.shape[0] != 512 or img.shape[1] != 512:
                    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # To tensor and normalize
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_t = TF.normalize(img_t, DOTA_MEAN, DOTA_STD)

                # Mask to binary tensor
                mask_t = torch.from_numpy(mask).float() / 255.0
                mask_t = (mask_t > 0.5).float().unsqueeze(0)

                return img_t, mask_t, {"id": fname}

            except Exception as e:
                print(f"[WARN] Error loading {fname}: {e}. Trying next...")
                idx = (idx + 1) % len(self.images)

        # If we exhausted all attempts
        raise RuntimeError(
            f"Failed to load any valid image after {max_attempts} attempts. "
            f"Check dataset integrity."
        )
