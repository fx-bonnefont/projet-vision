"""
Extract and cache backbone features for faster decoder training.

This script pre-computes DINOv3 backbone features for all tiles,
allowing decoder training without recomputing backbone features each epoch.

Usage:
    python extract_features.py -m small-plus --data-dir segdata/DOTA/DOTA_PLANES_TILED
    python extract_features.py -m large-sat --data-dir segdata/DOTA/DOTA_PLANES_TILED
"""
import argparse
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from dataset import DOTA_MEAN, DOTA_STD
from model import DINOv3Backbone, LAYER_INDICES, DINOV3_MODELS


def extract_and_save_features(
    model_size: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 16,
    device: str = "cuda",
    use_fp16: bool = True
):
    """
    Extract backbone features for all tiles and save to disk.

    Args:
        model_size: DINOv3 model size (small-plus, large-sat, etc.)
        data_dir: Path to tiled dataset (with train/test subdirs)
        output_dir: Base output directory for cached features
        batch_size: Batch size for feature extraction
        device: Device to use for extraction
        use_fp16: Save features as float16 to reduce disk space
    """
    # Create output directory structure
    cache_dir = os.path.join(output_dir, model_size)
    os.makedirs(cache_dir, exist_ok=True)

    # Load backbone
    print(f"Loading backbone: {model_size}")
    backbone = DINOv3Backbone(model_size, freeze_backbone=True).to(device)
    backbone.eval()

    layer_indices = LAYER_INDICES[model_size]
    patch_size = backbone.patch_size
    embed_dim = backbone.embed_dim

    print(f"  Embed dim: {embed_dim}")
    print(f"  Patch size: {patch_size}")
    print(f"  Layer indices: {layer_indices}")
    print(f"  Output format: {'float16' if use_fp16 else 'float32'}")

    # Process train and test splits
    for split in ["train", "test"]:
        split_input_dir = os.path.join(data_dir, split, "image")
        split_output_dir = os.path.join(cache_dir, split)

        if not os.path.exists(split_input_dir):
            print(f"Skipping {split}: {split_input_dir} not found")
            continue

        os.makedirs(split_output_dir, exist_ok=True)

        # Get list of images
        image_files = sorted([
            f for f in os.listdir(split_input_dir)
            if f.endswith(".png") and not f.startswith("._")
        ])

        print(f"\nProcessing {split}: {len(image_files)} tiles")

        # Check for already processed files
        existing = set(f.replace(".pt", ".png") for f in os.listdir(split_output_dir) if f.endswith(".pt"))
        to_process = [f for f in image_files if f not in existing]

        if len(to_process) < len(image_files):
            print(f"  Skipping {len(image_files) - len(to_process)} already processed tiles")

        if not to_process:
            print(f"  All tiles already processed")
            continue

        # Process in batches
        for i in tqdm(range(0, len(to_process), batch_size), desc=f"Extracting {split}"):
            batch_files = to_process[i:i + batch_size]
            batch_images = []

            for fname in batch_files:
                img_path = os.path.join(split_input_dir, fname)
                img = Image.open(img_path).convert("RGB")
                img_t = TF.to_tensor(img)
                img_t = TF.normalize(img_t, DOTA_MEAN, DOTA_STD)
                batch_images.append(img_t)

            batch_tensor = torch.stack(batch_images).to(device)

            with torch.no_grad():
                # Extract features from intermediate layers
                features = backbone.get_intermediate_layers(batch_tensor, layer_indices)

                # Reshape to 2D feature maps
                B = batch_tensor.shape[0]
                H, W = batch_tensor.shape[2], batch_tensor.shape[3]
                patch_h, patch_w = H // patch_size, W // patch_size

                features_2d = [
                    f.permute(0, 2, 1).contiguous().reshape(B, -1, patch_h, patch_w)
                    for f in features
                ]

            # Save each tile's features
            for j, fname in enumerate(batch_files):
                # Stack all 4 feature maps into a single tensor
                tile_features = torch.stack([f[j] for f in features_2d])  # (4, embed_dim, H, W)

                if use_fp16:
                    tile_features = tile_features.half()

                tile_features = tile_features.cpu()

                # Save with same name but .pt extension
                out_fname = fname.replace(".png", ".pt")
                out_path = os.path.join(split_output_dir, out_fname)
                torch.save(tile_features, out_path)

    # Save metadata
    metadata = {
        "model_size": model_size,
        "model_name": DINOV3_MODELS[model_size],
        "embed_dim": embed_dim,
        "patch_size": patch_size,
        "layer_indices": layer_indices,
        "num_layers": 4,
        "feature_shape": f"(4, {embed_dim}, 32, 32)",
        "dtype": "float16" if use_fp16 else "float32",
    }

    metadata_path = os.path.join(cache_dir, "metadata.pt")
    torch.save(metadata, metadata_path)
    print(f"\nMetadata saved to: {metadata_path}")

    # Calculate disk usage
    total_size = 0
    for split in ["train", "test"]:
        split_dir = os.path.join(cache_dir, split)
        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                total_size += os.path.getsize(os.path.join(split_dir, f))

    print(f"Total cache size: {total_size / (1024**3):.2f} GB")


def get_cache_dir(data_dir: str) -> str:
    """Get the cache directory path based on data_dir."""
    return data_dir.rstrip('/') + "_CACHED_FEATURES"


def main():
    parser = argparse.ArgumentParser(description="Extract and cache backbone features")
    parser.add_argument("-m", required=True,
                        choices=["small", "small-plus", "base", "large", "huge", "giant", "large-sat", "giant-sat"],
                        help="DINOv3 backbone size")
    parser.add_argument("--data-dir", default="segdata/DOTA/DOTA_PLANES_TILED",
                        help="Path to tiled dataset")
    parser.add_argument("-b", type=int, default=16,
                        help="Batch size for extraction")
    parser.add_argument("--fp32", action="store_true",
                        help="Save as float32 instead of float16")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = get_cache_dir(args.data_dir)
    print(f"Cache directory: {output_dir}")

    extract_and_save_features(
        model_size=args.m,
        data_dir=args.data_dir,
        output_dir=output_dir,
        batch_size=args.b,
        device=device,
        use_fp16=not args.fp32
    )

    print("\nDone! Training will automatically use cached features.")


if __name__ == "__main__":
    main()
