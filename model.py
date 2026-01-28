"""
Model architecture: Frozen backbone + simple segmentation head.
Supports multiple backbones via backbones.py
"""
import torch
import torch.nn as nn

from backbones import create_backbone, list_backbones, BaseBackbone


# Default backbone
DEFAULT_BACKBONE = 'dinov2_vits14'


class SimpleSegmenter(nn.Module):
    """
    Simple segmentation model using frozen backbone features.

    Architecture:
    - Frozen backbone (DINOv2, SAM3, ResNet, etc.)
    - Segmentation head: Conv2d -> ReLU -> Conv2d -> 1-channel heatmap
    """

    def __init__(self, backbone: BaseBackbone):
        super().__init__()

        self.backbone = backbone
        self.patch_size = backbone.config.patch_size
        self.hidden_size = backbone.config.hidden_size

        # Simple segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_size, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        print(f"SimpleSegmenter initialized:")
        print(f"  Backbone: {backbone.config.name}")
        print(f"  Hidden size: {self.hidden_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Heatmap logits [B, 1, H', W']
        """
        # Extract features with frozen backbone
        features = self.backbone.extract_features(x)  # [B, D, H', W']

        # Apply segmentation head
        out = self.head(features)  # [B, 1, H', W']

        return out

    def get_feature_size(self, img_size: int) -> int:
        """Get feature map size for given input image size."""
        return self.backbone.get_feature_size(img_size)


def build_model(backbone_name: str = DEFAULT_BACKBONE, device: str = 'cpu') -> SimpleSegmenter:
    """
    Build a segmentation model with specified backbone.

    Args:
        backbone_name: Name of backbone (dinov2_vits14, sam3, resnet50, etc.)
        device: Device to load model on

    Returns:
        Initialized SimpleSegmenter model
    """
    backbone = create_backbone(backbone_name)
    model = SimpleSegmenter(backbone)
    model = model.to(device)
    return model


def load_model(
    weights_path: str,
    backbone_name: str | None = None,
    device: str = 'cpu'
) -> SimpleSegmenter:
    """
    Load the segmentation model.
    Auto-detects backbone if saved in the new format.
    """
    print(f"Loading model from {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    # Detect format
    if isinstance(checkpoint, dict) and 'backbone_name' in checkpoint:
        # New format
        saved_backbone = checkpoint['backbone_name']
        print(f"  Detected backbone in checkpoint: {saved_backbone}")
        
        if backbone_name is not None and backbone_name != saved_backbone:
            print(f"  WARNING: Requested backbone '{backbone_name}' differs from saved '{saved_backbone}'. Using saved.")
        
        backbone_name = saved_backbone
        state_dict = checkpoint['state_dict']
    else:
        # Old format (just state_dict)
        print("  Legacy checkpoint format (weights only).")
        if backbone_name is None:
            backbone_name = DEFAULT_BACKBONE
            print(f"  No backbone specified, using default: {backbone_name}")
        state_dict = checkpoint

    model = build_model(backbone_name, device)
    model.head.load_state_dict(state_dict)

    return model


def save_model(model: SimpleSegmenter, path: str):
    """Save model weights and configuration."""
    checkpoint = {
        'backbone_name': model.backbone.config.name,
        'state_dict': model.head.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path} (backbone: {model.backbone.config.name})")


def get_image_size(backbone_name: str) -> int:
    """
    Get recommended image size for a backbone.
    Must be multiple of patch size.
    """
    if 'dinov3' in backbone_name:
        return 512  # 32 * 16
    elif 'dinov2' in backbone_name:
        return 518  # 37 * 14
    elif 'sam3' in backbone_name:
        return 504  # 126 * 4 (SAM3 FPN at 1/4 res)
    elif 'resnet' in backbone_name:
        return 512  # 16 * 32
    else:
        return 512


def get_feature_size(backbone_name: str, img_size: int) -> int:
    """Get feature map size for backbone and image size."""
    if 'dinov3' in backbone_name:
        return img_size // 16
    elif 'dinov2' in backbone_name:
        return img_size // 14
    elif 'sam3' in backbone_name:
        return img_size // 4  # SAM3 FPN outputs at 1/4 resolution
    elif 'resnet' in backbone_name:
        return img_size // 32
    else:
        return img_size // 16
