"""
SegDino Model Architecture.

Backbone: DINOv3 (ViT) - Frozen
Decoder: Heavy UNet with Pyramid Input Injection
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

# DINOv3 HuggingFace models
DINOV3_MODELS = {
    "vit-small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vit-small-plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vit-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vit-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vit-huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit-giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "vit-large-sat": "facebook/dinov3-vitl16-pretrain-sat493m",
    "vit-giant-sat": "facebook/dinov3-vit7b16-pretrain-sat493m",
}

LAYER_INDICES = {
    "vit-small": [2, 5, 8, 11],           # Small (12 layers)
    "vit-small-plus": [2, 5, 8, 11],      # Small+ (12 layers)
    "vit-base": [2, 5, 8, 11],            # Base (12 layers)
    "vit-large": [5, 11, 17, 23],         # Large (24 layers)
    "vit-huge": [7, 15, 23, 31],          # Huge+ (32 layers)
    "vit-giant": [9, 19, 29, 39],         # Giant/7B (40 layers)
    "vit-large-sat": [5, 11, 17, 23],     # Large SAT (24 layers)
    "vit-giant-sat": [9, 19, 29, 39],     # Giant SAT (40 layers)
}


class DINOv3Backbone(nn.Module):
    """DINOv3 Vision Transformer backbone."""

    def __init__(self, model_size: str = "vit-base", freeze_backbone: bool = True):
        super().__init__()
        from transformers import AutoModel

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

        model_name = DINOV3_MODELS[model_size]
        self.model = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

        # Handle register tokens if present
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 4)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[Backbone] {model_name} (FROZEN)")
        else:
            print(f"[Backbone] {model_name} (TRAINABLE)")

    def get_intermediate_layers(self, x: torch.Tensor, layer_ids: list) -> list:
        """Extract features from intermediate layers."""
        outputs = self.model(x, output_hidden_states=True)
        # Skip embedding layer (index 0), take hidden states
        layer_outputs = outputs.hidden_states[1:]
        selected = [layer_outputs[i] for i in layer_ids]

        # Remove CLS token and register tokens
        skip = 1 + self.num_register_tokens
        return [feat[:, skip:, :] for feat in selected]


class ResBlock(nn.Module):
    """Residual block with BatchNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


class PyramidUpBlock(nn.Module):
    """
    Upsampling block with Pyramid Input Injection.

    Architecture:
        1. Upsample features (2x)
        2. Inject SHARP RGB image at new resolution
        3. Process with Conv + ResBlock
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 3, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResBlock(out_channels)
        )

    def forward(self, x: torch.Tensor, rgb_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map to upsample
            rgb_img: Full resolution RGB image (will be resized)

        Returns:
            Upsampled features with RGB injection
        """
        # 1. Upsample features
        x = self.upsample(x)

        # 2. Resize RGB to match NEW feature resolution
        _, _, h, w = x.shape
        rgb_resized = F.interpolate(rgb_img, size=(h, w), mode='bilinear', align_corners=True)

        # 3. Concatenate and process
        x = torch.cat([x, rgb_resized], dim=1)
        x = self.conv(x)
        return x


class SegDino(nn.Module):
    """
    SegDino: Semantic Segmentation with DINOv3 + Heavy UNet + Pyramid Injection.
    """

    def __init__(
        self,
        model_size: str = "vit-base",
        nclass: int = 1,
        features: int = 512,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone = DINOv3Backbone(model_size, freeze_backbone=freeze_backbone)
        self.layer_indices = LAYER_INDICES[model_size]
        self.patch_size = self.backbone.patch_size
        embed_dim = self.backbone.embed_dim

        # Feature fusion: Concat 4 layers from backbone
        self.project_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(features),
            ResBlock(features),
            ResBlock(features)
        )

        # Pyramid Decoder: 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1
        self.up1 = PyramidUpBlock(features, 256)  # 1/16 -> 1/8
        self.up2 = PyramidUpBlock(256, 128)       # 1/8 -> 1/4
        self.up3 = PyramidUpBlock(128, 64)        # 1/4 -> 1/2
        self.up4 = PyramidUpBlock(64, 32)         # 1/2 -> 1/1

        # Output head
        self.output = nn.Conv2d(32, nclass, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input RGB image (B, 3, H, W)

        Returns:
            Logits (B, 1, H, W)
        """
        B, _, H, W = x.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Extract multi-scale features from backbone
        features = self.backbone.get_intermediate_layers(x, self.layer_indices)

        # Reshape to 2D feature maps
        features_2d = [
            f.permute(0, 2, 1).contiguous().reshape(B, -1, patch_h, patch_w)
            for f in features
        ]

        # Fuse and process in bottleneck
        fused = torch.cat(features_2d, dim=1)
        feat = self.project_fuse(fused)
        feat = self.bottleneck(feat)  # Resolution: 1/16

        # Pyramid decoder with RGB injection
        feat = self.up1(feat, x)  # 1/8
        feat = self.up2(feat, x)  # 1/4
        feat = self.up3(feat, x)  # 1/2
        feat = self.up4(feat, x)  # 1/1

        # Output
        logits = self.output(feat)
        return logits
