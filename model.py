"""
SegDino Model Architecture.

Backbone: DINOv3 (ViT) - Frozen
Decoders: Modular (HeavyUNet, Medium, Light)
"""
import os

import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

# DINOv3 HuggingFace models
DINOV3_MODELS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small-plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "huge": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "giant": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "large-sat": "facebook/dinov3-vitl16-pretrain-sat493m",
    "giant-sat": "facebook/dinov3-vit7b16-pretrain-sat493m",
}

LAYER_INDICES = {
    "small": [2, 5, 8, 11],           # Small (12 layers)
    "small-plus": [2, 5, 8, 11],      # Small+ (12 layers)
    "base": [2, 5, 8, 11],            # Base (12 layers)
    "large": [5, 11, 17, 23],         # Large (24 layers)
    "huge": [7, 15, 23, 31],          # Huge+ (32 layers)
    "giant": [9, 19, 29, 39],         # Giant/7B (40 layers)
    "large-sat": [5, 11, 17, 23],     # Large SAT (24 layers)
    "giant-sat": [9, 19, 29, 39],     # Giant SAT (40 layers)
}


# =============================================================================
# Backbone
# =============================================================================

class DINOv3Backbone(nn.Module):
    """DINOv3 Vision Transformer backbone."""

    def __init__(self, model_size: str = "base", freeze_backbone: bool = True):
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
        layer_outputs = outputs.hidden_states[1:]
        selected = [layer_outputs[i] for i in layer_ids]
        skip = 1 + self.num_register_tokens
        return [feat[:, skip:, :] for feat in selected]


# =============================================================================
# Building Blocks
# =============================================================================

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
    """Upsample + Conv + ResBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResBlock(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


# =============================================================================
# Decoders
# =============================================================================

class HeavyUNetDecoder(nn.Module):
    """
    Heavy UNet Decoder with ResBlocks at every stage.
    ~50M params. Best quality, needs more data.
    """
    name = "heavy_unet"

    def __init__(self, embed_dim: int, nclass: int = 1, features: int = 512):
        super().__init__()
        # Feature fusion: 4 layers concatenated
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
        self.up1 = PyramidUpBlock(features, 256)
        self.up2 = PyramidUpBlock(256, 128)
        self.up3 = PyramidUpBlock(128, 64)
        self.up4 = PyramidUpBlock(64, 32)

        # Output head
        self.output = nn.Conv2d(32, nclass, kernel_size=1)

    def forward(self, features_2d: list) -> torch.Tensor:
        """
        Args:
            features_2d: List of 4 feature maps from backbone, each (B, C, H, W)
        Returns:
            Logits (B, nclass, H*16, W*16)
        """
        fused = torch.cat(features_2d, dim=1)
        feat = self.project_fuse(fused)
        feat = self.bottleneck(feat)
        feat = self.up1(feat)
        feat = self.up2(feat)
        feat = self.up3(feat)
        feat = self.up4(feat)
        return self.output(feat)


class MediumDecoder(nn.Module):
    """
    Medium decoder with fewer ResBlocks.
    ~10M params. Good balance.
    """
    name = "medium"

    def __init__(self, embed_dim: int, nclass: int = 1, features: int = 256):
        super().__init__()
        self.project_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = ResBlock(features)

        # Simpler upsampling path
        self.up1 = PyramidUpBlock(features, 128)
        self.up2 = PyramidUpBlock(128, 64)
        self.up3 = PyramidUpBlock(64, 32)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(32, nclass, kernel_size=1)

    def forward(self, features_2d: list) -> torch.Tensor:
        fused = torch.cat(features_2d, dim=1)
        feat = self.project_fuse(fused)
        feat = self.bottleneck(feat)
        feat = self.up1(feat)
        feat = self.up2(feat)
        feat = self.up3(feat)
        feat = self.up4(feat)
        return self.output(feat)


class LightDecoder(nn.Module):
    """
    Lightweight decoder with bilinear upsampling + conv 1x1.
    ~500K params. Fast, for testing or limited data.
    """
    name = "light"

    def __init__(self, embed_dim: int, nclass: int = 1, features: int = 128):
        super().__init__()
        self.project_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(features, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(32, nclass, kernel_size=1)

    def forward(self, features_2d: list) -> torch.Tensor:
        fused = torch.cat(features_2d, dim=1)
        feat = self.project_fuse(fused)
        feat = self.upsample(feat)
        return self.output(feat)


# =============================================================================
# Decoder Registry
# =============================================================================

DECODERS = {
    "heavy_unet": HeavyUNetDecoder,
    "medium": MediumDecoder,
    "light": LightDecoder,
}


def get_decoder(name: str, embed_dim: int, nclass: int = 1) -> nn.Module:
    """Get decoder by name."""
    if name not in DECODERS:
        raise ValueError(f"Unknown decoder: {name}. Available: {list(DECODERS.keys())}")
    return DECODERS[name](embed_dim=embed_dim, nclass=nclass)


# =============================================================================
# Main Model
# =============================================================================

class SegDino(nn.Module):
    """
    SegDino: Semantic Segmentation with DINOv3 + Modular Decoder.
    """

    def __init__(
        self,
        model_size: str = "base",
        decoder_name: str = "heavy_unet",
        nclass: int = 1,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_size = model_size
        self.decoder_name = decoder_name

        self.backbone = DINOv3Backbone(model_size, freeze_backbone=freeze_backbone)
        self.layer_indices = LAYER_INDICES[model_size]
        self.patch_size = self.backbone.patch_size
        embed_dim = self.backbone.embed_dim

        self.decoder = get_decoder(decoder_name, embed_dim, nclass)
        print(f"[Decoder] {decoder_name} ({sum(p.numel() for p in self.decoder.parameters()):,} params)")

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

        # Decode
        logits = self.decoder(features_2d)
        return logits

    def get_config(self) -> dict:
        """Return model configuration for saving."""
        return {
            "model_size": self.model_size,
            "decoder_name": self.decoder_name,
        }
