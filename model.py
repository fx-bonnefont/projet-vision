"""
SegDino Model Architecture.

Backbone: DINOv3 (ViT) - Frozen
Decoder: Light (bilinear upsampling + conv)
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
    "small": [2, 5, 8, 11],
    "small-plus": [2, 5, 8, 11],
    "base": [2, 5, 8, 11],
    "large": [5, 11, 17, 23],
    "huge": [7, 15, 23, 31],
    "giant": [9, 19, 29, 39],
    "large-sat": [5, 11, 17, 23],
    "giant-sat": [9, 19, 29, 39],
}


class DINOv3Backbone(nn.Module):
    """DINOv3 Vision Transformer backbone."""

    def __init__(self, model_size: str = "base", freeze_backbone: bool = True):
        super().__init__()
        from transformers import AutoModel

        model_name = DINOV3_MODELS[model_size]
        self.model = AutoModel.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        self.embed_dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 4)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def get_intermediate_layers(self, x: torch.Tensor, layer_ids: list) -> list:
        """Extract features from intermediate layers."""
        outputs = self.model(x, output_hidden_states=True)
        layer_outputs = outputs.hidden_states[1:]
        selected = [layer_outputs[i] for i in layer_ids]
        skip = 1 + self.num_register_tokens
        return [feat[:, skip:, :] for feat in selected]


class LightDecoder(nn.Module):
    """
    Lightweight decoder with bilinear upsampling + conv.
    ~500K params. Fast, for center detection.
    """

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


class SegDino(nn.Module):
    """
    SegDino: Center detection with DINOv3 + Light Decoder.
    """

    def __init__(self, model_size: str = "base", freeze_backbone: bool = True):
        super().__init__()
        self.model_size = model_size

        self.backbone = DINOv3Backbone(model_size, freeze_backbone=freeze_backbone)
        self.layer_indices = LAYER_INDICES[model_size]
        self.patch_size = self.backbone.patch_size
        embed_dim = self.backbone.embed_dim

        self.decoder = LightDecoder(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        features = self.backbone.get_intermediate_layers(x, self.layer_indices)
        features_2d = [
            f.permute(0, 2, 1).contiguous().reshape(B, -1, patch_h, patch_w)
            for f in features
        ]

        return self.decoder(features_2d)

    def get_config(self) -> dict:
        return {"model_size": self.model_size}


class DecoderOnly(nn.Module):
    """
    Decoder-only model for training with pre-cached backbone features.
    """

    def __init__(self, model_size: str = "base"):
        super().__init__()
        self.model_size = model_size

        from transformers import AutoConfig
        model_name = DINOV3_MODELS[model_size]
        config = AutoConfig.from_pretrained(model_name)
        embed_dim = config.hidden_size

        self.embed_dim = embed_dim
        self.decoder = LightDecoder(embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features_2d = [features[:, i] for i in range(4)]
        return self.decoder(features_2d)

    def get_config(self) -> dict:
        return {"model_size": self.model_size}
