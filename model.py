"""
SegDino Model Architecture.
Backbone: DINOv3 (ViT)
Decoder: Heavy UNet with Pyramid Input Injection.
"""
import os
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()

DINOV3_MODELS = {
    "1": "facebook/dinov3-vits16-pretrain-lvd1689m",   # small
    "2": "facebook/dinov3-vitb16-pretrain-lvd1689m",   # base
    "3": "facebook/dinov3-vitl16-pretrain-lvd1689m",   # large
}

LAYER_INDICES = {
    "1": [2, 5, 8, 11],
    "2": [2, 5, 8, 11],
    "3": [6, 12, 18, 23],
}

class DINOv3Backbone(nn.Module):
    def __init__(self, model_size: str = "2", freeze_backbone: bool = True):
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

    def get_intermediate_layers(self, x: torch.Tensor, n: List[int]) -> List[torch.Tensor]:
        outputs = self.model(x, output_hidden_states=True)
        # Skip embedding layer (0), take hidden states
        layer_outputs = outputs.hidden_states[1:]
        selected = [layer_outputs[i] for i in n]
        # Remove CLS + register tokens
        skip = 1 + self.num_register_tokens
        return [feat[:, skip:, :] for feat in selected]

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SegDino(nn.Module):
    def __init__(self, model_size: str = "1", nclass: int = 1, features: int = 512, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = DINOv3Backbone(model_size, freeze_backbone=freeze_backbone)
        self.layer_indices = LAYER_INDICES[model_size]
        self.patch_size = self.backbone.patch_size
        embed_dim = self.backbone.embed_dim

        # Fusion
        self.project_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(ResBlock(features), ResBlock(features), ResBlock(features))

        # Decoder Stages (Pyramid Injection)
        # Input image (3 channels) is injected at each stage
        
        # Stage 1: 1/16 -> 1/8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features + 3, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResBlock(256)
        )

        # Stage 2: 1/8 -> 1/4
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256 + 3, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResBlock(128)
        )

        # Stage 3: 1/4 -> 1/2
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128 + 3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64)
        )

        # Stage 4: 1/2 -> 1/1
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64 + 3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32)
        )

        self.output = nn.Conv2d(32, nclass, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_img = x
        B, C, H, W = x.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Backbone
        features = self.backbone.get_intermediate_layers(x, self.layer_indices)
        features_2d = [f.permute(0, 2, 1).contiguous().reshape(B, -1, patch_h, patch_w) for f in features]
        
        # Fuse & Bottleneck
        x = self.project_fuse(torch.cat(features_2d, dim=1))
        x = self.bottleneck(x) # 1/16

        # Decoder with Injection
        def inject(feat, target_shape):
            img_resize = F.interpolate(input_img, size=target_shape, mode='bilinear', align_corners=True)
            return torch.cat([feat, img_resize], dim=1)

        x = self.up1(inject(x, (H//16, W//16))) # -> 1/8
        x = self.up2(inject(x, (H//8, W//8)))   # -> 1/4
        x = self.up3(inject(x, (H//4, W//4)))   # -> 1/2
        x = self.up4(inject(x, (H//2, W//2)))   # -> 1/1

        return self.output(x)