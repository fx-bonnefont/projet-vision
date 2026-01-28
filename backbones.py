"""
Modular backbone support for feature extraction.
Supports: DINOv2, SAM3, ResNet
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BackboneConfig:
    """Configuration for a backbone."""
    name: str
    patch_size: int
    hidden_size: int
    num_register_tokens: int = 0


class BaseBackbone(ABC, nn.Module):
    """Abstract base class for backbones."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from input images.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Spatial features [B, hidden_size, H', W']
        """
        pass

    def get_feature_size(self, img_size: int) -> int:
        """Get output feature map size for given input size."""
        return img_size // self.config.patch_size


# =============================================================================
# DINOv2 Backbone (via torch.hub)
# =============================================================================

class DINOv2Backbone(BaseBackbone):
    """
    DINOv2 backbone via torch.hub.

    Available models:
    - dinov2_vits14: ViT-S/14, 384-dim
    - dinov2_vitb14: ViT-B/14, 768-dim
    - dinov2_vitl14: ViT-L/14, 1024-dim
    - dinov2_vitg14: ViT-g/14, 1536-dim
    - *_reg variants: with register tokens
    """

    CONFIGS = {
        'dinov2_vits14': BackboneConfig('dinov2_vits14', patch_size=14, hidden_size=384),
        'dinov2_vitb14': BackboneConfig('dinov2_vitb14', patch_size=14, hidden_size=768),
        'dinov2_vitl14': BackboneConfig('dinov2_vitl14', patch_size=14, hidden_size=1024),
        'dinov2_vitg14': BackboneConfig('dinov2_vitg14', patch_size=14, hidden_size=1536),
        'dinov2_vits14_reg': BackboneConfig('dinov2_vits14_reg', patch_size=14, hidden_size=384, num_register_tokens=4),
        'dinov2_vitb14_reg': BackboneConfig('dinov2_vitb14_reg', patch_size=14, hidden_size=768, num_register_tokens=4),
        'dinov2_vitl14_reg': BackboneConfig('dinov2_vitl14_reg', patch_size=14, hidden_size=1024, num_register_tokens=4),
        'dinov2_vitg14_reg': BackboneConfig('dinov2_vitg14_reg', patch_size=14, hidden_size=1536, num_register_tokens=4),
    }

    def __init__(self, model_name: str = 'dinov2_vits14'):
        if model_name not in self.CONFIGS:
            raise ValueError(f"Unknown DINOv2 model: {model_name}. Available: {list(self.CONFIGS.keys())}")

        config = self.CONFIGS[model_name]
        super().__init__(config)

        print(f"Loading {model_name} via torch.hub...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        h_patches = H // self.config.patch_size
        w_patches = W // self.config.patch_size

        with torch.no_grad():
            features = self.model.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']  # [B, N, D]

        # Reshape to spatial grid [B, D, H', W']
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(B, self.config.hidden_size, h_patches, w_patches)

        return patch_tokens


# =============================================================================
# DINOv3 Backbone (via HuggingFace transformers)
# =============================================================================

class DINOv3Backbone(BaseBackbone):
    """
    DINOv3 backbone via HuggingFace transformers.

    Available models:
    - facebook/dinov3-vits16-pretrain-lvd1689m: ViT-S/16, 384-dim
    - facebook/dinov3-vitb16-pretrain-lvd1689m: ViT-B/16, 768-dim
    - facebook/dinov3-vitl16-pretrain-lvd1689m: ViT-L/16, 1024-dim
    - facebook/dinov3-vit7b16-pretrain-lvd1689m: ViT-7B/16, 1536-dim
    - facebook/dinov3-vit7b16-pretrain-sat493m: ViT-7B/16 satellite pretrained
    """

    CONFIGS = {
        'dinov3_vits16': ('facebook/dinov3-vits16-pretrain-lvd1689m', 384),
        'dinov3_vitb16': ('facebook/dinov3-vitb16-pretrain-lvd1689m', 768),
        'dinov3_vitl16': ('facebook/dinov3-vitl16-pretrain-lvd1689m', 1024),
        'dinov3_vit7b16': ('facebook/dinov3-vit7b16-pretrain-lvd1689m', 1536),
        'dinov3_vit7b16_sat': ('facebook/dinov3-vit7b16-pretrain-sat493m', 1536),
    }

    def __init__(self, model_name: str = 'dinov3_vit7b16_sat'):
        if model_name not in self.CONFIGS:
            raise ValueError(f"Unknown DINOv3 model: {model_name}. Available: {list(self.CONFIGS.keys())}")

        model_id, hidden_size = self.CONFIGS[model_name]
        config = BackboneConfig(model_name, patch_size=16, hidden_size=hidden_size, num_register_tokens=4)
        super().__init__(config)

        print(f"Loading DINOv3 from HuggingFace ({model_id})...")
        from transformers import AutoModel, AutoImageProcessor

        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        print(f"  DINOv3 loaded: hidden_size={hidden_size}, patch_size=16")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch features from DINOv3.
        Returns spatial features [B, D, H', W'].
        """
        B, _, H, W = x.shape
        h_patches = H // self.config.patch_size
        w_patches = W // self.config.patch_size

        with torch.no_grad():
            outputs = self.model(pixel_values=x)
            # last_hidden_state: [B, 1 + num_register_tokens + num_patches, hidden_size]
            last_hidden_state = outputs.last_hidden_state

            # Skip CLS token (index 0) and register tokens (indices 1..4)
            patch_tokens = last_hidden_state[:, 1 + self.config.num_register_tokens:, :]  # [B, N, D]

        # Reshape to spatial grid [B, D, H', W']
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(B, self.config.hidden_size, h_patches, w_patches)

        return patch_tokens


# =============================================================================
# SAM3 Backbone (via HuggingFace transformers)
# =============================================================================

class SAM3Backbone(BaseBackbone):
    """
    SAM3 vision encoder backbone via HuggingFace.
    Uses model.get_vision_features() for feature extraction.

    Model ID: facebook/sam3
    - ViT backbone: 1024-dim hidden size
    - Patch size: 14
    - Default image size: 1008 (but we use 1008 or smaller)
    """

    def __init__(self, model_id: str = "facebook/sam3"):
        # SAM3 ViT config: hidden_size=1024, patch_size=14
        config = BackboneConfig('sam3', patch_size=14, hidden_size=256)  # FPN output is 256
        super().__init__(config)

        print(f"Loading SAM3 from HuggingFace ({model_id})...")
        from transformers import Sam3Model, Sam3Processor

        self.model = Sam3Model.from_pretrained(model_id)
        self.processor = Sam3Processor.from_pretrained(model_id)

        # Freeze
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        print(f"  SAM3 loaded successfully")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features from SAM3.
        Returns FPN features at 1/4 resolution (256 channels).
        """
        with torch.no_grad():
            # get_vision_features returns Sam3VisionEncoderOutput
            vision_output = self.model.get_vision_features(pixel_values=x)
            # Use the multi-scale features from FPN
            # vision_output contains feature_maps at different scales
            # We use the highest resolution one (1/4 scale)
            features = vision_output.feature_maps[0]  # [B, 256, H/4, W/4]

        return features

    def get_feature_size(self, img_size: int) -> int:
        """SAM3 FPN outputs at 1/4 resolution."""
        return img_size // 4


# =============================================================================
# ResNet Backbone (torchvision, simple fallback)
# =============================================================================

class ResNetBackbone(BaseBackbone):
    """
    ResNet backbone from torchvision.
    Simple fallback that doesn't require external downloads.
    """

    def __init__(self, model_name: str = 'resnet50'):
        # ResNet has effective patch size of 32 (5 downsampling stages)
        hidden_sizes = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
        }

        if model_name not in hidden_sizes:
            raise ValueError(f"Unknown ResNet: {model_name}. Available: {list(hidden_sizes.keys())}")

        config = BackboneConfig(model_name, patch_size=32, hidden_size=hidden_sizes[model_name])
        super().__init__(config)

        print(f"Loading {model_name} from torchvision...")
        import torchvision.models as models

        weights = 'IMAGENET1K_V1'
        self.model = getattr(models, model_name)(weights=weights)

        # Remove classification head, keep feature extractor
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        )

        # Freeze
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.features(x)  # [B, C, H/32, W/32]
        return features


# =============================================================================
# Factory function
# =============================================================================

BACKBONE_REGISTRY = {
    # DINOv2 (torch.hub)
    'dinov2_vits14': lambda: DINOv2Backbone('dinov2_vits14'),
    'dinov2_vitb14': lambda: DINOv2Backbone('dinov2_vitb14'),
    'dinov2_vitl14': lambda: DINOv2Backbone('dinov2_vitl14'),
    'dinov2_vits14_reg': lambda: DINOv2Backbone('dinov2_vits14_reg'),
    'dinov2_vitb14_reg': lambda: DINOv2Backbone('dinov2_vitb14_reg'),
    # DINOv3 (HuggingFace)
    'dinov3_vits16': lambda: DINOv3Backbone('dinov3_vits16'),
    'dinov3_vitb16': lambda: DINOv3Backbone('dinov3_vitb16'),
    'dinov3_vitl16': lambda: DINOv3Backbone('dinov3_vitl16'),
    'dinov3_vit7b16': lambda: DINOv3Backbone('dinov3_vit7b16'),
    'dinov3_vit7b16_sat': lambda: DINOv3Backbone('dinov3_vit7b16_sat'),  # Satellite pretrained!
    # SAM3 (HuggingFace)
    'sam3': lambda: SAM3Backbone(),
    # ResNet (torchvision)
    'resnet18': lambda: ResNetBackbone('resnet18'),
    'resnet50': lambda: ResNetBackbone('resnet50'),
    'resnet101': lambda: ResNetBackbone('resnet101'),
}


def create_backbone(name: str) -> BaseBackbone:
    """
    Create a backbone by name.

    Args:
        name: Backbone name (dinov2_vits14, sam3, resnet50, etc.)

    Returns:
        Initialized backbone
    """
    if name not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Unknown backbone: {name}. Available: {available}")

    return BACKBONE_REGISTRY[name]()


def list_backbones() -> list[str]:
    """List available backbone names."""
    return list(BACKBONE_REGISTRY.keys())
