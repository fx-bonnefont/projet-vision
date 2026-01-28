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
    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract spatial features from input images.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            List of feature maps [B, hidden_size, H', W']
            Usually ordered from shallow to deep.
        """
        pass

    def get_feature_size(self, img_size: int) -> int:
        """Get output feature map size for given input size (deepest layer)."""
        return img_size // self.config.patch_size


# =============================================================================
# DINOv2 Backbone (via torch.hub)
# =============================================================================

class DINOv2Backbone(BaseBackbone):
    """
    DINOv2 backbone.
    """
    # ... (Rest of config dict matches original)
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
        super().__init__(self.CONFIGS[model_name])
        # Force loading output_hidden_states is hard with torch.hub generally, 
        # normally it just returns last layer. 
        # For DINOv2 hub model, forward_features returns a dict.
        # It's harder to get intermediates easily without hacking. 
        # For now, we return a list with ONE element for DINOv2 to match interface.
        print(f"Loading {model_name} via torch.hub...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        for param in self.model.parameters(): param.requires_grad = False
        self.model.eval()

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, _, H, W = x.shape
        h_patches = H // self.config.patch_size
        w_patches = W // self.config.patch_size

        with torch.no_grad():
            features = self.model.forward_features(x)
            patch_tokens = features['x_norm_patchtokens']  # [B, N, D]

        patch_tokens = patch_tokens.permute(0, 2, 1).contiguous()
        patch_tokens = patch_tokens.reshape(B, self.config.hidden_size, h_patches, w_patches)
        return [patch_tokens] # Single scale for DINOv2 (hub limitation)


# =============================================================================
# DINOv3 Backbone (via HuggingFace transformers)
# =============================================================================

class DINOv3Backbone(BaseBackbone):
    """
    DINOv3 backbone.
    """
    # ... (Configs match original)
    CONFIGS = {
        'dinov3_vits16': ('facebook/dinov3-vits16-pretrain-lvd1689m', 384),
        'dinov3_vitb16': ('facebook/dinov3-vitb16-pretrain-lvd1689m', 768),
        'dinov3_vitl16': ('facebook/dinov3-vitl16-pretrain-lvd1689m', 1024),
        'dinov3_vit7b16': ('facebook/dinov3-vit7b16-pretrain-lvd1689m', 1536),
        'dinov3_vit7b16_sat': ('facebook/dinov3-vit7b16-pretrain-sat493m', 1536),
    }

    def __init__(self, model_name: str = 'dinov3_vit7b16_sat'):
        # ... (Init logic matches original, just copy paste ensuring usage of keys)
        if model_name not in self.CONFIGS:
             raise ValueError(f"Unknown DINOv3 model: {model_name}")
        model_id, hidden_size = self.CONFIGS[model_name]
        config = BackboneConfig(model_name, patch_size=16, hidden_size=hidden_size, num_register_tokens=4)
        super().__init__(config)
        
        print(f"Loading DINOv3 from HuggingFace ({model_id})...")
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id)
        
        for param in self.model.parameters(): param.requires_grad = False
        self.model.eval()
        
        if hasattr(self.model.config, 'hidden_size'):
            real_hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, 'embed_dim'):
            real_hidden_size = self.model.config.embed_dim
        else:
            real_hidden_size = hidden_size
            
        if real_hidden_size != hidden_size:
            self.config.hidden_size = real_hidden_size
        
        print(f"  DINOv3 loaded: hidden_size={self.config.hidden_size}")

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multiscale features.
        Returns 4 feature maps from layers [3, 6, 9, 12] (approx).
        """
        B, _, H, W = x.shape
        h_patches = H // self.config.patch_size
        w_patches = W // self.config.patch_size

        with torch.no_grad():
            outputs = self.model(pixel_values=x, output_hidden_states=True)
            # Tuple of hidden states (embeddings + layers)
            hidden_states = outputs.hidden_states 
            
            # Select 4 evenly spaced layers
            # e.g. for S16 (12 layers -> 13 states): indices [-1, -4, -7, -10] => [12, 9, 6, 3]
            # Order: Deep to Shallow or Shallow to Deep? 
            # Usually U-Net expects: [Deepest] for bottleneck, then skips [Deep-1, Deep-2...]
            # Or [Shallow, Mid, Deep]
            # Let's return ordered from Shallow to Deep: [3, 6, 9, 12]
            
            indices = [-10, -7, -4, -1] # 3, 6, 9, 12 roughly
            
            selected_features = []
            for idx in indices:
                # [B, N, D]
                if abs(idx) > len(hidden_states):
                    # Fallback for shallow models
                    state = hidden_states[-1]
                else:
                    state = hidden_states[idx]
                
                # Reshape
                patch_tokens = state[:, 1 + self.config.num_register_tokens:, :] # Skip CLS + Reg
                patch_tokens = patch_tokens.permute(0, 2, 1).contiguous()
                patch_tokens = patch_tokens.reshape(B, self.config.hidden_size, h_patches, w_patches)
                selected_features.append(patch_tokens)
                
        return selected_features


# =============================================================================
# SAM3 Backbone
# =============================================================================

class SAM3Backbone(BaseBackbone):
    """
    SAM3 backbone.
    """
    def __init__(self, model_id: str = "facebook/sam3"):
        config = BackboneConfig('sam3', patch_size=14, hidden_size=256) 
        super().__init__(config)
        print(f"Loading SAM3...")
        from transformers import Sam3Model
        self.model = Sam3Model.from_pretrained(model_id)
        for param in self.model.parameters(): param.requires_grad = False
        self.model.eval()

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            vision_output = self.model.get_vision_features(pixel_values=x)
            # SAM3 already returns feature pyramid in feature_maps
            # feature_maps is a list of tensors at 1/4, 1/8, 1/16 etc?
            # Actually standard SAM returns 1/4. SAM3 might differ.
            # For safety, we wrap the main one in a list.
            features = vision_output.feature_maps[0] 
            
        return [features]


# =============================================================================
# ResNet Backbone
# =============================================================================

class ResNetBackbone(BaseBackbone):
    """ResNet Backbone"""
    def __init__(self, model_name: str = 'resnet50'):
        # Config setup...
        hidden_sizes = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048}
        config = BackboneConfig(model_name, patch_size=32, hidden_size=hidden_sizes.get(model_name, 2048))
        super().__init__(config)
        import torchvision.models as models
        self.model = getattr(models, model_name)(weights='IMAGENET1K_V1')
        
        # Split layers
        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        
        # Freeze
        for m in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in m.parameters(): param.requires_grad = False
            m.eval()

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            x0 = self.layer0(x) # 1/4
            x1 = self.layer1(x0) # 1/4 (256ch usually)
            x2 = self.layer2(x1) # 1/8
            x3 = self.layer3(x2) # 1/16
            x4 = self.layer4(x3) # 1/32
        
        # Return [1/4, 1/8, 1/16, 1/32]
        return [x1, x2, x3, x4]

# Factory... match original
BACKBONE_REGISTRY = {
    'dinov2_vits14': lambda: DINOv2Backbone('dinov2_vits14'),
    'dinov2_vitb14': lambda: DINOv2Backbone('dinov2_vitb14'),
    'dinov2_vitl14': lambda: DINOv2Backbone('dinov2_vitl14'),
    'dinov2_vits14_reg': lambda: DINOv2Backbone('dinov2_vits14_reg'),
    'dinov2_vitb14_reg': lambda: DINOv2Backbone('dinov2_vitb14_reg'),
    'dinov3_vits16': lambda: DINOv3Backbone('dinov3_vits16'),
    'dinov3_vitb16': lambda: DINOv3Backbone('dinov3_vitb16'),
    'dinov3_vitl16': lambda: DINOv3Backbone('dinov3_vitl16'),
    'dinov3_vit7b16': lambda: DINOv3Backbone('dinov3_vit7b16'),
    'dinov3_vit7b16_sat': lambda: DINOv3Backbone('dinov3_vit7b16_sat'),
    'sam3': lambda: SAM3Backbone(),
    'resnet18': lambda: ResNetBackbone('resnet18'),
    'resnet50': lambda: ResNetBackbone('resnet50'),
    'resnet101': lambda: ResNetBackbone('resnet101'),
}

def create_backbone(name: str):
    return BACKBONE_REGISTRY[name]()

def list_backbones():
    return list(BACKBONE_REGISTRY.keys())
