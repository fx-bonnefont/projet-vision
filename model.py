"""
Model architecture: Frozen backbone + simple segmentation head.
Supports multiple backbones via backbones.py
"""
import torch
import torch.nn as nn

from backbones import create_backbone, list_backbones, BaseBackbone


# Default backbone
DEFAULT_BACKBONE = 'dinov3_vitl16'


class UpBlock(nn.Module):
    """Upsampling block with ConvTranspose2d."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class AdapterBlock(nn.Module):
    """
    Adapts a feature map (e.g. from ViT) to a target channels/size.
    Used to normalize channel counts from different backbone layers.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block: Upsample + Cat(Skip) + Conv
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Upsampling layer
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # Conv after concat
        # Input channels = in_channels (from up) + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        
        if skip is not None:
            # Handle potential size mismatch due to padding
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    True U-Net Decoder that expects a list of multi-scale features.
    Adapted for ViT which may output features all at 1/16 scale.
    """
    def __init__(self, feature_channels: list[int], out_channels=1):
        super().__init__()
        
        # Expecting features ordered from shallow to deep
        # Example DINOv3: [384, 384, 384, 384] (all same scale)
        # Example ResNet: [256, 512, 1024, 2048] (scales 1/4, 1/8, 1/16, 1/32)
        
        self.feature_channels = feature_channels
        base_ch = 64 # Base decoder width
        
        # Adapters to align channels (useful for ViT)
        # We project everything to base_ch*N to have standard U-Net widths
        # C4 (Deepest) -> 512
        # C3 -> 256
        # C2 -> 128
        # C1 (Shallowest) -> 64
        
        # Enforce consistency:
        # Bottleneck takes C4.
        # Decoder 1: Up(Bottleneck) + C3
        # Decoder 2: Up(Dec1) + C2
        # Decoder 3: Up(Dec2) + C1
        # Final: Up(Dec3) -> Output
        
        # Note: If ViT outputs are all 1/16, we simple Upsample 2x at each step anyway 
        # but we feed the skip connection from the earlier layer. 
        # Even if C3 is same res as C4, the "Up" operation will double resolution. 
        # Wait, if C3 is 1/16 and we Up(C4) to 1/8, we can't Cat(1/16).
        # SOLUTION: For ViT (Constant Scale), we must UPSAMPLE the skip if needed OR 
        # treat the ViT features as a "Single Stage" dense block and then upsample pure.
        
        # HYBRID STRATEGY:
        # If backbone is isotropic (ViT, all 1/16):
        #   We effectively have 4 features at 1/16.
        #   We fuse them all into a dense 1/16 block first?
        #   OR we assume the user WANTS standard U-Net behavior and we artificially
        #   upsample the skips? No, that's wasteful.
        
        # IMPROVED STRATEGY for ViT:
        # DINOv3 layers contain different semantic levels but SAME spatial 1/16.
        # We process them (Adapter) to reduce channels.
        # Then we Concatenate them all? 
        # Let's Implement "Simple Feature Pyramid" (ViTDet style) approximation:
        # Fuse [F1, F2, F3, F4] all at 1/16 scale. 
        # Then just standard upsampling 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1.
        # This uses the multi-layer info without fake skip distortions.
        
        # However, for ResNet, we DO need proper skips.
        # So we detect if "isotropic".
        
        self.isotropic = len(set(feature_channels)) == 1 and feature_channels[0] > 0
        
        if self.isotropic:
            # ViT Mode: Dense Fusion then Upsample
            print("  Mode: Isotropic (ViT) - Using Dense Fusion Decoder")
            self.hidden_dim = feature_channels[0]
            
            # Adapters to compress each layer
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, 128, 1),
                    nn.ReLU()
                ) for c in feature_channels
            ])
            
            # Fusion: 4 * 128 = 512 channels
            fusion_dim = 128 * len(feature_channels)
            
            # Pure Upsampling layers (1/16 -> 1/1)
            # 512 -> 256 (1/8)
            self.up1 = UpBlock(fusion_dim, 256)
            # 256 -> 128 (1/4)
            self.up2 = UpBlock(256, 128)
            # 128 -> 64 (1/2)
            self.up3 = UpBlock(128, 64)
            # 64 -> 32 (1/1)
            self.up4 = UpBlock(64, 32)
            
        else:
            # ResNet Mode: Standard U-Net
            print("  Mode: Hierarchical (ResNet) - Using Standard U-Net Decoder")
            c1, c2, c3, c4 = feature_channels
            
            # Bottleneck (C4)
            self.center = nn.Conv2d(c4, 512, 3, padding=1)
            
            # Decoders
            # 512 -> 256 (fuse C3)
            self.dec3 = DecoderBlock(512, c3, 256)
            # 256 -> 128 (fuse C2)
            self.dec2 = DecoderBlock(256, c2, 128)
            # 128 -> 64 (fuse C1)
            self.dec1 = DecoderBlock(128, c1, 64)
            
            # Final upsample (1/4 -> 1/1)
            self.final_up = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=4), # x4 upsampling from 1/4
                nn.BatchNorm2d(32),
                nn.ReLU()
            )

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor]):
        
        if self.isotropic:
            # ViT Logic: Fuse all 1/16 features
            # Reshape all to match first one (just in case)
            target_h, target_w = features[0].shape[-2:]
            
            processed = []
            for i, adapters in enumerate(self.adapters):
                f = adapters(features[i])
                if f.shape[-2:] != (target_h, target_w):
                    f = F.interpolate(f, size=(target_h, target_w), mode='bilinear')
                processed.append(f)
            
            # [B, 512, H/16, W/16]
            x = torch.cat(processed, dim=1)
            
            # Progressive Upsampling
            x = self.up1(x) # 1/8
            x = self.up2(x) # 1/4
            x = self.up3(x) # 1/2
            x = self.up4(x) # 1/1
            
        else:
            # ResNet Logic (c1, c2, c3, c4) = (1/4, 1/8, 1/16, 1/32)
            c1, c2, c3, c4 = features
            
            x = self.center(c4)
            x = self.dec3(x, c3) # -> 1/16
            x = self.dec2(x, c2) # -> 1/8
            x = self.dec1(x, c1) # -> 1/4
            x = self.final_up(x) # -> 1/1
            
        return self.final_conv(x)


class SimpleSegmenter(nn.Module):
    """
    Full pipeline: Backbone + UNet Decoder.
    Supports multi-scale backbones.
    """

    def __init__(self, backbone: BaseBackbone):
        super().__init__()

        self.backbone = backbone
        self.patch_size = backbone.config.patch_size
        self.hidden_size = backbone.config.hidden_size
        
        # Determine feature channels
        # Perform a dummy forward pass to get shapes?
        # Or deduce from config. 
        # For DINOv3, it returns 4 tensors of size hidden_size.
        if 'resnet' in backbone.config.name:
            # ResNet standard channels: [256, 512, 1024, 2048]
            if backbone.config.hidden_size == 2048: # R50/101
                channels = [256, 512, 1024, 2048]
            else: # R18/34
                channels = [64, 128, 256, 512]
        elif 'sam' in backbone.config.name:
            channels = [256] # Single scale FPN
        else:
            # ViT (DINOv2/v3): 4 layers of same size
            channels = [self.hidden_size] * 4

        # Replace simple head with UNet Decoder
        self.head = UNetDecoder(feature_channels=channels, out_channels=1)
        
        print(f"SimpleSegmenter initialized:")
        print(f"  Backbone: {backbone.config.name}")
        print(f"  Channels: {channels}")
        print(f"  Head: Multi-scale UNetDecoder")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        # Extract features (returns list)
        features = self.backbone.extract_features(x)
        
        # Decode
        logits = self.head(features)
        
        return logits
    
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
