"""
Loss functions for SegDino.

Supports both segmentation (mask) and center detection (heatmap) modes.
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)
             Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W) - NOT sigmoid activated
            target: Binary ground truth (B, 1, H, W) in {0, 1}

        Returns:
            Dice loss scalar
        """
        pred = torch.sigmoid(logits)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        cardinality_sum = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality_sum + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    """
    Combination of BCE and Dice Loss for segmentation.

    Default: 0.5 * BCE + 0.5 * Dice
    """

    def __init__(self, alpha: float = 0.5, pos_weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(logits, target)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in center detection.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Good for sparse targets like center heatmaps.
    """

    def __init__(self, alpha: float = 2.0, gamma: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W)
            target: Gaussian heatmap (B, 1, H, W) in [0, 1]

        Returns:
            Focal loss scalar
        """
        pred = torch.sigmoid(logits)
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)  # Numerical stability

        # Positive and negative focal weights
        pos_weight = (1 - pred) ** self.alpha
        neg_weight = pred ** self.alpha

        # Focal loss
        pos_loss = -target * pos_weight * torch.log(pred)
        neg_loss = -(1 - target) ** self.gamma * neg_weight * torch.log(1 - pred)

        loss = pos_loss + neg_loss
        return loss.mean()


class MSELoss(nn.Module):
    """
    MSE Loss for center heatmap regression.

    Simple but effective for Gaussian heatmaps.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W)
            target: Gaussian heatmap (B, 1, H, W) in [0, 1]

        Returns:
            MSE loss scalar
        """
        pred = torch.sigmoid(logits)
        return self.mse(pred, target)


class CenterLoss(nn.Module):
    """
    Combined loss for center detection: MSE + Focal.

    Default: 0.5 * MSE + 0.5 * Focal
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = MSELoss()
        self.focal = FocalLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(logits, target)
        focal_loss = self.focal(logits, target)
        return self.alpha * mse_loss + (1.0 - self.alpha) * focal_loss


# Registry for easy access
LOSSES = {
    "combo": ComboLoss,      # For mask segmentation
    "dice": DiceLoss,        # For mask segmentation
    "mse": MSELoss,          # For center detection
    "focal": FocalLoss,      # For center detection
    "center": CenterLoss,    # For center detection (MSE + Focal)
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """Get loss function by name."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSSES.keys())}")
    return LOSSES[name](**kwargs)
