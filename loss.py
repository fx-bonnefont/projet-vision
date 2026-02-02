"""
Loss functions for SegDino.

Mathematical correctness verified.
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Formula: Dice = 2 * |A ∩ B| / (|A| + |B|)
             Loss = 1 - Dice

    Note: Smoothing is added to numerator and denominator for numerical stability.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W) - NOT sigmoidactivated
            target: Binary ground truth (B, 1, H, W) in {0, 1}

        Returns:
            Dice loss scalar
        """
        pred = torch.sigmoid(logits)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()

        # Correct formula: |A| + |B| = sum(A) + sum(B) because binary
        cardinality_sum = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality_sum + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    """
    Combination of BCE and Dice Loss.

    Default: 0.5 * BCE + 0.5 * Dice
    """

    def __init__(self, alpha: float = 0.5, pos_weight: torch.Tensor = None):
        """
        Args:
            alpha: Weight for BCE (1-alpha for Dice). Default: 0.5
            pos_weight: Positive class weight for BCE. Default: None (balanced)
        """
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W)
            target: Binary ground truth (B, 1, H, W) in {0, 1}

        Returns:
            Combined loss scalar
        """
        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(logits, target)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss
