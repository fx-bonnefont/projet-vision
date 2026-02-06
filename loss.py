"""
Loss function for SegDino center detection.
"""
import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Combined loss for center detection: MSE + Focal.

    MSE provides smooth regression, Focal handles class imbalance.
    """

    def __init__(self, alpha: float = 0.5, focal_alpha: float = 2.0, focal_gamma: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.mse = nn.MSELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (B, 1, H, W)
            target: Gaussian heatmap (B, 1, H, W) in [0, 1]

        Returns:
            Combined loss scalar
        """
        pred = torch.sigmoid(logits)

        # MSE loss
        mse_loss = self.mse(pred, target)

        # Focal loss
        pred_clamped = pred.clamp(min=1e-6, max=1 - 1e-6)
        pos_weight = (1 - pred_clamped) ** self.focal_alpha
        neg_weight = pred_clamped ** self.focal_alpha
        pos_loss = -target * pos_weight * torch.log(pred_clamped)
        neg_loss = -(1 - target) ** self.focal_gamma * neg_weight * torch.log(1 - pred_clamped)
        focal_loss = (pos_loss + neg_loss).mean()

        return self.alpha * mse_loss + (1.0 - self.alpha) * focal_loss
