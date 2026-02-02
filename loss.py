"""
Loss functions for SegDino.
"""
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    """
    Weighted combination of BCE and Dice Loss.
    """
    def __init__(self, alpha=0.5, bce_weight=None):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=bce_weight)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        return self.alpha * loss_bce + (1 - self.alpha) * loss_dice
