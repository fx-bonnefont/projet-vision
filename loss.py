import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.
    Penalizes low overlap between prediction and ground truth.
    Smooth parameter prevents division by zero.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Flatten
        # logits: [B, 1, H, W] -> sigmoid -> [B, N]
        # targets: [B, 1, H, W] -> [B, N]
        
        preds = torch.sigmoid(logits)
        
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        
        # Dice = 2 * (A n B) / (|A| + |B|)
        dice_score = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice_score

class BCEDiceLoss(nn.Module):
    """
    Combination of BCE With Logits Loss and Dice Loss.
    BCE handles pixel-wise classification.
    Dice handles global shape and overlap.
    """
    def __init__(self, pos_weight=None, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()
            
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
