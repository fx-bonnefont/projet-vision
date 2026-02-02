"""
Metrics for binary segmentation.

All metrics are mathematically consistent with loss.py
"""
import torch


def calculate_dice_iou(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """
    Calculate Dice coefficient and IoU for binary segmentation.

    Mathematically consistent with DiceLoss in loss.py:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        IoU = |A ∩ B| / |A ∪ B|

    Args:
        logits: Raw model output (B, 1, H, W)
        target: Binary ground truth (B, 1, H, W) in {0, 1}

    Returns:
        tuple: (dice, iou) as floats
    """
    pred = (torch.sigmoid(logits) > 0.5).float()

    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Intersection
    intersection = (pred_flat * target_flat).sum()

    # Dice: 2 * |A ∩ B| / (|A| + |B|)
    cardinality_sum = pred_flat.sum() + target_flat.sum()
    dice = (2.0 * intersection / (cardinality_sum + 1e-6)).item()

    # IoU: |A ∩ B| / |A ∪ B|
    # Since |A ∪ B| = |A| + |B| - |A ∩ B|
    union = cardinality_sum - intersection
    iou = (intersection / (union + 1e-6)).item()

    return dice, iou


def get_model_stats(model: torch.nn.Module) -> tuple[float, float]:
    """
    Compute L2 norms of gradients and weights.

    Handles both regular and DDP-wrapped models.

    Args:
        model: PyTorch model (can be DDP-wrapped)

    Returns:
        tuple: (grad_norm, weight_norm) as floats
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Unwrap DDP if necessary
    if isinstance(model, DDP):
        model = model.module

    grad_norm_sq = 0.0
    weight_norm_sq = 0.0

    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += param.grad.data.norm(2).item() ** 2
        if param.data is not None:
            weight_norm_sq += param.data.norm(2).item() ** 2

    return grad_norm_sq ** 0.5, weight_norm_sq ** 0.5
