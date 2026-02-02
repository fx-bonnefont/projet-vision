"""
Visualization utilities for inference.
"""
import torch
import cv2
import numpy as np


# DOTA dataset statistics (from dataset.py)
DOTA_MEAN = (0.4733, 0.4668, 0.4366)
DOTA_STD = (0.1846, 0.1832, 0.1800)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image tensor using DOTA statistics.

    Args:
        tensor: Normalized image (C, H, W) or (B, C, H, W)

    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(DOTA_MEAN).view(-1, 1, 1)
    std = torch.tensor(DOTA_STD).view(-1, 1, 1)

    # Handle batch dimension
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def draw_bboxes(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes around mask regions.

    Args:
        img: BGR image (H, W, 3)
        mask: Binary mask (H, W)
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    out = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:  # Filter tiny noise
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

    return out
