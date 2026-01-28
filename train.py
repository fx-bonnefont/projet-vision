"""
Training script for the binary segmentation model.
Supports multiple backbones via --backbone argument.
"""
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

from backbones import list_backbones
from dataset import get_dataloader
from model import build_model, save_model, get_image_size, get_feature_size, DEFAULT_BACKBONE


def train(
    image_dir: str,
    label_dir: str,
    val_image_dir: str | None = None,
    val_label_dir: str | None = None,
    output_path: str = 'model.pth',
    backbone_name: str = DEFAULT_BACKBONE,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str | None = None
):
    """Train the segmentation model."""

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Get sizes for this backbone
    img_size = get_image_size(backbone_name)
    feat_size = get_feature_size(backbone_name, img_size)

    # Create train dataloader
    train_dataloader = get_dataloader(
        image_dir=image_dir,
        label_dir=label_dir,
        img_size=img_size,
        feat_size=feat_size,
        batch_size=batch_size,
        shuffle=True
    )

    # Create val dataloader if paths provided
    val_dataloader = None
    if val_image_dir and val_label_dir:
        print(f"Validation enabled using {val_image_dir}")
        val_dataloader = get_dataloader(
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            img_size=img_size,
            feat_size=feat_size,
            batch_size=batch_size,
            shuffle=False
        )

    # Create model
    model = build_model(backbone_name=backbone_name, device=device)

    # Only train the head (backbone is frozen)
    trainable_params = list(model.head.parameters())
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(trainable_params, lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            predictions = model(images)

            # Compute loss
            loss = criterion(predictions, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation loop
        val_info = ""
        if val_dataloader:
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for images, masks in val_dataloader:
                    images = images.to(device)
                    masks = masks.to(device)
                    predictions = model(images)
                    loss = criterion(predictions, masks)
                    total_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            val_info = f" | Val Loss: {avg_val_loss:.4f}"

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}{val_info}")

    # Save model
    save_model(model, output_path)
    print(f"Training complete! Model saved to {output_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train binary segmentation model')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                        help='Directory containing label files')
    parser.add_argument('--val-images', type=str, default=None,
                        help='Directory containing validation images (optional)')
    parser.add_argument('--val-labels', type=str, default=None,
                        help='Directory containing validation label files (optional)')
    parser.add_argument('--output', type=str, default='model.pth',
                        help='Output path for model weights')
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE,
                        choices=list_backbones(),
                        help=f'Backbone to use (default: {DEFAULT_BACKBONE})')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, mps, cpu)')

    args = parser.parse_args()

    train(
        image_dir=args.images,
        label_dir=args.labels,
        val_image_dir=args.val_images,
        val_label_dir=args.val_labels,
        output_path=args.output,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )


if __name__ == '__main__':
    main()
