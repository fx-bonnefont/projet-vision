"""
Training script for the binary segmentation model.
Supports multiple backbones via --backbone argument.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from system_logger import SystemLogger

from backbones import list_backbones
from dataset import get_dataloader
from model import build_model, save_model, get_image_size, get_feature_size, DEFAULT_BACKBONE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import BCEDiceLoss


def train(
    image_dir: str,
    label_dir: str,
    val_image_dir: str | None = None,
    val_label_dir: str | None = None,
    output_path: str = 'model.pth',
    backbone_name: str = DEFAULT_BACKBONE,
    epochs: int = 20,
    batch_size: int = 4,
    cache_data: bool = False,
    lr: float = 1e-3,
    pos_weight: float = 1.0,
    unfreeze_backbone: bool = False,
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
    
    # -----------------------------------------------------
    # SANITY CHECK: Verify architecture compatibility
    # -----------------------------------------------------
    print("Performing architecture sanity check...")
    try:
        dummy_model = build_model(backbone_name, device)
        dummy_size = get_image_size(backbone_name)
        dummy_in = torch.randn(1, 3, dummy_size, dummy_size).to(device)
        with torch.no_grad():
            dummy_out = dummy_model(dummy_in)
        if dummy_out.shape != (1, 1, dummy_size, dummy_size):
             raise ValueError(f"Output shape mismatch: {dummy_out.shape} vs (1, 1, {dummy_size}, {dummy_size})")
        print("  ✅ Architecture check passed!")
        del dummy_model, dummy_in, dummy_out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"\n❌ ARCHITECTURE CHECK FAILED: {str(e)}")
        print("Aborting training to save time.")
        exit(1)
    # -----------------------------------------------------

    # Get sizes for this backbone
    img_size = get_image_size(backbone_name)
    # feat_size = get_feature_size(backbone_name, img_size) # No longer needed for masks

    # Create train dataloader
    train_dataloader = get_dataloader(
        image_dir=image_dir,
        label_dir=label_dir,
        img_size=img_size,
        # feat_size=feat_size, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, # Optimal for cached data
        cache=cache_data
    )

    # Create val dataloader if paths provided
    val_dataloader = None
    if val_image_dir and val_label_dir:
        print(f"Validation enabled using {val_image_dir}")
        val_dataloader = get_dataloader(
            image_dir=val_image_dir,
            label_dir=val_label_dir,
            img_size=img_size,
            # feat_size=feat_size,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            cache=cache_data
        )

    # Create model
    model = build_model(backbone_name=backbone_name, device=device)

    # Setup trainable parameters
    trainable_params = list(model.head.parameters())
    
    if unfreeze_backbone:
        print("Unfreezing backbone (last layers)...")
        # Try to unfreeze the backbone model
        if hasattr(model.backbone, 'model'):
             for param in model.backbone.model.parameters():
                param.requires_grad = True
             trainable_params += list(model.backbone.model.parameters())
        elif hasattr(model.backbone, 'features'): # ResNet
             for param in model.backbone.features.parameters():
                 param.requires_grad = True
             trainable_params += list(model.backbone.features.parameters())

    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Loss and optimizer
    # Use BCEDiceLoss to enforce better shape boundaries
    pos_weight_tensor = torch.tensor(pos_weight).to(device) if pos_weight > 1.0 else None
    
    criterion = BCEDiceLoss(pos_weight=pos_weight_tensor, bce_weight=0.5, dice_weight=0.5)
    print(f"Using Hybrid Loss: 50% BCE (pos_weight={pos_weight}) + 50% Dice")

    optimizer = Adam(trainable_params, lr=lr)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Initialize Logger
    logger = SystemLogger(output_dir=os.path.dirname(output_path) if output_path else '.')

    min_val_loss = float('inf')

    # Training loop
    # Use tqdm for the epoch loop
    epoch_pbar = tqdm(range(epochs), desc="Total Progress")
    
    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        # Use tqdm for the batch loop (leave=False removes it after completion)
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass with AMP
            if device == 'cpu':
                predictions = model(images)
                loss = criterion(predictions, masks)
            else:
                with torch.amp.autocast(device_type="mps", dtype=torch.float16) if device == "mps" else torch.amp.autocast(device_type=device):
                    predictions = model(images)
                    loss = criterion(predictions, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Compute Gradient Norm (Diagnostic)
            total_norm = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Clip gradients? (Optional, but good for stability)
            # torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1
            
            # Metrics for Logger
            pos_ratio = (masks > 0.5).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.log_batch(
                epoch=epoch + 1,
                batch=num_train_batches,
                mode='train',
                loss=loss.item(),
                grad_norm=total_norm,
                pos_pixel_ratio=pos_ratio,
                lr=current_lr
            )
            
            # Update progress bar
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{total_norm:.2f}", pos=f"{pos_ratio:.3f}")

        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation loop
        val_info = ""
        if val_dataloader:
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            # Add progress bar for validation
            val_pbar = tqdm(val_dataloader, desc="Validation", leave=False)
            
            with torch.no_grad():
                for images, masks in val_pbar:
                    images = images.to(device)
                    masks = masks.to(device)
                    predictions = model(images)
                    loss = criterion(predictions, masks)
                    total_val_loss += loss.item()
                    num_val_batches += 1
                    
                    # Log validation batch (sampled)
                    pos_ratio = (masks > 0.5).float().mean().item()
                    logger.log_batch(
                        epoch=epoch + 1,
                        batch=num_val_batches,
                        mode='val',
                        loss=loss.item(),
                        grad_norm=0.0,
                        pos_pixel_ratio=pos_ratio,
                        lr=lr
                    )
            
            avg_val_loss = total_val_loss / num_val_batches
            val_info = f" | Val Loss: {avg_val_loss:.4f}"

        # Write to tqdm output instead of print to avoid messing up the bars
        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}{val_info}")

        # Save best model
        if val_dataloader and avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_model_path = output_path.replace('.pth', '_best.pth')
            save_model(model, best_model_path)
            tqdm.write(f"  --> New best model saved! (Val Loss: {min_val_loss:.4f})")
            
        # Step scheduler
        if val_dataloader:
            scheduler.step(avg_val_loss)



    # Save model
    save_model(model, output_path)
    print(f"Training complete! Model saved to {output_path}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DINOv3 Segmentation Model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--output', type=str, default=None, help='Path to save model')
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE, 
                        help=f'Backbone name (default: {DEFAULT_BACKBONE}). Available: {list_backbones()}')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (increased for optimization)')
    parser.add_argument('--cache', action='store_true',
                        help='Cache all data in RAM (recommended for small datasets)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--pos-weight', type=float, default=10.0,
                        help='Weight for positive class (objects) in loss to handle imbalance')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze the last blocks of the backbone')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu, cuda, mps)')

    args = parser.parse_args()

    # Define paths based on strict structure
    train_images = os.path.join(args.data, 'images', 'train')
    train_labels = os.path.join(args.data, 'labels', 'train')
    
    val_images = os.path.join(args.data, 'images', 'test')
    val_labels = os.path.join(args.data, 'labels', 'test')

    # Check existence
    if not os.path.exists(train_images) or not os.path.exists(train_labels):
        raise FileNotFoundError(f"Missing train data. Expected:\n  {train_images}\n  {train_labels}")

    # Check for validation data
    has_val = os.path.exists(val_images) and os.path.exists(val_labels)
    if not has_val:
        print(f"No validation data found (expected 'test' folders), skipping validation.")
        val_images = None
        val_labels = None

    # Auto-generate output filename if not provided
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = f"model_{args.backbone}_{timestamp}.pth"
        print(f"Auto-generated output filename: {args.output}")

    train(
        image_dir=train_images,
        label_dir=train_labels,
        val_image_dir=val_images,
        val_label_dir=val_labels,
        output_path=args.output,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cache_data=args.cache,
        lr=args.lr,
        pos_weight=args.pos_weight,
        unfreeze_backbone=args.unfreeze,
        device=args.device
    )



