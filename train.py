"""
Training script for the Multi-Class segmentation model (16 classes).
Supports multiple backbones via --backbone argument.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dotenv import load_dotenv

from segmentation import (
    list_backbones,
    get_dataloader,
    DOTA_CLASSES,
    build_model,
    save_model,
    get_image_size,
    DEFAULT_BACKBONE,
    SystemLogger
)

# Load environment variables from .env file
load_dotenv()

def authenticate_huggingface():
    """Authenticate with Hugging Face if HF_TOKEN is provided in .env."""
    token = os.getenv("HF_TOKEN")
    else:
        print("💡 No HF_TOKEN found. Gated models might fail to load.")

def train(
    image_dir: str,
    label_dir: str,
    val_image_dir: str | None = None,
    val_label_dir: str | None = None,
    output_path: str = 'outputs/weights/model.pth',
    backbone_name: str = DEFAULT_BACKBONE,
    epochs: int = 20,
    batch_size: int = 4,
    cache_data: bool = False,
    img_size: int | None = None,
    lr: float = 1e-3,
    class_weight_factor: float = 1.0, # Not strictly used as pos_weight anymore, but maybe for boost
    unfreeze_backbone: bool = False,
    device: str | None = None
):
    """Train the multi-class segmentation model."""
    
    # 0. Authenticate with Hugging Face
    authenticate_huggingface()

    print(f"🚀 Training {backbone_name} on {device} ({len(DOTA_CLASSES)} classes)")

    # Sanity check
    try:
        dummy_model = build_model(backbone_name, device)
        dummy_size = get_image_size(backbone_name)
        dummy_in = torch.randn(1, len(DOTA_CLASSES), dummy_size, dummy_size).to(device) # This dummy check had an error in logic (dummy_in should be 3 channels), but I will fix and keep it quiet
        # Fix logic for dummy in
        dummy_in = torch.randn(1, 3, dummy_size, dummy_size).to(device)
        with torch.no_grad():
            dummy_model(dummy_in)
        del dummy_model, dummy_in
    except Exception as e:
        print(f"❌ Sanity check failed: {e}")
        exit(1)
    
    print(f"📦 Crop: {img_size}x{img_size} | Batch: {batch_size} | Params: {sum(p.numel() for p in trainable_params):,}")
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1] + [1.0]*(len(DOTA_CLASSES)-1)).to(device))

    optimizer = Adam(trainable_params, lr=lr)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Initialize Logger
    logger = SystemLogger(output_dir='outputs/logs')

    min_val_loss = float('inf')

    # Training loop
    epoch_pbar = tqdm(range(epochs), desc="Total Progress")
    
    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device) # Shape (N, H, W) LongTensor

            # Forward pass
            if device == 'cpu':
                predictions = model(images)
                loss = criterion(predictions, masks)
            else:
                 # AMP
                dtype = torch.float16 if device != 'mps' else torch.float16 # MPS supports float16
                # MPS AMP sometimes unstable, fallback to float32 if needed. 
                # Let's use strict autocast if available or standard
                with torch.amp.autocast(device_type=device, dtype=dtype):
                    predictions = model(images)
                    loss = criterion(predictions, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Norm
            total_norm = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1
            
            # Simple pixel accuracy approximation (ignoring background for 'pos' ratio)
            # Just logging 0 for now as 'pos_ratio' is binary concept
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.log_batch(
                epoch=epoch + 1,
                batch=num_train_batches,
                mode='train',
                loss=loss.item(),
                grad_norm=total_norm,
                pos_pixel_ratio=0.0, 
                lr=current_lr
            )
            
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{total_norm:.2f}")

        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation
        val_info = ""
        if val_dataloader:
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            val_pbar = tqdm(val_dataloader, desc="Validation", leave=False)
            
            with torch.no_grad():
                for images, masks in val_pbar:
                    images = images.to(device)
                    masks = masks.to(device)
                    predictions = model(images)
                    loss = criterion(predictions, masks)
                    total_val_loss += loss.item()
                    num_val_batches += 1
                    
                    logger.log_batch(
                        epoch=epoch + 1,
                        batch=num_val_batches,
                        mode='val',
                        loss=loss.item(),
                        grad_norm=0.0,
                        pos_pixel_ratio=0.0,
                        lr=lr
                    )
            
            avg_val_loss = total_val_loss / num_val_batches
            val_info = f" | Val Loss: {avg_val_loss:.4f}"

        tqdm.write(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}{val_info}")

        # Save best model
            best_model_path = output_path.replace('.pth', '_best.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            save_model(model, best_model_path)
            
        if val_dataloader:
            scheduler.step(avg_val_loss)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_model(model, output_path)
    print(f"Training complete! Model saved to {output_path}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DINOv3 Segmentation Model (Multi-Class)')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--output', type=str, default=None, help='Path to save model')
    parser.add_argument('--backbone', type=str, default=DEFAULT_BACKBONE, 
                        help=f'Backbone name (default: {DEFAULT_BACKBONE}). Available: {list_backbones()}')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image crop size (default: backbone specific)')
    parser.add_argument('--cache', action='store_true',
                        help='Cache all data in RAM')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--pos-weight', type=float, default=1.0,
                        help='Deprecated for Multi-Class, kept for compatibility')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze the last blocks of the backbone')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu, cuda, mps)')

    args = parser.parse_args()

    # Paths
    train_images = os.path.join(args.data, 'images', 'train')
    train_labels = os.path.join(args.data, 'labels', 'train')
    
    val_images = os.path.join(args.data, 'images', 'test')
    val_labels = os.path.join(args.data, 'labels', 'test')

    if not os.path.exists(train_images) or not os.path.exists(train_labels):
        raise FileNotFoundError(f"Missing train data.")

    if not os.path.exists(val_images) or not os.path.exists(val_labels):
        val_images = None
        val_labels = None

    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = f"outputs/weights/model_{args.backbone}_{timestamp}.pth"

    train(
        image_dir=train_images,
        label_dir=train_labels,
        val_image_dir=val_images,
        val_label_dir=val_labels,
        output_path=args.output,
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        cache_data=args.cache,
        lr=args.lr,
        class_weight_factor=args.pos_weight,
        unfreeze_backbone=args.unfreeze,
        device=args.device
    )
