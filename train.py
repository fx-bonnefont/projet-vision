"""
SegDino training entry point for single-device (local) runs.

Supports both segmentation (mask) and center detection modes.
"""
import argparse
import os
import random
import time
import uuid
from datetime import datetime

import torch
from tqdm import tqdm

from model import SegDino, DECODERS
from dataset import PreTiledDataset
from loss import get_loss, LOSSES
from utils import calculate_dice_iou, get_model_stats


def segdino_collate(batch):
    """
    Custom collate fn. Stacks images and targets, returns meta as a list of dicts.
    This is needed because `meta['centers']` can have variable length.
    """
    imgs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    metas = [item[2] for item in batch]
    return imgs, targets, metas


def generate_run_id(decoder_name: str, target_type: str) -> str:
    """Generate unique run ID with timestamp, decoder and target type."""
    return f"{datetime.now().strftime('%Y%m%d_%H')}_{decoder_name}_{target_type}_{uuid.uuid4().hex[:4]}"


def train_epoch(model, loader, criterion, optimizer, device, rank, target_type):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    total_grad_norm, total_weight_norm = 0.0, 0.0

    # Progress bar only on rank 0
    iterator = tqdm(loader, desc="Train") if rank == 0 else loader

    for imgs, targets, _ in iterator:
        imgs, targets = imgs.to(device), targets.to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)

        # Backward
        loss.backward()

        # Gradient stats (before clipping)
        grad_norm, weight_norm = get_model_stats(model)
        total_grad_norm += grad_norm
        total_weight_norm += weight_norm

        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

        # Cleanup
        del logits, loss, imgs, targets

    n = len(loader)
    return total_loss / n, total_grad_norm / n, total_weight_norm / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank, target_type):
    """Evaluation on validation set."""
    model.eval()
    total_loss = 0.0
    total_metric = 0.0  # IoU for mask, MSE for center

    # Progress bar only on rank 0
    iterator = tqdm(loader, desc="Eval") if rank == 0 else loader

    for imgs, targets, _ in iterator:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)

        total_loss += criterion(logits, targets).item()

        if target_type == "mask":
            # Use IoU for segmentation
            _, iou = calculate_dice_iou(logits, targets)
            total_metric += iou
        else:
            # Use negative MSE (higher is better) for center detection
            pred = torch.sigmoid(logits)
            mse = ((pred - targets) ** 2).mean().item()
            total_metric += (1.0 - mse)  # Convert to "higher is better"

        del logits, imgs, targets

    n = len(loader)
    return total_loss / n, total_metric / n


def save_checkpoint(model, path: str, target_type: str, loss_name: str):
    """Save checkpoint with model config."""
    config = model.get_config()
    config["target_type"] = target_type
    config["loss"] = loss_name
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="SegDino Training")
    parser.add_argument("--data_dir", default="segdata/DOTA/DOTA_PLANES_TILED")
    parser.add_argument(
        "--model_size",
        default="small",
        choices=["small", "small-plus", "base", "large", "huge", "giant", "large-sat", "giant-sat"],
    )
    parser.add_argument(
        "--decoder",
        default="light",
        choices=list(DECODERS.keys()),
        help=f"Decoder architecture. Available: {list(DECODERS.keys())}",
    )
    parser.add_argument(
        "--target_type",
        default="center",
        choices=["mask", "center"],
        help="Target type: 'mask' for segmentation, 'center' for center detection",
    )
    parser.add_argument(
        "--loss",
        default="mse",
        choices=list(LOSSES.keys()),
        help=f"Loss function. Available: {list(LOSSES.keys())}",
    )
    parser.add_argument("--sigma", type=float, default=8.0, help="Gaussian sigma for center heatmaps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU (effective batch = batch_size × num_gpus)",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ========================
    # Device & Seeding
    # ========================
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rank = 0
    world_size = 1

    print(f"[Device] Using {device}")

    # ========================
    # Run ID and Logging
    # ========================
    run_id = generate_run_id(args.decoder, args.target_type)
    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    best_pth_path = os.path.join(log_dir, f"{run_id}_best.pth")
    csv_path = os.path.join(log_dir, f"{run_id}_log.csv")

    print(f"\n{'='*60}")
    print(f"SegDino Training: {run_id}")
    print(f"{'='*60}")
    print(f"Backbone: {args.model_size}")
    print(f"Decoder: {args.decoder}")
    print(f"Target: {args.target_type}" + (f" (sigma={args.sigma})" if args.target_type == 'center' else ""))
    print(f"Loss: {args.loss}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr} | Epochs: {args.epochs}")
    print(f"Data: {args.data_dir}")
    print(f"{'='*60}\n")

    # ========================
    # Model
    # ========================
    model = SegDino(
        model_size=args.model_size,
        decoder_name=args.decoder,
        freeze_backbone=True
    ).to(device)

    # ========================
    # Optimizer & Scheduler
    # ========================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ========================
    # Loss Function
    # ========================
    criterion = get_loss(args.loss).to(device)

    # ========================
    # Data
    # ========================
    train_dataset = PreTiledDataset(args.data_dir, "train", target_type=args.target_type, sigma=args.sigma)
    val_dataset = PreTiledDataset(args.data_dir, "test", target_type=args.target_type, sigma=args.sigma)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        collate_fn=segdino_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        collate_fn=segdino_collate,
    )

    # ========================
    # Logging Init
    # ========================
    metric_name = "val_iou" if args.target_type == "mask" else "val_score"
    if rank == 0:
        with open(csv_path, "w") as f:
            f.write(f"# SegDino Training Log\n")
            f.write(f"# Run ID: {run_id}\n")
            f.write(f"# Backbone: {args.model_size}\n")
            f.write(f"# Decoder: {args.decoder}\n")
            f.write(f"# Target: {args.target_type}\n")
            f.write(f"# Loss: {args.loss}\n")
            f.write(f"# Sigma: {args.sigma}\n")
            f.write(f"# Batch size: {args.batch_size}\n")
            f.write(f"# Device: {device}\n")
            f.write(f"# LR: {args.lr}\n")
            f.write(f"# Epochs: {args.epochs}\n")
            f.write(f"# Data: {args.data_dir}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#\n")
            f.write(f"epoch,train_loss,val_loss,{metric_name},grad_norm,weight_norm,lr,duration\n")

    # ========================
    # Zero-Epoch Mode: Save initial weights and exit
    # ========================
    if args.epochs == 0:
        init_pth_path = os.path.join(log_dir, f"{run_id}_initial_weights.pth")
        print(f"\n--epochs=0 detected. Saving initial model weights...")
        save_checkpoint(model, init_pth_path, args.target_type, args.loss)
        print(f"Initial weights saved to: {init_pth_path}")
        print("Exiting.")
        return

    best_metric = 0.0

    # ========================
    # Training Loop
    # ========================
    epoch_iterator = tqdm(range(1, args.epochs + 1), desc="Total Progress", unit="epoch")

    for epoch in epoch_iterator:
        start_time = time.time()

        # Train & Evaluate
        t_loss, t_grad, t_weight = train_epoch(
            model, train_loader, criterion, optimizer, device, rank, args.target_type
        )
        v_loss, v_metric = evaluate(
            model, val_loader, criterion, device, rank, args.target_type
        )

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start_time

        # Logging (Rank 0 only)
        if rank == 0:
            metric_label = "ValIoU" if args.target_type == "mask" else "ValScore"
            log_msg = (
                f"Epoch {epoch}/{args.epochs} | "
                f"TrLoss: {t_loss:.4f} | {metric_label}: {v_metric:.4f} | "
                f"Grad: {t_grad:.2f} | LR: {current_lr:.2e} | Time: {duration:.1f}s"
            )
            tqdm.write(log_msg)

            with open(csv_path, "a") as f:
                f.write(
                    f"{epoch},{t_loss:.4f},{v_loss:.4f},{v_metric:.4f},"
                    f"{t_grad:.4f},{t_weight:.4f},{current_lr:.2e},{duration:.1f}\n"
                )

            # Save best model
            if v_metric > best_metric:
                best_metric = v_metric
                save_checkpoint(model, best_pth_path, args.target_type, args.loss)
                tqdm.write(f"  >> New Best {metric_label}: {best_metric:.4f} (Saved)")

    # ========================
    # Cleanup
    # ========================
    if rank == 0:
        metric_label = "IoU" if args.target_type == "mask" else "Score"
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation {metric_label}: {best_metric:.4f}")
        print(f"Model: {best_pth_path}")
        print(f"Logs: {csv_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
