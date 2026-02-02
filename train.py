"""
SegDino Training Script with DistributedDataParallel Support.

Automatically detects single-GPU vs multi-GPU mode.
"""
import argparse
import os
import random
import time
import uuid
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import SegDino
from dataset import PreTiledDataset
from loss import ComboLoss
from utils import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    get_rank,
    get_world_size,
    calculate_dice_iou,
    get_model_stats,
)


def generate_run_id() -> str:
    """Generate unique run ID with timestamp."""
    return f"{datetime.now().strftime('%Y%m%d_%H')}_{uuid.uuid4().hex[:4]}"


def train_epoch(model, loader, criterion, optimizer, device, rank):
    """Single training epoch."""
    model.train()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    total_grad_norm, total_weight_norm = 0.0, 0.0

    # Progress bar only on rank 0
    iterator = tqdm(loader, desc="Train") if rank == 0 else loader

    for imgs, masks, _ in iterator:
        imgs, masks = imgs.to(device), masks.to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)

        # Backward
        loss.backward()

        # Gradient stats (before clipping)
        grad_norm, weight_norm = get_model_stats(model)
        total_grad_norm += grad_norm
        total_weight_norm += weight_norm

        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # Metrics
        dice, iou = calculate_dice_iou(logits, masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        # Cleanup
        del logits, loss, imgs, masks

    n = len(loader)
    return (
        total_loss / n,
        total_dice / n,
        total_iou / n,
        total_grad_norm / n,
        total_weight_norm / n,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, rank):
    """Evaluation on validation set."""
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0

    # Progress bar only on rank 0
    iterator = tqdm(loader, desc="Eval") if rank == 0 else loader

    for imgs, masks, _ in iterator:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)

        total_loss += criterion(logits, masks).item()
        dice, iou = calculate_dice_iou(logits, masks)
        total_dice += dice
        total_iou += iou

        del logits, imgs, masks

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser(description="SegDino Training")
    parser.add_argument("--data_dir", default="segdata/DOTA/DOTA_PLANES_TILED")
    parser.add_argument(
        "--model_size",
        default="vit-base",
        choices=[
            "vit-small",
            "vit-small-plus",
            "vit-base",
            "vit-large",
            "vit-huge",
            "vit-giant",
            "vit-large-sat",
            "vit-giant-sat",
        ],
    )
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
    # Distributed Setup
    # ========================
    is_distributed = ("RANK" in os.environ) or ("SLURM_PROCID" in os.environ)

    if is_distributed:
        rank, local_rank, world_size = setup_ddp()
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

        # Different seed per rank for data diversity
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)

        if is_main_process():
            print(f"[DDP] Initialized: rank={rank}, world_size={world_size}")
            print(f"[DDP] Effective batch: {args.batch_size} × {world_size} = {args.batch_size * world_size}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        print(f"[Single GPU] Device: {device}")

    # ========================
    # Run ID and Logging
    # ========================
    if is_main_process():
        run_id = generate_run_id()
        log_dir = "runs"
        os.makedirs(log_dir, exist_ok=True)
        best_pth_path = os.path.join(log_dir, f"{run_id}_best.pth")
        csv_path = os.path.join(log_dir, f"{run_id}_log.csv")

        print(f"\n{'='*60}")
        print(f"SegDino Training: {run_id}")
        print(f"{'='*60}")
        print(f"Model: {args.model_size}")
        print(f"Batch per GPU: {args.batch_size} | Total GPUs: {world_size}")
        print(f"LR: {args.lr} | Epochs: {args.epochs}")
        print(f"Data: {args.data_dir}")
        print(f"{'='*60}\n")

    # Broadcast run_id to all ranks
    if is_distributed:
        if is_main_process():
            run_id_list = [run_id]
        else:
            run_id_list = [None]
        dist.broadcast_object_list(run_id_list, src=0)
        run_id = run_id_list[0]

        # All ranks need paths
        log_dir = "runs"
        best_pth_path = os.path.join(log_dir, f"{run_id}_best.pth")
        csv_path = os.path.join(log_dir, f"{run_id}_log.csv")

    # ========================
    # Model
    # ========================
    model = SegDino(model_size=args.model_size, freeze_backbone=True).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ========================
    # Optimizer & Scheduler
    # ========================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Create criterion ONCE (not per epoch!)
    criterion = ComboLoss().to(device)

    # ========================
    # Data
    # ========================
    train_dataset = PreTiledDataset(args.data_dir, "train")
    val_dataset = PreTiledDataset(args.data_dir, "test")

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )

    # ========================
    # Logging Init
    # ========================
    if is_main_process():
        with open(csv_path, "w") as f:
            f.write(f"# SegDino Training Log\n")
            f.write(f"# Run ID: {run_id}\n")
            f.write(f"# Model: {args.model_size}\n")
            f.write(f"# Batch per GPU: {args.batch_size}\n")
            f.write(f"# Num GPUs: {world_size}\n")
            f.write(f"# Effective Batch: {args.batch_size * world_size}\n")
            f.write(f"# LR: {args.lr}\n")
            f.write(f"# Epochs: {args.epochs}\n")
            f.write(f"# Data: {args.data_dir}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#\n")
            f.write("epoch,train_loss,val_loss,val_dice,val_iou,grad_norm,weight_norm,lr,duration\n")

    # ========================
    # Zero-Epoch Mode: Save initial weights and exit
    # ========================
    if args.epochs == 0:
        if is_main_process():
            init_pth_path = os.path.join(log_dir, f"{run_id}_initial_weights.pth")
            print(f"\n--epochs=0 detected. Saving initial model weights...")
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(model_to_save.state_dict(), init_pth_path)
            print(f"Initial weights saved to: {init_pth_path}")
            print("Exiting.")
        
        if is_distributed:
            dist.barrier()
            cleanup_ddp()
        return # Exit main function

    best_iou = 0.0

    # ========================
    # Training Loop
    # ========================
    epoch_iterator = (
        tqdm(range(1, args.epochs + 1), desc="Total Progress", unit="epoch")
        if is_main_process()
        else range(1, args.epochs + 1)
    )

    for epoch in epoch_iterator:
        start_time = time.time()

        # Set epoch for DistributedSampler (shuffle differently each epoch)
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train & Evaluate
        t_loss, t_dice, t_iou, t_grad, t_weight = train_epoch(
            model, train_loader, criterion, optimizer, device, rank
        )
        v_loss, v_dice, v_iou = evaluate(model, val_loader, criterion, device, rank)

        # Synchronize metrics across GPUs
        if is_distributed:
            metrics = torch.tensor(
                [t_loss, v_loss, v_dice, v_iou, t_grad, t_weight], device=device
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            t_loss, v_loss, v_dice, v_iou, t_grad, t_weight = metrics.cpu().tolist()

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start_time

        # Logging (Rank 0 only)
        if is_main_process():
            log_msg = (
                f"Epoch {epoch}/{args.epochs} | "
                f"TrLoss: {t_loss:.4f} | ValIoU: {v_iou:.4f} | "
                f"Grad: {t_grad:.2f} | LR: {current_lr:.2e} | Time: {duration:.1f}s"
            )
            tqdm.write(log_msg)

            with open(csv_path, "a") as f:
                f.write(
                    f"{epoch},{t_loss:.4f},{v_loss:.4f},{v_dice:.4f},{v_iou:.4f},"
                    f"{t_grad:.4f},{t_weight:.4f},{current_lr:.2e},{duration:.1f}\n"
                )

            # Save best model
            if v_iou > best_iou:
                best_iou = v_iou
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(model_to_save.state_dict(), best_pth_path)
                tqdm.write(f"  >> New Best IoU: {best_iou:.4f} (Saved)")

        # Synchronization barrier
        if is_distributed:
            dist.barrier()

    # ========================
    # Cleanup
    # ========================
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation IoU: {best_iou:.4f}")
        print(f"Model: {best_pth_path}")
        print(f"Logs: {csv_path}")
        print(f"{'='*60}\n")

    if is_distributed:
        cleanup_ddp()


if __name__ == "__main__":
    main()
