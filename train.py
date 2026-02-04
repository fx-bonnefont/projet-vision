"""
SegDino training entry point for single-device (local) runs.

Supports both segmentation (mask) and center detection modes.
Supports training with pre-cached backbone features for faster experimentation.
"""
import argparse
import os
import random
import time
import uuid
from datetime import datetime

import torch
from tqdm import tqdm

from model import SegDino, DecoderOnly, DECODERS
from dataset import PreTiledDataset, CachedFeaturesDataset
from loss import get_loss, LOSSES
from utils import calculate_dice_iou, get_model_stats
from inference import find_peaks, match_centers
from extract_features import extract_and_save_features, get_cache_dir


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
def evaluate(model, loader, criterion, device, rank, target_type, threshold=0.3, match_radius=20.0):
    """
    Evaluation on validation set.

    For center detection mode, computes real detection metrics (F1 score)
    instead of pixel-wise MSE, using peak detection and center matching.
    """
    model.eval()
    total_loss = 0.0

    # For mask mode: accumulate IoU
    total_iou = 0.0

    # For center mode: accumulate TP, FP, FN for F1 calculation
    total_tp, total_fp, total_fn = 0, 0, 0

    # Progress bar only on rank 0
    iterator = tqdm(loader, desc="Eval") if rank == 0 else loader

    for imgs, targets, metas in iterator:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)

        total_loss += criterion(logits, targets).item()

        if target_type == "mask":
            # Use IoU for segmentation
            _, iou = calculate_dice_iou(logits, targets)
            total_iou += iou
        else:
            # Center detection: compute real detection metrics
            pred = torch.sigmoid(logits).cpu().numpy()

            for i in range(pred.shape[0]):
                # Get predicted centers from heatmap
                pred_heatmap = pred[i].squeeze()
                pred_centers = find_peaks(pred_heatmap, threshold=threshold)

                # Get GT centers from metadata
                gt_centers = metas[i].get('centers', [])

                # Match and count
                tp, fp, fn = match_centers(gt_centers, pred_centers, match_radius=match_radius)
                total_tp += tp
                total_fp += fp
                total_fn += fn

        del logits, imgs, targets

    n = len(loader)
    avg_loss = total_loss / n

    if target_type == "mask":
        return avg_loss, total_iou / n
    else:
        # Compute F1 score from accumulated TP, FP, FN
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return avg_loss, f1, precision, recall, total_tp, total_fp, total_fn


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


def check_cached_features(data_dir: str, model_size: str) -> bool:
    """Check if cached features exist for the given model_size."""
    cache_dir = get_cache_dir(data_dir)
    model_cache_dir = os.path.join(cache_dir, model_size)
    train_cache = os.path.join(model_cache_dir, "train")
    test_cache = os.path.join(model_cache_dir, "test")

    if not os.path.exists(train_cache) or not os.path.exists(test_cache):
        return False

    # Check if there are actual files
    train_files = [f for f in os.listdir(train_cache) if f.endswith(".pt")]
    test_files = [f for f in os.listdir(test_cache) if f.endswith(".pt")]

    return len(train_files) > 0 and len(test_files) > 0


def main():
    parser = argparse.ArgumentParser(description="SegDino Training")
    parser.add_argument("--data_dir", default="segdata/DOTA/DOTA_PLANES_TILED",
                        help="Path to tiled dataset")
    parser.add_argument(
        "--model_size",
        default="small",
        choices=["small", "small-plus", "base", "large", "huge", "giant", "large-sat", "giant-sat"],
    )
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable feature caching (force full model training)")
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
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Detection threshold for center mode validation metrics")
    parser.add_argument("--match_radius", type=float, default=20.0,
                        help="Radius for matching predicted centers to GT (in pixels)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU (effective batch = batch_size × num_gpus)",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no_warm_restarts", action="store_true",
                        help="Disable SGDR and use simple CosineAnnealing instead")
    parser.add_argument("--t0", type=int, default=10,
                        help="SGDR: number of epochs before first restart")
    parser.add_argument("--t_mult", type=int, default=1,
                        help="SGDR: multiplier for period after each restart (1 = constant period)")
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

    # ========================
    # Feature Cache Detection
    # ========================
    cache_dir = get_cache_dir(args.data_dir)
    use_cached = False

    if not args.no_cache:
        model_cache_path = os.path.join(cache_dir, args.model_size)
        if check_cached_features(args.data_dir, args.model_size):
            print(f"\n[Cache] Cached features FOUND for '{args.model_size}'")
            print(f"[Cache] Location: {model_cache_path}")
            print(f"[Cache] Using decoder-only training (fast mode)\n")
            use_cached = True
        else:
            print(f"\n[Cache] No cached features for '{args.model_size}'")
            print(f"[Cache] Extracting features to: {model_cache_path}")
            print(f"[Cache] This is a one-time operation...\n")
            extract_and_save_features(
                model_size=args.model_size,
                data_dir=args.data_dir,
                output_dir=cache_dir,
                batch_size=args.batch_size,
                device=device,
                use_fp16=True
            )
            use_cached = True
            print(f"\n[Cache] Feature extraction complete!")
            print(f"[Cache] Using decoder-only training (fast mode)\n")
    else:
        print(f"\n[Cache] Disabled (--no_cache). Using full model.\n")

    # ========================
    # Model
    # ========================
    if use_cached:
        model = DecoderOnly(
            model_size=args.model_size,
            decoder_name=args.decoder
        ).to(device)
    else:
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

    if args.no_warm_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        scheduler_name = "CosineAnnealing"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=1e-6
        )
        scheduler_name = f"SGDR(T0={args.t0}, Tmult={args.t_mult})"

    # ========================
    # Loss Function
    # ========================
    criterion = get_loss(args.loss).to(device)

    # ========================
    # Config Summary
    # ========================
    print(f"\n{'='*60}")
    print(f"SegDino Training: {run_id}")
    print(f"{'='*60}")
    print(f"Backbone: {args.model_size}" + (" (CACHED)" if use_cached else ""))
    print(f"Decoder: {args.decoder}")
    print(f"Target: {args.target_type}" + (f" (sigma={args.sigma})" if args.target_type == 'center' else ""))
    print(f"Loss: {args.loss}")
    if args.target_type == 'center':
        print(f"Detection threshold: {args.threshold} | Match radius: {args.match_radius}px")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr} | Epochs: {args.epochs} | Scheduler: {scheduler_name}")
    print(f"Data: {args.data_dir}")
    if use_cached:
        print(f"Cached features: {cache_dir}")
    print(f"{'='*60}\n")

    # ========================
    # Data
    # ========================
    if use_cached:
        train_dataset = CachedFeaturesDataset(
            cache_dir, args.data_dir, args.model_size,
            "train", target_type=args.target_type, sigma=args.sigma
        )
        val_dataset = CachedFeaturesDataset(
            cache_dir, args.data_dir, args.model_size,
            "test", target_type=args.target_type, sigma=args.sigma
        )
    else:
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
    if rank == 0:
        with open(csv_path, "w") as f:
            f.write(f"# SegDino Training Log\n")
            f.write(f"# Run ID: {run_id}\n")
            f.write(f"# Backbone: {args.model_size}\n")
            f.write(f"# Cached features: {use_cached}\n")
            f.write(f"# Decoder: {args.decoder}\n")
            f.write(f"# Target: {args.target_type}\n")
            f.write(f"# Loss: {args.loss}\n")
            f.write(f"# Sigma: {args.sigma}\n")
            if args.target_type == 'center':
                f.write(f"# Detection threshold: {args.threshold}\n")
                f.write(f"# Match radius: {args.match_radius}\n")
            f.write(f"# Batch size: {args.batch_size}\n")
            f.write(f"# Device: {device}\n")
            f.write(f"# LR: {args.lr}\n")
            f.write(f"# Scheduler: {scheduler_name}\n")
            f.write(f"# Epochs: {args.epochs}\n")
            f.write(f"# Data: {args.data_dir}\n")
            if use_cached:
                f.write(f"# Cached features dir: {cache_dir}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#\n")
            if args.target_type == "mask":
                f.write("epoch,train_loss,val_loss,val_iou,grad_norm,weight_norm,lr,duration\n")
            else:
                f.write("epoch,train_loss,val_loss,val_f1,val_precision,val_recall,tp,fp,fn,grad_norm,weight_norm,lr,duration\n")

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

        if args.target_type == "mask":
            v_loss, v_metric = evaluate(
                model, val_loader, criterion, device, rank, args.target_type
            )
            v_precision, v_recall = None, None
            v_tp, v_fp, v_fn = None, None, None
        else:
            v_loss, v_metric, v_precision, v_recall, v_tp, v_fp, v_fn = evaluate(
                model, val_loader, criterion, device, rank, args.target_type,
                threshold=args.threshold, match_radius=args.match_radius
            )

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start_time

        # Logging (Rank 0 only)
        if rank == 0:
            if args.target_type == "mask":
                log_msg = (
                    f"Epoch {epoch}/{args.epochs} | "
                    f"TrLoss: {t_loss:.4f} | ValIoU: {v_metric:.4f} | "
                    f"Grad: {t_grad:.2f} | LR: {current_lr:.2e} | Time: {duration:.1f}s"
                )
                tqdm.write(log_msg)

                with open(csv_path, "a") as f:
                    f.write(
                        f"{epoch},{t_loss:.4f},{v_loss:.4f},{v_metric:.4f},"
                        f"{t_grad:.4f},{t_weight:.4f},{current_lr:.2e},{duration:.1f}\n"
                    )
            else:
                log_msg = (
                    f"Epoch {epoch}/{args.epochs} | "
                    f"TrLoss: {t_loss:.4f} | F1: {v_metric:.4f} | "
                    f"P: {v_precision:.3f} R: {v_recall:.3f} | "
                    f"TP:{v_tp} FP:{v_fp} FN:{v_fn} | "
                    f"Grad: {t_grad:.2f} | LR: {current_lr:.2e} | Time: {duration:.1f}s"
                )
                tqdm.write(log_msg)

                with open(csv_path, "a") as f:
                    f.write(
                        f"{epoch},{t_loss:.4f},{v_loss:.4f},{v_metric:.4f},"
                        f"{v_precision:.4f},{v_recall:.4f},{v_tp},{v_fp},{v_fn},"
                        f"{t_grad:.4f},{t_weight:.4f},{current_lr:.2e},{duration:.1f}\n"
                    )

            # Save best model
            metric_label = "ValIoU" if args.target_type == "mask" else "F1"
            if v_metric > best_metric:
                best_metric = v_metric
                save_checkpoint(model, best_pth_path, args.target_type, args.loss)
                tqdm.write(f"  >> New Best {metric_label}: {best_metric:.4f} (Saved)")

    # ========================
    # Cleanup
    # ========================
    if rank == 0:
        metric_label = "IoU" if args.target_type == "mask" else "F1"
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation {metric_label}: {best_metric:.4f}")
        print(f"Model: {best_pth_path}")
        print(f"Logs: {csv_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
