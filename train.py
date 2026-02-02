"""
SegDino Training Script (Simplified).
"""
import argparse
import os
import random
import time
import uuid
from datetime import datetime

import torch
from tqdm import tqdm

from model import SegDino, DINOV3_MODELS
from dataset import PreTiledDataset
from loss import ComboLoss

def generate_run_id():
    return f"{datetime.now().strftime('%Y%m%d_%H')}_{uuid.uuid4().hex[:4]}"

def calculate_metrics(pred, target):
    """Returns Dice and IoU."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * inter / (union + 1e-6)).item()
    iou = (inter / (union - inter + 1e-6)).item()
    return dice, iou

def get_model_stats(model):
    """Computes L2 norm of weights and gradients."""
    total_norm = 0.0
    weight_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
        if p.data is not None:
            weight_norm += p.data.norm(2).item() ** 2
    return total_norm ** 0.5, weight_norm ** 0.5

def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = ComboLoss().to(device)
    total_loss, total_dice, total_iou = 0, 0, 0
    total_grad_norm, total_weight_norm = 0, 0

    for imgs, masks, _ in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        
        # Stats
        grad_norm, weight_norm = get_model_stats(model)
        total_grad_norm += grad_norm
        total_weight_norm += weight_norm
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
            
        dice, iou = calculate_metrics(logits, masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n, total_grad_norm / n, total_weight_norm / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = ComboLoss().to(device)
    total_loss, total_dice, total_iou = 0, 0, 0

    for imgs, masks, _ in tqdm(loader, desc="Eval"):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        dice, iou = calculate_metrics(logits, masks)
        total_dice += dice
        total_iou += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Volumes/X9Pro/DOTA_PLANES_TILED")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    run_id = generate_run_id()
    
    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    best_pth_path = os.path.join(log_dir, f"{run_id}_best.pth")
    csv_path = os.path.join(log_dir, f"{run_id}_log.csv")

    print(f"--- SegDino Run: {run_id} ---")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} | Batch: {args.batch_size} | LR: {args.lr}")

    model = SegDino(model_size="1", freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_loader = torch.utils.data.DataLoader(
        PreTiledDataset(args.data_dir, "train"), 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        PreTiledDataset(args.data_dir, "test"), 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_dice,val_iou,grad_norm,weight_norm,lr,time\n")

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        t_loss, t_dice, t_iou, t_grad, t_weight = train_epoch(model, train_loader, optimizer, device)
        v_loss, v_dice, v_iou = evaluate(model, val_loader, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {t_loss:.4f} | IoU: {v_iou:.4f} | Grad: {t_grad:.2f} | LR: {current_lr:.2e}")
        
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{t_loss:.4f},{v_loss:.4f},{v_dice:.4f},{v_iou:.4f},{t_grad:.4f},{t_weight:.4f},{current_lr:.2e},{duration:.1f}\n")

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), best_pth_path)
            print(f"  --> New Best IoU: {best_iou:.4f} (Saved)")

if __name__ == "__main__":
    main()