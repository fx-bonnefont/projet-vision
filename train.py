"SegDino Training Script."
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
    iou = (inter / (union - inter + 1e-6)).item() # Correct IoU: I / (A+B-I)
    return dice, iou

def train_epoch(model, loader, optimizer, device, accum_iter):
    model.train()
    criterion = ComboLoss().to(device)
    total_loss, total_dice, total_iou = 0, 0, 0
    optimizer.zero_grad()

    for i, (imgs, masks, _) in enumerate(tqdm(loader, desc="Train")):
        imgs, masks = imgs.to(device), masks.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, masks)
        
        (loss / accum_iter).backward()
        
        if (i + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        dice, iou = calculate_metrics(logits, masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_iter", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(42)
    random.seed(42)
    run_id = generate_run_id()
    
    log_dir = "runs"
    os.makedirs(log_dir, exist_ok=True)
    best_pth_path = os.path.join(log_dir, f"{run_id}_best.pth")
    csv_path = os.path.join(log_dir, f"{run_id}_log.csv")

    print(f"--- SegDino Run: {run_id} ---")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} | Batch (Eff): {args.batch_size*args.accum_iter}")

    # Model
    model = SegDino(model_size="1", freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Data
    train_loader = torch.utils.data.DataLoader(
        PreTiledDataset(args.data_dir, "train"), 
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        PreTiledDataset(args.data_dir, "test"), 
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Logging Init
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_dice,val_iou,lr,time\n")

    best_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        
        t_loss, t_dice, t_iou = train_epoch(model, train_loader, optimizer, device, args.accum_iter)
        v_loss, v_dice, v_iou = evaluate(model, val_loader, device)
        
        scheduler.step()
        duration = time.time() - start
        
        # Log
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {t_loss:.4f} | Val IoU: {v_iou:.4f} | Time: {duration:.1f}s")
        
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{t_loss:.4f},{v_loss:.4f},{v_dice:.4f},{v_iou:.4f},{optimizer.param_groups[0]['lr']:.2e},{duration:.1f}\n")

        # Save Best
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), best_pth_path)
            print(f"  --> New Best IoU: {best_iou:.4f} (Saved)")

if __name__ == "__main__":
    main()
