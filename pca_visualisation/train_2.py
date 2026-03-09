"""
SegDino training for center detection.
"""
import argparse
import os
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from codenames import generate_codename
from dataset import CachedFeaturesDataset
from extract_features import extract_and_save_features, get_cache_dir
from inference import find_peaks, match_centers
from loss import CenterLoss
from model import DecoderOnly

torch.backends.cudnn.benchmark = True

# Hardcoded configuration
DATA_DIR = "data/DOTA/DOTA_PLANES_TILED"
SEED = 42
NUM_WORKERS = 4
SIGMA = 12.0
THRESHOLD = 0.3
MATCH_RADIUS = 20.0


def segdino_collate(batch):
    imgs = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    metas = [item[2] for item in batch]
    return imgs, targets, metas


def generate_run_id(model_size: str) -> str:
    codename = generate_codename()
    return f"{datetime.now().strftime('%d_%m-%H_%M_%S')}_{model_size.upper()}_{codename}"


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for imgs, targets, _ in tqdm(loader, desc="train", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        # FP16 training
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
def find_peaks_torch(hm, threshold):
    mask = hm > threshold
    ys, xs = torch.where(mask)
    return torch.stack([xs, ys], dim=1)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0

    for imgs, targets, metas in tqdm(loader, desc="eval", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)

        # FP16 inference
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(imgs)

        # Loss (toujours en FP32 dans criterion)
        total_loss += criterion(logits, targets).item()

        # Heatmaps en GPU
        pred = torch.sigmoid(logits)

        for i in range(pred.shape[0]):
            hm = pred[i, 0]  # (H, W) sur GPU

            # Version GPU de find_peaks
            pred_centers = find_peaks_torch(hm, threshold=THRESHOLD)

            # match_centers attend une liste Python
            pred_centers = pred_centers.cpu().tolist()

            gt_centers = metas[i].get('centers', [])

            tp, fp, fn = match_centers(gt_centers, pred_centers, match_radius=MATCH_RADIUS)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_loss = total_loss / len(loader)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return avg_loss, f1


def save_checkpoint(model, path: str):
    torch.save({"model_state_dict": model.state_dict(), "config": model.get_config()}, path)


def check_cached_features(model_size: str) -> bool:
    cache_dir = get_cache_dir(DATA_DIR)
    train_cache = os.path.join(cache_dir, model_size, "train")
    test_cache = os.path.join(cache_dir, model_size, "test")
    if not os.path.exists(train_cache) or not os.path.exists(test_cache):
        return False
    return any(f.endswith(".pt") for f in os.listdir(train_cache))


def main():
    parser = argparse.ArgumentParser(description="SegDino Training")
    parser.add_argument("-m", default="small",
        choices=["small", "small-plus", "base", "large", "huge", "giant", "large-sat", "giant-sat"])
    parser.add_argument("-e", type=int, default=20)
    parser.add_argument("-l", type=float, default=5e-4)
    parser.add_argument("-b", type=int, default=12)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    run_id = generate_run_id(args.m)

    torch.manual_seed(SEED)
    random.seed(SEED)
    os.makedirs("runs", exist_ok=True)
    pth_path = os.path.join("runs", f"{run_id}.pth")
    csv_path = os.path.join("runs", f"{run_id}.csv")

    # Cache
    cache_dir = get_cache_dir(DATA_DIR)
    train_cache = os.path.join(cache_dir, args.m, "train")
    test_cache = os.path.join(cache_dir, args.m, "test")
    if check_cached_features(args.m):
        n_train = len([f for f in os.listdir(train_cache) if f.endswith(".pt")])
        n_test = len([f for f in os.listdir(test_cache) if f.endswith(".pt")])
        print(f"[cache] train: {n_train}, test: {n_test}")
    else:
        print(f"[cache] extracting features for {args.m}...")
        extract_and_save_features(args.m, DATA_DIR, cache_dir, args.b, device, True)
        print("[cache] done")

    # Model
    model = DecoderOnly(model_size=args.m).to(device)
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.l)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.e, eta_min=1e-6)
    criterion = CenterLoss().to(device)

    # Data
    train_loader = DataLoader(
        CachedFeaturesDataset(cache_dir, DATA_DIR, args.m, "train", sigma=SIGMA),
        batch_size=args.b, shuffle=True, num_workers=NUM_WORKERS,
        drop_last=True, collate_fn=segdino_collate, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        CachedFeaturesDataset(cache_dir, DATA_DIR, args.m, "test", sigma=SIGMA),
        batch_size=args.b, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=segdino_collate, pin_memory=True, persistent_workers=True
    )

    # CSV header
    with open(csv_path, "w") as f:
        f.write(f"# lr={args.l}, batch={args.b}\n")
        f.write("epoch,train_loss,val_loss,f1\n")

    best_f1 = 0.0
    for epoch in tqdm(range(1, args.e + 1), desc=run_id):
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{t_loss:.4f},{v_loss:.4f},{v_f1:.4f}\n")

        saved = ""
        if v_f1 > best_f1:
            best_f1 = v_f1
            save_checkpoint(model, pth_path)
            saved = " *"

        tqdm.write(f"{epoch:3d} | train: {t_loss:.4f} | val: {v_loss:.4f} | f1: {v_f1:.4f}{saved}")

    del train_loader, val_loader


if __name__ == "__main__":
    main()
