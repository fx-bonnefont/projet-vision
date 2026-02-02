# SegDino: Small Object Segmentation on Satellite Imagery

SegDino is a specialized deep learning model designed for segmenting small objects (e.g., planes) in high-resolution satellite imagery (DOTA dataset). It leverages the power of Self-Supervised Vision Transformers (DINOv3) combined with a custom Heavy U-Net decoder featuring Multi-Scale Input Injection to resolve fine details.

## Architecture & Design Choices

### 1. Backbone: DINOv3 (Frozen)
We use Meta's **DINOv3** (ViT-Small/Base/Large) as the feature extractor.
- **Why Frozen?** DINOv3 features are robust enough to capture semantic concepts without needing full fine-tuning, which saves VRAM and prevents overfitting on small datasets.
- **Why ViT?** Transformers offer a global receptive field, crucial for understanding context in large aerial scenes.

### 2. Decoder: Heavy U-Net with Pyramid Input Injection
Standard decoders (like DPT) fail on small objects because the backbone resolution (1/16) destroys high-frequency details (edges).
- **Heavy U-Net:** A deep decoder with Residual Blocks at every stage to process complex features.
- **Pyramid Input Injection:** To solve the resolution gap, we inject the **raw RGB image** (resized) at every stage of the decoder (1/8, 1/4, 1/2). This gives the decoder a direct "visual guide" to draw precise boundaries around objects, guided by the semantic map from DINO.

### 3. Training Strategy
- **Offline Tiling:** Large DOTA images are pre-tiled (512x512) to maximize IO throughput.
- **Smart Sampling:** We keep all positive tiles (with planes) and a small ratio of negative tiles (background) to balance the dataset.
- **Loss:** **ComboLoss** (0.5 * BCE + 0.5 * Dice) to handle class imbalance (few object pixels vs massive background).
- **Optimizer:** AdamW with Cosine Annealing scheduler.

## Experiment Tracking & Logging

SegDino implements a rigorous logging system to ensure reproducibility and monitor model health during long training sessions. All outputs are centralized in the `runs/` directory.

### Comprehensive CSV Logs
Each run generates a unique CSV file (`{RUN_ID}_log.csv`) featuring:
- **Detailed Metadata Header:** Commented lines (`#`) recording the exact configuration: backbone model, decoder architecture details, tiling strategy, and hyper-parameters.
- **Performance Metrics:** Per-epoch tracking of Train Loss, Validation Loss, Dice score, and IoU.
- **Model Health Check:** Logging of **Gradient L2 Norm** and **Weight L2 Norm** to detect vanishing/exploding gradients or regularization issues.
- **Time Tracking:** Duration of each epoch and cumulative training time.

### Smart Checkpointing
To save disk space, the system only persists the **best model weights** (`{RUN_ID}_best.pth`) based on the highest Validation IoU achieved during the run.

---

## Project Structure

```
segdino/
├── train.py           # Main training script (with Experiment Tracking)
├── model.py           # SegDino architecture (Heavy UNet + Pyramid Injection)
├── dataset.py         # Efficient PreTiledDataset loader
├── loss.py            # ComboLoss (BCE + Dice) definition
├── inference.py       # Inference & Visualization script
├── prepare_data.py    # Offline tiling script
├── runs/              # Centralized logs (.csv) and best weights (.pth)
└── segdata/           # Symlinks to raw large images
```

---

## Setup & Data Preparation

### 1. Prerequisites
- Python 3.8+
- PyTorch, Torchvision
- OpenCV, Tqdm, Transformers, HuggingFace Hub

### 2. Data Structure (Symlinks)
Your raw DOTA dataset should be accessible. Create symlinks in `segdata/dota` to point to your external SSD or dataset folder.

```bash
mkdir -p segdata/dota/train
mkdir -p segdata/dota/test

# Example linking (Adjust paths to your system)
ln -s /Volumes/SSD/DOTA/train/images segdata/dota/train/image
ln -s /Volumes/SSD/DOTA/train/labelTxt segdata/dota/train/label  # Or mask folder
```

### 3. Offline Tiling (Crucial Step)
Before training, you must slice the huge satellite images into 512x512 tiles. This script also handles filtering to create a balanced dataset.

```bash
# This will create a new folder DOTA_PLANES_TILED on your SSD
python prepare_data.py
```
*Note: Edit `prepare_data.py` to change source/target paths if necessary.*

---

## Training

To launch training with the recommended configuration (Model Base, Batch 10, 25 Epochs):

```bash
python train.py --model_size 2 --batch_size 10 --epochs 25 --data_dir /Volumes/X9Pro/DOTA_PLANES_TILED
```

**Arguments:**
- `--model_size`: `1` (Small), `2` (Base), `3` (Large).
- `--batch_size`: Physical batch size (dependant on your VRAM).
- `--lr`: Learning rate (Default: 5e-4).

---

## Inference & Visualization

To test your model and visualize predictions (Green = GT, Orange = Pred):

```bash
python inference.py --ckpt runs/YOUR_RUN_ID_best.pth --save_dir inference_results
```

The script will generate side-by-side comparisons for inspection.
