# SegDino: Small Object Segmentation on Satellite Imagery

SegDino is a specialized deep learning model designed for segmenting small objects (e.g., planes) in high-resolution satellite imagery (DOTA dataset). It leverages the power of Self-Supervised Vision Transformers (DINOv3) combined with a custom Heavy U-Net decoder featuring Multi-Scale Input Injection to resolve fine details.

## Recent Updates

The codebase has been refactored for correctness and maintainability:

1. **Dice Loss Formula**: Corrected mathematical formula. Previous implementation counted intersection twice in the denominator. Now uses correct formula: `Dice = 2 * |A ∩ B| / (|A| + |B|)`

2. **Pyramid Input Injection**: Fixed architecture to inject RGB image AFTER upsampling features (not before), preserving sharp details at each decoder level.

3. **Code Organization**: Created `utils/` package with modular components:
   - `utils/distributed.py`: DDP utilities for multi-GPU training
   - `utils/metrics.py`: Mathematically correct Dice/IoU metrics
   - `utils/visualization.py`: Inference and visualization helpers

4. **Robustness**: Added bounded retry mechanism in dataset loader to prevent infinite loops on corrupted files.

---

## Architecture & Design Choices

### 1. Backbone: DINOv3 (Frozen)
We use Meta's **DINOv3** (ViT-Small/Base/Large/Huge/Giant) as the feature extractor.
- **Why Frozen?** DINOv3 features are robust enough to capture semantic concepts without needing full fine-tuning, which saves VRAM and prevents overfitting on small datasets.
- **Why ViT?** Transformers offer a global receptive field, crucial for understanding context in large aerial scenes.

### 2. Decoder: Heavy U-Net with Pyramid Input Injection
Standard decoders (like DPT) fail on small objects because the backbone resolution (1/16) destroys high-frequency details (edges).
- **Heavy U-Net:** A deep decoder with Residual Blocks at every stage to process complex features.
- **Pyramid Input Injection:** At each decoder stage, we first upsample features, then inject the raw RGB image at the new resolution. This preserves sharp details that guide precise boundary delineation.

### 3. Training Strategy
- **Offline Tiling:** Large DOTA images are pre-tiled (512x512) to maximize IO throughput.
- **Smart Sampling:** We keep all positive tiles (with planes) and a small ratio of negative tiles (background) to balance the dataset.
- **Loss:** **ComboLoss** (0.5 * BCE + 0.5 * Dice) to handle class imbalance (few object pixels vs massive background).
- **Optimizer:** AdamW with Cosine Annealing scheduler.
- **Multi-GPU:** DistributedDataParallel (DDP) with automatic detection for SLURM clusters.

## Project Structure

```
segdino/
├── train.py              # Main training script (DDP-enabled)
├── train_ddp.sh          # SLURM submission script for multi-GPU
├── test_single_gpu.sh    # Quick test script
├── verify_setup.py       # Pre-deployment verification
├── model.py              # SegDino architecture (Heavy UNet + Pyramid Injection)
├── dataset.py            # Efficient PreTiledDataset loader
├── loss.py               # ComboLoss (BCE + Dice) definition
├── inference.py          # Unified inference script (sliding window on full images)
├── utils/
│   ├── distributed.py    # DDP utilities
│   ├── metrics.py        # Dice/IoU metrics
│   └── visualization.py  # Inference helpers
├── runs/                 # Training logs (.csv) and checkpoints (.pth)
└── segdata/              # Symlinks to raw large images
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (for GPU training)

### Installation

This project uses `uv` for fast and reproducible environment management.

```bash
# 1. Create the virtual environment
# (Requires python3.12, or specify your python version)
uv venv python3.12 .venv
source .venv/bin/activate

# 2. Sync dependencies from the lock file
uv sync
```

### HuggingFace Authentication

DINOv3 models require HuggingFace access. Create a `.env` file:

```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

Get your token from: https://huggingface.co/settings/tokens

---

## Data Preparation

This project requires a specific data structure, which you can set up with a single symbolic link.

Your main data folder (e.g., `DOTA`) should contain subdirectories for the full images (`DOTA_PLANES`) and the pre-tiled images (`DOTA_PLANES_TILED`).

Create a symbolic link named `DOTA` inside the project's `segdata` directory that points to your main data folder.

```bash
# Example: If your main DOTA folder is in your home directory
ln -s ~/DOTA segdata/DOTA
```

The code will then automatically find the correct datasets for training and inference within `segdata/DOTA/`.

---

## Training

### Available Models

| Model | Params | Layers | Pre-training | Recommended Use |
|-------|--------|--------|--------------|-----------------|
| `vit-small` | 21.6M | 12 | ImageNet | Quick experiments |
| `vit-small-plus` | 28.7M | 12 | ImageNet | Quick experiments |
| `vit-base` | 85.7M | 12 | ImageNet | General purpose |
| `vit-large` | 0.3B | 24 | ImageNet | High accuracy |
| `vit-huge` | 0.8B | 32 | ImageNet | Maximum capacity |
| `vit-giant` | 7B | 40 | ImageNet | Research |
| **`vit-large-sat`** | 0.3B | 24 | **Satellite (493M)** | **Recommended for aerial** |
| **`vit-giant-sat`** | 7B | 40 | **Satellite (493M)** | Research on aerial |

**Note**: SAT variants are pre-trained on satellite imagery and may perform better on DOTA.

### Single GPU Training

```bash
python train.py \
    --model_size vit-base \
    --batch_size 10 \
    --epochs 25
```

**Common configurations:**
- **vit-base** on 16GB GPU: `--batch_size 10`
- **vit-large** on 24GB GPU: `--batch_size 16`

### Multi-GPU Training (SLURM Cluster)

The code automatically detects and uses DistributedDataParallel (DDP) when run via SLURM.

#### Quick Start

```bash
# 1. Verify setup
python verify_setup.py

# 2. Test on single GPU (optional)
./test_single_gpu.sh

# 3. Edit DATA_DIR in train_ddp.sh (line 72)
nano train_ddp.sh

# 4. Submit to cluster
sbatch train_ddp.sh

# 5. Monitor progress
tail -f logs/slurm_<JOB_ID>.out
```

#### Custom Parameters

```bash
# vit-large with 16 batch per GPU (32 effective)
sbatch --export=ALL,MODEL_SIZE=vit-large,BATCH_SIZE=16,EPOCHS=25,DATA_DIR=/your/data/path train_ddp.sh

# vit-large-sat (satellite-pretrained)
sbatch --export=ALL,MODEL_SIZE=vit-large-sat,BATCH_SIZE=16,EPOCHS=25,DATA_DIR=/your/data/path train_ddp.sh
```

#### Memory Estimates (2x NVIDIA 3090, 24GB each)

| Model | Batch/GPU | Effective Batch | VRAM/GPU | Status |
|-------|-----------|-----------------|----------|--------|
| vit-large | 16 | 32 | ~18-20GB | Safe |
| vit-large | 20 | 40 | ~22-23GB | Tight |
| vit-large-sat | 16 | 32 | ~18-20GB | Safe |
| vit-huge | 8 | 16 | ~18-20GB | Safe |
| vit-huge | 12 | 24 | ~23-24GB | Tight |

**Recommended**: Use `--batch_size 16` for vit-large/vit-large-sat on 3090s.

#### Expected Training Time

With 2x 3090 and vit-large:
- **Per epoch**: ~10-15 minutes (depends on dataset size)
- **25 epochs**: ~4-6 hours

---

## Training Outputs

All outputs are saved to `runs/`:

```
runs/
├── 20260202_15_a3f4_best.pth    # Best model checkpoint
└── 20260202_15_a3f4_log.csv     # Training log with metadata
```

### CSV Log Format

Each log includes:
- **Metadata header** (commented lines): model config, batch size, num GPUs, etc.
- **Per-epoch metrics**: train_loss, val_loss, val_dice, val_iou, grad_norm, weight_norm, lr, duration

**View final metrics:**
```bash
grep -v "^#" runs/<RUN_ID>_log.csv | tail -1
```

---

## Inference

The `inference.py` script performs inference on full-size images using a sliding window approach with Gaussian blending to produce smooth predictions. It generates a side-by-side visualization comparing the ground truth bounding boxes (left, green) with the predicted bounding boxes (right, orange).

```bash
python inference.py \
    --ckpt runs/YOUR_RUN_ID_best.pth \
    --model_size vit-base \
    --save_dir inference_results \
    --limit 10
```

- `--limit`: (Optional) Restricts the number of images to process.
- The script automatically handles finding `image`/`images` and `mask`/`masks` subdirectories based on its default path.

---

## Multi-GPU Implementation Details

### How DDP Works

The training script automatically detects distributed mode via environment variables:

**Supported Launchers:**
- SLURM: via `srun` (detects `SLURM_PROCID`)
- torchrun: `torchrun --nproc_per_node=2 train.py ...`
- Single GPU: Falls back automatically if no DDP env vars found

**Key Features:**
- **Automatic detection**: No `--distributed` flag needed
- **Data sharding**: DistributedSampler splits dataset across GPUs (no overlap)
- **Gradient sync**: DDP automatically averages gradients during backward()
- **Metric averaging**: `dist.all_reduce()` averages metrics across all ranks
- **Single checkpoint**: Only rank 0 saves models and logs

**Effective Batch Size:**
```
Effective Batch = batch_size × num_gpus
```

Example: `--batch_size 16` with 2 GPUs = **32 effective batch size**

### SLURM Script Configuration

Edit `train_ddp.sh` before submitting:

**Line 10** - Adjust GPU partition name:
```bash
#SBATCH --partition=gpu  # Change to your cluster's partition name
```

**Line 72** - Set data directory:
```bash
DATA_DIR=${DATA_DIR:-"/actual/path/to/DOTA_PLANES_TILED"}
```

**Line 55** - Load CUDA module (if required by cluster):
```bash
module load cuda/11.8  # Uncomment and adjust version
```

---

## Troubleshooting

### Job Fails on Cluster

```bash
# Check error log
cat logs/slurm_<JOB_ID>.err

# Common issues:
# 1. Data path wrong -> verify DATA_DIR in train_ddp.sh
# 2. CUDA module not loaded -> uncomment module load in train_ddp.sh
# 3. Virtual environment missing -> create .venv on cluster
```

### Out of Memory

Reduce batch size:
```bash
sbatch --export=ALL,BATCH_SIZE=12,... train_ddp.sh
```

### Slow Data Loading

If data is on network storage:
```bash
sbatch --export=ALL,NUM_WORKERS=4,... train_ddp.sh
```

### Training Diverges

Check gradient norms in CSV log:
```bash
grep -v "^#" runs/<RUN_ID>_log.csv | awk -F, '{print $6}' | tail -10
```

If grad_norm > 100, reduce learning rate.

---

## Comparing Models

To compare vit-large vs vit-large-sat:

```python
import pandas as pd

# Load logs
df_large = pd.read_csv('runs/<RUN_ID_large>_log.csv', comment='#')
df_sat = pd.read_csv('runs/<RUN_ID_sat>_log.csv', comment='#')

# Compare final IoU
print(f"vit-large:     {df_large['val_iou'].iloc[-1]:.4f}")
print(f"vit-large-sat: {df_sat['val_iou'].iloc[-1]:.4f}")
print(f"Improvement:   {(df_sat['val_iou'].iloc[-1] - df_large['val_iou'].iloc[-1]):.4f}")
```

---

## Citation

If using DINOv3 in your research, please cite:

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```
