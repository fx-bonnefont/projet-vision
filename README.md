# SegDino: Small Object Segmentation on Satellite Imagery

SegDino is a specialized deep learning model designed for segmenting small objects (e.g., planes) in high-resolution satellite imagery (DOTA dataset). It leverages the power of a frozen Self-Supervised Vision Transformer (DINOv3) backbone combined with a custom Heavy U-Net decoder to resolve fine details from coarse semantic features.

## Recent Updates

The codebase has been refactored for correctness and maintainability:

1. **Dice Loss Formula**: Corrected mathematical formula. Previous implementation counted intersection twice in the denominator. Now uses correct formula: `Dice = 2 * |A ∩ B| / (|A| + |B|)`

2. **Code Organization**: Created `utils/` package with modular components:
   - `utils/distributed.py`: DDP utilities for multi-GPU training
   - `utils/metrics.py`: Mathematically correct Dice/IoU metrics
   - `utils/visualization.py`: Inference and visualization helpers

3. **Robustness**: Added bounded retry mechanism in dataset loader to prevent infinite loops on corrupted files.

---

## Architecture & Design Choices

### 1. Backbone: DINOv3 (Frozen)
We use Meta's **DINOv3** (ViT-Small/Base/Large/Huge/Giant) as the feature extractor.
- **Why Frozen?** DINOv3 features are robust enough to capture semantic concepts without needing full fine-tuning, which saves VRAM and prevents overfitting on small datasets.
- **Why ViT?** Transformers offer a global receptive field, crucial for understanding context in large aerial scenes.

### 2. Decoder: Heavy U-Net
The primary challenge is reconstructing high-resolution masks from the coarse (e.g., 1/16) feature map produced by the ViT, which lacks fine spatial details.
- **Heavy U-Net Decoder:** To solve this, we use a deep and powerful convolutional decoder. It consists of multiple upsampling stages, each containing `ResBlock`s to progressively refine the features and rebuild spatial information.
- **No Shortcuts:** This decoder architecture is "purist"—it relies **only** on the semantic features from the DINOv3 backbone. It does not receive any "shortcut" connections from the raw input image. This is a deliberate choice to ensure that any analysis performed on the model (e.g., adversarial attacks) is a true test of the backbone's features, not the decoder's ability to leverage raw pixel data.

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
├── model.py              # SegDino architecture (Heavy UNet Decoder)
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

This project is configured to run on the SLURM cluster described in `CLUSTER.md`. The `train_ddp.sh` script is designed to be autonomous, handling environment setup automatically.

#### 1. Interactive Test (Recommended First Step)

Before submitting a long training job, it's best to run a quick test in an interactive session on a compute node.

**a. Request an interactive session with one GPU:**
```bash
# On the cluster gateway (gpu-gw)
sinteractive --partition=3090 --gres=gpu:1 --time=01:00:00
```

**b. Once on the compute node, run a quick training test:**
```bash
# cd to the project directory
# Make sure your environment is activated: source .venv/bin/activate
python train.py --model_size vit-small --epochs 1 --batch_size 4
```
If this command completes without error, your setup is correct. You can then `exit` the interactive session.

#### 2. Submit a Batch Training Job

To run a full training, submit the `train_ddp.sh` script to the SLURM scheduler.

**a. Launch with default parameters (vit-large, 25 epochs):**
```bash
sbatch train_ddp.sh
```

**b. Launch with custom parameters:**

You can override variables like `MODEL_SIZE` and `EPOCHS` directly with `--export`.
```bash
# Example: Train vit-base for 100 epochs
sbatch --export=ALL,MODEL_SIZE=vit-base,EPOCHS=100 train_ddp.sh
```

#### 3. Monitor the Job
Use standard SLURM commands to monitor your job.
```bash
# See your running/pending jobs
squeue --me

# Follow the output log in real-time
# (The job ID is shown after you run sbatch)
tail -f logs/slurm_[JOB_ID].out
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
