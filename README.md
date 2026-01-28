# DOTA Multi-Class Segmentation

Semantic segmentation system for aerial imagery using U-Net architecture with DINOv3/ResNet backbones. Supports 16-class segmentation on DOTA dataset (planes, ships, vehicles, bridges, etc.).

## Features

- **Multi-Class Segmentation**: 16 object classes + background
- **Smart Cropping**: Preserves object resolution in high-res satellite images (4K+)
- **Sliding Window Inference**: Full-resolution predictions via tiling
- **Frozen Backbone Training**: Fast convergence by training only decoder
- **Multi-Backbone Support**: DINOv3 (ViT), ResNet50/101
- **Class-Weighted Loss**: Handles severe class imbalance
- **Colab Ready**: Automated training notebook for GPU acceleration

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/fx-bonnefont/projet-vision.git
cd projet-vision

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Training

```bash
python train.py \
  --data /path/to/DOTA \
  --backbone dinov3_vits16 \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-3 \
  --output checkpoints/model.pth
```

**Key Arguments:**
- `--data`: Root directory containing `images/train`, `labels/train`, etc.
- `--backbone`: Model backbone (`dinov3_vits16`, `dinov3_vitl16`, `resnet50`)
- `--cache`: Load entire dataset into RAM (recommended for small datasets)
- `--unfreeze`: Fine-tune backbone (not recommended, use frozen decoder)

### Inference

```bash
python inference.py \
  --model checkpoints/model_best.pth \
  --data /path/to/DOTA \
  --num-images 10 \
  --output checkpoints/inference_output
```

Generates 4-panel visualizations:
- **Top-Left**: Ground truth (if available in `debug/` folder)
- **Top-Right**: Foreground confidence heatmap
- **Bottom-Left**: Predicted class mask (color-coded)
- **Bottom-Right**: Bounding boxes with class labels

## Project Structure

```
projet-vision/
├── segmentation/          # Core package
│   ├── __init__.py
│   ├── model.py          # U-Net + backbone integration
│   ├── dataset.py        # DOTA data loader with smart cropping
│   ├── backbones.py      # DINOv3, ResNet implementations
│   ├── loss.py           # BCEDiceLoss (legacy, not used in multi-class)
│   └── logger.py         # Training metrics logger
├── scripts/              # Utility scripts
│   ├── benchmark_system.py
│   └── prepare_debug_data.py  # Generate GT visualizations
├── notebooks/            # Jupyter notebooks
│   └── run_on_colab.ipynb     # Automated Colab training
├── logs/                 # Training logs (CSV)
├── checkpoints/          # Model weights and inference outputs
├── train.py              # Training entry point
├── inference.py          # Inference entry point
└── README.md
```

## Dataset Format

Expected directory structure:

```
DOTA/
├── images/
│   ├── train/
│   │   ├── P0000.png
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── P0000.txt
│   │   └── ...
│   └── test/
│       └── ...
└── debug/  (optional, for GT visualization)
    ├── train/
    │   ├── visu_P0000.png
    │   └── ...
    └── test/
        └── ...
```

**Label Format (DOTA):**
```
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

## Supported Classes

1. plane
2. ship
3. storage-tank
4. baseball-diamond
5. tennis-court
6. basketball-court
7. ground-track-field
8. harbor
9. bridge
10. large-vehicle
11. small-vehicle
12. helicopter
13. roundabout
14. soccer-ball-field
15. swimming-pool

## Training on Google Colab

1. Push your code to GitHub
2. Open `notebooks/run_on_colab.ipynb` in Colab
3. Mount Google Drive (store DOTA dataset there)
4. Run all cells

The notebook automatically:
- Clones your repo
- Installs dependencies
- Authenticates with Hugging Face (for DINOv3)
- Launches training with optimal settings

## Architecture Details

### Model
- **Encoder**: Frozen DINOv3/ResNet (pre-trained on ImageNet)
- **Decoder**: U-Net with skip connections
- **Output**: 16 channels (one per class)

### Training Strategy
- **Loss**: CrossEntropyLoss with class weights (background=0.1, objects=1.0)
- **Optimizer**: Adam (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau
- **Data Augmentation**: Smart cropping (80% object-centered, 20% random)

### Inference Strategy
- **Small Images (<512px)**: Direct resize + predict
- **Large Images (>512px)**: Sliding window with 512x512 tiles
- **Post-processing**: Argmax → class mask → contour detection → bounding boxes

## Performance Tips

1. **Use `--cache`** if dataset fits in RAM (~10GB for 1000 images)
2. **Freeze backbone** (default) for faster convergence
3. **Increase batch size** to 16-32 on GPUs with 16GB+ VRAM
4. **Use DINOv3-Large** for best accuracy (requires 24GB RAM)

## Troubleshooting

### Hugging Face Authentication Error
```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### Out of Memory
- Reduce `--batch-size`
- Use smaller backbone (`dinov3_vits16` instead of `vitl16`)
- Disable `--cache`

### Poor Segmentation Quality
- Train for more epochs (50-100)
- Verify dataset labels are correct
- Check class distribution (use `scripts/benchmark_system.py`)

## Citation

If you use this code, please cite the DOTA dataset:

```bibtex
@article{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and others},
  journal={CVPR},
  year={2018}
}
```

## License

MIT License - See LICENSE file for details
