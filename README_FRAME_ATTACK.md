# Frame Attack (`frame_attack.py`)

This document explains how `frame_attack.py` is structured and how to use it.

## What This Script Does

`frame_attack.py` trains an adversarial **frame attack** against a frozen SegDino detector.

- The model is loaded from a checkpoint and frozen.
- A trainable tensor (`attack`) is optimized.
- For each object center in a tile, the script draws a learned border around the object region.
- The optimization objective is selected from multiple attack losses.

The script logs metrics to MLflow and saves the learned attack tensor to `runs/<run_id>_attack.pt`.

## High-Level Architecture

Main components in `frame_attack.py`:

- `AttackConfig`: dataclass holding all runtime options.
- `FramePatchAttack`: core class that handles:
  - model loading/freeze
  - data loader creation
  - attack tensor init/load/save
  - attack application to images
  - objective computation
  - training + validation loops
  - optional visualization helpers
- `parse_args()`: CLI argument parsing + validation.
- `main()`: entrypoint that builds config, trains, and prints the resulting `run_id`.

## How Attack Application Works

The attack tensor shape is:

- `[1, 3, thickness, 4 * TILE_SIZE]`

It is split into 4 strips:

- left, top, right, bottom

For each object center (from metadata), the script:

1. Converts object area to a square radius.
2. Computes a box around the center.
3. Resizes each strip to match border dimensions.
4. Writes the strips around the box on the normalized input image tensor.

## Available Objectives

`--objective` supports:

- `suppress_count` (default)
  - Minimizes a soft thresholded count:
  - `sigmoid(temp * (prob - threshold)).mean()`
  - Best aligned with reducing detections above threshold.
- `suppress_confidence`
  - Minimizes average attacked probability map.
- `divergence`
  - Minimizes `-MSE(attacked_prob, clean_prob)` (equivalent to maximizing map difference).
- `center_like`
  - Uses a CenterLoss-like distance (MSE + focal-like term) against clean prediction as target.
  - The attack minimizes the negative of that distance (maximizes discrepancy).

Related args:

- `--count-threshold`, `--count-temperature` for `suppress_count`
- `--center-alpha`, `--center-focal-alpha`, `--center-focal-gamma` for `center_like`

## Dataset Expectations

`--data-dir` must contain:

- `train/image/*.png`
- `train/mask/*.png`
- `test/image/*.png`
- `test/mask/*.png`

## Default Checkpoint

Default model checkpoint:

- `runs/16_02-21_05_12_BASE_animal-variation.pth`

You can override with `--checkpoint`.

## Command Line Usage

Run from repository root.

### 1) Minimal command (all defaults)

```bash
./.venv/bin/python frame_attack.py
```

### 2) Full command (all explicit defaults)

```bash
./.venv/bin/python frame_attack.py \
  --checkpoint "runs/16_02-21_05_12_BASE_animal-variation.pth" \
  --epochs 2 \
  --learning-rate 0.5 \
  --batch-size 16 \
  --thickness 24 \
  --batch-repetition 1 \
  --early-stop 1000000 \
  --data-dir "segdata/DOTA/DOTA_PLANES_TILED" \
  --workers 4 \
  --validation-ratio 0.3 \
  --save-every 50 \
  --seed 1619 \
  --objective suppress_count \
  --count-threshold 0.3 \
  --count-temperature 10.0 \
  --center-alpha 0.5 \
  --center-focal-alpha 2.0 \
  --center-focal-gamma 4.0
```

Notes:

- `--skip-validation` is a flag (default is disabled).
- `--attack-id` and `--attack-path` default to `None` (only pass when resuming/initializing from an existing attack).

## Resume / Warm Start Options

Initialize attack from previous artifacts:

- From MLflow run:

```bash
./.venv/bin/python frame_attack.py --attack-id <run_id>
```

- From local file:

```bash
./.venv/bin/python frame_attack.py --attack-path runs/<run_id>_attack.pt
```

If both are provided, `--attack-path` is used first.

## Output and Logging

During training, the script:

- logs metrics to MLflow (`training_loss`, max probs, validation metrics)
- periodically saves `runs/<run_id>_attack.pt`
- prints epoch summaries with objective + train/validation losses

Final line includes the MLflow run id:

- `Attack trained and saved with run_id: <run_id>`

## Notebook Visualization

After running training from CLI, use the `run_id` in a notebook:

```python
from frame_attack import AttackConfig, FramePatchAttack
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = "<RUN_ID_FROM_CLI>"

cfg = AttackConfig(
    checkpoint_path="runs/16_02-21_05_12_BASE_animal-variation.pth",
    data_dir="segdata/DOTA/DOTA_PLANES_TILED",
    attack_id=run_id,
)

attacker = FramePatchAttack(cfg)
attacker.plot_interesting_images()
```

Alternative: load from local file with `attack_path="runs/<run_id>_attack.pt"`.

## Troubleshooting

- `ModuleNotFoundError` (e.g., `cv2`): run through your project venv (`./.venv/bin/python ...`).
- Missing artifacts via `--attack-id`: ensure the same MLflow tracking URI is used in CLI and notebook.
- Empty/invalid dataset errors: verify `image/` and `mask/` folder names and PNG files.
