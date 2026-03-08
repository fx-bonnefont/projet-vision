# Frame Attack (`frame_attack.py`)

This file documents the current behavior and CLI of `frame_attack.py`.

## What It Does

`frame_attack.py` trains an adversarial border attack against a frozen SegDino model.

- Loads a checkpoint and freezes model weights.
- Trains a tensor `attack` with shape `[1, 3, thickness, 4 * attack_tile_size]`.
- Splits the tensor into `left/top/right/bottom` strips.
- Draws those strips around each object center (radius derived from object area).
- Optimizes one of several attack objectives.
- Logs metrics/artifacts to MLflow and saves `runs/<run_id>_attack.pt`.

## Main Components

- `AttackConfig`: runtime config dataclass.
- `FramePatchAttack`: data loading, attack init/load/apply, loss, train/validation/test evaluation, plotting.
- `parse_args()`: CLI parsing and validation.
- `main()`: CLI entrypoint.

## Objectives

`--objective` choices:

- `suppress_count` (default): minimizes `sigmoid(temp * (prob - threshold)).mean()`.
- `suppress_confidence`: minimizes mean attacked probability.
- `divergence`: minimizes `-MSE(attacked_prob, clean_prob)` (maximizes difference).
- `center_like`: minimizes negative of a CenterLoss-like distance (MSE + focal-like term) to clean predictions.

Objective-specific args:

- `--count-threshold`, `--count-temperature` for `suppress_count`.
- `--center-alpha`, `--center-focal-alpha`, `--center-focal-gamma` for `center_like`.

## Dataset Layout

`--data-dir` must provide:

- `train/image/*.png`
- `train/mask/*.png`
- `test/image/*.png`
- `test/mask/*.png`

By default, empty tiles are filtered out for speed. Use `--include-empty-tiles` to disable this filtering.

## Default Checkpoint

- `runs/16_02-21_05_12_BASE_animal-variation.pth`

Override with `--checkpoint`.

## CLI Usage

Run from repo root.

Minimal run:

```bash
uv run python frame_attack.py
```

Full command with explicit static defaults:

```bash
uv run python frame_attack.py \
  --checkpoint "runs/16_02-21_05_12_BASE_animal-variation.pth" \
  --epochs 2 \
  --learning-rate 0.5 \
  --batch-size 16 \
  --attack-tile-size 512 \
  --thickness 24 \
  --batch-repetition 1 \
  --early-stop 1000000 \
  --data-dir "segdata/DOTA/DOTA_PLANES_TILED" \
  --workers 4 \
  --validation-ratio 0.3 \
  --save-every 50 \
  --seed 1619 \
  --experiment-name "patch-attack" \
  --eval-threshold 0.3 \
  --match-radius 20.0 \
  --objective suppress_count \
  --count-threshold 0.3 \
  --count-temperature 10.0 \
  --center-alpha 0.5 \
  --center-focal-alpha 2.0 \
  --center-focal-gamma 4.0
```

Notes:

- `--run-name` defaults to an auto-generated codename.
- `--mlflow-tracking-uri` defaults to the currently configured MLflow tracking URI.
- `--skip-validation` is disabled by default.
- `--attack-id` / `--attack-path` default to `None`.

## Warm Start / Resume

From MLflow run artifact:

```bash
uv run python frame_attack.py --attack-id <run_id>
```

From local attack tensor:

```bash
uv run python frame_attack.py --attack-path runs/<run_id>_attack.pt
```

If both are provided, `--attack-path` takes priority.  
If MLflow download fails for `--attack-id`, the script tries local fallback `runs/<run_id>_attack.pt`.
When a loaded attack width implies a different tile size (`width / 4`), `attack_tile_size` is auto-synced.

## Logging and Outputs

During training the script:

- Logs MLflow params and training metrics (`training_loss`, max clean/attacked prediction, throughput metrics).
- Optionally logs validation metrics each epoch (`loss`, max prob, accuracy/precision/recall/F1, TP/FP/FN).
- Periodically saves attack checkpoints when `--save-every > 0`.
- Always saves the final attack to `runs/<run_id>_attack.pt`.

Final CLI output includes:

- `Attack trained and saved with run_id: <run_id>`

## Evaluation Metrics

Validation/test evaluation is center-based:

- peak extraction threshold: `--eval-threshold`
- matching radius: `--match-radius`
- reported metrics for clean and attacked predictions:
  - accuracy, precision, recall, F1
  - TP, FP, FN counts

## Visualization from Notebook

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

You can also load with `attack_path="runs/<run_id>_attack.pt"`.

## Troubleshooting

- Missing Python packages (for example `cv2`): run through project env (`uv run ...` or venv Python).
- `--attack-id` cannot fetch artifacts: check MLflow tracking URI and run id; local fallback path is `runs/<run_id>_attack.pt`.
- No objects / split errors: verify mask files and object annotations exist in `train`/`test`.
