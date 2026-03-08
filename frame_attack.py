"""Train and inspect a frame-like adversarial attack for SegDino tiles.
---
How to use:
uv run python frame_attack.py

Extended command-line help:
uv run python frame_attack.py \
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
  --experiment-name "patch-attack"
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from codenames import generate_codename
from dataset import DOTA_MEAN, DOTA_STD, PreTiledDataset, mask_to_centers
from inference import draw_centers, find_peaks, get_gaussian_weight_map, load_checkpoint, match_centers
from train import segdino_collate

# Default project configuration.
DEFAULT_TILE_SIZE = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_DATA_DIR = "segdata/DOTA/DOTA_PLANES_TILED"
DEFAULT_CHECKPOINT_PATH = "runs/16_02-21_05_12_BASE_animal-variation.pth"
DEFAULT_NUM_WORKERS = 4
DEFAULT_VALIDATION_RATIO = 0.3
DEFAULT_THRESHOLD = 0.3
DEFAULT_MATCH_RADIUS = 20.0
DEFAULT_INTERESTING_IMAGES = ("P2269_obj_1.png","P2523_obj_3.png", "P2804_obj_5.png", "P2790_obj_6.png")
DEFAULT_OBJECTIVE = "suppress_count"
DEFAULT_COUNT_THRESHOLD = 0.3
DEFAULT_COUNT_TEMPERATURE = 10.0
DEFAULT_CENTER_ALPHA = 0.5
DEFAULT_CENTER_FOCAL_ALPHA = 2.0
DEFAULT_CENTER_FOCAL_GAMMA = 4.0
DEFAULT_EXPERIMENT_NAME = "patch-attack"


def select_device() -> str:
    """Return the best available device for inference/training."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducible attack initialization/splits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(slots=True)
class AttackConfig:
    """Runtime configuration for frame attack training."""

    checkpoint_path: str
    data_dir: str = DEFAULT_DATA_DIR
    epochs: int = 2
    learning_rate: float = 0.5
    batch_size: int = DEFAULT_BATCH_SIZE
    attack_tile_size: int = DEFAULT_TILE_SIZE
    thickness: int = 24
    batch_repetition: int = 1
    early_stop: int = 1_000_000
    num_workers: int = DEFAULT_NUM_WORKERS
    validation_ratio: float = DEFAULT_VALIDATION_RATIO
    save_every: int = 50
    attack_id: Optional[str] = None
    attack_path: Optional[str] = None
    skip_validation: bool = False
    seed: int = 1619
    run_name: Optional[str] = None
    objective: str = DEFAULT_OBJECTIVE
    count_threshold: float = DEFAULT_COUNT_THRESHOLD
    count_temperature: float = DEFAULT_COUNT_TEMPERATURE
    center_alpha: float = DEFAULT_CENTER_ALPHA
    center_focal_alpha: float = DEFAULT_CENTER_FOCAL_ALPHA
    center_focal_gamma: float = DEFAULT_CENTER_FOCAL_GAMMA
    experiment_name: str = DEFAULT_EXPERIMENT_NAME
    mlflow_tracking_uri: Optional[str] = None
    filter_empty_tiles: bool = True
    eval_threshold: float = DEFAULT_THRESHOLD
    match_radius: float = DEFAULT_MATCH_RADIUS


class FramePatchAttack:
    """Train a border attack that surrounds each detected object with strips."""

    def __init__(self, config: AttackConfig):
        """Initialize data, model, optimizer, and trainable attack tensor."""
        self.config = config
        self.device = select_device()
        self.tile_size = DEFAULT_TILE_SIZE
        self.attack_tile_size = self.config.attack_tile_size

        # set_seed(self.config.seed)
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        self.mlflow_client = mlflow.tracking.MlflowClient()

        self.model = load_checkpoint(self.config.checkpoint_path, self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.train_loader, self.validation_loader, self.test_loader = self._build_dataloaders()
        self.attack = torch.nn.Parameter(self._init_attack_tensor().to(self.device))
        self.thickness = int(self.attack.shape[2])

        # Keep attack values in valid normalized image range.
        self.norm_min = TF.normalize(torch.zeros_like(self.attack), DOTA_MEAN, DOTA_STD)
        self.norm_max = TF.normalize(torch.ones_like(self.attack), DOTA_MEAN, DOTA_STD)

        self.optimizer = torch.optim.AdamW([self.attack], lr=self.config.learning_rate)

        self.run_id: Optional[str] = None
        self.attack_save_path: Optional[str] = None
        self.interesting_images = list(DEFAULT_INTERESTING_IMAGES)

    def _subset_object_tiles(self, dataset: PreTiledDataset, split_name: str) -> Subset:
        """Create a subset containing only tiles with at least one valid object."""
        object_indices = []
        for index, image_name in enumerate(dataset.images):
            mask_path = os.path.join(dataset.mask_dir, image_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and len(mask_to_centers(mask)) > 0:
                object_indices.append(index)

        if not object_indices:
            raise ValueError(f"No object tiles found in split '{split_name}'.")

        print(f"[{split_name}] tiles with objects: {len(object_indices)}/{len(dataset)}")
        return Subset(dataset, object_indices)

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test dataloaders for tiled data."""
        train_dataset = PreTiledDataset(self.config.data_dir, "train", return_empty=True)
        test_dataset = PreTiledDataset(self.config.data_dir, "test", return_empty=True)
        if self.config.filter_empty_tiles:
            train_dataset = self._subset_object_tiles(train_dataset, "train")
            test_dataset = self._subset_object_tiles(test_dataset, "test")

        if len(train_dataset) < 2:
            raise ValueError("Training dataset must contain at least two samples.")

        n_val = int(len(train_dataset) * self.config.validation_ratio)
        n_val = max(1, min(n_val, len(train_dataset) - 1))
        n_train = len(train_dataset) - n_val

        generator = torch.Generator().manual_seed(self.config.seed)
        dataset_train, dataset_validation = random_split(
            train_dataset,
            [n_train, n_val],
            generator=generator,
        )

        persistent_workers = self.config.num_workers > 0

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
            collate_fn=segdino_collate,
            persistent_workers=persistent_workers,
        )

        validation_loader = DataLoader(
            dataset_validation,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
            collate_fn=segdino_collate,
            persistent_workers=persistent_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
            collate_fn=segdino_collate,
            persistent_workers=persistent_workers,
        )

        return train_loader, validation_loader, test_loader

    def _init_attack_tensor(self) -> torch.Tensor:
        """Initialize trainable attack tensor from random state or loaded artifact."""
        if self.config.attack_path:
            return self._load_attack_file(self.config.attack_path)

        if self.config.attack_id:
            return self._load_attack_artifact(self.config.attack_id)

        width = 4 * self.attack_tile_size
        return torch.rand((1, 3, self.config.thickness, width), dtype=torch.float32)

    def _load_attack_file(self, path: str) -> torch.Tensor:
        """Load an attack tensor from a local `.pt` file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attack file not found: {path}")

        attack = torch.load(path, map_location=self.device, weights_only=False)
        return self._validate_attack_tensor(attack, source=path)

    def _local_attack_path_from_run_id(self, run_id: str) -> str:
        """Return the conventional local attack checkpoint path for a run id."""
        return os.path.join("runs", f"{run_id}_attack.pt")

    def _load_attack_artifact(self, run_id: str) -> torch.Tensor:
        """Load an attack tensor from MLflow, with local fallback on access failures."""
        try:
            artifacts = self.mlflow_client.list_artifacts(run_id)
            attack_artifact = next((item.path for item in artifacts if item.path.endswith(".pt")), None)
            if attack_artifact is None:
                raise FileNotFoundError(f"No .pt artifact found for run id '{run_id}'.")

            local_path = self.mlflow_client.download_artifacts(run_id, attack_artifact)
            attack = torch.load(local_path, map_location=self.device, weights_only=False)
            return self._validate_attack_tensor(attack, source=f"mlflow:{run_id}/{attack_artifact}")
        except Exception as exc:
            fallback_path = self._local_attack_path_from_run_id(run_id)
            if os.path.exists(fallback_path):
                return self._load_attack_file(fallback_path)

            tracking_uri = mlflow.get_tracking_uri()
            raise RuntimeError(
                "Failed to load attack from MLflow and no local fallback was found. "
                f"run_id='{run_id}', tracking_uri='{tracking_uri}', "
                f"expected local fallback='{fallback_path}'."
            ) from exc

    def _sync_attack_tile_size_from_loaded_attack(self, attack: torch.Tensor, source: str) -> None:
        """Update attack tile size to match a loaded tensor width when possible."""
        loaded_width = int(attack.shape[3])
        if loaded_width % 4 != 0:
            raise ValueError(
                f"Attack width must be divisible by 4, got {loaded_width} from {source}."
            )

        loaded_tile_size = loaded_width // 4
        if loaded_tile_size == self.attack_tile_size:
            return

        print(
            "Loaded attack width does not match configured attack tile size. "
            f"Overriding attack_tile_size from {self.attack_tile_size} to {loaded_tile_size} "
            f"based on {source}."
        )
        self.attack_tile_size = loaded_tile_size
        self.config.attack_tile_size = loaded_tile_size

    def _validate_attack_tensor(self, attack: torch.Tensor, source: str) -> torch.Tensor:
        """Validate attack tensor shape and return float tensor."""
        if not isinstance(attack, torch.Tensor):
            raise TypeError(f"Attack from {source} is not a torch.Tensor.")

        if attack.ndim != 4 or attack.shape[0] != 1 or attack.shape[1] != 3:
            raise ValueError(
                f"Attack from {source} must have shape [1, 3, thickness, width], "
                f"got {tuple(attack.shape)}."
            )

        self._sync_attack_tile_size_from_loaded_attack(attack, source=source)

        return attack.float()

    def _split_attack(self, attack: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split attack tensor into left/top/right/bottom strips."""
        left = attack[:, :, :, : self.attack_tile_size]
        top = attack[:, :, :, self.attack_tile_size : 2 * self.attack_tile_size]
        right = attack[:, :, :, 2 * self.attack_tile_size : 3 * self.attack_tile_size]
        bottom = attack[:, :, :, 3 * self.attack_tile_size :]
        return left, top, right, bottom

    def _resize_horizontal_strip(self, strip: torch.Tensor, width: int) -> torch.Tensor:
        """Resize a horizontal strip to a target width."""
        return F.interpolate(
            strip,
            size=(self.thickness, width),
            mode="bilinear",
            align_corners=False,
        )

    def _resize_vertical_strip(self, strip: torch.Tensor, height: int) -> torch.Tensor:
        """Resize a vertical strip to a target height."""
        strip_vertical = strip.permute(0, 1, 3, 2)
        return F.interpolate(
            strip_vertical,
            size=(height, self.thickness),
            mode="bilinear",
            align_corners=False,
        )

    def _apply_border_around_center(
        self,
        image: torch.Tensor,
        strips: tuple[torch.Tensor, ...],
        center_x: int,
        center_y: int,
        radius: int,
    ) -> None:
        """Write attack strips around one object center on a single image tensor."""
        left_strip, top_strip, right_strip, bottom_strip = strips
        _, height, width = image.shape
        thickness = self.thickness

        # Centers are (x, y): columns first, rows second.
        row0 = max(center_y - radius, 0)
        row1 = min(center_y + radius, height)
        col0 = max(center_x - radius, 0)
        col1 = min(center_x + radius, width)

        top_row0, top_row1 = max(row0 - thickness, 0), row0
        if top_row1 > top_row0 and col1 > col0:
            needed_h = top_row1 - top_row0
            needed_w = col1 - col0
            top = self._resize_horizontal_strip(top_strip, needed_w)
            image[:, top_row0:top_row1, col0:col1] = top[0, :, :needed_h, :]

        bottom_row0, bottom_row1 = row1, min(row1 + thickness, height)
        if bottom_row1 > bottom_row0 and col1 > col0:
            needed_h = bottom_row1 - bottom_row0
            needed_w = col1 - col0
            bottom = self._resize_horizontal_strip(bottom_strip, needed_w)
            image[:, bottom_row0:bottom_row1, col0:col1] = bottom[0, :, :needed_h, :]

        left_col0, left_col1 = max(col0 - thickness, 0), col0
        if left_col1 > left_col0 and row1 > row0:
            needed_h = row1 - row0
            needed_w = left_col1 - left_col0
            left = self._resize_vertical_strip(left_strip, needed_h)
            image[:, row0:row1, left_col0:left_col1] = left[0, :, :, :needed_w]

        right_col0, right_col1 = col1, min(col1 + thickness, width)
        if right_col1 > right_col0 and row1 > row0:
            needed_h = row1 - row0
            needed_w = right_col1 - right_col0
            right = self._resize_vertical_strip(right_strip, needed_h)
            image[:, row0:row1, right_col0:right_col1] = right[0, :, :, :needed_w]

    def _compute_radius(self, area: float) -> int:
        """Convert object area to a square half-size radius."""
        return max(1, int(math.sqrt(max(float(area), 1.0)) / 2.0))

    def apply_attack(
        self,
        images: torch.Tensor,
        metas: list[dict[str, Any]],
        attack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attack borders around every object in a batch."""
        attack_tensor = self.attack if attack is None else attack
        strips = self._split_attack(attack_tensor)

        attacked_images = images.clone()
        # Use index-based access instead of tensor iteration (unbind views),
        # which can conflict when this method is called in both inference and grad modes.
        for image_idx in range(attacked_images.shape[0]):
            image = attacked_images[image_idx]
            meta = metas[image_idx]
            centers = meta.get("centers", [])
            areas = meta.get("areas", [1.0] * len(centers))

            for (center_x, center_y), area in zip(centers, areas):
                radius = self._compute_radius(area)
                self._apply_border_around_center(
                    image=image,
                    strips=strips,
                    center_x=int(center_x),
                    center_y=int(center_y),
                    radius=radius,
                )

        return attacked_images

    def _attack_loss(
        self,
        clean_prob: Optional[torch.Tensor],
        attacked_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the selected attack objective."""
        if self.config.objective == "divergence":
            if clean_prob is None:
                raise ValueError("clean_prob is required for objective='divergence'.")
            return -F.mse_loss(attacked_prob, clean_prob)

        if self.config.objective == "suppress_confidence":
            return attacked_prob.mean()

        if self.config.objective == "suppress_count":
            soft_count = torch.sigmoid(
                self.config.count_temperature * (attacked_prob - self.config.count_threshold)
            )
            return soft_count.mean()

        if self.config.objective == "center_like":
            if clean_prob is None:
                raise ValueError("clean_prob is required for objective='center_like'.")
            return -self._center_like_distance(attacked_prob, clean_prob)

        raise ValueError(f"Unknown objective: {self.config.objective}")

    def _center_like_distance(
        self,
        attacked_prob: torch.Tensor,
        target_prob: torch.Tensor,
    ) -> torch.Tensor:
        """CenterLoss-like distance on probabilities (MSE + focal-like term)."""
        attacked_clamped = attacked_prob.clamp(min=1e-6, max=1.0 - 1e-6)
        mse_term = F.mse_loss(attacked_clamped, target_prob)

        pos_weight = (1.0 - attacked_clamped) ** self.config.center_focal_alpha
        neg_weight = attacked_clamped ** self.config.center_focal_alpha
        pos_loss = -target_prob * pos_weight * torch.log(attacked_clamped)
        neg_loss = (
            -(1.0 - target_prob) ** self.config.center_focal_gamma
            * neg_weight
            * torch.log(1.0 - attacked_clamped)
        )
        focal_term = (pos_loss + neg_loss).mean()

        return (
            self.config.center_alpha * mse_term
            + (1.0 - self.config.center_alpha) * focal_term
        )

    @staticmethod
    def _compute_detection_scores(tp: int, fp: int, fn: int) -> dict[str, float]:
        """Compute accuracy/precision/recall/F1 from TP/FP/FN counts."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    @torch.inference_mode()
    def _evaluate_loader(
        self,
        loader: DataLoader,
        split_name: str,
        attack: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Evaluate attack on a dataloader with center-matching detection metrics."""
        loss_sum = 0.0
        max_clean = []
        max_attacked = []
        n_batches = 0

        clean_tp = clean_fp = clean_fn = 0
        attacked_tp = attacked_fp = attacked_fn = 0

        for images, _, metas, _ in tqdm(loader, desc=split_name, leave=False):
            images = images.to(self.device, non_blocking=True)
            clean_prob = torch.sigmoid(self.model(images))
            attacked_images = self.apply_attack(images, metas, attack=attack)
            attacked_prob = torch.sigmoid(self.model(attacked_images))

            loss = self._attack_loss(clean_prob, attacked_prob)
            loss_sum += loss.item()
            n_batches += 1

            max_clean.append(clean_prob.max().item())
            max_attacked.append(attacked_prob.max().item())

            clean_maps = clean_prob[:, 0].detach().cpu().numpy()
            attacked_maps = attacked_prob[:, 0].detach().cpu().numpy()

            for image_idx, meta in enumerate(metas):
                gt_centers = meta.get("centers", [])

                pred_clean = find_peaks(clean_maps[image_idx], threshold=self.config.eval_threshold)
                pred_attacked = find_peaks(attacked_maps[image_idx], threshold=self.config.eval_threshold)

                tp_clean, fp_clean, fn_clean = match_centers(
                    gt_centers,
                    pred_clean,
                    match_radius=self.config.match_radius,
                )
                tp_attacked, fp_attacked, fn_attacked = match_centers(
                    gt_centers,
                    pred_attacked,
                    match_radius=self.config.match_radius,
                )

                clean_tp += tp_clean
                clean_fp += fp_clean
                clean_fn += fn_clean

                attacked_tp += tp_attacked
                attacked_fp += fp_attacked
                attacked_fn += fn_attacked

        if n_batches == 0:
            return {
                "loss": 0.0,
                "clean_max_prob": 0.0,
                "attacked_max_prob": 0.0,
                "clean_accuracy": 0.0,
                "clean_precision": 0.0,
                "clean_recall": 0.0,
                "clean_f1": 0.0,
                "attacked_accuracy": 0.0,
                "attacked_precision": 0.0,
                "attacked_recall": 0.0,
                "attacked_f1": 0.0,
            }

        clean_scores = self._compute_detection_scores(clean_tp, clean_fp, clean_fn)
        attacked_scores = self._compute_detection_scores(attacked_tp, attacked_fp, attacked_fn)

        return {
            "loss": loss_sum / n_batches,
            "clean_max_prob": float(np.mean(max_clean)),
            "attacked_max_prob": float(np.mean(max_attacked)),
            "clean_accuracy": clean_scores["accuracy"],
            "clean_precision": clean_scores["precision"],
            "clean_recall": clean_scores["recall"],
            "clean_f1": clean_scores["f1"],
            "attacked_accuracy": attacked_scores["accuracy"],
            "attacked_precision": attacked_scores["precision"],
            "attacked_recall": attacked_scores["recall"],
            "attacked_f1": attacked_scores["f1"],
            "clean_tp": float(clean_tp),
            "clean_fp": float(clean_fp),
            "clean_fn": float(clean_fn),
            "attacked_tp": float(attacked_tp),
            "attacked_fp": float(attacked_fp),
            "attacked_fn": float(attacked_fn),
        }

    @torch.inference_mode()
    def validate_attack(self, attack: Optional[torch.Tensor] = None) -> dict[str, float]:
        """Evaluate current attack on the validation split."""
        return self._evaluate_loader(self.validation_loader, split_name="validation", attack=attack)

    @torch.inference_mode()
    def evaluate_test_attack(self, attack: Optional[torch.Tensor] = None) -> dict[str, float]:
        """Evaluate current attack on the test split."""
        return self._evaluate_loader(self.test_loader, split_name="test", attack=attack)

    def save_attack(self) -> None:
        """Persist attack tensor and log it to MLflow."""
        if self.attack_save_path is None:
            raise RuntimeError("attack_save_path is not set. Call train_attack first.")

        torch.save(self.attack.detach().cpu(), self.attack_save_path)
        mlflow.log_artifact(self.attack_save_path)

    def train_attack(self) -> tuple[torch.Tensor, str]:
        """Train the attack tensor against the frozen detector."""
        os.makedirs("runs", exist_ok=True)
        stopped_early = False
        global_step = 0
        skipped_empty_batches = 0
        skipped_no_grad_steps = 0

        mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run(run_name=self.config.run_name) as run:
            print("Connecting to MLflow tracking server at:", mlflow.get_tracking_uri())
            self.run_id = run.info.run_id
            self.attack_save_path = os.path.join("runs", f"{self.run_id}_attack.pt")
            print("Connected to MLflow. Run ID:", self.run_id)
            
            mlflow.log_params(
                {
                    "checkpoint_path": self.config.checkpoint_path,
                    "data_dir": self.config.data_dir,
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "attack_tile_size": self.attack_tile_size,
                    "thickness": self.thickness,
                    "batch_repetition": self.config.batch_repetition,
                    "early_stop": self.config.early_stop,
                    "validation_ratio": self.config.validation_ratio,
                    "seed": self.config.seed,
                    "objective": self.config.objective,
                    "count_threshold": self.config.count_threshold,
                    "count_temperature": self.config.count_temperature,
                    "center_alpha": self.config.center_alpha,
                    "center_focal_alpha": self.config.center_focal_alpha,
                    "center_focal_gamma": self.config.center_focal_gamma,
                    "experiment_name": self.config.experiment_name,
                    "mlflow_tracking_uri": mlflow.get_tracking_uri(),
                    "filter_empty_tiles": self.config.filter_empty_tiles,
                    "eval_threshold": self.config.eval_threshold,
                    "match_radius": self.config.match_radius,
                }
            )

            for epoch in tqdm(range(1, self.config.epochs + 1), desc=self.run_id):
                epoch_loss = 0.0
                n_updates = 0
                epoch_skipped_empty_batches = 0
                epoch_skipped_no_grad_steps = 0
                epoch_wall_start_s = time.perf_counter()
                epoch_step_time_s = 0.0
                epoch_unique_images = 0
                epoch_effective_image_passes = 0

                for images, _, metas, _ in tqdm(self.train_loader, desc="train", leave=False):
                    # If a batch has no objects, the attack is not applied anywhere
                    # and the loss has no dependency on the trainable attack tensor.
                    if all(len(meta.get("centers", [])) == 0 for meta in metas):
                        skipped_empty_batches += 1
                        epoch_skipped_empty_batches += 1
                        continue

                    images = images.to(self.device, non_blocking=True)
                    batch_size_current = int(images.shape[0])
                    epoch_unique_images += batch_size_current
                    clean_prob: Optional[torch.Tensor] = None
                    if self.config.objective in {"divergence", "center_like"}:
                        with torch.no_grad():
                            clean_prob = torch.sigmoid(self.model(images))

                    for _ in range(self.config.batch_repetition):
                        step_start_s = time.perf_counter()
                        self.optimizer.zero_grad(set_to_none=True)

                        attacked_images = self.apply_attack(images, metas, attack=self.attack)
                        attacked_prob = torch.sigmoid(self.model(attacked_images))
                        loss = self._attack_loss(clean_prob, attacked_prob)

                        if not loss.requires_grad:
                            skipped_no_grad_steps += 1
                            epoch_skipped_no_grad_steps += 1
                            continue

                        loss.backward()
                        self.optimizer.step()

                        with torch.no_grad():
                            self.attack.clamp_(self.norm_min, self.norm_max)

                        epoch_step_time_s += time.perf_counter() - step_start_s
                        epoch_effective_image_passes += batch_size_current
                        global_step += 1
                        n_updates += 1
                        epoch_loss += loss.item()

                        mlflow.log_metric("training_loss", loss.item(), step=global_step)
                        if clean_prob is not None:
                            mlflow.log_metric(
                                "max_prediction_unattacked",
                                clean_prob.max().item(),
                                step=global_step,
                            )
                        mlflow.log_metric(
                            "max_prediction_attacked",
                            attacked_prob.max().item(),
                            step=global_step,
                        )

                        if self.config.save_every > 0 and global_step % self.config.save_every == 0:
                            self.save_attack()

                        if global_step >= self.config.early_stop:
                            stopped_early = True
                            break

                    if stopped_early:
                        break

                avg_train_loss = epoch_loss / max(n_updates, 1)
                epoch_wall_time_s = time.perf_counter() - epoch_wall_start_s
                seconds_per_image_effective = (
                    epoch_step_time_s / max(epoch_effective_image_passes, 1)
                )
                images_per_second_effective = (
                    epoch_effective_image_passes / max(epoch_step_time_s, 1e-12)
                )
                message = (
                    f"EPOCH {epoch:03d} | objective={self.config.objective} "
                    f"| train_loss={avg_train_loss:.6f} "
                    f"| s/img_eff={seconds_per_image_effective:.6f} "
                    f"| img/s_eff={images_per_second_effective:.2f}"
                )
                if epoch_skipped_empty_batches > 0 or epoch_skipped_no_grad_steps > 0:
                    message += (
                        f" | skipped_empty_batches={epoch_skipped_empty_batches}"
                        f" | skipped_no_grad_steps={epoch_skipped_no_grad_steps}"
                    )

                if not self.config.skip_validation:
                    validation = self.validate_attack(attack=self.attack)
                    mlflow.log_metric(
                        "validation_loss", 
                        validation["loss"], 
                        step=epoch)
                    mlflow.log_metric(
                        "avg_train_loss",
                        avg_train_loss,
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_clean_max_prob",
                        validation["clean_max_prob"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_attacked_max_prob",
                        validation["attacked_max_prob"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_clean_accuracy",
                        validation["clean_accuracy"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_clean_precision",
                        validation["clean_precision"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_clean_recall",
                        validation["clean_recall"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_clean_f1",
                        validation["clean_f1"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_attacked_accuracy",
                        validation["attacked_accuracy"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_attacked_precision",
                        validation["attacked_precision"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_attacked_recall",
                        validation["attacked_recall"],
                        step=epoch,
                    )
                    mlflow.log_metric(
                        "validation_attacked_f1",
                        validation["attacked_f1"],
                        step=epoch,
                    )

                    message += (
                        f" | val_loss={validation['loss']:.6f}"
                        f" | max_prob={validation['clean_max_prob']:.4f}"
                        f"->{validation['attacked_max_prob']:.4f}"
                        f" | P={validation['clean_precision']:.3f}->{validation['attacked_precision']:.3f}"
                        f" | R={validation['clean_recall']:.3f}->{validation['attacked_recall']:.3f}"
                        f" | F1={validation['clean_f1']:.3f}->{validation['attacked_f1']:.3f}"
                        f" | Acc={validation['clean_accuracy']:.3f}->{validation['attacked_accuracy']:.3f}"
                    )

                mlflow.log_metric("train_epoch_wall_time_s", epoch_wall_time_s, step=epoch)
                mlflow.log_metric("train_epoch_step_time_s", epoch_step_time_s, step=epoch)
                mlflow.log_metric(
                    "train_epoch_unique_images",
                    float(epoch_unique_images),
                    step=epoch,
                )
                mlflow.log_metric(
                    "train_epoch_effective_image_passes",
                    float(epoch_effective_image_passes),
                    step=epoch,
                )
                mlflow.log_metric(
                    "train_seconds_per_image_effective",
                    seconds_per_image_effective,
                    step=epoch,
                )
                mlflow.log_metric(
                    "train_images_per_second_effective",
                    images_per_second_effective,
                    step=epoch,
                )

                tqdm.write(message)

                if stopped_early:
                    tqdm.write(f"Early stop reached at global_step={global_step}.")
                    break

            if skipped_empty_batches > 0 or skipped_no_grad_steps > 0:
                mlflow.log_metric("skipped_empty_batches", skipped_empty_batches)
                mlflow.log_metric("skipped_no_grad_steps", skipped_no_grad_steps)

            self.save_attack()

        if self.run_id is None:
            raise RuntimeError("Training ended without an MLflow run id.")

        return self.attack.detach(), self.run_id

    def image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert RGB uint8 numpy image to normalized model input tensor."""
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_t = TF.normalize(image_t, DOTA_MEAN, DOTA_STD)
        return image_t.unsqueeze(0).to(self.device)

    def tensor_to_image(self, image_t: torch.Tensor) -> np.ndarray:
        """Convert normalized model tensor back to displayable uint8 RGB image."""
        mean = torch.tensor(DOTA_MEAN, device=image_t.device).view(1, 3, 1, 1)
        std = torch.tensor(DOTA_STD, device=image_t.device).view(1, 3, 1, 1)
        denorm = (image_t * std + mean).clamp(0.0, 1.0)
        image_np = denorm[0].detach().cpu().permute(1, 2, 0).numpy()
        return (image_np * 255.0).astype(np.uint8)

    @torch.inference_mode()
    def show_attack_results(
        self,
        image: np.ndarray,
        meta: list[dict[str, Any]],
        attack: Optional[torch.Tensor] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        """Visualize clean/attacked predictions and their probability maps."""
        image_t = self.image_to_tensor(image)
        weight_map = get_gaussian_weight_map(self.tile_size, self.device)

        clean_prob_t = torch.sigmoid(self.model(image_t)) * weight_map
        attacked_image_t = self.apply_attack(image_t, meta, attack=attack)
        attacked_prob_t = torch.sigmoid(self.model(attacked_image_t)) * weight_map

        clean_prob = clean_prob_t[0, 0].detach().cpu().numpy()
        attacked_prob = attacked_prob_t[0, 0].detach().cpu().numpy()

        clean_centers = find_peaks(clean_prob, threshold=threshold)
        attacked_centers = find_peaks(attacked_prob, threshold=threshold)

        clean_img = draw_centers(self.tensor_to_image(image_t), clean_centers, (0, 165, 255))
        attacked_img = draw_centers(
            self.tensor_to_image(attacked_image_t),
            attacked_centers,
            (0, 165, 255),
        )

        clean_heatmap = plt.get_cmap("magma")(
            (clean_prob - clean_prob.min()) / (clean_prob.max() - clean_prob.min() + 1e-8)
        )[..., :3]
        attacked_heatmap = plt.get_cmap("magma")(
            (attacked_prob - attacked_prob.min())
            / (attacked_prob.max() - attacked_prob.min() + 1e-8)
        )[..., :3]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(clean_img)
        axes[0, 0].set_title(f"Clean prediction (max={clean_prob.max():.3f})")
        axes[0, 1].imshow(clean_heatmap)
        axes[0, 1].set_title("Clean probability")

        axes[1, 0].imshow(attacked_img)
        axes[1, 0].set_title(f"Attacked prediction (max={attacked_prob.max():.3f})")
        axes[1, 1].imshow(attacked_heatmap)
        axes[1, 1].set_title("Attacked probability")

        for axis in axes.ravel():
            axis.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_interesting_images(self, run_id: Optional[str] = None) -> None:
        """Render a fixed set of test images with the current or loaded attack."""
        if run_id is None:
            attack = self.attack.detach()
        else:
            attack = self._load_attack_artifact(run_id).to(self.device)

        img_dir = os.path.join(self.config.data_dir, "test", "image")
        mask_dir = os.path.join(self.config.data_dir, "test", "mask")

        for image_name in self.interesting_images:
            image_path = os.path.join(img_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name)

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image_bgr is None or mask is None:
                print(f"Skipping {image_name}: image/mask not found.")
                continue

            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            centers, areas = mask_to_centers(mask, return_areas=True)
            meta = [{"centers": centers, "areas": areas, "num_objects": len(centers)}]

            self.show_attack_results(image=image, meta=meta, attack=attack)


def parse_args() -> AttackConfig:
    """Parse CLI arguments and map them to AttackConfig."""
    parser = argparse.ArgumentParser(description="Train a frame adversarial attack on SegDino")
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to SegDino checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument("-e", "--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.5,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for train/validation/test loaders",
    )
    parser.add_argument(
        "--attack-tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help="Base size used for each attack strip chunk (attack width = 4 * attack_tile_size).",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=24,
        help="Border thickness of the trainable frame strips",
    )
    parser.add_argument(
        "--batch-repetition",
        type=int,
        default=1,
        help="Number of optimizer steps per sampled batch",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=1_000_000,
        help="Stop after this many optimizer updates",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Dataset root directory (expects train/test with image+mask folders)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Fraction of train split used for validation",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save attack artifact every N optimization steps (0 disables)",
    )
    parser.add_argument(
        "--attack-id",
        default=None,
        help="MLflow run id to initialize attack from a saved artifact",
    )
    parser.add_argument(
        "--attack-path",
        default=None,
        help="Local .pt file to initialize attack from",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Disable validation after each epoch",
    )
    parser.add_argument("--seed", type=int, default=1619, help="Random seed")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional MLflow run name (defaults to generated codename)",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT_NAME})",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help=(
            "MLflow tracking URI to use for this run "
            "(example: http://127.0.0.1:5000 or sqlite:///mlflow.db)."
        ),
    )
    parser.add_argument(
        "--include-empty-tiles",
        action="store_true",
        help="Include empty tiles in dataloaders (default is to filter them out for speed).",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Detection threshold used for validation/test center matching metrics.",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=DEFAULT_MATCH_RADIUS,
        help="Distance radius used to match predicted centers to ground truth.",
    )
    parser.add_argument(
        "--objective",
        default=DEFAULT_OBJECTIVE,
        choices=["divergence", "suppress_confidence", "suppress_count", "center_like"],
        help=(
            "Attack objective: 'divergence' maximizes clean/attacked difference, "
            "'suppress_confidence' minimizes average attacked probability, "
            "'suppress_count' minimizes a soft thresholded count (default), "
            "'center_like' maximizes a CenterLoss-like discrepancy to clean predictions."
        ),
    )
    parser.add_argument(
        "--count-threshold",
        type=float,
        default=DEFAULT_COUNT_THRESHOLD,
        help="Probability threshold used by objective='suppress_count'.",
    )
    parser.add_argument(
        "--count-temperature",
        type=float,
        default=DEFAULT_COUNT_TEMPERATURE,
        help="Sigmoid temperature used by objective='suppress_count'.",
    )
    parser.add_argument(
        "--center-alpha",
        type=float,
        default=DEFAULT_CENTER_ALPHA,
        help="MSE mix factor used by objective='center_like' (same role as CenterLoss alpha).",
    )
    parser.add_argument(
        "--center-focal-alpha",
        type=float,
        default=DEFAULT_CENTER_FOCAL_ALPHA,
        help="Focal alpha exponent used by objective='center_like'.",
    )
    parser.add_argument(
        "--center-focal-gamma",
        type=float,
        default=DEFAULT_CENTER_FOCAL_GAMMA,
        help="Focal gamma exponent used by objective='center_like'.",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.attack_tile_size <= 0:
        parser.error("--attack-tile-size must be > 0")
    if args.thickness <= 0:
        parser.error("--thickness must be > 0")
    if args.batch_repetition <= 0:
        parser.error("--batch-repetition must be > 0")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("--validation-ratio must be in (0, 1)")
    if not 0.0 <= args.eval_threshold <= 1.0:
        parser.error("--eval-threshold must be in [0, 1]")
    if args.match_radius <= 0:
        parser.error("--match-radius must be > 0")
    if not 0.0 <= args.count_threshold <= 1.0:
        parser.error("--count-threshold must be in [0, 1]")
    if args.count_temperature <= 0:
        parser.error("--count-temperature must be > 0")
    if not 0.0 <= args.center_alpha <= 1.0:
        parser.error("--center-alpha must be in [0, 1]")
    if args.center_focal_alpha < 0:
        parser.error("--center-focal-alpha must be >= 0")
    if args.center_focal_gamma < 0:
        parser.error("--center-focal-gamma must be >= 0")

    run_name = args.run_name or generate_codename()

    return AttackConfig(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        attack_tile_size=args.attack_tile_size,
        thickness=args.thickness,
        batch_repetition=args.batch_repetition,
        early_stop=args.early_stop,
        num_workers=args.workers,
        validation_ratio=args.validation_ratio,
        save_every=args.save_every,
        attack_id=args.attack_id,
        attack_path=args.attack_path,
        skip_validation=args.skip_validation,
        seed=args.seed,
        run_name=run_name,
        objective=args.objective,
        count_threshold=args.count_threshold,
        count_temperature=args.count_temperature,
        center_alpha=args.center_alpha,
        center_focal_alpha=args.center_focal_alpha,
        center_focal_gamma=args.center_focal_gamma,
        experiment_name=args.experiment_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        filter_empty_tiles=not args.include_empty_tiles,
        eval_threshold=args.eval_threshold,
        match_radius=args.match_radius,
    )


def main() -> None:
    """CLI entrypoint for training frame attack parameters."""
    config = parse_args()

    print("Initializing frame attack trainer...")
    trainer = FramePatchAttack(config)

    print(f"Training attack with objective='{config.objective}'...")
    _, run_id = trainer.train_attack()
    print(f"Attack trained and saved with run_id: {run_id}")


if __name__ == "__main__":
    main()
