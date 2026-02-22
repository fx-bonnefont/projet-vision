"""Train and inspect a frame-like adversarial attack for SegDino tiles."""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from codenames import generate_codename
from dataset import DOTA_MEAN, DOTA_STD, PreTiledDataset, mask_to_centers
from inference import draw_centers, find_peaks, get_gaussian_weight_map, load_checkpoint
from train import segdino_collate

# Default project configuration.
DEFAULT_TILE_SIZE = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_DATA_DIR = "segdata/DOTA/DOTA_PLANES_TILED"
DEFAULT_NUM_WORKERS = 4
DEFAULT_VALIDATION_RATIO = 0.3
DEFAULT_THRESHOLD = 0.3
DEFAULT_INTERESTING_IMAGES = ("P0023_obj_27.png",)
DEFAULT_LOG_INTERVAL = 20
DEFAULT_LOG_LEVEL = "INFO"


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


def setup_logging(level: str) -> logging.Logger:
    """Configure and return the module logger."""
    logger = logging.getLogger("frame_attack")
    logger.setLevel(level.upper())

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


@dataclass(slots=True)
class AttackConfig:
    """Runtime configuration for frame attack training."""

    checkpoint_path: str
    data_dir: str = DEFAULT_DATA_DIR
    epochs: int = 2
    learning_rate: float = 0.5
    batch_size: int = DEFAULT_BATCH_SIZE
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
    log_interval: int = DEFAULT_LOG_INTERVAL
    log_level: str = DEFAULT_LOG_LEVEL


class FramePatchAttack:
    """Train a border attack that surrounds each detected object with strips."""

    def __init__(self, config: AttackConfig, logger: Optional[logging.Logger] = None):
        """Initialize data, model, optimizer, and trainable attack tensor."""
        self.config = config
        self.logger = logger or logging.getLogger("frame_attack")
        self.device = select_device()
        self.tile_size = DEFAULT_TILE_SIZE

        set_seed(self.config.seed)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.logger.info("Using device: %s", self.device)

        self.model = load_checkpoint(self.config.checkpoint_path, self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.logger.info("Loaded checkpoint: %s", self.config.checkpoint_path)

        self.train_loader, self.validation_loader, self.test_loader = self._build_dataloaders()
        self.attack = torch.nn.Parameter(self._init_attack_tensor().to(self.device))
        self.thickness = int(self.attack.shape[2])
        self.logger.info(
            "Attack tensor shape: %s",
            tuple(self.attack.shape),
        )

        # Keep attack values in valid normalized image range.
        self.norm_min = TF.normalize(torch.zeros_like(self.attack), DOTA_MEAN, DOTA_STD)
        self.norm_max = TF.normalize(torch.ones_like(self.attack), DOTA_MEAN, DOTA_STD)

        self.optimizer = torch.optim.AdamW([self.attack], lr=self.config.learning_rate)
        self.logger.info("Optimizer: AdamW(lr=%s)", self.config.learning_rate)

        self.run_id: Optional[str] = None
        self.attack_save_path: Optional[str] = None
        self.interesting_images = list(DEFAULT_INTERESTING_IMAGES)

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test dataloaders for tiled data."""
        train_dataset = PreTiledDataset(self.config.data_dir, "train", return_empty=True)
        test_dataset = PreTiledDataset(self.config.data_dir, "test", return_empty=True)

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

        self.logger.info(
            "Dataloaders ready | train=%d | val=%d | test=%d | batch_size=%d | workers=%d",
            len(dataset_train),
            len(dataset_validation),
            len(test_dataset),
            self.config.batch_size,
            self.config.num_workers,
        )

        return train_loader, validation_loader, test_loader

    def _init_attack_tensor(self) -> torch.Tensor:
        """Initialize trainable attack tensor from random state or loaded artifact."""
        if self.config.attack_path:
            self.logger.info("Initializing attack from local file: %s", self.config.attack_path)
            return self._load_attack_file(self.config.attack_path)

        if self.config.attack_id:
            self.logger.info("Initializing attack from MLflow run: %s", self.config.attack_id)
            return self._load_attack_artifact(self.config.attack_id)

        width = 4 * self.tile_size
        self.logger.info(
            "Initializing random attack tensor with thickness=%d and width=%d",
            self.config.thickness,
            width,
        )
        return torch.rand((1, 3, self.config.thickness, width), dtype=torch.float32)

    def _load_attack_file(self, path: str) -> torch.Tensor:
        """Load an attack tensor from a local `.pt` file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attack file not found: {path}")

        attack = torch.load(path, map_location=self.device, weights_only=False)
        validated_attack = self._validate_attack_tensor(attack, source=path)
        self.logger.info("Loaded attack tensor from local file: %s", path)
        return validated_attack

    def _load_attack_artifact(self, run_id: str) -> torch.Tensor:
        """Load an attack tensor from an MLflow run artifact."""
        artifacts = self.mlflow_client.list_artifacts(run_id)
        attack_artifact = next((item.path for item in artifacts if item.path.endswith(".pt")), None)
        if attack_artifact is None:
            raise FileNotFoundError(f"No .pt artifact found for run id '{run_id}'.")

        local_path = self.mlflow_client.download_artifacts(run_id, attack_artifact)
        attack = torch.load(local_path, map_location=self.device, weights_only=False)
        validated_attack = self._validate_attack_tensor(attack, source=f"mlflow:{run_id}/{attack_artifact}")
        self.logger.info("Loaded attack tensor from MLflow artifact: %s", attack_artifact)
        return validated_attack

    def _validate_attack_tensor(self, attack: torch.Tensor, source: str) -> torch.Tensor:
        """Validate attack tensor shape and return float tensor."""
        if not isinstance(attack, torch.Tensor):
            raise TypeError(f"Attack from {source} is not a torch.Tensor.")

        if attack.ndim != 4 or attack.shape[0] != 1 or attack.shape[1] != 3:
            raise ValueError(
                f"Attack from {source} must have shape [1, 3, thickness, width], "
                f"got {tuple(attack.shape)}."
            )

        if attack.shape[3] != 4 * self.tile_size:
            raise ValueError(
                f"Attack width must be {4 * self.tile_size}, got {attack.shape[3]} "
                f"from {source}."
            )

        return attack.float()

    def _split_attack(self, attack: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split attack tensor into left/top/right/bottom strips."""
        left = attack[:, :, :, : self.tile_size]
        top = attack[:, :, :, self.tile_size : 2 * self.tile_size]
        right = attack[:, :, :, 2 * self.tile_size : 3 * self.tile_size]
        bottom = attack[:, :, :, 3 * self.tile_size :]
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
        for image_idx, image in enumerate(attacked_images):
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

    def _attack_loss(self, clean_prob: torch.Tensor, attacked_prob: torch.Tensor) -> torch.Tensor:
        """Return objective minimized during training (maximize map discrepancy)."""
        return -F.mse_loss(attacked_prob, clean_prob)

    @torch.inference_mode()
    def validate_attack(self, attack: Optional[torch.Tensor] = None) -> dict[str, float]:
        """Evaluate current attack on the validation split."""
        loss_sum = 0.0
        max_clean = []
        max_attacked = []
        n_batches = 0

        for images, _, metas, _ in tqdm(self.validation_loader, desc="validation", leave=False):
            images = images.to(self.device, non_blocking=True)
            clean_prob = torch.sigmoid(self.model(images))
            attacked_images = self.apply_attack(images, metas, attack=attack)
            attacked_prob = torch.sigmoid(self.model(attacked_images))

            loss = self._attack_loss(clean_prob, attacked_prob)
            loss_sum += loss.item()
            n_batches += 1

            max_clean.append(clean_prob.max().item())
            max_attacked.append(attacked_prob.max().item())

        if n_batches == 0:
            return {
                "loss": 0.0,
                "clean_max_prob": 0.0,
                "attacked_max_prob": 0.0,
            }

        return {
            "loss": loss_sum / n_batches,
            "clean_max_prob": float(np.mean(max_clean)),
            "attacked_max_prob": float(np.mean(max_attacked)),
        }

    def save_attack(self) -> None:
        """Persist attack tensor and log it to MLflow."""
        if self.attack_save_path is None:
            raise RuntimeError("attack_save_path is not set. Call train_attack first.")

        torch.save(self.attack.detach().cpu(), self.attack_save_path)
        mlflow.log_artifact(self.attack_save_path)
        self.logger.info("Saved attack artifact: %s", self.attack_save_path)

    def train_attack(self) -> tuple[torch.Tensor, str]:
        """Train the attack tensor against the frozen detector."""
        os.makedirs("runs", exist_ok=True)
        stopped_early = False
        global_step = 0
        self.logger.info("Starting frame attack training.")
        self.logger.info("Training configuration: %s", asdict(self.config))

        with mlflow.start_run(run_name=self.config.run_name) as run:
            self.run_id = run.info.run_id
            self.attack_save_path = os.path.join("runs", f"{self.run_id}_attack.pt")
            self.logger.info("MLflow run id: %s", self.run_id)
            self.logger.info("Attack checkpoints will be written to: %s", self.attack_save_path)

            mlflow.log_params(
                {
                    "checkpoint_path": self.config.checkpoint_path,
                    "data_dir": self.config.data_dir,
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "thickness": self.thickness,
                    "batch_repetition": self.config.batch_repetition,
                    "early_stop": self.config.early_stop,
                    "validation_ratio": self.config.validation_ratio,
                    "seed": self.config.seed,
                }
            )

            for epoch in tqdm(range(1, self.config.epochs + 1), desc=self.run_id):
                epoch_loss = 0.0
                n_updates = 0
                interval_loss = 0.0
                interval_updates = 0

                for images, _, metas, _ in tqdm(self.train_loader, desc="train", leave=False):
                    images = images.to(self.device, non_blocking=True)

                    with torch.no_grad():
                        clean_prob = torch.sigmoid(self.model(images))

                    for _ in range(self.config.batch_repetition):
                        self.optimizer.zero_grad(set_to_none=True)

                        attacked_images = self.apply_attack(images, metas, attack=self.attack)
                        attacked_prob = torch.sigmoid(self.model(attacked_images))
                        loss = self._attack_loss(clean_prob, attacked_prob)

                        loss.backward()
                        self.optimizer.step()

                        with torch.no_grad():
                            self.attack.clamp_(self.norm_min, self.norm_max)

                        global_step += 1
                        n_updates += 1
                        epoch_loss += loss.item()
                        interval_loss += loss.item()
                        interval_updates += 1

                        mlflow.log_metric("training_loss", loss.item(), step=global_step)
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

                        if (
                            self.config.log_interval > 0
                            and global_step % self.config.log_interval == 0
                        ):
                            avg_interval_loss = interval_loss / max(interval_updates, 1)
                            interval_message = (
                                f"step={global_step:06d} "
                                f"| avg_loss={avg_interval_loss:.6f} "
                                f"| max_prob={clean_prob.max().item():.4f}"
                                f"->{attacked_prob.max().item():.4f}"
                            )
                            tqdm.write(interval_message)
                            self.logger.info(interval_message)
                            interval_loss = 0.0
                            interval_updates = 0

                        if self.config.save_every > 0 and global_step % self.config.save_every == 0:
                            self.save_attack()

                        if global_step >= self.config.early_stop:
                            stopped_early = True
                            break

                    if stopped_early:
                        break

                avg_train_loss = epoch_loss / max(n_updates, 1)
                message = f"EPOCH {epoch:03d} | train_loss={avg_train_loss:.6f}"

                if not self.config.skip_validation:
                    validation = self.validate_attack(attack=self.attack)
                    mlflow.log_metric("validation_loss", validation["loss"], step=global_step)
                    mlflow.log_metric(
                        "validation_clean_max_prob",
                        validation["clean_max_prob"],
                        step=global_step,
                    )
                    mlflow.log_metric(
                        "validation_attacked_max_prob",
                        validation["attacked_max_prob"],
                        step=global_step,
                    )

                    message += (
                        f" | val_loss={validation['loss']:.6f}"
                        f" | max_prob={validation['clean_max_prob']:.4f}"
                        f"->{validation['attacked_max_prob']:.4f}"
                    )

                tqdm.write(message)
                self.logger.info(message)

                if stopped_early:
                    early_stop_message = f"Early stop reached at global_step={global_step}."
                    tqdm.write(early_stop_message)
                    self.logger.warning(early_stop_message)
                    break

            self.save_attack()

        if self.run_id is None:
            raise RuntimeError("Training ended without an MLflow run id.")

        self.logger.info("Training complete. Final attack stored at: %s", self.attack_save_path)
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
                self.logger.warning("Skipping %s: image/mask not found.", image_name)
                continue

            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            centers, areas = mask_to_centers(mask, return_areas=True)
            meta = [{"centers": centers, "areas": areas, "num_objects": len(centers)}]

            self.show_attack_results(image=image, meta=meta, attack=attack)


def parse_args() -> AttackConfig:
    """Parse CLI arguments and map them to AttackConfig."""
    parser = argparse.ArgumentParser(description="Train a frame adversarial attack on SegDino")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to SegDino checkpoint")
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
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help="Log one training update every N optimizer steps (0 disables step logs)",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.thickness <= 0:
        parser.error("--thickness must be > 0")
    if args.batch_repetition <= 0:
        parser.error("--batch-repetition must be > 0")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if args.log_interval < 0:
        parser.error("--log-interval must be >= 0")
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("--validation-ratio must be in (0, 1)")

    run_name = args.run_name or generate_codename()

    return AttackConfig(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
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
        log_interval=args.log_interval,
        log_level=args.log_level,
    )


def main() -> None:
    """CLI entrypoint for training frame attack parameters."""
    config = parse_args()
    logger = setup_logging(config.log_level)
    logger.info("CLI configuration: %s", asdict(config))

    logger.info("Initializing frame attack trainer...")
    trainer = FramePatchAttack(config, logger=logger)

    logger.info("Training attack...")
    _, run_id = trainer.train_attack()
    logger.info("Attack trained and saved with run_id: %s", run_id)


if __name__ == "__main__":
    main()
