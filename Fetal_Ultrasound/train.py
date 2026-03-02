from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

try:
    from .model import build_model
    from .utils import (
        collect_image_mask_pairs,
        configure_logging,
        enable_gpu_memory_growth,
        ensure_dir,
        get_segmentation_custom_objects,
        save_json,
        set_global_seed,
        split_counts,
        training_root,
        validate_dataset_structure,
        validate_runtime_requirements,
    )
except ImportError:
    from model import build_model
    from utils import (
        collect_image_mask_pairs,
        configure_logging,
        enable_gpu_memory_growth,
        ensure_dir,
        get_segmentation_custom_objects,
        save_json,
        set_global_seed,
        split_counts,
        training_root,
        validate_dataset_structure,
        validate_runtime_requirements,
    )

import logging

LOGGER = logging.getLogger("fetal_ultrasound.train")


@dataclass
class TrainConfig:
    dataset_dir: Path
    output_dir: Path
    model_type: str = "unet"
    img_height: int = 256
    img_width: int = 256
    input_channels: int = 1
    filters_base: int = 32
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 5e-4
    augmentation: bool = True
    rotation_range: float = 15.0
    width_shift: float = 0.1
    height_shift: float = 0.1
    zoom_range: float = 0.15
    horizontal_flip: bool = True
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 6
    reduce_lr_factor: float = 0.5
    seed: int = 42

    @property
    def img_size(self) -> Tuple[int, int]:
        return (self.img_height, self.img_width)


class SegmentationDataGenerator:
    """Paired image/mask generator with synchronized augmentation."""

    def __init__(
        self,
        dataset_split_dir: Path,
        batch_size: int,
        img_size: Tuple[int, int],
        augment: bool,
        aug_config: Dict[str, float],
        seed: int,
    ):
        import numpy as np
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        image_dir = dataset_split_dir / "images"
        mask_dir = dataset_split_dir / "masks"

        self.pairs = collect_image_mask_pairs(image_dir, mask_dir)
        if not self.pairs:
            raise ValueError(f"No image/mask pairs found in {dataset_split_dir}")

        self.batch_size = batch_size
        self.img_size = img_size
        self.steps_per_epoch = max(1, math.floor(len(self.pairs) / batch_size))

        if augment:
            datagen_kwargs = {
                "rescale": 1.0 / 255.0,
                "rotation_range": aug_config["rotation_range"],
                "width_shift_range": aug_config["width_shift"],
                "height_shift_range": aug_config["height_shift"],
                "zoom_range": aug_config["zoom_range"],
                "horizontal_flip": aug_config["horizontal_flip"],
                "fill_mode": "nearest",
            }
        else:
            datagen_kwargs = {"rescale": 1.0 / 255.0}

        image_datagen = ImageDataGenerator(**datagen_kwargs)
        mask_datagen = ImageDataGenerator(**datagen_kwargs)

        self.image_flow = image_datagen.flow_from_directory(
            str(dataset_split_dir),
            classes=["images"],
            class_mode=None,
            color_mode="grayscale",
            target_size=img_size,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
        )

        self.mask_flow = mask_datagen.flow_from_directory(
            str(dataset_split_dir),
            classes=["masks"],
            class_mode=None,
            color_mode="grayscale",
            target_size=img_size,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
        )

        self.np = np

    def __iter__(self):
        return self

    def __next__(self):
        images = next(self.image_flow)
        masks = next(self.mask_flow)

        if not hasattr(self, "logged_mask_info"):
            import numpy as np
            LOGGER.info(
                f"Mask batch diagnostics: min={np.min(masks)}, max={np.max(masks)}, dtype={masks.dtype}"
            )
            self.logged_mask_info = True

        masks = (masks > 0.5).astype(self.np.float32)
        return images, masks


def train(config: TrainConfig) -> Dict[str, str]:
    validate_runtime_requirements(require_tensorflow=True)

    import tensorflow as tf
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import AdamW

    set_global_seed(config.seed)
    enable_gpu_memory_growth()

    validate_dataset_structure(config.dataset_dir)
    counts = split_counts(config.dataset_dir)
    LOGGER.info("Dataset counts: %s", json.dumps(counts, indent=2))

    train_split = config.dataset_dir / "train"
    val_split = config.dataset_dir / "val"

    aug_config = {
        "rotation_range": config.rotation_range,
        "width_shift": config.width_shift,
        "height_shift": config.height_shift,
        "zoom_range": config.zoom_range,
        "horizontal_flip": config.horizontal_flip,
    }

    train_gen = SegmentationDataGenerator(
        dataset_split_dir=train_split,
        batch_size=config.batch_size,
        img_size=config.img_size,
        augment=config.augmentation,
        aug_config=aug_config,
        seed=config.seed,
    )

    val_gen = SegmentationDataGenerator(
        dataset_split_dir=val_split,
        batch_size=config.batch_size,
        img_size=config.img_size,
        augment=False,
        aug_config=aug_config,
        seed=config.seed,
    )

    custom = get_segmentation_custom_objects()
    model = build_model(
        model_type=config.model_type,
        input_size=(config.img_height, config.img_width, config.input_channels),
        filters_base=config.filters_base,
    )

    model.compile(
        optimizer=AdamW(learning_rate=config.learning_rate),
        loss=custom["combined_loss"],
        metrics=[
            custom["dice_coef"],
            custom["iou_score"],
            custom["pixel_accuracy"],
            custom["sensitivity"],
            custom["specificity"],
            "binary_crossentropy",
        ],
    )

    LOGGER.info("Model summary: %s parameters", f"{model.count_params():,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fetal_ultrasound_{config.model_type}_{timestamp}"

    ensure_dir(config.output_dir)
    best_model_path = config.output_dir / f"{run_name}_best.h5"
    final_model_path = config.output_dir / f"{run_name}_final.h5"
    config_path = config.output_dir / f"{run_name}_config.json"
    labels_path = config.output_dir / f"{run_name}_labels.json"
    history_plot_path = config.output_dir / f"{run_name}_history.json"
    log_csv_path = config.output_dir / f"{run_name}_training_log.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        CSVLogger(str(log_csv_path), append=False),
    ]

    config_payload = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    config_payload.update(
        {
            "img_size": list(config.img_size),
            "run_name": run_name,
            "best_model_path": str(best_model_path),
            "final_model_path": str(final_model_path),
            "preprocessing": {
                "color_mode": "grayscale",
                "normalize": "image / 255.0",
                "mask_binarization_threshold": 0.5,
                "training_dataset_assumption": "dataset images are already CLAHE+denoise preprocessed",
            },
        }
    )
    save_json(config_payload, config_path)

    save_json(
        {
            "task": "fetal_ultrasound_head_segmentation",
            "classes": {"0": "background", "1": "fetal_head"},
        },
        labels_path,
    )

    LOGGER.info("Starting training for %d epochs", config.epochs)

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_gen.steps_per_epoch,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model_path)

    save_json({k: [float(vv) for vv in v] for k, v in history.history.items()}, history_plot_path)

    best_epoch = int(tf.math.argmax(history.history["val_dice_coef"]).numpy())
    best_val_dice = float(history.history["val_dice_coef"][best_epoch])
    best_val_iou = float(history.history["val_iou_score"][best_epoch])
    LOGGER.info(
        "Best epoch=%d val_dice=%.4f val_iou=%.4f",
        best_epoch + 1,
        best_val_dice,
        best_val_iou,
    )

    contract_path = config.output_dir / "contract_latest.json"
    save_json(
        {
            "disease_type": "fetal_ultrasound",
            "model_path": str(best_model_path),
            "model_type": "segmentation",
            "img_size": config.img_height,
            "labels": ["Fetal Head Contour Segmentation"],
            "labels_path": str(labels_path),
            "architecture": "Attention U-Net" if config.model_type == "attention_unet" else "U-Net",
            "description": "Performs semantic segmentation to outline fetal head contour on ultrasound images.",
            "metrics": {
                "best_epoch": best_epoch + 1,
                "best_val_dice": best_val_dice,
                "best_val_iou": best_val_iou,
            },
            "created_at": datetime.now().isoformat(),
        },
        contract_path,
    )

    return {
        "run_name": run_name,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "config_path": str(config_path),
        "labels_path": str(labels_path),
        "log_csv_path": str(log_csv_path),
        "history_json_path": str(history_plot_path),
        "contract_path": str(contract_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fetal ultrasound segmentation model")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--output-dir", type=Path, default=training_root())
    parser.add_argument("--model-type", choices=["unet", "attention_unet"], default="attention_unet")
    parser.add_argument("--img-height", type=int, default=256)
    parser.add_argument("--img-width", type=int, default=256)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--filters-base", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate setup without training")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.check_only:
        counts = split_counts(args.dataset_dir)
        print(json.dumps({"dataset_dir": str(args.dataset_dir), "counts": counts}, indent=2))
        return

    cfg = TrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        img_height=args.img_height,
        img_width=args.img_width,
        input_channels=args.input_channels,
        filters_base=args.filters_base,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        augmentation=not args.disable_augmentation,
        seed=args.seed,
    )

    artifacts = train(cfg)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
