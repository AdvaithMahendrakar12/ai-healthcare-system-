from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

try:
    from .utils import (
        collect_image_mask_pairs,
        configure_logging,
        dice_np,
        get_segmentation_custom_objects,
        iou_np,
        pixel_accuracy_np,
        resolve_model_path,
        sensitivity_np,
        specificity_np,
        split_counts,
        summarize_environment,
        training_root,
        validate_dataset_structure,
        validate_runtime_requirements,
    )
except ImportError:
    from utils import (
        collect_image_mask_pairs,
        configure_logging,
        dice_np,
        get_segmentation_custom_objects,
        iou_np,
        pixel_accuracy_np,
        resolve_model_path,
        sensitivity_np,
        specificity_np,
        split_counts,
        summarize_environment,
        training_root,
        validate_dataset_structure,
        validate_runtime_requirements,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fetal ultrasound segmentation model")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-dir", type=Path, default=training_root())
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("results/fetal_eval_metrics.json"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    validate_dataset_structure(args.dataset_dir)
    model_path = resolve_model_path(args.model_path, args.model_dir)

    if args.check_only:
        payload = {
            "environment": summarize_environment(),
            "dataset_counts": split_counts(args.dataset_dir),
            "split": args.split,
            "resolved_model_path": str(model_path),
        }
        print(json.dumps(payload, indent=2))
        return

    validate_runtime_requirements(require_tensorflow=True)

    import cv2
    import numpy as np
    from tensorflow.keras.models import load_model

    custom_objects = get_segmentation_custom_objects()
    model = load_model(str(model_path), custom_objects=custom_objects, compile=False)

    split_dir = args.dataset_dir / args.split
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    pairs = collect_image_mask_pairs(image_dir, mask_dir)
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {split_dir}")

    images: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for image_path, mask_path in pairs:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        image = cv2.resize(image, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype("float32") / 255.0
        mask = (mask.astype("float32") / 255.0) > 0.5

        images.append(np.expand_dims(image, axis=-1))
        masks.append(np.expand_dims(mask.astype("float32"), axis=-1))

    if not images:
        raise RuntimeError("No readable image/mask pairs were found for evaluation.")

    x = np.array(images, dtype="float32")
    y_true = np.array(masks, dtype="float32")

    y_pred_prob = model.predict(x, batch_size=args.batch_size, verbose=1)
    y_pred_bin = (y_pred_prob > args.threshold).astype("float32")

    dice_scores: List[float] = []
    iou_scores: List[float] = []
    pixel_accs: List[float] = []
    sensitivities: List[float] = []
    specificities: List[float] = []

    for idx in range(y_true.shape[0]):
        gt = y_true[idx]
        pred = y_pred_bin[idx]
        dice_scores.append(dice_np(gt, pred))
        iou_scores.append(iou_np(gt, pred))
        pixel_accs.append(pixel_accuracy_np(gt, pred))
        sensitivities.append(sensitivity_np(gt, pred))
        specificities.append(specificity_np(gt, pred))

    metrics = {
        "model_path": str(model_path),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "num_samples": int(y_true.shape[0]),
        "threshold": args.threshold,
        "dice_coefficient": _summary(dice_scores),
        "iou_score": _summary(iou_scores),
        "pixel_accuracy": _summary(pixel_accs),
        "sensitivity": _summary(sensitivities),
        "specificity": _summary(specificities),
        "mean_probability": float(y_pred_prob.mean()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
