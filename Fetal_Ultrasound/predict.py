from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .utils import (
        configure_logging,
        create_overlay,
        resolve_model_path,
        summarize_environment,
        training_root,
        get_segmentation_custom_objects,
        preprocess_ultrasound_image,
        validate_runtime_requirements,
    )
except ImportError:
    from utils import (
        configure_logging,
        create_overlay,
        resolve_model_path,
        summarize_environment,
        training_root,
        get_segmentation_custom_objects,
        preprocess_ultrasound_image,
        validate_runtime_requirements,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fetal ultrasound segmentation inference")
    parser.add_argument("--image", type=Path, required=True, help="Input ultrasound image path")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit model file (.h5/.keras)")
    parser.add_argument("--model-dir", type=Path, default=training_root(), help="Directory to auto-discover model")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for overlay outputs")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--assume-preprocessed", action="store_true", help="Skip CLAHE+denoise")
    parser.add_argument("--check-only", action="store_true", help="Validate env and model path only")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    model_path = resolve_model_path(args.model_path, args.model_dir)

    if args.check_only:
        payload = {
            "environment": summarize_environment(),
            "resolved_model_path": str(model_path),
            "image_exists": args.image.exists(),
        }
        print(json.dumps(payload, indent=2))
        return

    validate_runtime_requirements(require_tensorflow=True)

    import numpy as np
    from tensorflow.keras.models import load_model

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    custom_objects = get_segmentation_custom_objects()
    model = load_model(str(model_path), custom_objects=custom_objects, compile=False)

    model_input = preprocess_ultrasound_image(
        image_path=args.image,
        img_size=(args.img_size, args.img_size),
        apply_clahe_eq=not args.assume_preprocessed,
        denoise=not args.assume_preprocessed,
    )

    prediction = model.predict(model_input, verbose=0)[0, :, :, 0]
    binary_mask = (prediction > args.threshold).astype("float32")

    positive_region = prediction[binary_mask > 0.5]
    mean_positive_prob = float(positive_region.mean()) if positive_region.size else 0.0
    coverage_percent = float(binary_mask.mean() * 100.0)

    result_dir = args.output_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    overlay_path, mask_path = create_overlay(
        original_image_path=args.image,
        mask_binary=binary_mask,
        output_path=result_dir / f"{args.image.stem}_overlay.png",
        img_size=(args.img_size, args.img_size),
    )

    payload = {
        "model_path": str(model_path),
        "image_path": str(args.image),
        "threshold": args.threshold,
        "coverage_percent": coverage_percent,
        "mean_probability": float(prediction.mean()),
        "mean_positive_probability": mean_positive_prob,
        "max_probability": float(prediction.max()),
        "detected": coverage_percent > 0.5,
        "overlay_path": str(overlay_path),
        "mask_path": str(mask_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
