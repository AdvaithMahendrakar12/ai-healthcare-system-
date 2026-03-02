from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

LOGGER = logging.getLogger("fetal_ultrasound.preprocessing")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def mask_name_for_image(image_name: str) -> str:
    if image_name.endswith("_HC.png"):
        return image_name.replace("_HC.png", "_HC_Annotation.png")
    return f"{Path(image_name).stem}_Annotation.png"


def discover_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    image_dir = source_dir / "images"
    mask_dir = source_dir / "masks"

    pairs: List[Tuple[Path, Path]] = []
    if image_dir.exists() and mask_dir.exists():
        for image_path in sorted(image_dir.glob("*.png")):
            mask_path = mask_dir / mask_name_for_image(image_path.name)
            if mask_path.exists():
                pairs.append((image_path, mask_path))
        return pairs

    for image_path in sorted(source_dir.glob("*.png")):
        if image_path.name.endswith("_Annotation.png"):
            continue
        mask_path = source_dir / mask_name_for_image(image_path.name)
        if mask_path.exists():
            pairs.append((image_path, mask_path))

    return pairs


def prepare_split_dirs(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "masks").mkdir(parents=True, exist_ok=True)


def preprocess_and_write(
    image_path: Path,
    mask_path: Path,
    out_image_path: Path,
    out_mask_path: Path,
    img_size: int,
) -> None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(out_image_path), image)
    cv2.imwrite(str(out_mask_path), mask)


def run_preprocessing(
    source_dir: Path,
    output_dir: Path,
    img_size: int,
    val_split: float,
    test_split: float,
    seed: int,
) -> None:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("source_dir and output_dir cannot be identical")

    pairs = discover_pairs(source_dir)
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {source_dir}")

    LOGGER.info("Found %d valid image/mask pairs", len(pairs))

    train_pairs, temp_pairs = train_test_split(
        pairs,
        test_size=(val_split + test_split),
        random_state=seed,
        shuffle=True,
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=test_split / (val_split + test_split),
        random_state=seed,
        shuffle=True,
    )

    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    prepare_split_dirs(output_dir)

    skipped = 0
    processed = 0

    for split_name, split_pairs in splits.items():
        LOGGER.info("Processing %s split with %d pairs", split_name, len(split_pairs))

        for image_path, mask_path in tqdm(split_pairs, desc=f"{split_name} split"):
            try:
                out_image_path = output_dir / split_name / "images" / image_path.name
                out_mask_path = output_dir / split_name / "masks" / mask_path.name
                preprocess_and_write(image_path, mask_path, out_image_path, out_mask_path, img_size)
                processed += 1
            except Exception:
                skipped += 1
                LOGGER.warning("Skipping pair: %s | %s", image_path.name, mask_path.name, exc_info=True)

    LOGGER.info("Preprocessing done: processed=%d skipped=%d", processed, skipped)
    for split in ("train", "val", "test"):
        image_count = len(list((output_dir / split / "images").glob("*.png")))
        mask_count = len(list((output_dir / split / "masks").glob("*.png")))
        LOGGER.info("%s: images=%d masks=%d", split, image_count, mask_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and split fetal ultrasound dataset")
    parser.add_argument("--source-dir", type=Path, default=Path("../dataset/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("../dataset"))
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.val_split <= 0 or args.test_split <= 0 or (args.val_split + args.test_split) >= 1:
        raise ValueError("val_split and test_split must be >0 and sum to <1")

    run_preprocessing(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
