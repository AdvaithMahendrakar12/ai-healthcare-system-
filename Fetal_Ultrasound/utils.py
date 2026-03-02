from __future__ import annotations

import importlib.util
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("fetal_ultrasound")


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent logging for scripts."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parent


def dataset_root() -> Path:
    return project_root() / "dataset"


def training_root() -> Path:
    return project_root() / "training"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def python_version_ok() -> Tuple[bool, str]:
    """TensorFlow 2.15 is reliably supported on Python 3.10/3.11."""
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor in (10, 11)
    message = (
        f"Python {major}.{minor} detected. Recommended: Python 3.10 or 3.11 for TensorFlow 2.15."
    )
    return ok, message


def package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def dependency_report() -> Dict[str, bool]:
    return {
        "numpy": package_available("numpy"),
        "opencv-python": package_available("cv2"),
        "tensorflow": package_available("tensorflow"),
        "pillow": package_available("PIL"),
        "scikit-learn": package_available("sklearn"),
    }


def validate_runtime_requirements(require_tensorflow: bool = True) -> None:
    ok_python, py_message = python_version_ok()
    if not ok_python:
        raise RuntimeError(py_message)

    deps = dependency_report()
    missing = [name for name, present in deps.items() if not present]
    if require_tensorflow and "tensorflow" in missing:
        raise RuntimeError(
            "TensorFlow is not installed. Install dependencies with `pip install -r requirements.txt`."
        )
    if missing:
        raise RuntimeError(f"Missing dependencies: {', '.join(sorted(missing))}")


def set_global_seed(seed: int = 42, deterministic_ops: bool = True) -> None:
    random.seed(seed)

    if package_available("numpy"):
        import numpy as np

        np.random.seed(seed)

    if package_available("tensorflow"):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic_ops and hasattr(tf.config.experimental, "enable_op_determinism"):
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                LOGGER.warning("Could not enable TensorFlow deterministic ops", exc_info=True)


def enable_gpu_memory_growth() -> None:
    if not package_available("tensorflow"):
        return

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        LOGGER.info("No GPU detected. Running on CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            LOGGER.warning("Could not set memory growth for %s", gpu, exc_info=True)

    LOGGER.info("Enabled memory growth for %d GPU(s)", len(gpus))


def get_segmentation_custom_objects() -> Dict[str, object]:
    """Return custom objects required to load/compile the segmentation model."""
    if not package_available("tensorflow"):
        raise RuntimeError("TensorFlow is required for segmentation custom objects.")

    import tensorflow.keras.backend as K

    def dice_coef(y_true, y_pred, smooth: float = 1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
        )

    def dice_coef_loss(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred)

    def focal_loss(y_true, y_pred, alpha: float = 0.75, gamma: float = 2.0):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=-1))

    def combined_loss(y_true, y_pred):
        return 0.7 * dice_coef_loss(y_true, y_pred) + 0.3 * focal_loss(y_true, y_pred)

    def iou_score(y_true, y_pred, smooth: float = 1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    def pixel_accuracy(y_true, y_pred):
        return K.mean(K.equal(y_true, K.round(y_pred)))

    def sensitivity(y_true, y_pred, smooth: float = 1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(K.round(y_pred))
        true_positives = K.sum(y_true_f * y_pred_f)
        possible_positives = K.sum(y_true_f)
        return (true_positives + smooth) / (possible_positives + smooth)

    def specificity(y_true, y_pred, smooth: float = 1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(K.round(y_pred))
        true_negatives = K.sum((1 - y_true_f) * (1 - y_pred_f))
        possible_negatives = K.sum(1 - y_true_f)
        return (true_negatives + smooth) / (possible_negatives + smooth)

    return {
        "dice_coef": dice_coef,
        "dice_coef_loss": dice_coef_loss,
        "focal_loss": focal_loss,
        "combined_loss": combined_loss,
        "iou_score": iou_score,
        "pixel_accuracy": pixel_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def mask_filename_for_image(image_name: str) -> str:
    if image_name.endswith("_HC.png"):
        return image_name.replace("_HC.png", "_HC_Annotation.png")
    stem = image_name.rsplit(".", 1)[0]
    return f"{stem}_Annotation.png"


def split_counts(dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / split / "images"
        mask_dir = dataset_dir / split / "masks"
        images = len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0
        masks = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
        counts[split] = {"images": images, "masks": masks}
    return counts


def validate_dataset_structure(dataset_dir: Path) -> None:
    missing: List[str] = []
    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            expected = dataset_dir / split / sub
            if not expected.exists():
                missing.append(str(expected))
    if missing:
        raise FileNotFoundError("Missing dataset directories:\n" + "\n".join(missing))


def collect_image_mask_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / mask_filename_for_image(image_path.name)
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def discover_model_files(model_dir: Path) -> List[Path]:
    if not model_dir.exists():
        return []

    patterns = [
        "*_best.h5",
        "*.keras",
        "model.h5",
        "best_model.keras",
        "*.pt",
        "*.pth",
    ]

    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(model_dir.glob(pattern))

    unique = sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)
    return unique


def resolve_model_path(explicit_path: Optional[str], model_dir: Path) -> Path:
    if explicit_path:
        model_path = Path(explicit_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path

    candidates = discover_model_files(model_dir)
    if not candidates:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    return candidates[0]


def infer_config_path_from_model(model_path: Path) -> Optional[Path]:
    stem = model_path.stem
    for suffix in ("_best", "_final"):
        if stem.endswith(suffix):
            config_name = f"{stem[:-len(suffix)]}_config.json"
            config_path = model_path.with_name(config_name)
            if config_path.exists():
                return config_path
    return None


def save_json(payload: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def preprocess_ultrasound_image(
    image_path: Path,
    img_size: Tuple[int, int] = (256, 256),
    apply_clahe_eq: bool = True,
    denoise: bool = True,
):
    """Match preprocessing used during dataset cleaning before inference."""
    if not package_available("cv2") or not package_available("numpy"):
        raise RuntimeError("OpenCV and NumPy are required for image preprocessing.")

    import cv2
    import numpy as np

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)

    if apply_clahe_eq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    if denoise:
        image = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image


def create_overlay(
    original_image_path: Path,
    mask_binary,
    output_path: Path,
    img_size: Tuple[int, int] = (256, 256),
) -> Tuple[Path, Path]:
    if not package_available("cv2") or not package_available("numpy"):
        raise RuntimeError("OpenCV and NumPy are required for overlay generation.")

    import cv2
    import numpy as np

    original = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not read image: {original_image_path}")

    original = cv2.resize(original, img_size, interpolation=cv2.INTER_AREA)
    mask_uint8 = (mask_binary.astype("float32") * 255).astype("uint8")

    overlay = original.copy()
    overlay[:, :, 1] = np.where(mask_uint8 > 127, 255, overlay[:, :, 1])
    blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), blended)

    mask_path = output_path.with_name(f"{output_path.stem}_mask{output_path.suffix}")
    cv2.imwrite(str(mask_path), mask_uint8)
    return output_path, mask_path


def dice_np(y_true, y_pred, smooth: float = 1e-6) -> float:
    import numpy as np

    y_true_f = y_true.flatten().astype("float32")
    y_pred_f = y_pred.flatten().astype("float32")
    intersection = float(np.sum(y_true_f * y_pred_f))
    return float((2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))


def iou_np(y_true, y_pred, smooth: float = 1e-6) -> float:
    import numpy as np

    y_true_f = y_true.flatten().astype("float32")
    y_pred_f = y_pred.flatten().astype("float32")
    intersection = float(np.sum(y_true_f * y_pred_f))
    union = float(np.sum(y_true_f) + np.sum(y_pred_f) - intersection)
    return float((intersection + smooth) / (union + smooth))


def sensitivity_np(y_true, y_pred, smooth: float = 1e-6) -> float:
    import numpy as np

    y_true_f = y_true.flatten().astype("float32")
    y_pred_f = y_pred.flatten().astype("float32")
    tp = float(np.sum(y_true_f * y_pred_f))
    positives = float(np.sum(y_true_f))
    return float((tp + smooth) / (positives + smooth))


def specificity_np(y_true, y_pred, smooth: float = 1e-6) -> float:
    import numpy as np

    y_true_f = y_true.flatten().astype("float32")
    y_pred_f = y_pred.flatten().astype("float32")
    tn = float(np.sum((1 - y_true_f) * (1 - y_pred_f)))
    negatives = float(np.sum(1 - y_true_f))
    return float((tn + smooth) / (negatives + smooth))


def pixel_accuracy_np(y_true, y_pred) -> float:
    import numpy as np

    return float(np.mean(y_true.astype("float32") == y_pred.astype("float32")))


def summarize_environment() -> Dict[str, object]:
    ok_python, py_message = python_version_ok()
    return {
        "python_ok": ok_python,
        "python_message": py_message,
        "dependencies": dependency_report(),
    }
