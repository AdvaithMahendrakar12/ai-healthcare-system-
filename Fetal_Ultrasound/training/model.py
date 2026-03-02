"""Backward-compatible exports for the refactored model module."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import build_attention_unet as AttentionUNet  # noqa: E402
from model import build_model, build_unet as UNet  # noqa: E402

__all__ = ["AttentionUNet", "UNet", "build_model"]
