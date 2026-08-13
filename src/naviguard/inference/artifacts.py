"""naviguard.inference.artifacts — load & cache trained model/scaler/metadata."""

import json
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import tensorflow as tf

from naviguard.config import MODEL_PATH, SCALER_PATH, META_PATH
from naviguard.models.attention import AttentionPooling


class ArtifactsNotFoundError(RuntimeError):
    """Raised when the model/scaler haven't been trained yet."""


@dataclass
class Artifacts:
    model: tf.keras.Model
    scaler: object
    meta: dict


_cache: Optional[Artifacts] = None


def artifacts_exist() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


def get_artifacts(force_reload: bool = False) -> Artifacts:
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    missing = [p for p in (MODEL_PATH, SCALER_PATH) if not os.path.exists(p)]
    if missing:
        raise ArtifactsNotFoundError(
            f"Model artifacts not found: {missing}. "
            "Run `naviguard preprocess` then `naviguard train` first."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"AttentionPooling": AttentionPooling}
    )
    scaler = joblib.load(SCALER_PATH)
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

    _cache = Artifacts(model=model, scaler=scaler, meta=meta)
    return _cache
