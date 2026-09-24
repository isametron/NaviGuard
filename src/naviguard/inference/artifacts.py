"""naviguard.inference.artifacts — load & cache trained model/scaler/metadata.

The cache is thread-safe and keyed on the artifact files' mtimes, so a
retrain while the API is running is picked up on the next request.
"""

import json
import os
import threading
from dataclasses import dataclass
from typing import Optional

import joblib
import tensorflow as tf

from naviguard.config import MODEL_PATH, SCALER_PATH, META_PATH
from naviguard.errors import ArtifactsInvalidError
from naviguard.models.attention import AttentionPooling, LastValueSkip


class ArtifactsNotFoundError(RuntimeError):
    """Raised when the model/scaler haven't been trained yet."""


@dataclass
class Artifacts:
    model: tf.keras.Model
    scaler: object
    meta: dict


_cache: Optional[Artifacts] = None
_cache_key: Optional[tuple] = None
_lock = threading.Lock()


def artifacts_exist() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


def _signature() -> tuple:
    return tuple(os.path.getmtime(p) if os.path.exists(p) else None
                 for p in (MODEL_PATH, SCALER_PATH, META_PATH))


def _validate(model, meta: dict) -> None:
    shape = getattr(model, "input_shape", None)
    if not meta or not shape:
        return
    if meta.get("seq_len") not in (None, shape[1]):
        raise ArtifactsInvalidError(
            f"model_meta.json seq_len={meta['seq_len']} does not match the saved model's "
            f"input length {shape[1]}. Retrain with `naviguard train`."
        )
    out = getattr(model, "output_shape", None)
    if out and meta.get("horizon") not in (None, out[-1]):
        raise ArtifactsInvalidError(
            f"model_meta.json horizon={meta['horizon']} does not match the saved model's "
            f"output size {out[-1]}. Retrain with `naviguard train`."
        )


def get_artifacts(force_reload: bool = False) -> Artifacts:
    global _cache, _cache_key
    with _lock:
        key = _signature()
        if _cache is not None and not force_reload and key == _cache_key:
            return _cache

        missing = [p for p in (MODEL_PATH, SCALER_PATH) if not os.path.exists(p)]
        if missing:
            raise ArtifactsNotFoundError(
                f"Model artifacts not found: {missing}. "
                "Run `naviguard preprocess` then `naviguard train` first."
            )

        model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"AttentionPooling": AttentionPooling, "LastValueSkip": LastValueSkip}
        )
        scaler = joblib.load(SCALER_PATH)
        meta = {}
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
        _validate(model, meta)

        _cache = Artifacts(model=model, scaler=scaler, meta=meta)
        _cache_key = key
        return _cache
