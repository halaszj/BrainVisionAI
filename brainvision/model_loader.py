from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


class ModelNotAvailableError(FileNotFoundError):
    """Raised when the trained model artifact is unavailable."""


@lru_cache(maxsize=2)
def load_keras_model(model_path: str | Path) -> keras.Model:
    path = Path(model_path)
    if not path.exists():
        raise ModelNotAvailableError(
            f"Trained model not found at {path}. "
            "Run the training notebook and place the exported .keras model in model/."
        )

    model = keras.models.load_model(path, compile=False)
    if not isinstance(model, keras.Model):
        raise TypeError(f"Unsupported model object loaded from {path}")

    return model


def clear_model_cache() -> None:
    load_keras_model.cache_clear()
    tf.keras.backend.clear_session()
