from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from tensorflow import keras

from .config import BrainVisionSettings, get_settings
from .gradcam import generate_gradcam_heatmap, overlay_heatmap
from .model_loader import load_keras_model
from .preprocessing import prepare_for_inference
from .schemas import ClassProbability, PredictionResult


class PredictionError(RuntimeError):
    """Raised when inference fails or model output is invalid."""


def _validate_probability_vector(
    probabilities: np.ndarray,
    class_count: int,
) -> np.ndarray:
    vector = np.asarray(probabilities, dtype=np.float32).reshape(-1)

    if vector.size != class_count:
        raise PredictionError(
            f"Model returned {vector.size} outputs, but {class_count} classes are configured."
        )
    if not np.all(np.isfinite(vector)):
        raise PredictionError("Model output contains invalid numerical values.")

    total = float(vector.sum())
    if total <= 0:
        raise PredictionError("Model output does not contain valid probabilities.")

    # Softmax models should already sum to 1. Normalize defensively.
    vector = vector / total
    return vector


def predict_image(
    source: bytes | bytearray | BinaryIO | str | Path,
    *,
    filename: str | None = None,
    settings: BrainVisionSettings | None = None,
    model: keras.Model | None = None,
    include_gradcam: bool = True,
    gradcam_layer_name: str | None = None,
) -> PredictionResult:
    settings = settings or get_settings()
    image, image_batch = prepare_for_inference(source, settings)
    model = model or load_keras_model(settings.model_path)

    raw_output = model.predict(image_batch, verbose=0)
    probabilities = _validate_probability_vector(
        raw_output[0],
        len(settings.class_names),
    )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = settings.class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    class_probabilities = [
        ClassProbability(
            class_name=class_name,
            display_name=settings.display_names.get(
                class_name,
                class_name.replace("_", " ").title(),
            ),
            probability=float(probability),
        )
        for class_name, probability in zip(settings.class_names, probabilities)
    ]
    class_probabilities.sort(key=lambda item: item.probability, reverse=True)

    heatmap = None
    overlay = None
    if include_gradcam:
        heatmap, _ = generate_gradcam_heatmap(
            model=model,
            image_batch=image_batch,
            class_index=predicted_index,
            layer_name=gradcam_layer_name,
        )
        overlay = overlay_heatmap(
            original_image=image,
            heatmap=heatmap,
            alpha=settings.gradcam_alpha,
        )

    return PredictionResult(
        predicted_class=predicted_class,
        predicted_display_name=settings.display_names.get(
            predicted_class,
            predicted_class.replace("_", " ").title(),
        ),
        confidence=confidence,
        probabilities=class_probabilities,
        model_version=settings.model_version,
        timestamp=datetime.now(timezone.utc),
        image_width=image.width,
        image_height=image.height,
        source_filename=filename,
        heatmap=heatmap,
        overlay=overlay,
    )
