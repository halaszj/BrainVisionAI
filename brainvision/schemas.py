from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageValidationResult:
    is_valid: bool
    message: str
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    file_size_bytes: int | None = None


@dataclass(frozen=True)
class ClassProbability:
    class_name: str
    display_name: str
    probability: float


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_display_name: str
    confidence: float
    probabilities: list[ClassProbability]
    model_version: str
    timestamp: datetime
    image_width: int
    image_height: int
    source_filename: str | None = None
    heatmap: np.ndarray | None = None
    overlay: np.ndarray | None = None

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "predicted_class": self.predicted_class,
            "predicted_display_name": self.predicted_display_name,
            "confidence": float(self.confidence),
            "probabilities": [
                {
                    "class_name": item.class_name,
                    "display_name": item.display_name,
                    "probability": float(item.probability),
                }
                for item in self.probabilities
            ],
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "source_filename": self.source_filename,
        }

        if include_arrays:
            payload["heatmap"] = self.heatmap.tolist() if self.heatmap is not None else None
            payload["overlay"] = self.overlay.tolist() if self.overlay is not None else None

        return payload
