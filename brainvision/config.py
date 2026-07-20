from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "training_config.yaml"


@dataclass(frozen=True)
class BrainVisionSettings:
    project_root: Path
    model_path: Path
    labels_path: Path
    metrics_path: Path
    image_size: tuple[int, int]
    class_names: tuple[str, ...]
    display_names: dict[str, str]
    model_version: str
    max_upload_mb: int = 15
    gradcam_alpha: float = 0.42

    @property
    def allowed_extensions(self) -> tuple[str, ...]:
        return (".jpg", ".jpeg", ".png", ".bmp")


def load_yaml_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_labels(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_settings() -> BrainVisionSettings:
    config = load_yaml_config()
    model_path = PROJECT_ROOT / config["outputs"]["best_model"]
    labels_path = PROJECT_ROOT / config["outputs"]["labels"]
    metrics_path = PROJECT_ROOT / config["outputs"]["metrics"]

    labels = load_labels(labels_path)
    class_names = tuple(labels.get("class_names", config["dataset"]["classes"]))

    return BrainVisionSettings(
        project_root=PROJECT_ROOT,
        model_path=model_path,
        labels_path=labels_path,
        metrics_path=metrics_path,
        image_size=tuple(config["training"]["image_size"]),
        class_names=class_names,
        display_names=dict(config["dataset"]["display_names"]),
        model_version=str(config["project"]["model_version"]),
    )
