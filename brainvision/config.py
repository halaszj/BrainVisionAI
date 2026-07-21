from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_ROOT / "model"
REPORT_DIR = APP_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "brainvision_efficientnetb0_final.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

IMAGE_SIZE = (224, 224)
MEDICAL_DISCLAIMER = (
    "Educational use only. This system is not intended to diagnose, "
    "treat, cure, or prevent any medical condition."
)
