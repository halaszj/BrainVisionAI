from datetime import datetime, timezone

from PIL import Image

from brainvision.reporting import build_prediction_pdf


def test_prediction_pdf_is_created() -> None:
    prediction = {
        "predicted_class": "glioma",
        "predicted_display_name": "Glioma",
        "confidence": 0.91,
        "probabilities": [
            {"class_name": "glioma", "display_name": "Glioma", "probability": 0.91},
            {"class_name": "meningioma", "display_name": "Meningioma", "probability": 0.05},
            {"class_name": "pituitary", "display_name": "Pituitary", "probability": 0.03},
            {"class_name": "notumor", "display_name": "No Tumor", "probability": 0.01},
        ],
        "model_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_width": 224,
        "image_height": 224,
        "source_filename": "test.png",
    }
    image = Image.new("RGB", (224, 224), "gray")
    pdf = build_prediction_pdf(
        prediction=prediction,
        original_image=image,
        overlay_image=image,
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000
