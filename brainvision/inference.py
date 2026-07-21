import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        labels = json.load(file)
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.json must contain a non-empty JSON list.")
    return labels

def load_metadata(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def prepare_image(image: Image.Image, image_size=(224, 224)):
    image = image.convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    batch = tf.keras.applications.efficientnet.preprocess_input(batch)
    return image, batch

def predict_image(model, labels, image: Image.Image, image_size=(224, 224)):
    display_image, batch = prepare_image(image, image_size)
    probabilities = model.predict(batch, verbose=0)[0]

    if len(probabilities) != len(labels):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs, but labels.json contains "
            f"{len(labels)} labels."
        )

    predicted_index = int(np.argmax(probabilities))
    return {
        "display_image": display_image,
        "batch": batch,
        "predicted_index": predicted_index,
        "predicted_class": labels[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            labels[index]: float(probabilities[index])
            for index in range(len(labels))
        },
    }
