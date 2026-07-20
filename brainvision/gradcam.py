from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM cannot be generated."""


def find_last_convolutional_layer(model: keras.Model) -> str:
    candidates: list[str] = []

    def visit(layer: keras.layers.Layer) -> None:
        if isinstance(layer, keras.Model):
            for child in layer.layers:
                visit(child)
        output_shape = getattr(layer, "output_shape", None)
        if isinstance(output_shape, tuple) and len(output_shape) == 4:
            candidates.append(layer.name)

    for layer in model.layers:
        visit(layer)

    if not candidates:
        raise GradCAMError("No four-dimensional convolutional feature layer was found.")

    return candidates[-1]


def _find_layer_recursive(model: keras.Model, layer_name: str) -> keras.layers.Layer:
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if isinstance(layer, keras.Model):
            try:
                return _find_layer_recursive(layer, layer_name)
            except ValueError:
                pass
    raise ValueError(f"Layer '{layer_name}' was not found.")


def generate_gradcam_heatmap(
    model: keras.Model,
    image_batch: np.ndarray,
    class_index: int | None = None,
    layer_name: str | None = None,
) -> tuple[np.ndarray, int]:
    if image_batch.ndim != 4 or image_batch.shape[0] != 1:
        raise GradCAMError("Grad-CAM expects one image batch with shape (1, H, W, C).")

    selected_layer_name = layer_name or find_last_convolutional_layer(model)
    selected_layer = _find_layer_recursive(model, selected_layer_name)

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[selected_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        convolution_outputs, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, convolution_outputs)
    if gradients is None:
        raise GradCAMError(
            f"Gradients could not be calculated for layer '{selected_layer_name}'."
        )

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = convolution_outputs[0]
    weighted_maps = feature_maps * pooled_gradients
    heatmap = tf.reduce_sum(weighted_maps, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)

    return heatmap.numpy().astype(np.float32), int(class_index)


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    normalized = np.clip(heatmap, 0.0, 1.0)
    image = Image.fromarray(np.uint8(normalized * 255), mode="L")
    image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    import matplotlib

    normalized = np.clip(heatmap, 0.0, 1.0)
    colormap = matplotlib.colormaps.get_cmap("jet")
    rgba = colormap(normalized)
    return np.uint8(rgba[..., :3] * 255)


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be between 0 and 1.")

    base = np.asarray(original_image.convert("RGB"), dtype=np.float32)
    resized = resize_heatmap(heatmap, original_image.size)
    colored = colorize_heatmap(resized).astype(np.float32)

    overlay = (1.0 - alpha) * base + alpha * colored
    return np.uint8(np.clip(overlay, 0, 255))
