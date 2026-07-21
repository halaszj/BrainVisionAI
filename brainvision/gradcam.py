import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def find_nested_feature_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            return layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested feature-extractor model was found.")

def find_last_conv_layer(feature_model):
    for layer in reversed(feature_model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("No 4D convolutional layer was found.")

def build_classifier_head(model, feature_model):
    feature_shape = tuple(feature_model.output.shape[1:])
    head_input = tf.keras.Input(shape=feature_shape)
    x = head_input

    collect = False
    for layer in model.layers:
        if layer.name == feature_model.name:
            collect = True
            continue
        if collect:
            x = layer(x)

    return tf.keras.Model(head_input, x)

def make_gradcam_heatmap(model, processed_batch, class_index=None):
    feature_model = find_nested_feature_model(model)
    last_conv_name = find_last_conv_layer(feature_model)
    last_conv_layer = feature_model.get_layer(last_conv_name)

    conv_model = tf.keras.Model(
        feature_model.inputs,
        [last_conv_layer.output, feature_model.output],
    )
    classifier_head = build_classifier_head(model, feature_model)

    with tf.GradientTape() as tape:
        conv_output, feature_output = conv_model(processed_batch)
        predictions = classifier_head(feature_output)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_output)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_gradients, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (maximum + tf.keras.backend.epsilon())

    return heatmap.numpy(), last_conv_name

def overlay_heatmap(original_image: Image.Image, heatmap, alpha=0.40):
    original_image = original_image.convert("RGB")
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_image = Image.fromarray(heatmap_uint8).resize(original_image.size)
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    color_map = plt.get_cmap("jet")
    colored = color_map(heatmap_array)[..., :3]
    colored = np.uint8(colored * 255)

    original = np.asarray(original_image, dtype=np.float32)
    overlay = np.clip(
        (1.0 - alpha) * original + alpha * colored,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(overlay)
