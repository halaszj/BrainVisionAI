import numpy as np
from PIL import Image

from brainvision.gradcam import overlay_heatmap


def test_overlay_heatmap_shape() -> None:
    image = Image.new("RGB", (128, 96), "gray")
    heatmap = np.ones((8, 8), dtype=np.float32)
    overlay = overlay_heatmap(image, heatmap, alpha=0.4)
    assert overlay.shape == (96, 128, 3)
    assert overlay.dtype == np.uint8
