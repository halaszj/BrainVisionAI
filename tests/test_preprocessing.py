from io import BytesIO

from PIL import Image

from brainvision.config import get_settings
from brainvision.preprocessing import prepare_for_inference, validate_image


def create_test_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (256, 256), "gray").save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_image_accepts_png() -> None:
    settings = get_settings()
    result = validate_image(create_test_image(), settings)
    assert result.is_valid
    assert result.width == 256
    assert result.height == 256


def test_prepare_for_inference_shape() -> None:
    settings = get_settings()
    image, tensor = prepare_for_inference(create_test_image(), settings)
    assert image.size == (256, 256)
    assert tensor.shape == (1, settings.image_size[0], settings.image_size[1], 3)
