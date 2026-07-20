from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import BrainVisionSettings
from .schemas import ImageValidationResult


class ImageValidationError(ValueError):
    """Raised when an uploaded image is not suitable for inference."""


def _read_bytes(source: bytes | bytearray | BinaryIO | str | Path) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "read"):
        current_position = None
        if hasattr(source, "tell"):
            try:
                current_position = source.tell()
            except Exception:
                current_position = None
        data = source.read()
        if current_position is not None and hasattr(source, "seek"):
            try:
                source.seek(current_position)
            except Exception:
                pass
        return data
    raise TypeError("Unsupported image source type.")


def validate_image(
    source: bytes | bytearray | BinaryIO | str | Path,
    settings: BrainVisionSettings,
) -> ImageValidationResult:
    try:
        data = _read_bytes(source)
    except Exception as exc:
        return ImageValidationResult(False, f"Unable to read image data: {exc}")

    if not data:
        return ImageValidationResult(False, "The uploaded file is empty.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        return ImageValidationResult(
            False,
            f"The image exceeds the {settings.max_upload_mb} MB upload limit.",
            file_size_bytes=len(data),
        )

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return ImageValidationResult(
            False,
            f"The uploaded file is not a readable image: {exc}",
            file_size_bytes=len(data),
        )

    allowed_formats = {"JPEG", "PNG", "BMP"}
    if image_format not in allowed_formats:
        return ImageValidationResult(
            False,
            f"Unsupported image format '{image_format or 'unknown'}'. "
            "Use JPG, JPEG, PNG, or BMP.",
            width=width,
            height=height,
            mode=mode,
            file_size_bytes=len(data),
        )

    if width < 64 or height < 64:
        return ImageValidationResult(
            False,
            "The image is too small. Minimum dimensions are 64 × 64 pixels.",
            width=width,
            height=height,
            mode=mode,
            file_size_bytes=len(data),
        )

    if width > 10000 or height > 10000:
        return ImageValidationResult(
            False,
            "The image dimensions are too large for safe processing.",
            width=width,
            height=height,
            mode=mode,
            file_size_bytes=len(data),
        )

    return ImageValidationResult(
        True,
        "Image validation passed.",
        width=width,
        height=height,
        mode=mode,
        file_size_bytes=len(data),
    )


def load_rgb_image(
    source: bytes | bytearray | BinaryIO | str | Path,
    settings: BrainVisionSettings,
) -> Image.Image:
    validation = validate_image(source, settings)
    if not validation.is_valid:
        raise ImageValidationError(validation.message)

    data = _read_bytes(source)
    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB").copy()


def prepare_image_array(
    image: Image.Image,
    image_size: tuple[int, int],
) -> np.ndarray:
    resized = ImageOps.fit(
        image,
        image_size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def prepare_for_inference(
    source: bytes | bytearray | BinaryIO | str | Path,
    settings: BrainVisionSettings,
) -> tuple[Image.Image, np.ndarray]:
    image = load_rgb_image(source, settings)
    tensor = prepare_image_array(image, settings.image_size)
    return image, tensor
