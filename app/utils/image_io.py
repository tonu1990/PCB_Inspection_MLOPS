"""
Image I/O utilities.

Goal: take an uploaded image and return a NumPy RGB array (H, W, 3) dtype=uint8.
We keep this tiny and dependency-light (Pillow + numpy).
"""

from typing import Tuple
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import UploadFile


def load_upload_as_rgb_numpy(upload: UploadFile, max_pixels: int = 20_000_000) -> np.ndarray:
    """
    Read an uploaded image (FastAPI UploadFile) into an RGB NumPy array.

    - Converts to RGB (discard alpha if present).
    - Guards against gigantic images by checking pixel count.

    Returns:
        np.ndarray with shape (H, W, 3), dtype=uint8

    Raises:
        ValueError for invalid/corrupt images or oversized images.
    """
    # Read all bytes from the stream (FastAPI gives us a SpooledTemporaryFile)
    raw_bytes = upload.file.read()
    if not raw_bytes:
        raise ValueError("Empty upload or unable to read file bytes")

    try:
        with Image.open(BytesIO(raw_bytes)) as img:
            # Convert to RGB to standardize downstream processing
            img = img.convert("RGB")
            w, h = img.size
            if w * h > max_pixels:
                raise ValueError(f"Image too large ({w}x{h} > {max_pixels} pixels)")
            # Convert to NumPy (H, W, 3) uint8
            arr = np.asarray(img, dtype=np.uint8)
            return arr
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")
