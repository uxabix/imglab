"""
Utility functions for loading and saving images in NumPy array format.

This module provides simple helpers to:
- Load an image from disk into a NumPy RGB array.
- Save a NumPy RGB array back to an image file.

All images are automatically converted to RGB to standardize processing.
"""

import os

import numpy as np
from PIL import Image


def load_image(filename: str, folder_path: str = "input") -> np.ndarray:
    """Load an image file and return it as a NumPy RGB array.

    Args:
        filename: Name of the image file to load.
        folder_path: Directory where the image is located.

    Returns:
        A NumPy array of shape (H, W, 3) in RGB format.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If Pillow fails to read the file.
    """
    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input image not found: {file_path}")

    image = Image.open(file_path).convert("RGB")
    return np.array(image)


def save_image(filename: str, array: np.ndarray, folder_path: str = "output") -> None:
    """Save a NumPy array as an image file.

    Args:
        filename: Name of the output file.
        array: NumPy array representing the image (H, W, 3), dtype uint8 recommended.
        folder_path: Directory to save the image into.

    Raises:
        ValueError: If the array is not 2D or 3D with 3 channels.
    """
    if array.ndim not in (2, 3):
        raise ValueError(f"Invalid image array shape: {array.shape}")

    # Create directory if missing
    os.makedirs(folder_path, exist_ok=True)

    out_path = os.path.join(folder_path, filename)
    Image.fromarray(array).save(out_path)
