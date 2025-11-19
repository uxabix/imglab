"""
Image transformation helpers and point-wise operations.

This module provides decorators for safe uint8 image processing and several
basic point-wise operations such as addition, subtraction, multiplication,
division, and gamma correction. It also includes a grayscale conversion helper.

All operations are intended for use with NumPy arrays representing images.
"""
from functools import wraps
import numpy as np


def apply_clip_uint8(func):
    """Clip output values to the 0–255 range and convert the result to uint8.

    This decorator is suitable for image operations where the function returns
    a NumPy array with arbitrary numeric type. The decorator ensures that:

    1. All values are clipped to the valid RGB channel range (0–255).
    2. The output array is converted to uint8.

    Args:
        func (callable): Function returning a NumPy array.

    Returns:
        callable: Wrapped function performing clipping and dtype conversion.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)

    return wrapper


def pointwise_operation(func):
    """Decorator for operations that take (array, value).

    It ensures:
    - Conversion to int16 (to avoid overflow),
    - Passing the modified array to the original function,
    - Delegation of clipping/uint8 conversion to apply_clip_uint8.

    Args:
        func (callable): Function that accepts (arr: np.ndarray, value: int/float).

    Returns:
        callable: Wrapped function with safe integer handling.
    """

    @wraps(func)
    def wrapper(array: np.ndarray, value):
        arr = array.astype(np.int16)  # Prevent overflow during arithmetic
        arr = func(arr, value)
        return arr  # Final clipping + uint8 handled by apply_clip_uint8

    return apply_clip_uint8(wrapper)


# --------------------------------------------------------------------------- #
# ----------------------------- POINTWISE OPS ------------------------------- #
# --------------------------------------------------------------------------- #

@pointwise_operation
def image_add(arr: np.ndarray, value: int) -> np.ndarray:
    """Add a constant value to each pixel."""
    return arr + value


@pointwise_operation
def image_subtract(arr: np.ndarray, value: int) -> np.ndarray:
    """Subtract a constant value from each pixel."""
    return arr - value


@pointwise_operation
def image_multiply(arr: np.ndarray, value: int | float) -> np.ndarray:
    """Multiply each pixel by a constant value."""
    return arr * value


@pointwise_operation
def image_divide(arr: np.ndarray, value: int | float) -> np.ndarray:
    """Divide each pixel by a constant value (integer division)."""
    return arr // value


@pointwise_operation
def gamma_correction(arr: np.ndarray, value: float) -> np.ndarray:
    """Apply gamma correction.

    Formula:
        output = (input / 255) ** gamma * 255

    Args:
        arr (np.ndarray): Image array as int16.
        value (float): Gamma exponent.

    Returns:
        np.ndarray: Gamma-corrected float data, later clipped and cast to uint8.
    """
    normalized = arr / 255.0
    corrected = np.power(normalized, value)
    return corrected * 255.0


# --------------------------------------------------------------------------- #
# ----------------------------- GRAY SCALE ---------------------------------- #
# --------------------------------------------------------------------------- #

@apply_clip_uint8
def gray_scale(arr: np.ndarray, method="luminosity") -> np.ndarray:
    """Convert an RGB image to grayscale using several available methods.

    Supported methods:
        - "mean":        Simple average of channels.
        - "luminosity":  Weighted sum: 0.299 R + 0.587 G + 0.114 B.
        - "RED":         Red channel only.
        - "GREEN":       Green channel only.
        - "BLUE":        Blue channel only.

    Args:
        arr (np.ndarray): Input RGB image, shape (H, W, 3).
        method (str): Grayscale method name.

    Returns:
        np.ndarray: Grayscale image replicated across 3 channels (H, W, 3).

    Raises:
        ValueError: If an invalid method is provided or image shape is incorrect.
    """

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("gray_scale expects an RGB image with shape (H, W, 3)")

    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    methods = {
        "mean": (R + G + B) / 3,
        "luminosity": 0.299 * R + 0.587 * G + 0.114 * B,
        "RED": R,
        "GREEN": G,
        "BLUE": B,
    }

    if method not in methods:
        raise ValueError(f"Invalid grayscale method: {method}")

    gray = methods[method]
    return np.stack([gray, gray, gray], axis=-1)
