from functools import wraps

import numpy as np


def apply_clip_uint8(func):
    """Decorator to clip result to 0-255 and convert to uint8."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        arr = func(*args, **kwargs)
        arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)

    return wrapper


def pointwise_operation(func):
    """Decorator for operations with 'value': handle int16 conversion, then clip/uint8."""

    @wraps(func)
    def wrapper(array: np.ndarray, value):
        arr = array.astype(np.int16)  # prevent overflow
        arr = func(arr, value)  # apply the operation
        return arr  # clip/astype In apply_clip_uint8

    return apply_clip_uint8(wrapper)


@pointwise_operation
def image_add(arr, value):
    return arr + value


@pointwise_operation
def image_subtract(arr, value):
    return arr - value


@pointwise_operation
def image_multiply(arr, value):
    return arr * value


@pointwise_operation
def image_divide(arr, value):
    return arr // value


@pointwise_operation
def gamma_correction(arr, value):
    return np.power(arr / 255.0, value) * 255.0


@apply_clip_uint8
def gray_scale(arr: np.ndarray, method="luminosity") -> np.ndarray:
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    methods = {
        "mean": (R + G + B) / 3,
        "luminosity": 0.299 * R + 0.587 * G + 0.114 * B,
        "RED": R,
        "GREEN": G,
        "BLUE": B
    }

    if method not in methods:
        raise ValueError(f"Invalid method: {method}")

    gray = methods[method]
    return np.stack([gray, gray, gray], axis=-1)
