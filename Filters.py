"""
Filtering operations for grayscale and RGB images.

This module implements basic convolution, smoothing, sharpening,
median filtering, Gaussian blur, and Sobel edge detection. Functions
operate on NumPy arrays and support both grayscale and RGB images.
"""

import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply 2D convolution using zero padding.

    Args:
        image: 2D grayscale image array.
        kernel: Convolution kernel.

    Returns:
        A 2D float32 array containing the filtered image.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)

    H, W = image.shape
    out = np.zeros_like(image, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            window = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(window * kernel)

    return out


def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply mean (box) filter.

    Args:
        image: Grayscale image.
        size: Kernel size.

    Returns:
        Filtered image.
    """
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return convolve2d(image, kernel)


def sharpen_filter(image: np.ndarray) -> np.ndarray:
    """Apply sharpening filter using a standard 3x3 kernel.

    Args:
        image: Grayscale image.

    Returns:
        Sharpened image.
    """
    kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32
    )
    return convolve2d(image, kernel)


def gaussian_filter(image: np.ndarray, sigma: float = 1.0, size: int = 3) -> np.ndarray:
    """Apply Gaussian blur.

    Args:
        image: Grayscale image.
        sigma: Standard deviation of the Gaussian.
        size: Kernel size.

    Returns:
        Blurred image.
    """
    x, y = np.mgrid[-size // 2:size // 2 + 1, -size // 2:size // 2 + 1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    g /= g.sum()
    return convolve2d(image, g)


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply median filter.

    Args:
        image: Grayscale image.
        ksize: Neighborhood size.

    Returns:
        Median-filtered image.
    """
    ph = ksize // 2
    padded = np.pad(image, ((ph, ph), (ph, ph)), mode="constant", constant_values=0)

    H, W = image.shape
    out = np.zeros_like(image)

    for i in range(H):
        for j in range(W):
            window = padded[i:i + ksize, j:j + ksize]
            out[i, j] = np.median(window)

    return out


SOBEL_KERNELS = {
    "0": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32),

    "45": np.array([
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]], dtype=np.float32),

    "90": np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], dtype=np.float32),

    "135": np.array([
        [2, 1, 0],
        [1, 0, -1],
        [0, -1, -2]], dtype=np.float32),
}


def sobel(image: np.ndarray, angles=("0", "90")) -> np.ndarray:
    """Apply Sobel edge detection.

    Args:
        image: Grayscale image.
        angles: Sobel directions: "0", "45", "90", "135".

    Returns:
        Magnitude of gradients as uint8 image.
    """
    responses = []

    for ang in angles:
        kernel = SOBEL_KERNELS[ang]
        responses.append(convolve2d(image, kernel))

    magnitude = np.sqrt(sum(r * r for r in responses))
    magnitude = np.clip(magnitude, 0, 255)

    return magnitude.astype(np.uint8)


def apply_filter_rgb(image: np.ndarray, filter_func, **kwargs) -> np.ndarray:
    """Apply a grayscale filter channel-wise to an RGB image.

    Args:
        image: RGB image (H × W × 3).
        filter_func: Function to apply to each channel.
        **kwargs: Additional filter parameters.

    Returns:
        RGB image with filter applied to each channel.
    """
    out = np.zeros_like(image)
    for c in range(3):
        out[:, :, c] = filter_func(image[:, :, c], **kwargs)
    return out
