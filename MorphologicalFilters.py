"""
Morphological filters for grayscale and RGB images.

Filters implemented:
- Dilation
- Erosion
- Opening
- Closing
- Hit-or-Miss (Thinning and Thickening)

No external libraries except NumPy are used.
"""

import numpy as np


def to_binary(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert grayscale image to binary (0 or 1)."""
    return (image > threshold).astype(np.uint8)


def binary(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Convert image to binary (0 or 255).
    Accepts RGB or grayscale.
    """
    image = _ensure_grayscale_2d(image)
    img_bin = to_binary(image, threshold)
    return (img_bin * 255).astype(np.uint8)


def _ensure_grayscale_2d(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is a 2D uint8 grayscale array.

    Accepts:
      - (H, W) grayscale
      - (H, W, 1) grayscale-with-channel
      - (H, W, 3) RGB
      - (H, W, 4) RGBA (uses first 3 channels)
    """
    img = np.asarray(image)

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            rgb = img[:, :, :3].astype(np.float32)
            gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape={img.shape}")

    # Normalize type/range to uint8
    if np.issubdtype(gray.dtype, np.floating):
        # If it's likely in 0..1 range, scale up
        if gray.size and gray.max() <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def pad_image(image: np.ndarray, se_shape):
    """Pad image with zeros according to SE size, safe for odd and even sizes."""
    kh, kw = se_shape
    ph_top = kh // 2
    ph_bottom = kh - ph_top
    pw_left = kw // 2
    pw_right = kw - pw_left
    return np.pad(image, ((ph_top, ph_bottom), (pw_left, pw_right)),
                  mode="constant", constant_values=0)


def _dilation_bin(img_bin: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Binary dilation. Expects a 2D array with values {0,1}."""
    if img_bin.ndim != 2:
        raise ValueError(f"_dilation_bin expects 2D binary image, got shape={img_bin.shape}")

    H, W = img_bin.shape
    kh, kw = se.shape
    padded = pad_image(img_bin, se.shape)
    out = np.zeros_like(img_bin)
    for i in range(H):
        for j in range(W):
            window = padded[i:i + kh, j:j + kw]
            if np.any(window[se == 1] == 1):
                out[i, j] = 1
    return out


def _erosion_bin(img_bin: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Binary erosion. Expects a 2D array with values {0,1}."""
    if img_bin.ndim != 2:
        raise ValueError(f"_erosion_bin expects 2D binary image, got shape={img_bin.shape}")

    H, W = img_bin.shape
    kh, kw = se.shape
    padded = pad_image(img_bin, se.shape)
    out = np.zeros_like(img_bin)
    for i in range(H):
        for j in range(W):
            window = padded[i:i + kh, j:j + kw]
            if np.all(window[se == 1] == 1):
                out[i, j] = 1
    return out


def dilation(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """
    Public dilation: accepts RGB/grayscale, converts to grayscale (if needed),
    binarizes, applies dilation, returns uint8 image (0..255).
    """
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)

    image = _ensure_grayscale_2d(image)
    img_bin = to_binary(image)
    out = _dilation_bin(img_bin, se)
    return (out * 255).astype(np.uint8)


def erosion(image: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """
    Public erosion: accepts RGB/grayscale, converts to grayscale (if needed),
    binarizes, applies erosion, returns uint8 image (0..255).
    """
    if se is None:
        se = np.ones((3, 3), dtype=np.uint8)

    image = _ensure_grayscale_2d(image)
    img_bin = to_binary(image)
    out = _erosion_bin(img_bin, se)
    return (out * 255).astype(np.uint8)


def dilation_wrapper(image: np.ndarray, se: np.ndarray = None) -> np.ndarray:
    """Backward-compatible wrapper name."""
    return dilation(image, se)


def erosion_wrapper(image: np.ndarray, se: np.ndarray = None) -> np.ndarray:
    """Backward-compatible wrapper name."""
    return erosion(image, se)


def opening(image: np.ndarray, se: np.ndarray = None) -> np.ndarray:
    """Opening = erosion followed by dilation."""
    return dilation(erosion(image, se), se)


def closing(image: np.ndarray, se: np.ndarray = None) -> np.ndarray:
    """Closing = dilation followed by erosion."""
    return erosion(dilation(image, se), se)


def _hit_or_miss_bin(img_bin: np.ndarray,
                     se_obj: np.ndarray,
                     se_bg: np.ndarray) -> np.ndarray:
    """
    Binary Hit-or-Miss mechanism.
    Expects img_bin with values {0,1}.
    Returns binary mask {0,1}.
    """
    eroded_obj = _erosion_bin(img_bin, se_obj)
    eroded_bg = _erosion_bin(1 - img_bin, se_bg)
    return np.logical_and(eroded_obj, eroded_bg).astype(np.uint8)


def thinning(image: np.ndarray, se_list: list, max_iter: int = 25) -> np.ndarray:
    """
    Hit-or-Miss thinning.

    - Iterative algorithm with convergence check
    - Each SE is applied on the state at iteration start
    - Stops early if no pixels are removed
    """
    image = _ensure_grayscale_2d(image)
    out = to_binary(image)  # values {0,1}

    # Iteratively applies structuring elements until convergence
    for _ in range(max_iter):
        prev = out.copy()
        change_happened = False

        for se_obj, se_bg in se_list:
            hm = _hit_or_miss_bin(prev, se_obj, se_bg)
            if hm.any():
                out[hm == 1] = 0
                change_happened = True

        if not change_happened:
            break

    return (out * 255).astype(np.uint8)


def thickening(image: np.ndarray, se_list: list, max_iter: int = 25) -> np.ndarray:
    """
    Hit-or-Miss thickening.

    - Iterative algorithm with convergence check
    - Each SE is applied on the state at iteration start
    - Stops early if no pixels are added
    """
    image = _ensure_grayscale_2d(image)
    out = to_binary(image)  # values {0,1}

    # Iteratively applies structuring elements until convergence
    for _ in range(max_iter):
        prev = out.copy()
        change_happened = False

        for se_obj, se_bg in se_list:
            hm = _hit_or_miss_bin(prev, se_obj, se_bg)
            if hm.any():
                out[hm == 1] = 1
                change_happened = True

        if not change_happened:
            break

    return (out * 255).astype(np.uint8)
