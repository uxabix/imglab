import numpy as np


def convolve2d(image, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2  # padding
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    H, W = image.shape
    out = np.zeros_like(image, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            window = padded[i:i + kh, j:j + kw]
            out[i, j] = np.sum(window * kernel)
    return out


def mean_filter(image, size=3):
    kh, kw = size, size
    kernel = np.ones((kh, kw), dtype=np.float32) / (kh * kw)
    return convolve2d(image, kernel)


def sharpen_filter(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]], dtype=np.float32)
    return convolve2d(image, kernel)


def gaussian_filter(image, sigma=1, size=3):
    kh, kw = size, size
    x, y = np.mgrid[-kh // 2:kh // 2 + 1, -kw // 2:kw // 2 + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return convolve2d(image, g)


def median_filter(image, ksize=3):
    ph = ksize // 2
    padded = np.pad(image, ((ph, ph), (ph, ph)), mode='constant', constant_values=0)
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


def sobel(image, angles=("0", "90")):
    responses = []
    for ang in angles:
        kernel = SOBEL_KERNELS[ang]
        responses.append(convolve2d(image, kernel))

    magnitude = np.sqrt(sum(r * r for r in responses))
    magnitude = np.clip(magnitude, 0, 255)

    return magnitude.astype(np.uint8)


def apply_filter_rgb(image, filter_func, **kwargs):
    out = np.zeros_like(image)
    for c in range(3):
        out[:, :, c] = filter_func(image[:, :, c], **kwargs)
    return out
