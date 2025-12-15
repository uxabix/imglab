"""
Generate example output images demonstrating filtering,
arithmetic operations, and morphological operations on an input image.

This script loads a sample image, applies a variety of
transformations (arithmetic, grayscale, smoothing, sharpening,
Gaussian blur, median filtering, Sobel edge detection, and
morphological filters), and saves all results into the output directory.
"""

import os
from typing import Callable, Dict, Tuple, Any

import numpy as np
from tqdm import tqdm

from ImageIO import load_image, save_image
import ArithmeticalOperations as ao
import Filters as f
import MorphologicalFilters as mf


def generate_examples(input_path: str, output_dir: str) -> None:
    """Generate example filtered images and save them to disk."""

    image = load_image(input_path)

    # Each filter now is called directly (decorator applies RGB logic)
    operations: Dict[str, Tuple[Callable[..., Any], dict]] = {
        # Arithmetic
        "add": (ao.image_add, {"value": 40}),
        "multiply": (ao.image_multiply, {"value": 2}),
        "subtract": (ao.image_subtract, {"value": 50}),
        "divide": (ao.image_divide, {"value": 2}),

        # Gamma correction
        "gamma_mt1": (ao.gamma_correction, {"value": 3.5}),
        "gamma_lt1": (ao.gamma_correction, {"value": 0.5}),

        # Grayscale
        "gray_RED": (ao.gray_scale, {"method": "RED"}),
        "gray_GREEN": (ao.gray_scale, {"method": "GREEN"}),
        "gray_BLUE": (ao.gray_scale, {"method": "BLUE"}),
        "gray_default": (ao.gray_scale, {}),

        # Filters
        "mean": (f.mean_filter, {}),
        "sharpen": (f.sharpen_filter, {}),
        "gaussian": (f.gaussian_filter, {}),
        "median": (f.median_filter, {}),
        "custom": (f.custom_kernel_filter, {"kernel": np.array([
            [1, -1, 0],
            [0, 1, 0],
            [0, -1, 1]])}),

        # Sobel with angles
        "sobel_0_90": (f.sobel, {"angles": ("0", "90")}),
        "sobel_all": (f.sobel, {"angles": ("0", "45", "90", "135")}),
        "sobel_45": (f.sobel, {"angles": ("45",)}),

        # Binary
        "binary":  (mf.binary, {}),
        # Morphological filters
        "morphological_dilation": (mf.dilation, {"se": np.ones((3,3), dtype=np.uint8)}),
        "morphological_erosion": (mf.erosion, {"se": np.ones((3,3), dtype=np.uint8)}),
        "morphological_opening": (mf.opening, {}),
        "morphological_closing": (mf.closing, {}),
        # Hit-or-Miss, Thinning и Thickening:
        # For demonstration use simple 3x3 SE
        "morphological_hit_or_miss": (
            mf.hit_or_miss,
            {"se_obj": np.array([[0,0,0],
                                [0,1,0],
                                [0,0,0]]),
             "se_bg": np.array([[1,1,1],
                                [1,0,1],
                                [1,1,1]])}
        ),
        "morphological_thinning": (
            mf.thinning,
            {"se_list": [
                (np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]]),
                 np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]]))
            ]}
        ),
        "morphological_thickening": (
            mf.thickening,
            {"se_list": [
                (np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]]),
                 np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]]))
            ]}
        ),
    }

    os.makedirs(output_dir, exist_ok=True)

    for name, (func, params) in tqdm(operations.items(),
                                     desc="Processing examples",
                                     unit="img"):
        result = func(image, **params)
        save_image(f"{name}.jpg", result, output_dir)


if __name__ == "__main__":
    generate_examples(
        input_path="example.jpg",
        output_dir="output/examples/"
    )
