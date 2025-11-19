"""
Generate example output images demonstrating filtering
and arithmetic operations on an input image.

This script loads a sample image, applies a variety of
transformations (arithmetic, grayscale, smoothing, sharpening,
Gaussian blur, median filtering, and Sobel edge detection),
and saves all results into the output directory.
"""

import os
from typing import Callable, Dict, Tuple, Any

from tqdm import tqdm

from ImageIO import load_image, save_image
import ArithmeticalOperations as ao
import Filters as f


def generate_examples(input_path: str, output_dir: str) -> None:
    """Generate example filtered images and save them to disk.

    Args:
        input_path: Path to the input image.
        output_dir: Directory where example results will be saved.

    Raises:
        FileNotFoundError: If the input image does not exist.
    """
    image = load_image(input_path)

    # Lazy operations — a dict of (callable, kwargs)
    operations: Dict[str, Tuple[Callable[..., Any], dict]] = {
        "add": (ao.image_add, {"value": 40}),
        "multiply": (ao.image_multiply, {"value": 2}),
        "subtract": (ao.image_subtract, {"value": 50}),
        "divide": (ao.image_divide, {"value": 2}),

        "gamma_mt1": (ao.gamma_correction, {"value": 3.5}),
        "gamma_lt1": (ao.gamma_correction, {"value": 0.5}),

        "gray_RED": (ao.gray_scale, {"method": "RED"}),
        "gray_GREEN": (ao.gray_scale, {"method": "GREEN"}),
        "gray_BLUE": (ao.gray_scale, {"method": "BLUE"}),
        "gray_default": (ao.gray_scale, {}),

        "mean": (f.apply_filter_rgb, {"filter_func": f.mean_filter}),
        "sharpen": (f.apply_filter_rgb, {"filter_func": f.sharpen_filter}),
        "gaussian": (f.apply_filter_rgb, {"filter_func": f.gaussian_filter}),
        "median": (f.apply_filter_rgb, {"filter_func": f.median_filter}),

        "sobel_0_90": (f.apply_filter_rgb, {"filter_func": f.sobel}),
        "sobel_all": (f.apply_filter_rgb, {"filter_func": f.sobel, "angles": ("0", "45", "90", "135")}),
        "sobel_45": (f.apply_filter_rgb, {"filter_func": f.sobel, "angles": ("45",)}),
    }

    os.makedirs(output_dir, exist_ok=True)

    # Apply each transformation one by one with progress bar
    for name, (func, params) in tqdm(operations.items(), desc="Processing examples", unit="img"):
        result = func(image, **params)
        save_image(f"{name}.jpg", result, output_dir)


if __name__ == "__main__":
    generate_examples(
        input_path="example.jpg",
        output_dir="output/examples/"
    )
