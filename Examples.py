"""
Generate example output images demonstrating filtering
and arithmetic operations on an input image.

This script loads a sample image, applies a variety of
transformations (arithmetic, grayscale, smoothing, sharpening,
Gaussian blur, median filtering, and Sobel edge detection),
and saves all results into the output directory.
"""

import os
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

    examples = {
        "add": ao.image_add(image, 40),
        "multiply": ao.image_multiply(image, 2),
        "subtract": ao.image_subtract(image, 50),
        "divide": ao.image_divide(image, 2),

        "gamma_mt1": ao.gamma_correction(image, 3.5),
        "gamma_lt1": ao.gamma_correction(image, 0.5),

        "gray_RED": ao.gray_scale(image, "RED"),
        "gray_GREEN": ao.gray_scale(image, "GREEN"),
        "gray_BLUE": ao.gray_scale(image, "BLUE"),
        "gray_default": ao.gray_scale(image),

        "mean": f.apply_filter_rgb(image, f.mean_filter),
        "sharpen": f.apply_filter_rgb(image, f.sharpen_filter),
        "gaussian": f.apply_filter_rgb(image, f.gaussian_filter),
        "median": f.apply_filter_rgb(image, f.median_filter),

        "sobel_0_90": f.apply_filter_rgb(image, f.sobel),
        "sobel_all": f.apply_filter_rgb(image, f.sobel, angles=("0", "45", "90", "135")),
        "sobel_45": f.apply_filter_rgb(image, f.sobel, angles=("45",)),
    }

    # Create directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save each example
    for name, img in examples.items():
        save_image(f"{name}.jpg", img, output_dir)


if __name__ == "__main__":
    generate_examples(
        input_path="example.jpg",
        output_dir="output/examples/"
    )
