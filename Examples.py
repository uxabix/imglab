from ImageIO import load_image, save_image
import ArithmeticalOperations as ao
import Filters as f

image = load_image("example.jpg")
examples = {"add": ao.image_add(image, 40), "multiply": ao.image_multiply(image, 2),
            "subtract": ao.image_subtract(image, 50), "divide": ao.image_divide(image, 2),
            "gamma correction mt1": ao.gamma_correction(image, 3.5),
            "gamma correction lt1": ao.gamma_correction(image, 0.5), "gray scale RED": ao.gray_scale(image, "RED"),
            "gray scale GREEN": ao.gray_scale(image, "GREEN"), "gray scale BLUE": ao.gray_scale(image, "BLUE"),
            "gray scale": ao.gray_scale(image),
            "mean": f.apply_filter_rgb(image, f.mean_filter),
            "sharpen": f.apply_filter_rgb(image, f.sharpen_filter),
            "gauss": f.apply_filter_rgb(image, f.gaussian_filter),
            "median": f.apply_filter_rgb(image, f.median_filter),
            "sobel0,90": f.apply_filter_rgb(image, f.sobel),
            "sobel0,45,90,135": f.apply_filter_rgb(image, f.sobel, angles=("0", "45", "90", "135")),
            "sobel45": f.apply_filter_rgb(image, f.sobel, angles=("45",))}

path = "output/examples/"
for example in examples:
    save_image(example + ".jpg", examples[example], path)
